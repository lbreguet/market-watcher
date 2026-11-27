"""Buy/sell signals from the SQLite price history.

Statistical signals, not financial advice. The model is evaluated on a held-out
later slice of time after paying the bid/ask spread and an estimated trading fee,
and the report says so plainly when it finds no edge.

    uv run python -m market_watcher.signals --days 14 --horizon 6
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import DB_PATH, now as server_now
except ImportError:  # running as a loose script
    from config import DB_PATH, now as server_now

BAR = 300          # seconds per bar (5 minutes)
FEE_RATE = 0.07    # ASSUMPTION: Kalshi taker fee ~ 0.07 * p * (1 - p) per contract per trade.
                   # Check Kalshi's current fee schedule and adjust.
FEATURES = ["level", "d1h", "d6h", "d24h", "vol6h", "z24"]
MIN_ROWS = 3000    # labelled rows needed before we train anything
DEFAULT_DB = DB_PATH


@dataclass
class Report:
    verdict: str
    stats: dict = field(default_factory=dict)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, market_id: int | None = None, top: int = 6) -> str:
        lines = [self.verdict]
        s = self.signals
        if market_id is not None and not s.empty:
            s = s[s["mid"] == market_id]
        if s.empty:
            lines.append("No current signals for this selection.")
        else:
            for r in s.head(top).itertuples():
                lines.append(f"{r.signal:<4} {r.contract[:22]:<22} @ {r.price:.3f} "
                             f"(up {r.p_up:.0%} / down {r.p_down:.0%})")
        return "\n".join(lines)

    def text(self, top: int = 12) -> str:
        out = [self.verdict]
        if self.stats:
            out.append("  " + "  ".join(f"{k}={v}" for k, v in self.stats.items()))
        if not self.signals.empty:
            out.append("")
            out.append(self.signals.head(top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return "\n".join(out)


# ---------------------------------------------------------------- data
def load_bars(db: Path, days: float) -> pd.DataFrame:
    since = server_now() - days * 86400
    with closing(sqlite3.connect(db, timeout=10)) as conn:
        return pd.read_sql_query(
            "SELECT contract_id AS cid, MIN(market_id) AS mid, "
            "CAST(ts / ? AS INTEGER) AS bar, AVG(price) AS price, AVG(spread) AS spread "
            "FROM samples WHERE ts >= ? GROUP BY contract_id, bar",
            conn, params=(BAR, since),
        )


def load_names(db: Path) -> tuple[dict, dict]:
    with closing(sqlite3.connect(db, timeout=10)) as conn:
        markets = dict(conn.execute("SELECT id, name FROM markets"))
        contracts = dict(conn.execute("SELECT id, name FROM contracts"))
    return markets, contracts


def featurize(g: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    p = g.set_index("bar")["price"].sort_index()
    p = p.reindex(range(int(p.index.min()), int(p.index.max()) + 1)).ffill(limit=12)  # bridges gaps up to 1h (backfilled data is hourly)
    f = pd.DataFrame({"price": p, "level": p})
    f["spread"] = g.set_index("bar")["spread"].sort_index().reindex(p.index).ffill(limit=12)
    f["d1h"] = p - p.shift(12)
    f["d6h"] = p - p.shift(72)
    f["d24h"] = p - p.shift(288)
    f["vol6h"] = p.diff().rolling(72).std()
    f["z24"] = (p - p.rolling(288).mean()) / (p.rolling(288).std() + 1e-3)
    f["fwd"] = p.shift(-horizon_bars) - p
    f.index.name = "bar"
    return f.reset_index()


def net_pnl(move: np.ndarray, price: np.ndarray, spread: np.ndarray, fallback_spread: float) -> np.ndarray:
    """Profit per contract for a round trip that gains `move` at mid-price: you pay the
    bid/ask spread (buy at the ask, sell at the bid) plus an estimated fee on each leg."""
    spread = np.where(np.isnan(spread), fallback_spread, spread)
    fee = FEE_RATE * price * (1 - price)
    return move - spread - 2 * fee


# ---------------------------------------------------------------- analysis
def analyze(db: Path = DEFAULT_DB, days: float = 14, horizon_hours: float = 6,
            min_move: float = 0.02, cost: float = 0.02, threshold: float = 0.4) -> Report:
    # `cost` is only the fallback spread for rows with no recorded bid/ask.
    h = max(1, int(horizon_hours * 3600 / BAR))
    if not Path(db).exists():
        return Report(f"No database at {db} yet. Run the terminal for a while first.")
    bars = load_bars(db, days)
    if bars.empty:
        return Report("Not enough history yet: no samples in the database.")

    frames = []
    for (cid, mid), g in bars.groupby(["cid", "mid"]):
        if len(g) < 288 + h + 50:
            continue
        frames.append(featurize(g, h).assign(cid=cid, mid=mid))
    if not frames:
        return Report("Not enough history yet: contracts need roughly 1.5+ days of samples "
                      "(24h lookback plus the forecast horizon).")
    data = pd.concat(frames, ignore_index=True)

    labelled = data.dropna(subset=FEATURES + ["fwd"]).copy()
    labelled["y"] = np.where(labelled["fwd"] > min_move, 1, np.where(labelled["fwd"] < -min_move, -1, 0))
    if len(labelled) < MIN_ROWS:
        return Report(f"Not enough history yet: {len(labelled)} usable rows, need {MIN_ROWS}+.")

    # Chronological split with a purge gap so overlapping forecast windows can't leak.
    cut = labelled["bar"].quantile(0.7)
    train = labelled[labelled["bar"] < cut - h]
    test = labelled[labelled["bar"] >= cut]
    if train["y"].nunique() < 2 or len(test) < 500:
        return Report("Not enough varied data to evaluate a model yet.")

    def fit(df: pd.DataFrame):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(df[FEATURES], df["y"])

    def probs(model, X):
        P = model.predict_proba(X[FEATURES])
        idx = {c: i for i, c in enumerate(model.classes_)}
        up = P[:, idx[1]] if 1 in idx else np.zeros(len(X))
        down = P[:, idx[-1]] if -1 in idx else np.zeros(len(X))
        return up, down

    # ---- out-of-sample evaluation
    model = fit(train)
    up, down = probs(model, test)
    buy = (up >= threshold) & (up > down)
    sell = (down >= threshold) & (down > up)
    fwd = test["fwd"].to_numpy()
    px, sp = test["price"].to_numpy(), test["spread"].to_numpy()
    pnl = np.concatenate([net_pnl(fwd[buy], px[buy], sp[buy], cost),
                          net_pnl(-fwd[sell], px[sell], sp[sell], cost)])
    n = len(pnl)
    stats = {"train_rows": len(train), "test_rows": len(test), "trades": n}
    if n >= 30:
        mean = float(pnl.mean())
        t = mean / (pnl.std(ddof=1) / np.sqrt(n) + 1e-12)
        stats.update(avg_net_per_share=round(mean, 4), hit_rate=round(float((pnl > 0).mean()), 3),
                     t_stat=round(float(t), 2))
        has_edge = n >= 200 and mean > 0 and t > 2
    else:
        has_edge = False

    if has_edge:
        verdict = ("Possible edge: the model made money out-of-sample after fees and slippage. "
                   "Trades overlap in time, so the t-stat is optimistic; treat as a lead, not proof.")
    else:
        verdict = ("No reliable edge found: the model did not beat fees and slippage on held-out data. "
                   "Treat the signals below as noise.")

    # ---- live signals from the latest bar, using a model refit on everything labelled
    live_model = fit(labelled)
    now_bar = int(server_now() // BAR)
    cur = (data.dropna(subset=FEATURES).sort_values("bar").groupby("cid").tail(1))
    cur = cur[cur["bar"] >= now_bar - 6].copy()  # stale contracts (no recent data) are skipped
    if cur.empty:
        return Report(verdict + " (No contracts have fresh data right now.)", stats)
    up, down = probs(live_model, cur)
    cur["p_up"], cur["p_down"] = up, down
    cur["signal"] = np.where((up >= threshold) & (up > down), "BUY",
                             np.where((down >= threshold) & (down > up), "SELL", "HOLD"))
    markets, contracts = load_names(db)
    cur["market"] = cur["mid"].map(markets)
    cur["contract"] = cur["cid"].map(contracts)
    cur["conf"] = np.maximum(up, down)
    sig = (cur[cur["signal"] != "HOLD"].sort_values("conf", ascending=False)
           [["mid", "market", "contract", "price", "signal", "p_up", "p_down"]]
           .reset_index(drop=True))
    return Report(verdict, stats, sig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--days", type=float, default=14, help="history window to use")
    ap.add_argument("--horizon", type=float, default=6, help="forecast horizon in hours")
    ap.add_argument("--min-move", type=float, default=0.02, help="price move that counts as up/down")
    ap.add_argument("--cost", type=float, default=0.02, help="spread assumed when none is recorded")
    a = ap.parse_args()
    print(analyze(a.db, a.days, a.horizon, a.min_move, a.cost).text())


if __name__ == "__main__":
    main()