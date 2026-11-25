"""Mini terminal for PredictIt markets: market list + live chart, history kept in SQLite."""
from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input
from textual_plotext import PlotextPlot

API = "https://www.predictit.org/api/marketdata/all"
POLL_SECONDS = 30.0
TOP_CONTRACTS = 6   # lines drawn per chart
MAX_POINTS = 500    # downsample long histories to this many points per line
DB_PATH = Path(os.environ.get("MW_DB", Path.home() / ".local/share/market-watcher/history.db"))
WINDOWS = [("15m", 900), ("1h", 3600), ("6h", 21600), ("24h", 86400), ("all", None)]

try:
    TZ: ZoneInfo | None = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:  # tzdata missing: fall back to the system timezone
    TZ = None


def stamp(ts: float, fmt: str = "%H:%M:%S") -> str:
    return datetime.fromtimestamp(ts, TZ).strftime(fmt)


# ---------------------------------------------------------------- storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY, market_id INTEGER NOT NULL, name TEXT NOT NULL, price REAL);
CREATE TABLE IF NOT EXISTS samples (
    ts REAL NOT NULL, market_id INTEGER NOT NULL, contract_id INTEGER NOT NULL, price REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_samples_market_ts ON samples (market_id, ts);
"""


class Store:
    """Thin SQLite wrapper. A fresh connection per call keeps it thread-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, ts: float, rows: list[tuple]) -> None:
        """rows: (market_id, market_name, contract_id, contract_name, price)"""
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO markets VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name",
                {(r[0], r[1]) for r in rows},
            )
            conn.executemany(
                "INSERT INTO contracts VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "market_id = excluded.market_id, name = excluded.name, price = excluded.price",
                [(r[2], r[0], r[3], r[4]) for r in rows],
            )
            conn.executemany(
                "INSERT INTO samples VALUES (?, ?, ?, ?)",
                [(ts, r[0], r[2], r[4]) for r in rows],
            )

    def load_state(self):
        """Names and latest prices, so the list is populated before the first fetch."""
        with closing(self._connect()) as conn:
            markets = dict(conn.execute("SELECT id, name FROM markets"))
            contract_names: dict[int, str] = {}
            latest: dict[int, dict[int, float]] = defaultdict(dict)
            for cid, mid, name, price in conn.execute(
                "SELECT id, market_id, name, price FROM contracts"
            ):
                contract_names[cid] = name
                if price is not None:
                    latest[mid][cid] = price
        return markets, contract_names, latest

    def history(self, market_id: int, since: float | None) -> dict[int, list[tuple[float, float]]]:
        query = "SELECT contract_id, ts, price FROM samples WHERE market_id = ?"
        args: list = [market_id]
        if since is not None:
            query += " AND ts >= ?"
            args.append(since)
        query += " ORDER BY ts"
        out: dict[int, list[tuple[float, float]]] = defaultdict(list)
        with closing(self._connect()) as conn:
            for cid, ts, price in conn.execute(query, args):
                out[cid].append((ts, price))
        for cid, pts in out.items():
            step = max(1, len(pts) // MAX_POINTS)
            if step > 1:
                last = pts[-1]
                out[cid] = pts[::step]
                if out[cid][-1] != last:
                    out[cid].append(last)
        return out


# ---------------------------------------------------------------- app
class MarketWatcher(App):
    TITLE = "market-watcher"
    CSS = """
    #left { width: 50%; }
    #filter { margin: 0 0 1 0; }
    #table { height: 1fr; }
    #chart { width: 1fr; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("escape", "focus_table", "List"),
        Binding("w", "cycle_window", "Time window"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.store = Store(DB_PATH)
        self.market_names: dict[int, str] = {}
        self.contract_names: dict[int, str] = {}
        self.latest: dict[int, dict[int, float]] = defaultdict(dict)
        self.selected: int | None = None
        self.filter_text = ""
        self.paused = False
        self.backoff_until = 0.0
        self.window_idx = 1  # start on 1h

    # ---------- layout ----------
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="/ to filter markets", id="filter")
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield PlotextPlot(id="chart")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("Market", key="market", width=34)
        table.add_column("Leader", key="leader", width=16)
        table.add_column("Price", key="price")
        table.focus()
        self.market_names, self.contract_names, self.latest = self.store.load_state()
        self.rebuild_table()
        self.set_status(f"loaded {len(self.market_names)} markets from {DB_PATH.name}")
        self.poll()
        self.set_interval(POLL_SECONDS, self.poll)

    # ---------- data ----------
    def poll(self) -> None:
        if not self.paused and time.time() >= self.backoff_until:
            self.fetch()

    @work(thread=True, exclusive=True)
    def fetch(self) -> None:
        try:
            r = requests.get(API, timeout=10)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                self.backoff_until = time.time() + wait
                self.call_from_thread(self.set_status, f"rate limited, retrying in {wait}s")
                return
            r.raise_for_status()
            markets = r.json()["markets"]
            ts = time.time()
            rows = []
            for m in markets:
                for c in m["contracts"]:
                    bid, ask = c.get("bestSellYesCost"), c.get("bestBuyYesCost")
                    price = (bid + ask) / 2 if bid is not None and ask is not None \
                        else c.get("lastTradePrice")
                    if price is None:
                        continue
                    rows.append((m["id"], str(m["shortName"]), c["id"], str(c["shortName"]), float(price)))
            self.store.save(ts, rows)
        except Exception as e:  # network or db hiccups shouldn't kill the UI
            self.call_from_thread(self.set_status, f"error: {e}")
            return
        self.call_from_thread(self.apply, rows, ts)

    def apply(self, rows: list[tuple], ts: float) -> None:
        for mid, mname, cid, cname, price in rows:
            self.market_names[mid] = mname
            self.contract_names[cid] = cname
            self.latest[mid][cid] = price

        table = self.query_one(DataTable)
        if {int(k.value) for k in table.rows} != set(self.visible_markets()):
            self.rebuild_table()
        else:
            for key in list(table.rows):
                leader, price = self.leader(int(key.value))
                table.update_cell(key, "leader", leader)
                table.update_cell(key, "price", price)
        self.redraw()
        self.set_status(f"updated {stamp(ts)} | {len(self.market_names)} markets")

    def leader(self, mid: int) -> tuple[str, str]:
        latest = self.latest.get(mid)
        if not latest:
            return "-", "-"
        cid = max(latest, key=latest.get)
        return self.contract_names[cid][:16], f"{latest[cid]:.3f}"

    # ---------- table ----------
    def visible_markets(self) -> list[int]:
        needle = self.filter_text.lower()
        ids = [m for m, n in self.market_names.items() if not needle or needle in n.lower()]
        return sorted(ids, key=lambda m: self.market_names[m].lower())

    def rebuild_table(self) -> None:
        table = self.query_one(DataTable)
        keep = self.selected
        table.clear()
        for mid in self.visible_markets():
            leader, price = self.leader(mid)
            table.add_row(self.market_names[mid], leader, price, key=str(mid))
        if keep is not None:
            try:
                table.move_cursor(row=table.get_row_index(str(keep)))
            except Exception:  # selected market is filtered out
                pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.selected = int(event.row_key.value)
            self.redraw()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value
        self.rebuild_table()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one(DataTable).focus()

    # ---------- chart ----------
    def redraw(self) -> None:
        chart = self.query_one(PlotextPlot)
        plt = chart.plt
        plt.clear_figure()
        mid = self.selected
        label, span = WINDOWS[self.window_idx]
        name = self.market_names.get(mid, "") if mid is not None else ""

        series = {}
        top: list[int] = []
        if mid is not None:
            since = time.time() - span if span else None
            series = self.store.history(mid, since)
            latest = self.latest.get(mid, {})
            top = sorted((c for c in series if c in latest), key=lambda c: latest[c], reverse=True)
            top = top[:TOP_CONTRACTS]

        if not top:  # nothing to draw (no samples yet, or none in this window)
            plt.title(f"{name} [{label}]: no data yet")
            chart.refresh()
            return

        short = span is not None and span <= 86400
        plt.date_form(input_form="Y-m-d H:M:S", output_form="H:M:S" if short else "m/d H:M")
        for cid in top:
            pts = series[cid]
            plt.plot([stamp(t, "%Y-%m-%d %H:%M:%S") for t, _ in pts],
                     [p for _, p in pts], label=self.contract_names.get(cid, str(cid)))

        prices = [p for cid in top for _, p in series[cid]]
        lo, hi = min(prices), max(prices)
        pad = max(0.01, (hi - lo) * 0.1)
        plt.ylim(max(0, lo - pad), min(1, hi + pad))

        n = len(series[top[0]])
        plt.title(f"{name} [{label}, {n} pts]")
        plt.xlabel("time")
        plt.ylabel("price ($)")
        chart.refresh()

    # ---------- actions ----------
    def set_status(self, text: str) -> None:
        self.sub_title = text + (" [PAUSED]" if self.paused else "")

    def action_focus_filter(self) -> None:
        self.query_one(Input).focus()

    def action_focus_table(self) -> None:
        self.query_one(DataTable).focus()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.set_status("paused" if self.paused else "resumed")

    def action_cycle_window(self) -> None:
        self.window_idx = (self.window_idx + 1) % len(WINDOWS)
        self.redraw()

    def action_cursor_down(self) -> None:
        self.query_one(DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(DataTable).action_cursor_up()


def main() -> None:
    MarketWatcher().run()


if __name__ == "__main__":
    main()