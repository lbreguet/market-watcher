"""Mini terminal for Kalshi markets: event list + live chart, history kept in SQLite."""
from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input
from textual.worker import get_current_worker
from textual_plotext import PlotextPlot

try:
    from .backfill import backfill_missing
    from . import config
    from .config import DB_PATH
    from .kalshi import Client, RateLimited, Tracker
    from .signals import analyze
except ImportError:  # running as a loose script
    from backfill import backfill_missing
    import config
    from config import DB_PATH
    from kalshi import Client, RateLimited, Tracker
    from signals import analyze

POLL_SECONDS = 5.0
DISCOVERY_SECONDS = 600.0   # re-rank which events are tracked every 10 minutes
TOP_EVENTS = int(os.environ.get("MW_EVENTS", 40))
MIN_AGE_HOURS = float(os.environ.get("MW_MIN_AGE_HOURS", 0))  # e.g. 6 skips markets opened today
SERIES = [s.strip() for s in os.environ.get("MW_SERIES", "").split(",") if s.strip()]
TOP_CONTRACTS = 6    # lines drawn per chart
MAX_POINTS = 500     # downsample long histories to this many points per line
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
    id TEXT PRIMARY KEY, name TEXT NOT NULL, series TEXT);
CREATE TABLE IF NOT EXISTS contracts (
    id TEXT PRIMARY KEY, market_id TEXT NOT NULL, name TEXT NOT NULL, price REAL, spread REAL);
CREATE TABLE IF NOT EXISTS samples (
    ts REAL NOT NULL, market_id TEXT NOT NULL, contract_id TEXT NOT NULL,
    price REAL NOT NULL, spread REAL);
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
        # WAL needs shared memory, which WSL's mounted Windows drives (/mnt/c) don't reliably provide
        on_windows_drive = str(self.path.resolve()).startswith("/mnt/")
        conn.execute(f"PRAGMA journal_mode={'DELETE' if on_windows_drive else 'WAL'}")
        return conn

    def save(self, ts: float, rows: list[tuple]) -> None:
        """rows: (market_id, market_name, series, contract_id, contract_name, price, spread)"""
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO markets VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "name = excluded.name, series = excluded.series",
                {(r[0], r[1], r[2]) for r in rows},
            )
            conn.executemany(
                "INSERT INTO contracts VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "market_id = excluded.market_id, name = excluded.name, "
                "price = excluded.price, spread = excluded.spread",
                [(r[3], r[0], r[4], r[5], r[6]) for r in rows],
            )
            conn.executemany(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                [(ts, r[0], r[3], r[5], r[6]) for r in rows],
            )

    def load_state(self):
        """Names and latest prices, so the list is populated before the first fetch."""
        with closing(self._connect()) as conn:
            markets = dict(conn.execute("SELECT id, name FROM markets"))
            contract_names: dict[str, str] = {}
            latest: dict[str, dict[str, float]] = defaultdict(dict)
            for cid, mid, name, price in conn.execute(
                "SELECT id, market_id, name, price FROM contracts"
            ):
                contract_names[cid] = name
                if price is not None:
                    latest[mid][cid] = price
        return markets, contract_names, latest

    def history(self, market_id: str, since: float | None) -> dict[str, list[tuple[float, float]]]:
        query = "SELECT contract_id, ts, price FROM samples WHERE market_id = ?"
        args: list = [market_id]
        if since is not None:
            query += " AND ts >= ?"
            args.append(since)
        query += " ORDER BY ts"
        out: dict[str, list[tuple[float, float]]] = defaultdict(list)
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
    TITLE = "market-watcher (Kalshi)"
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
        Binding("s", "signals", "Signals"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.store = Store(DB_PATH)
        self.client = Client()
        self.tracker = Tracker(self.client, top_events=TOP_EVENTS, series=SERIES, min_age_hours=MIN_AGE_HOURS)
        self.market_names: dict[str, str] = {}
        self.contract_names: dict[str, str] = {}
        self.latest: dict[str, dict[str, float]] = defaultdict(dict)
        self.active: set[str] | None = None  # events in the newest live snapshot
        self.selected: str | None = None
        self.filter_text = ""
        self.paused = False
        self.fetching = False
        self.backoff_until = 0.0
        self.next_discovery = 0.0
        self.backfill_pending = False
        self.window_idx = 1  # start on 1h
        self.report = None
        self.report_ts = 0.0

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
        if self.paused or self.fetching or time.time() < self.backoff_until:
            return
        self.fetching = True
        self.fetch()

    @work(thread=True, exclusive=True, group="fetch")
    def fetch(self) -> None:
        try:
            if time.time() >= self.next_discovery:
                self.call_from_thread(self.set_status, "discovering markets...")
                if self.tracker.refresh():
                    self.backfill_pending = True
                self.next_discovery = time.time() + DISCOVERY_SECONDS
            ts = config.now()  # server time, immune to a drifting system clock
            rows = self.tracker.snapshot()
            if not rows:
                st = self.tracker.scan_stats
                self.call_from_thread(self.set_status, (
                    f"no priced markets: scanned {st.get('scanned', 0)} open markets, "
                    f"{st.get('priced', 0)} with usable quotes, {st.get('events', 0)} events"))
                return
            self.store.save(ts, rows)
        except RateLimited as e:
            wait = max(e.retry_after, 5)
            self.backoff_until = time.time() + wait
            self.call_from_thread(self.set_status, f"rate limited, retrying in {wait:.0f}s")
            return
        except Exception as e:  # network or db hiccups shouldn't kill the UI
            self.call_from_thread(self.set_status, f"error: {e}")
            return
        finally:
            self.fetching = False
        self.call_from_thread(self.apply, rows, ts)

    def apply(self, rows: list[tuple], ts: float) -> None:
        self.active = set()
        for mid, mname, _series, cid, cname, price, _spread in rows:
            self.active.add(mid)
            self.market_names[mid] = mname
            self.contract_names[cid] = cname
            self.latest[mid][cid] = price

        table = self.query_one(DataTable)
        if {k.value for k in table.rows} != set(self.visible_markets()):
            self.rebuild_table()
        else:
            for key in list(table.rows):
                leader, price = self.leader(key.value)
                table.update_cell(key, "leader", leader)
                table.update_cell(key, "price", price)
        self.redraw()
        skew = config.OFFSET
        note = (f" | system clock {'behind' if skew > 0 else 'ahead of'} Kalshi by {abs(skew) / 3600:.1f}h"
                if abs(skew) > 120 else "")
        if self.tracker.scan_stats.get("relaxed"):
            note += " | nothing older than the age filter was found, showing newest"
        self.set_status(f"updated {stamp(ts)} | tracking {len(self.active)} events{note}")
        if self.backfill_pending:  # new events were discovered: fill their history
            self.backfill_pending = False
            self.run_backfill()

    def leader(self, mid: str) -> tuple[str, str]:
        latest = self.latest.get(mid)
        if not latest:
            return "-", "-"
        cid = max(latest, key=latest.get)
        return self.contract_names[cid][:16], f"{latest[cid]:.3f}"

    # ---------- table ----------
    def visible_markets(self) -> list[str]:
        needle = self.filter_text.lower()
        ids = [m for m, n in self.market_names.items()
               if (self.active is None or m in self.active) and (not needle or needle in n.lower())]
        return sorted(ids, key=lambda m: self.market_names[m].lower())

    def rebuild_table(self) -> None:
        table = self.query_one(DataTable)
        keep = self.selected
        table.clear()
        for mid in self.visible_markets():
            leader, price = self.leader(mid)
            table.add_row(self.market_names[mid], leader, price, key=mid)
        if keep is not None:
            try:
                table.move_cursor(row=table.get_row_index(keep))
            except Exception:  # selected market is filtered out
                pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.selected = event.row_key.value
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
        top: list[str] = []
        if mid is not None:
            since = config.now() - span if span else None
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
                     [p for _, p in pts], label=self.contract_names.get(cid, cid))

        prices = [p for cid in top for _, p in series[cid]]
        lo, hi = min(prices), max(prices)
        pad = max(0.01, (hi - lo) * 0.1)
        plt.ylim(max(0, lo - pad), min(1, hi + pad))

        n = len(series[top[0]])
        plt.title(f"{name[:60]} [{label}, {n} pts]")
        plt.xlabel("time")
        plt.ylabel("price ($)")
        chart.refresh()

    # ---------- backfill ----------
    @work(thread=True, exclusive=True, group="backfill")
    def run_backfill(self) -> None:
        worker = get_current_worker()

        def progress(done: int, total: int, mid: str, added: int) -> None:
            self.call_from_thread(self.on_backfill, done, total, mid, added)

        try:
            backfill_missing(self.client, DB_PATH, first=self.selected, on_progress=progress,
                             should_stop=lambda: worker.is_cancelled)
        except Exception as e:
            self.call_from_thread(self.set_status, f"backfill error: {e}")

    def on_backfill(self, done: int, total: int, mid: str, added: int) -> None:
        if done < total:
            self.set_status(f"backfilling history {done}/{total} ({added} rows)")
        elif added:
            self.set_status(f"history backfill complete ({added} rows added)")
        else:
            self.set_status("backfill added 0 rows: run the kalshi check for candlestick diagnostics")
        if mid == self.selected:
            self.redraw()

    # ---------- signals ----------
    @work(thread=True, exclusive=True, group="signals")  # own group so it doesn't cancel fetch
    def run_signals(self) -> None:
        try:
            if self.report is None or time.time() - self.report_ts > 600:
                self.report = analyze(DB_PATH)
                self.report_ts = time.time()
            text = self.report.summary(self.selected)
        except Exception as e:
            text = f"signals failed: {e}"
        self.call_from_thread(self.notify, text, title="Signals", timeout=20)

    # ---------- actions ----------
    def set_status(self, text: str) -> None:
        self.sub_title = text + (" [PAUSED]" if self.paused else "")

    def action_signals(self) -> None:
        self.notify("Training model...", timeout=3)
        self.run_signals()

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