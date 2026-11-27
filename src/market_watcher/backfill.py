"""Backfill price history from Kalshi candlesticks into the SQLite database.

Per contract it fetches hourly candles for the last `days` and 1-minute candles for the
last `fine_hours`, and inserts only hours that have no samples yet, so it is safe to rerun.
Each event is processed once (tracked in the `backfill` table).

    uv run python -m market_watcher.backfill --days 7
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import requests

try:
    from .config import DB_PATH, now as server_now
    from .kalshi import Client, RateLimited, candle_quote, trade_quote
except ImportError:  # running as a loose script
    from config import DB_PATH, now as server_now
    from kalshi import Client, RateLimited, candle_quote, trade_quote

DEFAULT_DB = DB_PATH


def _sleep(seconds: float, should_stop) -> bool:
    """Sleep in short slices so quitting the app isn't delayed. False means stop requested."""
    end = time.time() + seconds
    while time.time() < end:
        if should_stop():
            return False
        time.sleep(min(0.25, max(0.0, end - time.time())))
    return not should_stop()


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS backfill ("
                 "market_id TEXT PRIMARY KEY, fetched_at REAL, rows INTEGER)")


def _fetch_candles(client: Client, series: str, ticker: str, start: float, end: float,
                   interval: int, should_stop) -> list[tuple[float, float, float | None]] | None:
    """Candles as (ts, price, spread). Waits out rate limits; None if stopped or the market 404s."""
    while True:
        try:
            raw = client.candles(series, ticker, start, end, interval)
            return [q for q in map(candle_quote, raw) if q]
        except RateLimited as e:
            if not _sleep(max(e.retry_after, 5), should_stop):
                return None
        except requests.HTTPError:
            return []  # e.g. settled/archived market: nothing to fetch, move on


def _fetch_trade_points(client: Client, ticker: str, start: float,
                        should_stop) -> list[tuple[float, float]] | None:
    """Fallback history from public trades: last trade price per 5-minute bucket.
    None if stopped; [] if the market has no trades or 404s."""
    while True:
        try:
            pts = sorted(q for q in map(trade_quote, client.trades(ticker, start)) if q)
            buckets = {int(ts // 300): price for ts, price in pts}  # later trades overwrite earlier
            return [((b + 1) * 300.0, p) for b, p in sorted(buckets.items())]
        except RateLimited as e:
            if not _sleep(max(e.retry_after, 5), should_stop):
                return None
        except requests.HTTPError:
            return []


def backfill_event(client: Client, db: Path, event_id: str, series: str, tickers: list[str],
                   days: float, fine_hours: float, should_stop=lambda: False) -> int:
    now = server_now()
    fine_start = now - fine_hours * 3600
    segments = [(now - days * 86400, fine_start, 60), (fine_start, now, 1)]

    rows: list[tuple] = []
    for ticker in tickers:
        got_candles = False
        for start, end, interval in segments:
            if should_stop():
                return 0
            got = _fetch_candles(client, series, ticker, start, end, interval, should_stop)
            if got is None:
                return 0
            got_candles = got_candles or bool(got)
            rows += [(ts, event_id, ticker, price, spread) for ts, price, spread in got]
        if not got_candles:  # candlesticks empty: rebuild from the trade tape instead
            pts = _fetch_trade_points(client, ticker, now - days * 86400, should_stop)
            if pts is None:
                return 0
            rows += [(ts, event_id, ticker, price, None) for ts, price in pts]

    with closing(sqlite3.connect(db, timeout=10)) as conn, conn:
        ensure_table(conn)
        have = set(conn.execute(
            "SELECT DISTINCT contract_id, CAST(ts / 3600 AS INTEGER) FROM samples "
            "WHERE market_id = ? AND ts >= ?", (event_id, now - days * 86400 - 3600)))
        keep = [r for r in rows if (r[2], int(r[0] // 3600)) not in have]
        conn.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?)", keep)
        conn.execute("INSERT OR REPLACE INTO backfill VALUES (?, ?, ?)", (event_id, time.time(), len(keep)))
    return len(keep)


def backfill_missing(client: Client, db: Path = DEFAULT_DB, days: float = 7, fine_hours: float = 24,
                     first: str | None = None, on_progress=None, should_stop=lambda: False) -> None:
    """Backfill every tracked event that hasn't been backfilled yet."""
    db = Path(db)
    with closing(sqlite3.connect(db, timeout=10)) as conn, conn:
        ensure_table(conn)
    with closing(sqlite3.connect(db, timeout=10)) as conn:
        # events that added nothing are retried after 6 hours rather than being marked done forever
        done = {r[0] for r in conn.execute(
            "SELECT market_id FROM backfill WHERE rows > 0 OR fetched_at > ?", (time.time() - 6 * 3600,))}
        todo = [(mid, series) for mid, series in
                conn.execute("SELECT id, COALESCE(series, '') FROM markets ORDER BY id") if mid not in done]
        contracts: dict[str, list[str]] = {}
        for mid, cid in conn.execute("SELECT market_id, id FROM contracts"):
            contracts.setdefault(mid, []).append(cid)

    todo.sort(key=lambda t: t[0] != first)  # the event on screen goes first (stable sort)
    total = len(todo)
    added = 0
    for i, (mid, series) in enumerate(todo, 1):
        if should_stop():
            return
        try:
            added += backfill_event(client, db, mid, series or mid.split("-")[0], contracts.get(mid, []),
                           days, fine_hours, should_stop)
        except requests.RequestException:
            pass  # network trouble: skip, retried next launch
        if on_progress:
            on_progress(i, total, mid, added)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--days", type=float, default=7, help="hourly history to fetch")
    ap.add_argument("--fine-hours", type=float, default=24, help="1-minute history to fetch")
    a = ap.parse_args()
    backfill_missing(Client(), a.db, a.days, a.fine_hours,
                     on_progress=lambda i, n, m, a: print(f"{i}/{n} {m} ({a} rows added so far)", flush=True))


if __name__ == "__main__":
    main()