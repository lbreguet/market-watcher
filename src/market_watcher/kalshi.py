"""Kalshi data source (public REST endpoints, no API key needed).

Terminology: a Kalshi *event* is what the terminal calls a "market" (the row you pick),
and a Kalshi *market* is a "contract" (one line on the chart).

Run a live connectivity check any time:
    uv run python -m market_watcher.kalshi
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
from email.utils import parsedate_to_datetime

try:
    from . import config
except ImportError:  # running as a loose script
    import config

BASE = os.environ.get("KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2")
HEADERS = {"User-Agent": "market-watcher/0.2 (personal use)"}
MAX_RPS = 8.0           # self-imposed; the Basic tier allows roughly 20 reads/s
MAX_SPREAD = 0.10       # a wider bid/ask than this makes the mid-price meaningless
DISCOVERY_PAGES = 10    # pages of 1000 open markets scanned when ranking by volume


class RateLimited(Exception):
    def __init__(self, retry_after: float = 5.0) -> None:
        super().__init__(f"rate limited, retry in {retry_after:.0f}s")
        self.retry_after = retry_after


def num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def quote(m: dict) -> tuple[float, float | None] | None:
    """(mid-price, spread) from a market's yes bid/ask, falling back to the last trade."""
    bid, ask = num(m.get("yes_bid_dollars")), num(m.get("yes_ask_dollars"))
    if bid and ask and ask >= bid and ask - bid <= MAX_SPREAD:
        return (bid + ask) / 2, ask - bid
    last = num(m.get("last_price_dollars"))
    return (last, None) if last else None


def volume(m: dict) -> float:
    return num(m.get("volume_24h_fp")) or num(m.get("volume_fp")) or 0.0


def parse_iso(s) -> float | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def trade_quote(t: dict) -> tuple[float, float] | None:
    """(timestamp, yes price) from one public trade."""
    ts = parse_iso(t.get("created_time")) or num(t.get("ts"))
    price = num(t.get("yes_price_dollars"))
    if price is None:
        cents = num(t.get("yes_price"))
        price = cents / 100 if cents is not None else None
    return (ts, price) if ts and price else None


def candle_quote(c: dict) -> tuple[float, float, float | None] | None:
    """(timestamp, price, spread) from one candlestick; prices can be null in quiet periods."""
    ts = num(c.get("end_period_ts"))
    if ts is None:
        return None
    bid = num((c.get("yes_bid") or {}).get("close_dollars"))
    ask = num((c.get("yes_ask") or {}).get("close_dollars"))
    if bid and ask and ask >= bid and ask - bid <= MAX_SPREAD:
        return ts, (bid + ask) / 2, ask - bid
    price = num((c.get("price") or {}).get("close_dollars"))
    return (ts, price, None) if price else None


class Client:
    """Thin requests wrapper with one shared rate limiter (safe to use from several threads)."""

    def __init__(self, base: str = BASE) -> None:
        self.base = base
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._lock = threading.Lock()
        self._next = 0.0

    def get(self, path: str, **params) -> dict:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + 1.0 / MAX_RPS
        if wait > 0:
            time.sleep(wait)
        params = {k: v for k, v in params.items() if v is not None}
        r = self.session.get(self.base + path, params=params, timeout=20)
        self._sync_clock(r)
        if r.status_code == 429:  # Kalshi sends no Retry-After header
            raise RateLimited(float(r.headers.get("Retry-After", 5)))
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _sync_clock(r: requests.Response) -> None:
        """Compare the server's Date header with the local clock; ignore jitter under 5 seconds."""
        try:
            diff = parsedate_to_datetime(r.headers["Date"]).timestamp() - time.time()
        except (KeyError, TypeError, ValueError):
            return
        config.set_offset(diff if abs(diff) > 5 else 0.0)

    def candles(self, series: str, ticker: str, start_ts: float, end_ts: float, interval: int) -> list[dict]:
        """interval is minutes: 1, 60 or 1440."""
        d = self.get(f"/series/{series}/markets/{ticker}/candlesticks",
                     start_ts=int(start_ts), end_ts=int(end_ts), period_interval=interval)
        return d.get("candlesticks") or []

    def trades(self, ticker: str, min_ts: float, max_ts: float | None = None,
               max_rows: int = 5000) -> list[dict]:
        """Public trade history for one market, newest first, up to max_rows."""
        out: list[dict] = []
        cursor = None
        while len(out) < max_rows:
            page = self.get("/markets/trades", ticker=ticker, min_ts=int(min_ts),
                            max_ts=int(max_ts) if max_ts else None, limit=1000, cursor=cursor)
            out += page.get("trades") or []
            cursor = page.get("cursor")
            if not cursor:
                break
        return out[:max_rows]

    def fetch_quotes(self, tickers: list[str], events: list[str]) -> list[dict]:
        """Current market objects for these tickers. Falls back to per-event queries if the
        `tickers` filter is rejected or ignored."""
        got: list[dict] = []
        requested = set(tickers)
        try:
            for i in range(0, len(tickers), 100):
                page = self.get("/markets", tickers=",".join(tickers[i:i + 100]), limit=1000)
                got += page.get("markets") or []
            # A working filter only returns what we asked for; anything else means it was ignored.
            if not tickers or (got and all(m.get("ticker") in requested for m in got)):
                return got
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (400, 422):
                raise
        got = []
        for ev in events:
            got += self.get("/markets", event_ticker=ev, limit=1000).get("markets") or []
        return got


class Tracker:
    """Chooses which events to follow (highest volume) and snapshots their prices."""

    def __init__(self, client: Client, top_events: int = 40, per_event: int = 8,
                 series: list[str] | None = None, min_age_hours: float = 0.0) -> None:
        self.client = client
        self.min_age_hours = min_age_hours
        self.top_events = top_events
        self.per_event = per_event
        self.series = series or []
        # event_ticker -> {"title", "series", "markets": {market_ticker: label}}
        self.events: dict[str, dict] = {}
        self.scan_stats: dict = {}  # what the last discovery saw, for status messages

    def _page(self, cursor: str | None, filters: dict) -> tuple[dict, dict]:
        try:
            return self.client.get("/markets", status="open", mve_filter="exclude",
                                   limit=1000, cursor=cursor, **filters), filters
        except requests.HTTPError as e:
            if "max_created_ts" in filters and e.response is not None and e.response.status_code == 400:
                filters = {k: v for k, v in filters.items() if k != "max_created_ts"}  # unsupported: filter locally
                return self.client.get("/markets", status="open", mve_filter="exclude",
                                       limit=1000, cursor=cursor, **filters), filters
            raise

    def _scan(self, max_pages: int | None = None, **filters) -> list[dict]:
        out, cursor, pages = [], None, 0
        while True:
            page, filters = self._page(cursor, filters)
            out += page.get("markets") or []
            cursor, pages = page.get("cursor"), pages + 1
            if not cursor or (max_pages and pages >= max_pages):
                return out

    def event_info(self, event_ticker: str, fallback_title: str) -> tuple[str, str]:
        guess_series = event_ticker.split("-")[0]
        try:
            d = self.client.get(f"/events/{event_ticker}")
            e = d.get("event", d)
            return e.get("title") or fallback_title, e.get("series_ticker") or guess_series
        except requests.HTTPError:
            return fallback_title, guess_series

    def refresh(self) -> list[str]:
        """Re-rank open markets by volume. Returns event tickers that are newly tracked."""
        extra = {}
        if self.min_age_hours:  # ask the server to skip new markets; also enforced locally below
            extra["max_created_ts"] = int(config.now() - self.min_age_hours * 3600)
        found = (sum((self._scan(series_ticker=s, **extra) for s in self.series), [])
                 if self.series else self._scan(max_pages=DISCOVERY_PAGES, **extra))
        priced = [m for m in found if m.get("event_ticker") and quote(m) is not None]
        cutoff = config.now() - self.min_age_hours * 3600

        def group(use_age_filter: bool) -> dict[str, list[dict]]:
            out: dict[str, list[dict]] = defaultdict(list)
            for m in priced:
                if use_age_filter and self.min_age_hours and (parse_iso(m.get("open_time")) or 0) > cutoff:
                    continue  # opened too recently: no history to show yet
                out[m["event_ticker"]].append(m)
            return out

        by_event = group(True)
        relaxed = False
        if not by_event and self.min_age_hours and priced:  # never end up with an empty screen
            by_event, relaxed = group(False), True
        self.scan_stats = {"scanned": len(found), "priced": len(priced),
                           "events": len(by_event), "relaxed": relaxed}

        ranked = sorted(by_event.items(), key=lambda kv: sum(volume(m) for m in kv[1]), reverse=True)
        tracked: dict[str, dict] = {}
        new: list[str] = []
        for ev, ms in ranked[: self.top_events]:
            ms = sorted(ms, key=volume, reverse=True)[: self.per_event]
            info = self.events.get(ev)
            if info is None:
                title, series = self.event_info(ev, ms[0].get("title") or ev)
                info = {"title": title, "series": series}
                new.append(ev)
            info["markets"] = {m["ticker"]: m.get("yes_sub_title") or m.get("title") or m["ticker"]
                               for m in ms}
            tracked[ev] = info
        self.events = tracked
        return new

    def snapshot(self) -> list[tuple]:
        """rows: (event_ticker, event_title, series, market_ticker, label, price, spread)"""
        tickers = [t for info in self.events.values() for t in info["markets"]]
        by_ticker = {m["ticker"]: m for m in self.client.fetch_quotes(tickers, list(self.events))}
        rows = []
        for ev, info in self.events.items():
            for t, label in info["markets"].items():
                q = quote(by_ticker[t]) if t in by_ticker else None
                if q:
                    rows.append((ev, info["title"], info["series"], t, label, q[0], q[1]))
        return rows


def check() -> None:
    """Hit each endpoint we depend on and report what actually comes back."""
    c = Client()

    def ok(msg: str) -> None:
        print("OK   ", msg)

    def bad(msg: str) -> None:
        print("FAIL ", msg)

    def age_hours(x: dict) -> float | None:
        opened = parse_iso(x.get("open_time"))
        return (config.now() - opened) / 3600 if opened else None

    def established(x: dict) -> bool:
        return (age_hours(x) or 0) >= 3 and quote(x) is not None

    try:
        ms, cursor, pages = [], None, 0
        while pages < 3:  # page 1 can be all brand-new markets, so look further for established ones
            page = c.get("/markets", status="open", mve_filter="exclude", limit=1000, cursor=cursor)
            ms += page.get("markets") or []
            cursor, pages = page.get("cursor"), pages + 1
            if not cursor or any(established(x) for x in ms):
                break
        ok(f"GET /markets -> {len(ms)} markets over {pages} page(s)")
        if not ms:
            return
        ms = sorted([x for x in ms if quote(x)] or ms, key=volume, reverse=True)
        solid = [x for x in ms if established(x)]
        if solid:  # test a market open 3h+, since hourly candles only exist for closed hours
            ms = solid + [x for x in ms if x not in solid]
        m = ms[0]
        age = age_hours(m)
        print(f"      test market: {m.get('ticker')} (open for "
              f"{'unknown time' if age is None else f'{age:.1f}h'}"
              f"{'' if solid else '; NO established market found in the scanned pages'})")
        print("      top by volume:", [(x.get("ticker"), volume(x)) for x in ms[:3]])
        print("      fields:", ", ".join(sorted(m)))
        print("      sample:", m.get("ticker"), "| event", m.get("event_ticker"),
              "| bid", m.get("yes_bid_dollars"), "ask", m.get("yes_ask_dollars"),
              "| last", m.get("last_price_dollars"), "| vol", m.get("volume_fp"),
              "| quote ->", quote(m))
    except Exception as e:
        return bad(f"GET /markets: {e}")

    want = [x["ticker"] for x in ms[:3]]
    try:
        got = [x["ticker"] for x in c.get("/markets", tickers=",".join(want), limit=1000).get("markets", [])]
        (ok if sorted(got) == sorted(want) else bad)(f"tickers filter: asked {len(want)}, got {len(got)}")
    except Exception as e:
        bad(f"tickers filter: {e} (the app falls back to per-event queries)")

    ev = m.get("event_ticker")
    series = ev.split("-")[0]
    try:
        d = c.get(f"/events/{ev}")
        e = d.get("event", d)
        series = e.get("series_ticker") or series
        ok(f"GET /events/{ev} -> title={e.get('title')!r}, series={e.get('series_ticker')!r}")
    except Exception as e:
        bad(f"GET /events/{ev}: {e}")

    off = config.OFFSET
    (bad if abs(off) > 60 else ok)(
        f"clock: your system is {abs(off) / 3600:.1f}h {'behind' if off > 0 else 'ahead of'} Kalshi's server"
        + (" (the app compensates, but you should fix it)" if abs(off) > 60 else ""))
    now = config.now()
    all_ok = True
    for interval, span in ((60, 7 * 86400), (1, 3600)):
        try:
            cs = c.candles(series, m["ticker"], now - span, now, interval)
            first = next((candle_quote(x) for x in cs if candle_quote(x)), None)
            too_young = interval == 60 and (age_hours(m) or 99) < 2
            if first:
                ok(f"candlesticks interval={interval}m -> {len(cs)} candles; first usable: {first}")
            elif too_young:  # not a failure: the market's first hour hasn't closed yet
                print(f"SKIP  candlesticks interval=60m -> 0 candles, but the market is only "
                      f"{age_hours(m) * 60:.0f} min old (hourly candles need a closed hour)")
            else:
                bad(f"candlesticks interval={interval}m -> {len(cs)} candles (no usable candles)")
                all_ok = False
        except Exception as e:
            bad(f"candlesticks interval={interval}m: {e}")
            all_ok = False

    try:  # the fallback the backfill uses when candles are empty
        tr = c.trades(m["ticker"], now - 86400, max_rows=1000)
        pts = [q for q in map(trade_quote, tr) if q]
        (ok if pts else bad)(f"trades last 24h -> {len(tr)} trades, {len(pts)} parsed into (time, price)")
        if tr:
            print("      first raw trade:", tr[0])
    except Exception as e:
        bad(f"trades: {e}")

    if not all_ok:
        _candle_diagnostics(c, series, ms[:3], now)


def _candle_diagnostics(c: Client, series: str, top: list[dict], now: float) -> None:
    """Probe why candles come back empty: clock skew, window size, market age, other markets."""
    m = top[0]
    t = m["ticker"]
    single = f"/series/{series}/markets/{t}/candlesticks"

    def show(name: str, path: str, **params) -> None:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = c.session.get(c.base + path, params=params, timeout=20)
            try:
                js = r.json()
            except ValueError:
                js = None
            n = None
            if isinstance(js, dict):
                if isinstance(js.get("candlesticks"), list):
                    n = len(js["candlesticks"])
                elif isinstance(js.get("markets"), list) and js["markets"]:
                    n = len(js["markets"][0].get("candlesticks") or [])
            print(f"  {name:<34} HTTP {r.status_code} candles={n}")
            if n in (None, 0):
                print("     raw:", r.text[:200].replace("\n", " "))
        except Exception as e:
            print(f"  {name:<34} error: {e}")
        time.sleep(0.15)

    print(f"\nCandlestick diagnostics for {t} (series {series}):")
    try:  # is the local clock far from the server's?
        r = c.session.get(c.base + "/exchange/status", timeout=20)
        print(f"  server Date header: {r.headers.get('Date')} | local UTC now: "
              f"{datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')}")
    except Exception as e:
        print("  clock check failed:", e)
    opened = parse_iso(m.get("open_time"))
    print(f"  market open_time={m.get('open_time')} status={m.get('status')}")

    show("single 60m / 24h", single, start_ts=int(now - 86400), end_ts=int(now), period_interval=60)
    show("single 1440m / 90d", single, start_ts=int(now - 90 * 86400), end_ts=int(now), period_interval=1440)
    if opened:
        show("single 1440m / since open_time", single, start_ts=int(opened), end_ts=int(now), period_interval=1440)
    show("single 60m / next 24h (future end)", single, start_ts=int(now - 86400), end_ts=int(now + 86400), period_interval=60)
    show("batch market_tickers 60m / 24h", "/markets/candlesticks", market_tickers=t,
         start_ts=int(now - 86400), end_ts=int(now), period_interval=60)
    for other in top[1:]:
        ser = str(other.get("event_ticker", "")).split("-")[0]
        show(f"other market {other['ticker'][:22]}", f"/series/{ser}/markets/{other['ticker']}/candlesticks",
             start_ts=int(now - 86400), end_ts=int(now), period_interval=60)
    try:
        r = c.session.get(c.base + "/historical/cutoff", timeout=20)
        print(f"  /historical/cutoff -> HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print("  /historical/cutoff failed:", e)

if __name__ == "__main__":
    check()