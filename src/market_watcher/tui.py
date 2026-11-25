"""Mini terminal for PredictIt markets: market list on the left, live chart on the right."""
from __future__ import annotations

import time
from collections import defaultdict

import requests
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input
from textual_plotext import PlotextPlot

API = "https://www.predictit.org/api/marketdata/all"
POLL_SECONDS = 30.0
TOP_CONTRACTS = 6  # lines drawn per chart


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
        Binding("escape", "focus_table", "Back to list"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.market_names: dict[int, str] = {}
        self.contract_names: dict[int, str] = {}
        # history[market_id][contract_id] = [(unix_ts, price), ...]
        self.history: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.selected: int | None = None
        self.filter_text = ""
        self.paused = False
        self.backoff_until = 0.0

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
        self.poll()
        self.set_interval(POLL_SECONDS, self.poll)

    # ---------- data ----------
    def poll(self) -> None:
        if not self.paused and time.time() >= self.backoff_until:
            self.fetch()

    @work(thread=True, exclusive=True)
    def fetch(self) -> None:
        try:
            r = requests.get(API, timeout=5)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                self.backoff_until = time.time() + wait
                self.call_from_thread(self.set_status, f"rate limited, retrying in {wait}s")
                return
            r.raise_for_status()
            markets = r.json()["markets"]
        except Exception as e:  # network hiccups shouldn't kill the UI
            self.call_from_thread(self.set_status, f"error: {e}")
            return
        self.call_from_thread(self.apply, markets, time.time())

    def apply(self, markets: list[dict], ts: float) -> None:
        for m in markets:
            mid = m["id"]
            self.market_names[mid] = str(m["shortName"])
            for c in m["contracts"]:
                bid = c.get("bestSellYesCost")  # best price to sell Yes into
                ask = c.get("bestBuyYesCost")   # best price to buy Yes at
                if bid is not None and ask is not None:
                    price = (bid + ask) / 2
                else:
                    price = c.get("lastTradePrice")
                if price is None:
                    continue

        table = self.query_one(DataTable)
        if table.row_count == 0:
            self.rebuild_table()
        else:
            for mid in list(table.rows):
                leader, price = self.leader(int(mid.value))
                table.update_cell(mid, "leader", leader)
                table.update_cell(mid, "price", price)
        self.redraw()
        self.set_status(f"updated {time.strftime('%H:%M:%S')} | {len(self.market_names)} markets")

    def leader(self, mid: int) -> tuple[str, str]:
        latest = {cid: pts[-1][1] for cid, pts in self.history[mid].items() if pts}
        if not latest:
            return "-", "-"
        cid = max(latest, key=latest.get)
        return self.contract_names[cid][:16], f"{latest[cid]:.2f}"

    # ---------- table ----------
    def rebuild_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        needle = self.filter_text.lower()
        for mid, name in sorted(self.market_names.items(), key=lambda kv: kv[1].lower()):
            if needle and needle not in name.lower():
                continue
            leader, price = self.leader(mid)
            table.add_row(name, leader, price, key=str(mid))

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
        plt.date_form("H:M:S")  # must come after clear_figure()
        if self.selected is not None and self.selected in self.history:
            series = self.history[self.selected]
            top = sorted(series, key=lambda cid: series[cid][-1][1], reverse=True)[:TOP_CONTRACTS]

            for cid in top:
                pts = series[cid]
                xs = [time.strftime("%H:%M:%S", time.localtime(t)) for t, _ in pts]
                plt.plot(xs, [p for _, p in pts], label=self.contract_names[cid])

            # auto-fit y-axis so small moves are visible
            prices = [p for cid in top for _, p in series[cid]]
            lo, hi = min(prices), max(prices)
            pad = max(0.01, (hi - lo) * 0.1)
            plt.ylim(max(0, lo - pad), min(1, hi + pad))

            n = len(series[top[0]])
            plt.title(f"{self.market_names.get(self.selected, '')} ({n} pts)")
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

    def action_cursor_down(self) -> None:
        self.query_one(DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(DataTable).action_cursor_up()


def main() -> None:
    MarketWatcher().run()


if __name__ == "__main__":
    main()