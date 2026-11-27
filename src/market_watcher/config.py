"""Shared settings. One place decides where the database lives and what time it is."""
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/market_watcher/config.py -> project root
DB_PATH = Path(os.environ.get("MW_DB", PROJECT_ROOT / ".db" / "market-watcher" / "kalshi.db"))

# WSL2's clock can freeze or drift (sleep/hibernate). The Kalshi client measures how far the local
# clock is from the server's and stores it here, so every timestamp we save or query is server time.
OFFSET = 0.0


def set_offset(seconds: float) -> None:
    global OFFSET
    OFFSET = seconds


def now() -> float:
    """Unix time as the Kalshi server sees it."""
    return time.time() + OFFSET
