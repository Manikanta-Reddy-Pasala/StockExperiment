"""SHARED core for momentum_pseudo_n100_adv — params + rebalance calendar
imported by BOTH backtest.py and live_signal.py so they can't drift.

Strategy: yearly-PIT pseudo-N100 = top-UNIV_SIZE by 20d ADV from N500 minus
Smallcap-250; uptrend (close > SMA_LONG); price <= MAX_PRICE; rank by LOOKBACK
return; hold rank-1 (retain band RETAIN); monthly rebalance (no mid-month).
The universe snapshot itself is shared via yearly_universes.json
(build_universe.py); these params + the calendar are shared from here.
"""
from __future__ import annotations

from tools.shared.rebalance_calendar import build_calendar

LOOKBACK = 30        # momentum window (TRADING days)
ADV_WIN = 20         # ADV averaging window
UNIV_SIZE = 100      # top-N by 20d ADV (pseudo-N100)
MAX_PRICE = 3000.0   # skip names priced above this at entry
SMA_LONG = 200       # uptrend filter: close > 200d SMA
RETAIN = 1           # exit band — top-1 rotation
