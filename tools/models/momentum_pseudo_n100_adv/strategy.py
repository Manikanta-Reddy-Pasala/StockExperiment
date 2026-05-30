"""SHARED core for momentum_pseudo_n100_adv — params + rebalance calendar
imported by BOTH backtest.py and live_signal.py so they can't drift.

Strategy: yearly-PIT pseudo-N100 = top-UNIV_SIZE by 20d ADV from N500 minus
Smallcap-250; uptrend (close > SMA_LONG); price <= MAX_PRICE; rank by LOOKBACK
return; hold while in top-RETAIN; monthly (1st trading day) + mid-month lead check.
The universe snapshot itself is shared via yearly_universes.json
(build_universe.py); these params + the calendar are shared from here.

2026-05-30 optimize sweep (both windows, real PIT engine): added mid-month +
RET5 + 3pp lead, upgrading from the old RET1/monthly-only config. New config
beats old on CAGR BOTH windows and on full-cycle DD:
  2023-26: +160.3% vs +150.7% CAGR | 22% vs 16% DD | Calmar 7.4 vs 9.3
  2021-26: +100.0% vs  +70.5% CAGR | 31% vs 44% DD | Calmar 3.2 vs 1.6
Top configs cluster (lead 0/3 x RET 3/5 all strong) = stable region, not a spike.
Only recent-window DD rises (16->22); full-cycle DD (the true worst case) drops
13pp while CAGR rises on both windows. Extra filters (momentum-floor, accel) were
swept and rejected — floor never binds under the SMA200 gate, accel hurts.
"""
from __future__ import annotations

from tools.shared.rebalance_calendar import (
    build_calendar, is_mid_month_check_day, MID_MONTH_FROM_DAY)

LOOKBACK = 30        # momentum window (TRADING days; 30 beat 15/20)
ADV_WIN = 20         # ADV averaging window
UNIV_SIZE = 100      # top-N by 20d ADV (pseudo-N100)
MAX_PRICE = 3000.0   # skip names priced above this at entry
SMA_LONG = 200       # uptrend filter: close > 200d SMA
RETAIN = 5           # exit band — hold while in top-5 (2026-05-30 sweep)
MIDMONTH_LEAD = 3.0  # mid-month rotates only if new rank-1 leads held by >= 3pp
