"""SHARED core for momentum_pseudo_n100_adv — params + rebalance calendar
imported by BOTH backtest.py and live_signal.py so they can't drift.

Strategy: yearly-PIT pseudo-N100 = top-UNIV_SIZE by 20d ADV from N500 minus
Smallcap-250; uptrend (close > SMA_LONG); price <= MAX_PRICE; rank by LOOKBACK
return; hold rank-1 (retain band RETAIN); monthly (1st trading day) rebalance.
The universe snapshot itself is shared via yearly_universes.json
(build_universe.py); these params + the calendar are shared from here.

Config history:
  - 2026-05-30: a sweep on the (then) backtest-start-anchored universe seemed to
    show mid-month + RET5 + 3pp lead beating RET1/monthly. SHIPPED, then REVERTED
    2026-05-31 — see below.
  - 2026-05-31: the universe anchor was made a FIXED calendar date (mid-May, to
    match live; see UNIVERSE_ANCHOR_*). On the CORRECT fixed anchor the mid-month
    + RET5 'win' disappeared — it was an artifact of the unstable start-anchor.
    Old RET1/monthly is clearly better on the fixed full-cycle anchor:
      RET1/monthly : +63.5% CAGR / 37.6% DD / Calmar 1.69
      mid-month/RET5: +57.5% CAGR / 66.1% DD / Calmar 0.87
    So pseudo is back to RET1 + monthly-only. The mid-month machinery stays in
    the code (backtest --mid-month-check, live --mid-month-check) but defaults
    OFF; MIDMONTH_LEAD is kept only for those opt-in paths.
"""
from __future__ import annotations

from tools.shared.rebalance_calendar import (
    build_calendar, is_mid_month_check_day, MID_MONTH_FROM_DAY)

LOOKBACK = 30        # momentum window (TRADING days; 30 beat 15/20)
ADV_WIN = 20         # ADV averaging window
UNIV_SIZE = 100      # top-N by 20d ADV (pseudo-N100)
MAX_PRICE = 3000.0   # skip names priced above this at entry
SMA_LONG = 200       # uptrend filter: close > 200d SMA
RETAIN = 1           # exit band — top-1 rotation (wins on the fixed anchor)
MIDMONTH_LEAD = 3.0  # only used by the opt-in --mid-month-check path (default OFF)

# Universe re-anchor date — FIXED calendar (month, day), NOT the backtest start.
# Live rebuilds yearly_universes.json once a year at the mid-May NSE rebalance
# (cron _yearly_universe fires May 15 -> build_universe --end-date ~May-13). The
# backtest must anchor at the SAME fixed date so its yearly ADV snapshots match
# live and its CAGR is not start-date-sensitive (2026-05-31 fix: was anchored to
# the backtest start month, which made absolute returns drift with the window).
UNIVERSE_ANCHOR_MONTH = 5
UNIVERSE_ANCHOR_DAY = 15
