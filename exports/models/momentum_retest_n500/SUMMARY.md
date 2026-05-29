# Retest Momentum (`momentum_retest_n500`)

**One line:** Each month, pick the 3 strongest-momentum liquid stocks (top-120 by
ADV from Nifty-500, in confirmed uptrends), but only **buy each on a pullback to
its 20-day EMA** (a "retest"), and hold the rest while it stays a momentum leader.

## Why it exists
Plain monthly momentum buys at the rebalance close — often a local high — which
bleeds in choppy years. Retest Momentum keeps momentum's trend-year power (a
rank-retain exit lets winners run) **and** adds a pullback/retest *entry* so it
buys leaders at a discount. That combination lifts the choppy years (2025/2026)
without giving up the trend years (2023/2024).

## How it works (rules)
**Universe** (true point-in-time, recomputed every month — no lookahead):
- Top **120** stocks by trailing 20-day ADV (close × volume) from the Nifty-500 pool
- Drop Nifty Smallcap-250 names (keeps it liquid mid/large-cap)

**Selection** (1st trading day of each month) — keep only stocks that pass ALL:
- Close **> 200-day SMA** (long-term uptrend)
- Price **≤ ₹3,000** (giant-loser / price guard)
- **30-day return > 10%** (real momentum, not "least-bad")
- **10-day return > 0** (accelerating, not rolling over)
- Rank survivors by 30-day return → the **top 3** are this month's targets.

**Entry** (checked daily within the month):
- Buy a target you don't yet hold **the day its price is within −1% to +8% of its
  20-day EMA** (a retest of support).
- If a target never pulls back that month, the slot **stays in CASH** (this is
  what improves entries in choppy markets).

**Exit** (monthly): sell a held name when it **drops out of the top-6 rank**
(retain band → lets winners run through trend years).

**Sizing:** 3 positions, equal-weight (⅓ capital each).

## Backtest (2023-05-15 → 2026-05-12, true-PIT, net of 0.15%/side cost)
| Metric | Value |
|---|---|
| **CAGR** | **+91.1%** |
| Total return | +594% (₹10L → ₹59.4L) |
| **Max drawdown** | **18.7%** |
| **Calmar** | **4.88** |
| Trades | 69 (win rate 67%) |

**Per calendar year** (positive every year):
| Year | Return | Intra-year DD | Note |
|---|---|---|---|
| 2023 (Apr-Dec) | **+107.7%** | 9.3% | partial |
| 2024 (full) | **+139.9%** | 15.9% | |
| 2025 (full) | **+27.5%** | 17.6% | market fell −7%; this beat it by ~35pp |
| 2026 (Jan-May) | **+10.0%** | 11.3% | partial; ≈ +47% annualized |

## Honest caveats
- **In-sample** on a bull-heavy 2022-2026 window → forward returns will be lower;
  the 18.7% DD could be deeper in an unseen crash (2008/2020-style).
- A few parameters (mom-floor 10%, accel, retest band 8%) are mildly tuned —
  economically sound rules, but confirm on pre-2023 data before sizing up.
- **2025 = +27.5% is the realistic ceiling** for that year — the steady large-cap
  winners (CANBK/MARUTI/SBILIFE) that rose +45-54% were *low-volatility quality*
  names that momentum can't capture (need fundamental/ROE data we don't have).
- True-PIT, net of cost, no lookahead — these numbers are honest, not inflated
  like the old lookahead pseudo (+144%).

## Files
- `tools/models/momentum_retest_n500/backtest.py` — canonical definition + backtest
- `exports/models/momentum_retest_n500/summary.json` — machine-readable metrics
- `exports/models/momentum_retest_n500/trade_ledger.json` — all 69 trades

## Live status
Backtest + model definition complete. Live execution (`live_signal.py` + `cron.py`
+ `data_pull.py`) — the intra-month retest entry needs daily monitoring, distinct
from the other models' rebalance-day execution — is the remaining build step
before deployment.
