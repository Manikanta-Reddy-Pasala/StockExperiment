# NIFTY 50 Weekly 0DTE Iron-Fly (`nifty50_weekly_0dte`)

**Status:** PAPER / RESEARCH — not live, no real capital.

Defined-risk 0DTE premium selling on NIFTY weekly expiry (Tuesday).

## Strategy — entry & exit
**Trade only on weekly expiry (Tuesday)** — on expiry day the option has one session left; time
value collapses to ~0 by close. Sell at the open, let it decay to settlement.

**ENTRY (9:15 open):** find ATM via put-call parity (spot ≈ median K+CE−PE);
**sell 1.2%-OTM CE + PE**; **buy wings 2% beyond each** (defines max loss).
Net credit = max profit; **max loss = wing width − credit** (fixed at entry, gap-proof).

**EXIT (first of):** hold to close if NIFTY stays between shorts (decays to ~max
profit) · **2× credit hard stop** intraday · expiry settlement.

## Results (backtest 2025-03-01..now, in-sample, daily-OHLC proxy)
| Metric | Value |
|---|---|
| Trades | 49 (37W / 12L) |
| Win rate | 75.5% |
| CAGR | 242.9% |
| Avg return / trade (margin) | 2.67% |
| Max drawdown | 12.6% |
| Worst trade | -7.1% (capped by wings) |

## How the % is derived
`return % = P&L ÷ margin deployed`, where **margin = wing width − credit** (the
defined-risk capital locked per iron-fly). Per-unit (lot-size independent).

## Capital simulation — 2 LOTS per trade (NIFTY lot 65 → 130 qty/leg)
Fixed **2 lots** (130 qty) per leg, both shorts and both wings (one basket order):
| Metric | Value |
|---|---|
| Size | 2 lots = 130 qty / leg |
| Margin / trade (≈) | ₹62,261 (= (wing−credit) × 130) |
| Avg P&L / trade | ₹1,381 |
| **Total P&L (49 trades)** | **₹67,678** |
| Best trade | ₹20,683 |
| Worst trade | ₹-4,459 (max loss capped by wings) |

*Profit pocketed per trade (not compounded). Margin varies per trade (defined
risk = wing − credit). NIFTY lot = 65.*

## Execution — BASKET / multi-leg order ONLY
The 4 legs are entered as **one basket (multi-leg) order**, never 4 individual
orders — legging in separately risks partial fills + the index moving between
legs, which breaks the defined-risk structure. Backtest/paper price all 4 legs
at the same instant (the basket). **Paper only — no real broker orders.**

### Year-by-year (2 lots)
| Year | Trades | Return % (margin) | P&L (2 lots) |
|---|---:|---:|---:|
| 2025 | 34 | 20.0% | ₹11,635 |
| 2026 | 15 | 110.7% | ₹56,043 |

## Caveats
- in-sample single regime (2025-26, seller-friendly)
- daily-OHLC proxy not true intraday (recorder accumulating real 5m)
- 49 trades = thin sample, no walk-forward yet
- live execution slippage will reduce returns

## Live paper
Paper-only (no orders). Crons enter 09:20 IST / settle 15:25 IST on its expiry days
→ table `paper_dte_trades` (model=`nifty50_weekly_0dte`).
`python tools/options/paper_dte_ironfly.py --report --model nifty50_weekly_0dte`.
