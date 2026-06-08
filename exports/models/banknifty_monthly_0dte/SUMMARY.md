# Bank Nifty Monthly 0DTE Iron-Fly (`banknifty_monthly_0dte`)

**Status:** PAPER / RESEARCH — not live, no real capital.

Defined-risk 0DTE premium selling on BANKNIFTY monthly expiry (last Tuesday).

## Strategy — entry & exit
**Trade only on monthly expiry (last Tuesday)** — on expiry day the option has one session left; time
value collapses to ~0 by close. Sell at the open, let it decay to settlement.

**ENTRY (9:15 open):** find ATM via put-call parity (spot ≈ median K+CE−PE);
**sell 1.2%-OTM CE + PE**; **buy wings 2% beyond each** (defines max loss).
Net credit = max profit; **max loss = wing width − credit** (fixed at entry, gap-proof).

**EXIT (first of):** hold to close if NIFTY stays between shorts (decays to ~max
profit) · **2× credit hard stop** intraday · expiry settlement.

## Results (backtest 2025-03-01..now, in-sample, daily-OHLC proxy)
| Metric | Value |
|---|---|
| Trades | 14 (12W / 2L) |
| Win rate | 85.7% |
| CAGR | 43.6% |
| Avg return / trade (margin) | 3.1% |
| Max drawdown | 4.2% |
| Worst trade | -4.2% (capped by wings) |

## How the % is derived
`return % = P&L ÷ margin deployed`, where **margin = wing width − credit** (the
defined-risk capital locked per iron-fly). Per-unit (lot-size independent).

## Capital simulation — 2 LOTS per trade (BANKNIFTY lot 30 → 60 qty/leg)
Fixed **2 lots** (60 qty) per leg, both shorts and both wings (one basket order):
| Metric | Value |
|---|---|
| Size | 2 lots = 60 qty / leg |
| Margin / trade (≈) | ₹63,066 (= (wing−credit) × 60) |
| Avg P&L / trade | ₹1,981 |
| **Total P&L (14 trades)** | **₹27,738** |
| Best trade | ₹4,470 |
| Worst trade | ₹-2,712 (max loss capped by wings) |

*Profit pocketed per trade (not compounded). Margin varies per trade (defined
risk = wing − credit). BANKNIFTY lot = 30.*

## Execution — BASKET / multi-leg order ONLY
The 4 legs are entered as **one basket (multi-leg) order**, never 4 individual
orders — legging in separately risks partial fills + the index moving between
legs, which breaks the defined-risk structure. Backtest/paper price all 4 legs
at the same instant (the basket). **Paper only — no real broker orders.**

### Year-by-year (2 lots)
| Year | Trades | Return % (margin) | P&L (2 lots) |
|---|---:|---:|---:|
| 2025 | 10 | 32.1% | ₹19,932 |
| 2026 | 4 | 11.4% | ₹7,806 |

## Caveats
- in-sample single regime (2025-26, seller-friendly)
- daily-OHLC proxy not true intraday (recorder accumulating real 5m)
- 14 trades = thin sample, no walk-forward yet
- live execution slippage will reduce returns
- monthly-only (~12/yr); BankNifty options less liquid than NIFTY; diversifier not standalone

## Live paper
Paper-only (no orders). Crons enter 09:20 IST / settle 15:25 IST on its expiry days
→ table `paper_dte_trades` (model=`banknifty_monthly_0dte`).
`python tools/options/paper_dte_ironfly.py --report --model banknifty_monthly_0dte`.
