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
| CAGR | 236.0% |
| Avg return / trade (margin) | 2.64% |
| Max drawdown | 12.6% |
| Worst trade | -11.0% (capped by wings) |

## How the % is derived
`return % = P&L ÷ margin deployed`, where **margin = wing width − credit** (the
defined-risk capital locked per iron-fly). Per-unit (lot-size independent).

## Capital simulation — ₹2,00,000 margin per trade
Deploy a fixed **₹2,00,000** of margin on each trade (rupee P&L = ₹2L × return%):
| Metric | Value |
|---|---|
| Margin in / trade | ₹2,00,000 |
| Avg P&L / trade | ₹5,288 |
| **Total P&L (49 trades)** | **₹259,118** |
| Best trade | ₹93,566 |
| Worst trade | ₹-22,069 (max loss capped by wings) |

*Fixed ₹2L per trade (profit pocketed, not compounded). Assumes ₹2L fully
deployed as margin; real lots are discrete (NIFTY lot 75, BankNifty 35) so actual
sizing rounds to whole lots.*

## Execution — BASKET / multi-leg order ONLY
The 4 legs are entered as **one basket (multi-leg) order**, never 4 individual
orders — legging in separately risks partial fills + the index moving between
legs, which breaks the defined-risk structure. Backtest/paper price all 4 legs
at the same instant (the basket). **Paper only — no real broker orders.**

### Year-by-year (₹2L/trade)
| Year | Trades | Return % (margin) | P&L (₹2L/trade) |
|---|---:|---:|---:|
| 2025 | 34 | 17.5% | ₹35,000 |
| 2026 | 15 | 112.0% | ₹224,000 |

## Caveats
- in-sample single regime (2025-26, seller-friendly)
- daily-OHLC proxy not true intraday (recorder accumulating real 5m)
- 49 trades = thin sample, no walk-forward yet
- live execution slippage will reduce returns

## Live paper
Paper-only (no orders). Crons enter 09:20 IST / settle 15:25 IST on its expiry days
→ table `paper_dte_trades` (model=`nifty50_weekly_0dte`).
`python tools/options/paper_dte_ironfly.py --report --model nifty50_weekly_0dte`.
