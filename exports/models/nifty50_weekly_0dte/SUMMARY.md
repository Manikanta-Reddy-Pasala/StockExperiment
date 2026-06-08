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
| CAGR | 234.9% |
| Avg return / trade (margin) | 2.64% |
| Max drawdown | 12.6% |
| Worst trade | -11.0% (capped by wings) |

### Year-by-year
| Year | Trades | Return % (margin) | Win % |
|---|---:|---:|---:|
| 2025 | 34 | 17.6% | 73.5% |
| 2026 | 15 | 111.7% | 80.0% |

## Caveats
- in-sample single regime (2025-26, seller-friendly)
- daily-OHLC proxy not true intraday (recorder accumulating real 5m)
- 49 trades = thin sample, no walk-forward yet
- live execution slippage will reduce returns

## Live paper
Paper-only (no orders). Crons enter 09:20 IST / settle 15:25 IST on its expiry days
→ table `paper_dte_trades` (model=`nifty50_weekly_0dte`).
`python tools/options/paper_dte_ironfly.py --report --model nifty50_weekly_0dte`.
