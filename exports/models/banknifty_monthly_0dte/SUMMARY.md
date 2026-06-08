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

### Year-by-year
| Year | Trades | Return % (margin) | Win % |
|---|---:|---:|---:|
| 2025 | 10 | 32.1% | 90.0% |
| 2026 | 4 | 11.4% | 75.0% |

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
