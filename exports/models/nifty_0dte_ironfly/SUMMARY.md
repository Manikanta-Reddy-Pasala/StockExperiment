# NIFTY 0DTE Iron-Fly (`nifty_0dte_ironfly`)

**Status:** PAPER / RESEARCH — not live, no real capital  

NIFTY weekly 0DTE iron-fly (defined risk). Defined-risk 0DTE premium selling — the only options model in this repo that clears >100% CAGR with a *bounded* worst case.

## Strategy — how it works (entry & exit, step by step)

**Idea:** On a NIFTY weekly *expiry day*, options have one trading session of life left. Time value (theta) collapses to zero by 3:30 PM. A seller who opens a position at 9:15 and lets it decay to settlement harvests that collapse — *as long as NIFTY stays inside the sold strikes*. The bought wings cap the loss if it doesn't.

**Trade only on expiry day.** No position on non-expiry days. Post-Sep-2025 NIFTY weekly expiry = Tuesday. (One trade per expiry → ~52/year.)

### ENTRY (at 9:15 AM open, expiry day)
1. **Find ATM** from the option chain itself via put-call parity (spot ≈ median of `strike + CE − PE` over liquid strikes) — no external spot feed, immune to data issues.
2. **Sell the body (collect premium):**
   - Sell 1 CE at **+1.2% OTM** (strike ≈ ATM × 1.012)
   - Sell 1 PE at **−1.2% OTM** (strike ≈ ATM × 0.988)
3. **Buy the wings (define the risk):**
   - Buy 1 CE **2% beyond** the short call (≈ +3.2% OTM)
   - Buy 1 PE **2% beyond** the short put (≈ −3.2% OTM)
4. **Net credit** = (short CE + short PE) − (long CE + long PE), collected up front. Max profit = this credit (kept if NIFTY expires between the short strikes).
5. **Max loss = wing width − credit** — fixed, known at entry. No gap, however large, can exceed it (the long wings absorb the rest). This is the whole point of the wings.

### EXIT (one of three, whichever comes first)
1. **Profit / hold to settlement** — if NIFTY stays between the short strikes, the body decays toward zero; position settles at the close (3:30 PM) near max profit.
2. **Hard stop (intraday)** — if the position's loss reaches **2× the credit collected**, exit immediately. Caps the typical bad day before it reaches max loss.
3. **Expiry settlement** — any remaining position is marked out at the close (intrinsic value at expiry).

**Worst case is structurally bounded** at the wing-defined max loss (≈ −24% of margin in the backtest) — even an −8% gap day cannot blow up the account, because the bought wings pay off in tandem. That is the trade-off vs a naked strangle (higher raw CAGR but unbounded tail).

## Trade rules

| When | Rule |
|---|---|
| **Days** | NIFTY weekly expiry days only (Tuesday, post Sep-2025) |
| **Entry** | At expiry-day open: sell 1.2%-OTM CE + PE; buy wings 2.0% beyond each (defined risk) |
| **Stop** | 2.0× credit loss (intraday) |
| **Exit** | expiry-day close or 2x stop |
| **Data** | historical_options expiry-day OHLC (daily bhavcopy proxy for 0DTE) |

## Results (2025-03-01..now)

| Metric | Value |
|---|---|
| Trades | 64 (49W / 15L) |
| Win rate | 76.6% |
| **CAGR** | **157.0%** |
| Avg return / trade (on margin) | 2.12% |
| Max drawdown | 24.2% |
| Worst trade | -24.2% (capped: yes — bought wings (worst day structurally bounded)) |

## Year-by-year

| Year | Trades | Return % (on margin) | Win % |
|---|---:|---:|---:|
| 2025 | 44 | 31.6% | 75.0% |
| 2026 | 20 | 104.3% | 80.0% |

## Caveats

- in-sample single regime (2025-26, seller-friendly)
- daily-OHLC proxy not true intraday (recorder accumulating real 5m)
- 64 trades = thin sample, no walk-forward yet
- live execution slippage will reduce returns

## Live paper trading
- Paper-only (no real orders). VM crons enter 09:20 IST / settle 15:25 IST on expiry days → table `paper_dte_trades`.
- `python tools/options/paper_dte_ironfly.py --report` for running paper results.
- Engine: `tools/options/opt_0dte.py`; regenerate this: `tools/options/gen_0dte_model.py`.
