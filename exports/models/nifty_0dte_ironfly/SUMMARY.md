# NIFTY 0DTE Iron-Fly (`nifty_0dte_ironfly`)

**Status:** PAPER / RESEARCH — not live, no real capital  

NIFTY weekly 0DTE iron-fly (defined risk). Defined-risk 0DTE premium selling — the only options model in this repo that clears >100% CAGR with a *bounded* worst case.

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
