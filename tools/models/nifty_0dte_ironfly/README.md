# nifty_0dte_ironfly

**Status: PAPER / RESEARCH — not live, no real capital.**

0DTE NIFTY weekly **iron-fly** (defined-risk premium selling). The only options
model in this repo that clears >100% CAGR with a *bounded* worst case.

## Strategy
- Trade only on NIFTY weekly **expiry days** (Tuesday, post Sep-2025).
- At the **open**: sell 1.2%-OTM CE + 1.2%-OTM PE; **buy wings 2% beyond** each
  (defined risk — the wings cap the max loss, so no gap can blow up the account).
- **Hard stop** at 2× credit loss (intraday).
- **Settle** at the close.

## Backtest (in-sample, 2025-03 → now)
| Metric | Value |
|---|---|
| Trades | 64 (weekly expiries) |
| Win rate | 76.6% |
| **CAGR** | **157%** |
| Max DD | 24.2% |
| Worst day | −24.2% (**structurally capped** by wings) |
| Per-year | 2025 +31.6% / 2026 +104.3% (ret on margin) |

See `summary.json` and `trade_ledger.json`.

## Honest caveats
- **In-sample, single regime** (2025-26 was seller-friendly), 64 trades — no
  walk-forward yet.
- Backtest uses the **expiry-day daily OHLC** as a 0DTE proxy (entry=open,
  settle=close, stop via day-high). Real intraday execution + slippage will
  reduce returns. Fyers serves no intraday history for *expired* contracts, so a
  recorder is accumulating real 5-min data for a true walk-forward later.
- Liquidity verified: all legs incl. far wings traded heavily (min ~160K
  contracts) on every expiry day — no stale/unfillable strikes.

## Code (engines live in `tools/options/`)
| File | Purpose |
|---|---|
| `tools/options/opt_0dte.py` | 0DTE backtest engine (expiry-day OHLC) |
| `tools/options/paper_dte_ironfly.py` | **live paper-trading model** (enter/settle/report) |
| `tools/options/record_intraday_options.py` | live 5-min option data recorder |
| `tools/options/gen_0dte_model.py` | regenerates summary.json + trade_ledger.json |

## Live paper crons (VM, paper-only, no orders)
- enter 09:20 IST (expiry days), settle 15:25 IST → table `paper_dte_trades`
- `python tools/options/paper_dte_ironfly.py --report` for running paper results

## Supporting options research (same `tools/options/`)
`opt_basket.py` (diversified stock strangle basket, ~17% CAGR/1% DD),
`opt_sweep.py` (index seller sweep), `opt_momcall.py` (directional — fails),
`opt_levhedge.py` (leverage + crash-hedge analysis). Full findings in memory
`stockexp-options-model-2026-06-08`.
