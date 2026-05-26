# mean_reversion_rsi_n100 — SUMMARY

**RSI-oversold mean-reversion on NSE Nifty 100. ❌ DISCARDED — CAGR too low. NOT a live model.**

> **STATUS: DISCARDED (5th-model exploration).** Explored 2026-05 as a candidate 5th model, then
> dropped by user decision. Entry/exit logic below is correct and code-grounded, but this model
> **does not trade live and has no scheduler wiring.**
>
> - Added: commit `198f038c` ("docs: 5th model exploration report + backtest scripts")
> - **Reverted: commit `5a4d2a15`** ("User decision: not worth pursuing")
> - Working tree now holds only an orphan `tools/models/mean_reversion_rsi_n100/__pycache__/backtest.cpython-310.pyc` (the revert missed it). Safe to delete.
> - **Source recoverable** from git: `git show 198f038c:tools/models/mean_reversion_rsi_n100/backtest.py` (+ `backtest_aggressive.py`). Full analysis: `git show 198f038c:docs/MODEL_EXPLORATION_5TH.md`.

## Why discarded

Genuinely de-correlated from the 4 momentum/breakout models (RSI-oversold ≠ momentum), BUT
**solo CAGR too low** — it would drag down the portfolio average. The 4 live models run
+65% → +149% CAGR; the best mean-reversion variant only reached +30.90% (N100) / +38.74% (N500).

| Variant | CAGR | Max DD | WR | Calmar |
|---|---:|---:|---:|---:|
| N100 single-pick (best) — RSI<30 / RSI>50 / +6% tgt / −4% stop / 10d hold | **+30.90%** | 15.3% | 68.1% | 2.02 |
| N500 single-pick (best) — RSI<30 / RSI>50 / +6% tgt / −4% stop / 10d hold | +38.74% | 21.5% | 55.1% | — |
| Code-default params (RSI>50 / +8% / −5% / 20d) | +21.38% | 21.8% | 65.6% | 0.98 |

Verdict from the report: *"too low to drag-down portfolio average. Could lift portfolio Sharpe
via low correlation if allocated 10–20% capital."* — not pursued.

## When it BUYS (entry rules)

Single position (`max_concurrent=1`), all-in, pick = **lowest RSI** (most oversold) among qualifiers:
1. Universe = NSE Nifty 100 (`src/data/symbols/nifty100.csv`).
2. **Uptrend gate** — close > 200-day SMA (`SMA_LONG=200`).
3. **Oversold** — RSI(14) < **30** (`RSI_WIN=14`, `RSI_LOW=30`; standard Wilder RSI).
- ⚠️ Docstring mentions a `MIN_BOUNCE` anti-falling-knife filter — **documented but never implemented** (no constant/code).

## When it SELLS (exit rules)

Unlike the rotation models, this one has **genuine price-based risk exits** (first wins). Code
defaults shown; the winning variant used +6% / −4% / 10d instead:

| Reason | Fires when | Code default |
|---|---|---|
| **RSI exit** | RSI(14) **> 50** (bounce confirmed) | `RSI_EXIT=50` |
| **TARGET** | close **≥ +8%** above entry | `TARGET_PCT=0.08` |
| **STOP** | close **≤ −5%** below entry | `STOP_PCT=0.05` |
| **MAX_HOLD** | **20 days** held | `MAX_HOLD=20` |
| FORCE_CLOSE_END | open position closed at backtest end | — |

CLI overrides: `--rsi-low / --rsi-exit / --target-pct / --stop-pct / --max-hold`. Charges via
`tools/live/broker_charges.compute_charges` (Fyers CNC). Backtest window 2023-05-15 → 2026-05-12, ₹1,000,000 start.
