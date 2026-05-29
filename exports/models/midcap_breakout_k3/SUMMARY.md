# Midcap Breakout K3 (midcap_breakout_k3)

Multi-holding (K=3) variant of `midcap_narrow_60d_breakout`. Same breakout
entry/exit + PIT midcap universe; the only change is holding up to **3
concurrent positions** instead of 1. Core logic shared backtest↔live via
`tools/models/midcap_breakout_k3/strategy.py`.

## Why this exists

The single-position model is catastrophically concentration-risked on current
point-in-time data: one bad breakout tanks the whole book. Validated by
reproducing it exactly at K=1:

| K | Full 2023-26 CAGR | Full maxDD | Calmar | trades |
|---|---|---|---|---|
| 1 (single, original) | **−28.3%** | **70.5%** | −0.40 | 11 |
| **3 (this model)** | **+8.3%** | **24.5%** | 0.34 | 27 |
| 5 | +1.1% | 32.6% | 0.03 | 42 |

Holding 3 names spreads the lumpy breakout outcomes → turns the −28%/70%DD
single-position blow-up into +8.3%/24%DD, and trades more often (27 vs 11). K=5
over-dilutes. **K=3 is the validated sweet spot.**

Recent window Mar-2025 → May-2026: K=3 = +17.2% / 10.5% DD (vs K=1 +25.6%/21.3%
— lower return but half the drawdown).

## How it works

- **Universe (PIT):** top-100 by 20-day ADV from the PIT-N500 ever-member union,
  minus PIT-N100 (mid/small caps only), restricted to freshly-trading names
  (staleness ≤ 5 sessions). Rebuilt each year-start.
- **Entry:** fresh 40-day high + close > 200-DMA + volume ≥ 2× 20-day avg. Rank
  competing breakouts by volume ratio; fill free slots; enter NEXT day's open.
- **Exit (shared core):** TARGET +100% / STOP −20% / TRAIL −20% off peak (armed
  at +10% profit) / MAX_HOLD 120 calendar days.
- **Sizing:** equal-weight, total_equity / 3 per slot.
- **Costs:** 10 bps slippage both legs, 0.10% STT on sells, ₹20/order.

## Files

| File | Role |
|---|---|
| `strategy.py` | shared core (params + PIT universe + breakout scan/exit) |
| `backtest.py` | offline backtest, `--k` parametrised (k=1 reproduces original) |
| `live_signal.py` | daily exits + breakout entries (multi_holding_service) |
| `cron.py` | emit 09:28 / execute 09:36 / data 20:44 |
| `data_pull.py` | N500 daily prefetch |

Reuses `tools/live/fyers_executor_multi.py` + `position_reconciler_multi.py`.

## Caveats

- Even at K=3 this is a modest absolute return (+8.3% CAGR full-period) — it's a
  diversifier/breakout-swing, not a primary momentum engine. Value = different
  signal (breakouts) + the K=3 fix removes the catastrophic single-position risk.
- The original model's old docstring claimed +141%/8% DD — that is STALE; current
  PIT data shows −28%/70% at K=1. K=3 is the corrected, validated version.
- Backtest, not live-proven. Ship `enabled=false` + ₹0 until funded.
