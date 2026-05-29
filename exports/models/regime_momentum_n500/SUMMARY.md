# Regime Momentum (regime_momentum_n500)

Multi-holding (K=5), regime-adaptive momentum on the liquid Nifty-500 pool.
Higher-beta sibling of `momentum_retest_n500`. Core logic is shared between the
backtest and the live signal via `tools/models/regime_momentum_n500/strategy.py`
so they cannot drift.

## How it works

**Universe** — top-80 by 20-day ADV (avg traded value = close×volume) within the
Nifty-500, price ≤ ₹3000. The top-80 ADV cut is deliberate *quality control*:
backtests showed widening the pool drags in small-cap blow-ups and HURTS returns.

**Ranking** — by 30-day return; require momentum > 0. Hold the top **5**
equal-weight. A held name is kept while it stays in the **top-6** rank (retain
band) — so winners ride instead of churning every month.

**Rebalance** — buys happen on the **1st trading day of each month**: sell any
holding that fell out of the top-6, then buy the top-5 not held. **No retest wait,
no take-profit** — winners run.

**Regime switch** — the Nifty-50 index vs its 200-DMA decides the risk posture:

| | Healthy (Nifty50 > 200DMA) | Bear (Nifty50 < 200DMA) |
|---|---|---|
| Stops | none — let trends run | hard stop **−10%** + trailing stop **−15% from peak** |
| Checked | — | **daily** |

In uptrends the model is pure let-it-ride momentum; in downtrends it arms stops +
trails (checked every day) to cut losers and lock gains. This regime-conditional
risk is what lifts the weak years without sacrificing the trend-year compounding.

## Backtest (true N500, net 0.15%/side)

**Full 2023-05 → 2026-05**

| CAGR | maxDD | Calmar | trades | win% |
|---|---|---|---|---|
| **+69.1%** | 27.4% | 2.52 | 133 | 60% |

Per-year: 2023 **+111.9%**, 2024 **+77.5%**, 2025 **+23.2%**, 2026 **+3.8%**
(2026 = partial bear year, Jan-May; Nifty −9% → model still positive).

**Recent windows**

| Window | Total | Annualized | maxDD | vs Nifty |
|---|---|---|---|---|
| Mar-2025 → May-2026 | +46.8% | +37.9% | 20.3% | +41 pts |
| Apr-2025 → Mar-2026 (down year) | +19.5% | +19.6% | 20.2% | +23 pts |

## Profile

Higher-beta than `momentum_retest_n500` (+91%/19% DD, K=3, retest entry). This
one is K=5, no-retest, regime-adaptive. The asymmetry is the point: ~+15-23% in
flat/down years, +70-110% in trend years.

## Files

| File | Role |
|---|---|
| `tools/models/regime_momentum_n500/strategy.py` | shared core (params + rank + regime + risk) |
| `tools/models/regime_momentum_n500/backtest.py` | offline backtest (imports strategy) |
| `tools/models/regime_momentum_n500/live_signal.py` | daily regime risk + monthly rebalance (imports strategy + multi_holding_service) |
| `tools/models/regime_momentum_n500/cron.py` | schedule glue (emit 09:27, execute 09:35, data 20:43) |
| `tools/models/regime_momentum_n500/data_pull.py` | N500 daily prefetch |
| `tools/live/fyers_executor_multi.py` | shared multi-holding executor (reused) |
| `tools/live/position_reconciler_multi.py` | shared multi-holding reconciler (reused) |

## Caveats

- Window 2023-05 → 2026-05 (NIFTY50-INDEX history limit for the 200-DMA regime gate).
- Backtest, not live-proven. Ship `enabled=false` + ₹0 capital until funded.
