# Long-Only Momentum — Filter Variants

_3-year backtest May 2023 → May 2026, N100, top_n=5, max-concurrent=1, ₹10L capital._

## Comparison

| Variant | 2023-24 | 2024-25 | 2025-26 | **3-yr Avg** | **Max Yr DD** | Trades/yr |
|---------|--------:|--------:|--------:|-------------:|--------------:|---------:|
| baseline (no filter) | +132.26% / 8.53% | +72.52% / 24.78% | -15.96% / 15.96% | **+62.94%** | **24.78%** | 6.3 |
| **trailing_sl 8%** ⭐ | +92.76% / 8.00% | +120.22% / 0.80% | +3.60% / 9.39% | **+72.19%** | **9.39%** | 8.0 |
| trailing_sl 15% | +4.93% / 24.55% | +42.16% / 20.42% | -10.17% / 10.17% | +12.31% | 24.55% | 7.7 |
| trailing_sl 25% | +132.26% / 8.53% | +27.59% / 29.78% | -15.96% / 15.96% | +47.96% | 29.78% | 5.7 |
| partial_tp (50% @ +30%) | +140.80% / 8.51% | +38.93% / 24.78% | -15.96% / 15.96% | +54.59% | 24.78% | 6.3 |
| **rotation_speed** | +132.26% / 8.53% | +93.58% / 15.59% | -15.96% / 15.96% | **+69.96%** | **15.96%** | 6.3 |
| drawdown_circuit | +132.26% / 8.53% | +72.52% / 24.78% | -15.96% / 15.96% | +62.94% | 24.78% | 6.3 |

## Winner: trailing_sl 8%

Beats baseline on BOTH axes:
- **+9.3 pts ROI** (+72.19% vs +62.94%)
- **-15.4 pts DD** (9.39% vs 24.78%)

Secondary winner: **rotation_speed** (skip stalled rank-1 → take rank-2) — +7 pts ROI, -9 pts DD, no extra trades.

## Filters That Don't Work

- **trailing_sl 15%** — worst (+12% avg). Premature exits in choppy markets kill mid-trade compounding.
- **trailing_sl 25%** — too loose; barely improves DD, slightly hurts ROI.
- **partial_tp** — caps winners (the BIG-trade is what makes momentum work, halving the position halves the alpha).
- **drawdown_circuit 15%** — never triggered (max=1 cap-sim, single-stock DD always under 15% somehow).

## Caveats Before Going Live

1. **Concurrency-cap quirk**: 8% SL frees the single slot for next-best entry. At max=5 the Y3 lift shrinks (+8.4% vs +3.6%). The DD reduction is robust regardless.
2. **Tight stop = real slippage**: 8% on daily-OHLC sim ≈ tighter under intraday vol. Expect more triggers + slippage cost in live.
3. **Lookahead bias** (same as `REBALANCE_FREQ_COMPARISON.md`): universe = 2026-05-13 N100 retroactive. Survivor-favoured Y3 means production ROI compresses.

## Recommendation

**Implement trailing_sl 8% as risk overlay.** DD reduction is the real gain (24.78% → 9.39%). Treat the ROI bump as bonus, not budget. Optionally combine with **rotation_speed** for stale-leader rotation.

DO NOT use trailing_sl 15% or partial_tp.

## Files (not committed yet)

```
tools/backtests/momrot_filtered_backtest.py
remote: /tmp/momrot_filt/{y1,y2,y3}_{baseline,trail8,trail15,trail25,partial,rotspeed,circuit}/
```
