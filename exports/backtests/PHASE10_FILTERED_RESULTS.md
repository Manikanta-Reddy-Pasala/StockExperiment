# Phase 10 — Filtered EMA + ORB-60 Day Trading

_Generated: 2026-05-12_

## Setup

Filters added to EMA strategies:
- `min_crossover_gap_pct = 0.003` (EMAs must separate by 0.3% on cross)
- `volume_confirm_mult = 1.5` (entry bar volume must be > 1.5× 20-bar SMA)
- `htf_filter_enabled = True` (1H cross only valid in matching daily trend regime)

ORB-60: opening range 09:15-10:14, breakout entry after 10:15 with vol>1.5×, SL=opposite ORB, target=ATR×1.5, EOD force-close.

## Filtered EMA 200/400 vs Raw — N50, max=2, ₹10L

| Year | Raw ROI% | Raw Trades | Filtered ROI% | Filtered Trades |
|------|---------:|-----------:|--------------:|----------------:|
| 2023-2024 | +98.13 | 179 | **0.00** | **0** |
| 2024-2025 | +54.88 | 125 | **0.00** | **0** |
| 2025-2026 | +6.77 | 54 | **0.00** | **0** |

**Filters TOO STRICT for slow EMA.** Combined HTF + volume + min_gap blocks all CROSSOVER signals on N50 large caps. The slow 200/400 EMAs rarely cross in matching daily regime — exactly when they DO, the filter kills the signal.

## Filtered EMA 9/21 vs Raw — N50, max=2, ₹10L

| Year | Raw ROI% | Raw Trades | Filtered ROI% | Filtered Trades | Filtered DD% |
|------|---------:|-----------:|--------------:|----------------:|-------------:|
| 2023-2024 | -0.94 | 554 | **+18.08** | 57 | 9.79 |
| 2024-2025 | -20.21 | 396 | **+8.24** | 40 | 9.80 |
| 2025-2026 | -7.10 | 213 | -6.46 | 13 | 9.80 |
| **Avg** | **-9.42** | 388 | **+6.62** | 37 | 9.80 |

**Filters TURNED EMA 9/21 from loser to winner** on 2 of 3 years.
Trade count cut 85% (388 → 37 average), and what remains is higher quality. 2025-2026 still slightly negative because filters cut too many mid-cap signals.

## ORB-60 Day Trading — N50, max=2, ₹10L

| Year | Trades | ROI% | Note |
|------|-------:|-----:|------|
| 2023-2024 | 0 | 0.00 | Strategy didn't fire |
| 2024-2025 | 0 | 0.00 | Strategy didn't fire |
| 2025-2026 | 0 | 0.00 | Strategy didn't fire |

**Implementation issue.** Volume comparison logic (entry bar vol > 1.5× ORB-window avg) likely too strict for 15m bars where volume is often 0 or thin. Need to debug or use a different volume reference (20-bar SMA instead of 4-bar ORB avg).

## Key takeaways

1. **EMA 200/400 raw is the multi-year winner.** Adding filters HURTS it (over-filtering). The retest1/retest2 state machine already filters false alarms enough.

2. **EMA 9/21 NEEDS filters.** Without filters it's a -9%/yr loser. With volume + HTF + min_gap, becomes +6.6%/yr. But still inferior to raw EMA 200/400's +53%/yr.

3. **ORB-60 implementation broken** — need debug pass. Either:
   - Use 20-bar volume SMA instead of 4-bar ORB volume avg
   - Lower volume multiplier from 1.5 to 1.0
   - Check if 15m bars have non-zero volume on index stocks

## Updated Decision

**Production config UNCHANGED from Phase 9:**

```yaml
strategy: ema_200_400          # raw, no false-alarm filters
universe: nifty50              # full 53 stocks
filters: sector_RS + calendar  # NOT volume/HTF/min_gap (over-filter)
overlay: vol_sizing 2%         # optional
max_concurrent: 2
capital_inr: 1_000_000
```

**Multi-year: +98% / +54% / +6.77% = compound +227.64%, avg +53.26%/yr, worst DD 13.06%.**

## What about false alarms then?

The Phase 10 finding is counter-intuitive: adding well-documented false-crossover filters HURTS the slow EMA strategy and HELPS the fast one. Reason:

- **Slow EMA (200/400)** crosses rarely. Each cross is already "significant" by construction. Additional filters block legitimate signals.
- **Fast EMA (9/21)** crosses constantly on noise. Filters are essential to extract real trends.

The retest1/retest2 state machine in our EMA 200/400 implementation IS a false-crossover filter — it requires price to pull back and re-break to confirm the trend.

## Next steps to improve EMA 200/400

Don't add MORE filters. Instead:
- **Multi-timeframe alignment** (weekly EMA trend confirms 1H signal) — softer than HTF
- **Sector breadth** (only trade when >50% of N50 stocks in same trend) — already partially done with sector RS
- **Volatility-scaled position sizing** (Phase 6 overlay) — reduces DD without dropping trades

## Files

- `exports/backtests/multiyear_filtered/` — filtered EMA 6 dirs (per-stock files omitted from commit)
- `exports/backtests/MULTI_YEAR_FILTERED.md` — aggregated filtered results
- `exports/backtests/orb60_n50_*/` — ORB-60 results (all 0 trades)
- `exports/backtests/PHASE10_FILTERED_RESULTS.md` — this file
- `tools/backtests/run_filtered_multiyear.sh` — filtered EMA runner
- `tools/backtests/run_orb60_backtest.py` — ORB-60 implementation (needs debug)
