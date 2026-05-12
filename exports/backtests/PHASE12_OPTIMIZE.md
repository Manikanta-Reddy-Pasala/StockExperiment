# Phase 12 Optimization Results

_Date: 2026-05-13 | Capital: ₹10,00,000 | Universe: Nifty 50_

## EMA 9/21 — Filter Variant Sweep (max=2)

| Variant | Filters | 2023-24 | 2024-25 | 2025-26 | Avg/yr |
|---------|---------|--------:|--------:|--------:|-------:|
| **v1 relaxed** | min_gap=0.001, vol=1.2 | **+33.39%** | +10.45% | +14.17% | **+19.34%** |
| v2 vol-only | vol=1.5 | -0.94% | -20.21% | -7.10% | -9.42% |
| **v3 htf-only** | HTF SMA filter | **+29.45%** | +14.65% | +14.79% | **+19.63%** |

**v3 (HTF-only) wins** — lowest DD (16-18% vs 27%) + consistent positive every year.

## ORB-60 — Vol/Target Variant Sweep

| Variant | Vol | TGT | max=5 ROI | DD% |
|---------|----:|----:|----------:|----:|
| v1 relaxed (max=5) | 1.2 | 1.5 | +12.38% / +11.86% / +1.03% | 5-9% |
| v2 wide-target | 1.0 | 2.0 | +0.68% / +5.69% / +1.91% | 6-9% |

Original (vol=1.5, atr=1.5) max=5 was best at 2024-25 (+24.68%) but inconsistent.

## Conclusions

**None of Phase 12 variants achieve ≥30%/yr every year.**

Best EMA 9/21 (htf-only): consistent positive but 14-30% range.
Best ORB-60: still 5-12% range, far from 30%.

## Phase 13 launched

Trying:
- N100 / N150 universes (wider stock pool, more momentum picks)
- EMA 200/400 swing on N100 + N150 (extend Phase 9 winner)
- EMA 9/21 htf-only on N100 + N150
- ORB-60 on N100 + N150

= 18 variants. ETA ~90 min on prod.

## Files

- `exports/backtests/optimize/ema921_*` and `orb_*` (15 Phase 12 dirs)
- `tools/backtests/optimize_phase13.sh` — Phase 13 runner
- `exports/backtests/optimize_p13/` (will be populated)
