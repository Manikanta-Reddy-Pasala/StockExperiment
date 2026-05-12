# Sweep Results — Phase 2 (top-N + max_concurrent)

_Generated: 2026-05-12_

## Best configs found

| Rank | Config | ROI% | MaxDD% | Trades | Notes |
|------|--------|-----:|-------:|-------:|-------|
| 1 | **N500 top-20 + max=2** | **+14.15** | 10.68 | 35 | Highest ROI |
| 2 | **N50 top-19 + max=3** | **+13.20** | **+7.86** | 44 | Best risk-adjusted |
| 3 | N50 top-25 + max=2 | +9.35 | 13.66 | 36 | |
| 4 | N50 top-19 + max=2 | +7.98 | 12.85 | 30 | |
| 5 | N500 top-20 + max=3 | +7.86 | 12.73 | 46 | |
| - | baseline N50 full + max=2 | +7.30 | 12.77 | 54 | |
| - | baseline N500 full + max=2 | -33.53 | 34.94 | 90 | |

## N50 top-N sweep (max=2..8)

| Top-N | max=2 | max=3 | max=5 | max=8 | Best DD% |
|------:|------:|------:|------:|------:|---------:|
|     5 | +2.21 | +0.61 | +5.02 | +11.04 | 4.19 (max=8) |
|    10 | -3.08 | -0.76 | +0.15 | +5.07  | 6.44 (max=8) |
|    15 | -4.19 | -0.01 | -4.27 | -5.90  | 11.40 |
| **19**| +7.98 | **+13.20** | +6.37 | +5.64 | **7.86 (max=3)** |
|    25 | +9.35 | +12.77 | +5.03 | +5.51  | 9.01 (max=3) |
| 53 (full) | +7.30 | -0.29 | -0.48 | +2.25 | 12.77 |

## N500 top-N sweep (max=2..8)

| Top-N | max=2 | max=3 | max=5 | max=8 |
|------:|------:|------:|------:|------:|
|    10 | -17.24 | -21.25 | -23.85 | -21.60 |
|    15 |  -2.64 |  -3.21 |  -6.09 |  -6.26 |
| **20**| **+14.15** |  +7.86 |  -2.29 |  -2.56 |
|    30 | -16.80 | -18.30 | -16.36 | -12.29 |
|    40 | -25.63 | -32.71 | -28.65 | -18.12 |
|    50 | -30.23 | -38.26 | -35.84 | -20.09 |

## Combined N50 top-19 + N500 top-19 = HURTS

38 stocks competing for limited slots — crowding effect:

| max | ROI% | DD% |
|----:|-----:|----:|
|   2 | -3.57 | 17.28 |
|   3 | -7.14 | 15.69 |
|   5 | +2.13 |  9.82 |

Crowding hurts even when component universes individually outperform.
Conclusion: pick ONE universe, not both.

## Recommendation

**Production config: N50 top-19 + max_concurrent=3**

Rationale:
- ROI +13.20% (only 0.95% behind N500 top-20 winner)
- **MaxDD 7.86% (3pp better than N500 top-20 alternative)**
- Larger-cap stocks = better real-world liquidity, lower slippage
- Smaller universe (19 vs 20) = simpler watchlist for live trading

## N50 top-19 final universe (in rank order)

```
HCLTECH, SBIN, HINDALCO, TMPV (Tata Motors PV), BHARTIARTL, HEROMOTOCO,
RELIANCE, TATASTEEL, JIOFIN, ITC, JSWSTEEL, BPCL, HINDUNILVR, CIPLA,
ULTRACEMCO, COALINDIA, ADANIPORTS, WIPRO, MARUTI
```

## Open follow-ups (not yet swept)

- HTF filter (200DMA trend gate)
- ATR stop multiplier variants
- min_crossover_gap thresholds
- Equal vs ATR-scaled position sizing
- SELL-only mode (cycle-direction filter — needs cap-sim enhancement)

These require strategy code changes or cap-sim enhancements. Deferred
to Phase 3 + 4. Current finding (+13.20%) already nearly doubles
baseline (+7.30%) so paper-trading viable as-is.
