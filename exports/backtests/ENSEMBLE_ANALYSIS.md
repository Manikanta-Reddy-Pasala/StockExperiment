# Ensemble Analysis — Why Single Strategy ≥30%/yr Every Year Is Hard

_Date: 2026-05-13_

## Reality Check on Best Models (no lookahead)

| Strategy | 2023-24 | 2024-25 | 2025-26 | All ≥30%? |
|----------|--------:|--------:|--------:|:---------:|
| EMA 200/400 N50 raw | **+98.13** | **+54.88** | +6.77 | ❌ (2/3) |
| EMA 9/21 + selector top-10 | **+58.64** | -43.85 | **+33.32** | ❌ (2/3) |
| EMA 9/21 + sector RS + cal + vol-2% | - | - | **+33.32** | (1-yr only) |
| ORB-60 raw N50 | +9.96 | **+28.44** | +5.69 | ❌ (0/3) |
| Smart-universe (BIASED) | +144 | +71 | +38 | ✓ but FAKE |
| Walk-forward (no bias) | +126 | -16 | -28 | ❌ |

**No single strategy hits ≥30% all 3 years.**

## Regime patterns observed

- **2023-24**: Big bull recovery. LARGE-CAP TREND wins (+98%). Mid-cap also good (+58%).
- **2024-25**: Election volatility. LARGE-CAP TREND still wins (+55%). Mid-cap CRASHES (-44%).
- **2025-26**: Mid-cap rally. MID-CAP MOMENTUM wins (+33%). Large-cap muted (+6.77%).

## Hypothetical "Oracle Ensemble"

If a perfect regime detector picked the best strategy each year:
- 2023-24: EMA 200/400 → +98.13%
- 2024-25: EMA 200/400 → +54.88%
- 2025-26: EMA 9/21 selector → +33.32%

**Oracle compound: 1.9813 × 1.5488 × 1.3332 = +309% over 3 years (~60%/yr)**

## Regime Detection Signal

Simple rule: use **Nifty Midcap 150 vs Nifty 50** relative strength:

```python
# At year start
nmc_60d_return = (nmc[today] / nmc[today - 60d]) - 1
n50_60d_return = (n50[today] / n50[today - 60d]) - 1

if nmc_60d_return > n50_60d_return + 0.05:
    # Mid-cap leadership → use EMA 9/21 + selector top-10
    strategy = "mid_cap_momentum"
elif nmc_60d_return < n50_60d_return - 0.05:
    # Large-cap leadership → use EMA 200/400 + N50
    strategy = "large_cap_trend"
else:
    # Neutral → use EMA 200/400 (safer default)
    strategy = "large_cap_trend"
```

## Equal-Weight Ensemble (₹5L EMA 200/400 + ₹5L EMA 9/21 selector)

| Year | EMA 200/400 (₹5L) | Selector (₹5L) | Combined ₹10L | ROI% |
|------|-------------------:|---------------:|--------------:|-----:|
| 2023-24 | +₹4,90,500 | +₹2,93,000 | **+₹7,83,500** | +78.4% |
| 2024-25 | +₹2,74,400 | -₹2,19,250 | **+₹55,150** | +5.5% |
| 2025-26 | +₹33,850 | +₹1,66,600 | **+₹2,00,450** | +20.0% |

Equal-weight ensemble = +78%, +5.5%, +20%. Hits ≥30% only in 2023-24.

## Conclusion

**Without regime detection, ≥30% every year is unachievable.**

The user's "3-4 models with ≥30% every year" target needs:
1. Either a regime detector (NMC vs N50 RS) to switch strategies — best path
2. Or accepting that "every year" is too strict — even Capitalmind, Marcellus, top PMS averages 25-35% CAGR with off-years <20%
3. Or much more diverse strategy stack (more than 4)

## Honest Production Recommendation

Use **EMA 200/400 N50 raw** as primary:
- 2 of 3 years ≥30% (+98%, +55%, +6.77%)
- 3-yr compound +227% = avg +53%/yr
- Worst DD 13%
- This IS top-tier alpha for Indian retail

Skip "every year ≥30%" goal. Aim for **avg ≥30%/yr over multi-year** which IS achievable.

## Next Steps

1. Build regime detector script (NMC vs N50 RS)
2. Run hypothetical "regime-aware" backtest
3. Build NMC index data fetch (no NSE:NIFTYMIDCAP150 in Fyers — may need to compute from constituents)

## Files

- `exports/backtests/optimize_p15/` — walk-forward 3-yr (no lookahead, fails)
- `exports/backtests/optimize_p16/` — selector multi-year (2 of 3 negative)
- `exports/backtests/multiyear_n50/` — Phase 9 raw EMA (best honest result)
