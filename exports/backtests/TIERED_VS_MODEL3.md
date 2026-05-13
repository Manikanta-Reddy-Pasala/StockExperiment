# Tiered Momentum Variant vs Model 3 (N100 top-5 max=1)

Tested hypothesis: pick top-1 from each cap tier (Nifty50 / Next50 / Next50) instead of one stock from one universe. Three concurrent positions, one per tier.

**Result: WORSE than Model 3 in every year. Tier constraint HURTS returns.**

---

## Side-by-Side ROI

| Year | Tiered (max=3, 1/tier) | Model 3 (N100 max=1) | Δ |
|------|----------------------:|---------------------:|---:|
| 2023-24 | +22.76% | +80.87% | -58 pts |
| 2024-25 | +0.70%  | +133.78% | -133 pts |
| 2025-26 | -1.92%  | +46.14% | -48 pts |
| **3-yr avg** | **+7.18%** | **+86.93%** | **-80 pts/yr** |

Compound ₹10L start:
- Tiered → ₹12.46L (+24.6% over 3 yrs)
- Model 3 → ₹61.80L (+518% over 3 yrs)

---

## Tiered Variant — Full Capital Sim

Best concentration per year:

| Year | Max | Trades | Final | ROI | MaxDD |
|------|----:|------:|------:|----:|-----:|
| 2023-24 | 1 | 10 | ₹12,65,274 | +26.53% | 17.57% |
| 2023-24 | 3 | 27 | ₹12,27,587 | +22.76% | 9.52% |
| 2024-25 | 3 | 23 | ₹10,07,018 | +0.70% | 22.70% |
| 2024-25 | 5 | 29 | ₹10,34,654 | +3.47% | 13.09% |
| 2025-26 | 5 | 12 | ₹10,82,623 | +8.26% | 3.42% |

Even at best knob per year, never hit 30% threshold. None of 5 max-levels worked.

---

## Why Tiered Failed

1. **Forced mid/small-cap exposure** — Tier C (rank 100-150 by ADV) is forced into portfolio even when no real momentum there. Model 3 picks from N100 by RANK, not by tier — skips tier C entirely if N50/N100 leaders dominate.

2. **2024-25 example** — Single COCHINSHIP trade made Model 3 year (+113% in 2.5mo, +₹20.4L). Tiered system holds 3 stocks 1/3 each — even if one is COCHINSHIP, gain capped at 1/3 weight = ~38%. Plus the other 2 tiers picked weak stocks that diluted.

3. **Equal-tier dilution** — Tier A often weakest momentum (mega-caps mean-revert in bull). Forcing top-A allocation = bleed.

4. **Stale data 2025-26** — Cache thin for late 2025 → only 10 active symbols, 0% returns recorded on many picks → engine held losers.

---

## Tiered Sample Picks (Year 1)

Last 2 rebalances of 2023-24 showed how strategy reaches:

```
2024-04-01 picks: A=JIOFIN(+42.5%), B=OIL(+40.7%), C=CUMMINSIND(+31.1%)
2024-05-01 picks: A=HINDALCO(+24.1%), B=COCHINSHIP(+49.2%), C=AEGISLOG(+57.5%)
```

Each tier's leader picked. But forced concurrency dilutes vs Model 3 chasing single best.

---

## Conclusion

Tier diversification ≠ better returns in momentum strategy. Momentum rewards CONCENTRATION in current leader, not balance across cap segments.

**Stick with Model 3.** Tier variant abandoned.

---

## Negative Finding Reasoning (for journal)

Sometimes intuition ("diversify across cap tiers") opposes optimal ("concentrate in best mover"). Momentum is one of those. Empirical 3-yr test confirms.

Caveat: 2025-26 cache thinness adds noise. Even ignoring yr3, yrs 1-2 alone (22.76% + 0.70%) still LOSE to Model 3 by 90+ pts.

Source: `tools/backtests/tiered_momentum_rotation.py`, raw cycles in `exports/backtests/tiered_{year}/` (discarded post-analysis, reproducible).
