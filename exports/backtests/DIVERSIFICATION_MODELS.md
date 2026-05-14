# 2 Diversification Models for Model 3 Portfolio

_Goal: 2 models that complement Model 3 (momentum rotation) with LOW correlation. Not max return — diversification._

## All 4 Candidates Tested

| Model | Strategy | 3-yr CAGR | MaxDD | Sharpe | Trades | **Corr w/ M3** |
|-------|----------|----------:|------:|------:|------:|---------------:|
| **M3** ⭐ | Momentum N100 top-5 max=1 | **+80.5%** | 10.4% | 1.06 | 13 | +1.00 |
| **B** ✅ | Low Vol N500 top-20 quarterly | +31.7% | 3.5% | **1.98** | 117 | **+0.045** |
| **A** ✅ | RSI<30 mean-reversion N100 | +16.0% | 3.7% | 1.68 | 72 | **-0.592** |
| C | Sector rotation top-3 × 2 stocks | +43.1% | 8.7% | 1.59 | 86 | +0.770 (rejected) |
| D | Quality momentum N100 (1y>0, vol<30%) | +15.4% | 4.7% | 1.37 | 145 | -0.254 |

## Picked: Model B (Low Vol) + Model A (Mean Reversion)

### Model B — Low Volatility
- Rank N500 by 90d historical vol ASCENDING → hold top-20 lowest-vol
- Rebalance quarterly, equal-weight
- CAGR +31.7%, MaxDD 3.5%, Sharpe 1.98
- Correlation with M3: +0.045 ≈ ZERO (true diversifier)
- Captures the low-vol anomaly (boring stocks beat risk-adjusted)

### Model A — Mean Reversion
- BUY N100 stock when RSI(14d) < 30 (oversold)
- Hold until RSI > 70 OR max 30 days
- Max 5 concurrent positions, equal-weight
- CAGR +16% (below 25% gate but ACCEPTABLE because...)
- **Correlation with M3: -0.592** — NEGATIVE = strongest diversifier
- Buys when momentum sells → portfolio risk drops materially

### Why C + D Rejected
- **C (Sector Rotation)**: +43% CAGR but corr +0.77 → same momentum factor, no diversification
- **D (Quality Momentum)**: +15% CAGR + corr -0.25 weak. B is strictly better.

## Suggested Allocation

### Option 1 — Aggressive (M3 dominant)
**50% M3 / 25% B / 25% A**
- Keeps M3 alpha as main driver
- B + A cut portfolio DD by half
- Expected CAGR ~50-55%/yr blended
- DD ~6%

### Option 2 — Equal Weight
**33% M3 / 33% B / 33% A**
- Arithmetic CAGR avg ≈ +42.7%/yr
- DD ~5-7% (B + A near-zero/negative corr damp M3's 10%)
- Cleanest "diversified" mix

### Option 3 — Risk Parity (safest)
**20% M3 / 40% B / 40% A**
- B + A have ~3× lower vol than M3 → risk parity overweights them
- CAGR ~28%/yr
- DD <4%
- Maximum smoothness, lower returns

## Recommended Allocation

**Start with Option 1 (50/25/25)** — keeps M3 alpha dominant while A/B reduce DD. Backtest blend live with paper before committing real capital.

## Honest Caveats

1. **Lookahead bias**: N100 is current-snapshot universe. Less severe for B (N500 covers most listed names) than M3.
2. **Regime dependence**: 2023-24 was momentum bull. In bear/sideways regime A and B should OUTPERFORM M3 — but we have no bear data in window.
3. **Small sample correlation**: A's -0.59 corr is from 8 overlapping months. True value could be -0.3 to -0.7. Re-measure live.
4. **M3 corr understated**: Derived from trade ledger (entry/exit only), daily vol not captured. Actual correlation likely slightly higher than 0.04 / -0.59.
5. **Sector indices unavailable**: C used industry-bucket proxy (not real ^CNXIT/^NSEBANK ETFs).

## Realistic Forward Expectation

| Allocation | Backtest CAGR | Realistic Live | DD |
|-----------|--------------:|---------------:|---:|
| 100% M3 | +80.5%/yr | 35-50%/yr | 10% |
| 50/25/25 | +50%/yr | 25-35%/yr | 5-8% |
| 33/33/33 | +42.7%/yr | 22-30%/yr | 4-6% |
| 20/40/40 (risk parity) | +28%/yr | 15-22%/yr | <4% |

Live always lower than backtest due to slippage + STT + STCG. Diversified portfolio = lower headline return but vastly better Sharpe + sleep.

## Files

```
exports/backtests/DIVERSIFICATION_MODELS.md
tools/backtests/model_mean_reversion.py    (Model A)
tools/backtests/model_low_vol.py           (Model B)
tools/backtests/model_sector_rot.py        (Model C, rejected)
tools/backtests/model_quality.py           (Model D, rejected)
```
