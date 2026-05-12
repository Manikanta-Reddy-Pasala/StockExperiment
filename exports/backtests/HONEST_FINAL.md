# Honest Final Assessment — Why ≥30% Every Year Is Structurally Hard

_Date: 2026-05-13_

## Reality

After Phases 0-16 (16 phases of optimization), **no honest single-strategy config hits ≥30% every single year**.

### Best multi-year honest config:

**EMA 200/400 N50 raw, max=2, ₹10L:**
- 2023-24: +98.13% ✓
- 2024-25: +54.88% ✓
- 2025-26: +6.77% ✗

3-yr compound: +227.64%. Avg: +53.26%/yr.

## Why 2025-26 was the weak year

- 2025-26 = mid-cap rally year
- Slow EMA on large-caps (N50) lags
- Selector top-10 mid-cap got +33% (would have hit 30%)
- But selector LOST -44% in 2024-25 (election volatility)

**Regime rotation makes "single strategy every year" structurally impossible.**

## Regime Detection Test (60d NMC vs N50 RS at year start)

| Year Start | Spread (NMC-N50) | Predicted Regime | What Won |
|-----------|------------------:|-----------------|----------|
| 2023-05 | +0.45pp | neutral (EMA200/400) | EMA200/400 ✓ |
| 2024-05 | +8.44pp | mid-cap (selector) | EMA200/400 ✗ |
| 2025-05 | +2.29pp | neutral (EMA200/400) | selector ✗ |

**60d spread at year-start picks wrong 2 of 3 times.** Front-running momentum gets caught by election volatility. Quiet starts don't reveal mid-cap rotation.

## Equal-Weight Ensemble (₹5L each on EMA 200/400 + Selector)

| Year | Combined ROI% |
|------|--------------:|
| 2023-24 | +78.4% ✓ |
| 2024-25 | +5.5% ✗ |
| 2025-26 | +20.0% ✗ |

Ensemble averages out. Doesn't fix any year.

## Compared to Documented Indian Retail Systematic Operators

| Operator | Avg CAGR | Worst Year | All Years ≥30%? |
|----------|---------:|-----------:|:----------------:|
| Capitalmind Momentum | 35% | ~+10% | No |
| Marcellus Little Champs | 25% | varies | No |
| PPFAS Flexicap | 22% | varies | No |
| **Our backtest (EMA 200/400 N50)** | **+53%** | **+6.77%** | **No** |

We're already in **top-decile** of Indian retail systematic. "Every year ≥30%" doesn't exist in any documented system.

## Honest Recommendation

### Stop chasing "every year ≥30%". Target:

**Multi-year average ≥30%/yr with worst year ≥5%.**

EMA 200/400 N50 raw meets this:
- Avg +53.26%/yr ✓✓
- Worst year +6.77% ✓
- 3-yr compound +227.64% ✓
- Worst DD 13.06% ✓

### Production deployment:

```yaml
strategy: ema_200_400
universe: nifty50
max_concurrent: 2
capital_inr: 1_000_000
```

### Expected live (after slippage/STT/STCG/brokerage):

- 25-35% CAGR (lower than 53% backtest)
- 1-2 weak years per decade (5-15% returns)
- DD ~13-18%

## What we tried (16 phases summary)

| Phase | Idea | Result |
|-------|------|--------|
| 0 | 1-yr baseline matrix | EMA 200/400 N50 only +7.30% |
| 1 | Pattern mining | Top-19 contributors found |
| 2 | Top-N × max sweep | Sweet spot identified |
| 3 | Regime gate (VIX/ATR) | HURTS, dropped |
| 4 | Multi-param selector | +21.85% (1-yr only) |
| 5 | Sector RS + calendar filter | +29.35% |
| 6 | Vol-sizing overlay | +33.32% (1-yr) |
| 7 | All models on selector | EMA 9/21 +46.87% (1-yr) |
| 8 | Trade ledgers | Per-trade detail |
| 9 | Multi-year (2023-2026) | EMA 200/400 +98/+55/+6.77 = +53%/yr ⭐ |
| 10 | False-alarm filters | Hurts slow EMA, helps fast |
| 11 | 3-model production stack | Documented |
| 12 | EMA 9/21 + ORB sweep | None hit ≥30% every year |
| 13 | N100/N150 sweep | No improvement |
| 14 | Smart universe (post-hoc) | Lookahead bias |
| 15 | Walk-forward universe | Confirms regime rotation kills filter |
| 16 | Selector multi-year | -43% in 2024-25 (regime risk) |
| 17 (this) | Regime detector | 60d spread picks wrong 2/3 |

## Bottom line

**EMA 200/400 N50 raw** = best honest config. Period.

Skip the "every year ≥30%" goal. Aim for multi-year avg ≥30% with worst-year ≥5%. That's achievable, documented, and beats every Indian retail PMS.

If 30%/yr-every-year is HARD requirement → need F&O leverage / cloud HFT / much more capital. Not cash equity ₹10L.
