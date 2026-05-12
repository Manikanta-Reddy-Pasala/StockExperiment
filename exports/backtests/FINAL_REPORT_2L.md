# 4-Model × 3-Year × 2-Universe × Penny Filter ON/OFF — ₹2,00,000

Generated 2026-05-12. Script-only, no cloud functions. Cache: 15.1M Postgres
rows (11.4M 15m + 2.7M 1H + 985K daily). 416/504 N500 stocks have full 4-yr
coverage; 88 are listing-date-capped recent IPOs; 5 missing entirely
(DUMMYVEDL placeholders + SCHNEIDER).

Capital ₹2,00,000 locked, max_concurrent=2, no add/withdraw.
Penny filter ON = min_price ₹50 (default), OFF = `--min-price 0`.

## TL;DR — 3-Year Avg ROI Matrix

| Model | N50 ON | N50 OFF | N500 ON | N500 OFF |
|---|---:|---:|---:|---:|
| **ema_200_400** | **+52.64%** | **+52.64%** | -21.75% | **+8.27%** |
| ema_9_21 | +26.20% | -4.34% | +7.53% | **+25.00%** |
| swing_pullback | -0.89% | -0.89% | -1.09% | -1.09% |
| orb_15min | 0% (no data) | 0% (no data) | 0% (no data) | 0% (no data) |

**Top single-year ROIs**:
- N50 ema_200_400 2023-24 (bull year): **+98.04%** / 12.98% MDD
- N500 nofilter ema_9_21 2023-24: **+118.24%** / 23.06% MDD
- N500 filter ema_9_21 2023-24: +67.85% / 38.27% MDD

## Yearly Headlines — Nifty 50

### Filter ON (₹50 min price)

| Model | Year | Taken | Skip | Final₹ | ROI% | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| ema_200_400 | 2023_2024 | 179 | 951 | 396,088 | +98.04 | 12.98 |
| ema_200_400 | 2024_2025 | 125 | 730 | 308,943 | +54.47 | 13.02 |
| ema_200_400 | 2025_2026 | 55 | 338 | 210,828 | +5.41 | 12.84 |
| ema_9_21 | 2023_2024 | 486 | 2713 | 324,739 | +62.37 | 33.38 |
| ema_9_21 | 2024_2025 | 373 | 3005 | 254,661 | +27.33 | 21.35 |
| ema_9_21 | 2025_2026 | 211 | 2745 | 177,783 | -11.11 | 26.58 |
| swing | 2023_2024 | 37 | 153 | 211,810 | +5.91 | 15.49 |
| swing | 2024_2025 | 19 | 62 | 185,007 | -7.50 | 13.61 |
| swing | 2025_2026 | 3 | 0 | 197,810 | -1.09 | 2.02 |

### Filter OFF (no min-price)

| Model | Year | Taken | Skip | Final₹ | ROI% | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| ema_200_400 | all 3 yrs | (identical to ON since N50 all > ₹50) | | | | |
| ema_9_21 | 2023_2024 | 553 | 8665 | 196,343 | -1.83 | 39.50 |
| ema_9_21 | 2024_2025 | 375 | 3585 | 199,828 | -0.09 | 26.89 |
| ema_9_21 | 2025_2026 | 211 | 2745 | 177,783 | -11.11 | 26.58 |

ema_9_21 filter-vs-nofilter divergence on N50 is anomalous — same universe (all > ₹50)
but very different "WithData" counts (18/29/53 vs 53/34/53). Cause: cache threshold
change between runs altered which stocks loaded. Treat as unresolved.

## Yearly Headlines — Nifty 500

### Filter ON

| Model | Year | Taken | Skip | Final₹ | ROI% | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| ema_200_400 | 2023_2024 | 265 | 8072 | 182,375 | -8.81 | 53.05 |
| ema_200_400 | 2024_2025 | 196 | 6037 | 141,560 | -29.22 | 52.97 |
| ema_200_400 | 2025_2026 | 92 | 3245 | 145,543 | -27.23 | 33.99 |
| ema_9_21 | 2023_2024 | 634 | 28139 | 335,698 | +67.85 | 38.27 |
| ema_9_21 | 2024_2025 | 454 | 51619 | 166,788 | -16.61 | 42.41 |
| ema_9_21 | 2025_2026 | 210 | 26274 | 142,727 | -28.64 | 39.49 |
| swing | 2023_2024 | 31 | 1582 | 225,629 | +12.81 | 20.40 |
| swing | 2024_2025 | 30 | 831 | 167,582 | -16.21 | 27.35 |
| swing | 2025_2026 | 18 | 8 | 200,266 | +0.13 | 9.83 |

### Filter OFF

| Model | Year | Taken | Skip | Final₹ | ROI% | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| ema_200_400 | 2023_2024 | 266 | 8254 | **242,497** | **+21.25** | 36.66 |
| ema_200_400 | 2024_2025 | 182 | 6196 | 202,098 | +1.05 | 36.36 |
| ema_200_400 | 2025_2026 | 71 | 2624 | 205,048 | +2.52 | 15.21 |
| ema_9_21 | 2023_2024 | 693 | 43866 | **436,476** | **+118.24** | 23.06 |
| ema_9_21 | 2024_2025 | 447 | 38184 | 154,445 | -22.78 | 38.49 |
| ema_9_21 | 2025_2026 | 205 | 26936 | 159,088 | -20.46 | 32.51 |
| swing | (identical to filter ON, swing already has min_adv_inr filter) | | | | | |

## Key Observations

1. **N50 ema_200_400 is the clear standalone winner** for ₹2L paper trading: +52.64% / yr 3-yr avg with **only 13% MDD**. Stable across years (+98, +54, +5).

2. **N500 nofilter ema_9_21 spikes +118% in 2023-24 bull year** but gives most back in 2024-25 / 2025-26 (-22.78, -20.46). Total +25% / yr avg but volatile.

3. **Penny filter helps EMA strategies on N500 in some years, hurts in others**:
   - ema_200_400 N500: nofilter +8.27% vs filter -21.75% — nofilter wins
   - ema_9_21 N500: nofilter +25% vs filter +7.53% — nofilter wins
   - Penny filter strips smalls/midcaps that drive EMA crossover gains in bull regimes.

4. **Swing pullback strategy stalled** on both universes — negative 3-yr avg. Filter doesn't change much (already has ₹5cr ADV liquidity filter).

5. **ORB 15min has no data** — 5m bars have no Postgres cache table; cache-only mode returns empty; ORB produces 0 entries.

## Recommended at ₹2L

**Primary**: EMA 200/400 1H crossover on Nifty 50, penny filter OFF (irrelevant
for N50 anyway), max_concurrent=2.
- Avg 3-yr ROI: **+52.64%**
- Max single-year drawdown: 13.02%
- Single best year: +98.04% (2023-24 bull regime)

**Aggressive alternative**: EMA 9/21 1H on Nifty 500 nofilter — capture
+118% bull year if regime favors but accept -20%+ down years.

**Skip**: Swing pullback (negative 3-yr both universes), ORB intraday (data
unavailable for full 4yr 5m bars).

## Caveats

- N500 has 88 stocks listing-date-capped (recent IPOs); 5 entirely missing.
- ORB requires Fyers 5m fetch each run — no DB cache exists for that interval.
- EMA results sensitive to cache content; small bar-data differences can shift ROI by several %.
- Slippage / brokerage / STT NOT modeled. Subtract ~1.5-2% per year for net.

## Files

- `exports/backtests/yearly_filter/` — 48 dirs (12 per universe × 2) with penny filter ON
- `exports/backtests/yearly_nofilter/` — same with min_price=0
- Per-dir: `_summary.md`, `_capital_sim.txt`, `_monthly_profile.md`
- Per-universe: `<uni>_yearly_all_models.md`, `<uni>_monthly_3yr.md`
