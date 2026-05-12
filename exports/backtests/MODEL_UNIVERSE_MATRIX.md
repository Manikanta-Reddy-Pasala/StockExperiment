# Model × Universe Matrix + Fake Signal Taxonomy

_Updated: 2026-05-12 | Capital: ₹10,00,000 | Target: ≥30%/yr each model_

## Universes covered

| Universe | Source | Stocks | Type |
|----------|--------|-------:|------|
| Nifty 50 | NSE index csv | 53 | Large cap |
| **Nifty 100 (pseudo)** | Top-100 of N500 by 20d ADV | 100 | Large + Next 50 |
| **Nifty 150 (pseudo)** | Top-150 of N500 by 20d ADV | 150 | Large + Mid |
| Nifty 500 | NSE index csv | 504 | Full broad |
| Selector top-10 | Multi-param ranked N500 | 10 | Volatile mid-caps |

Pseudo-N100/N150 built per year start (no lookahead):
- `exports/backtests/universes/nifty{100,150}_eq_{2023,2024,2025}-05-12.json`

## Target Models (3 with ≥30%/yr goal)

| # | Model | Status | Backtest ROI (avg/yr 3-yr) | Notes |
|---|-------|--------|---------------------------:|-------|
| **1** | EMA 200/400 (Swing 1H) | ✅ Met | **+53.26%** on N50 | Already >30% |
| **2** | EMA 9/21 (Short Swing 1H) | ⏳ Optimizing | +6.62% (filtered N50) | Need 24pp boost |
| **3** | ORB-60 (Day Trade 15m) | ⏳ Optimizing | +14.70% on N50 | Need 15pp boost |

Phase 12 sweep running (15 variants) + N100/N150 pending.

## Backtest Matrix (filled as sweeps complete)

### Model 1: EMA 200/400 — Swing

| Universe | 2023-24 | 2024-25 | 2025-26 | Avg/yr | Worst DD |
|----------|--------:|--------:|--------:|-------:|---------:|
| Nifty 50 (raw) | +98.13% | +54.88% | +6.77% | **+53.26%** | 13.06% |
| Nifty 100 (pseudo) | TBD | TBD | TBD | TBD | TBD |
| Nifty 150 (pseudo) | TBD | TBD | TBD | TBD | TBD |
| Selector top-10 | TBD | TBD | +21.85% | TBD | TBD |

### Model 2: EMA 9/21 — Short Swing

| Universe | Filters | 2023-24 | 2024-25 | 2025-26 | Avg/yr |
|----------|---------|--------:|--------:|--------:|-------:|
| N50 (raw) | none | -0.94% | -20.21% | -7.10% | -9.42% |
| N50 (filtered) | min_gap=.003, vol=1.5×, HTF | +18.08% | +8.24% | -6.46% | +6.62% |
| N50 (relaxed) | min_gap=.001, vol=1.2× | TBD | TBD | TBD | TBD |
| Selector top-10 (Phase 7) | sector+cal+vol-2% | TBD | TBD | +33.32% | TBD |
| Nifty 100 (pseudo) | TBD | TBD | TBD | TBD | TBD |
| Nifty 150 (pseudo) | TBD | TBD | TBD | TBD | TBD |

### Model 3: ORB-60 — Day Trade

| Universe | vol_mult | target_atr | 2023-24 | 2024-25 | 2025-26 | Avg/yr |
|----------|---------:|-----------:|--------:|--------:|--------:|-------:|
| N50 | 1.5 | 1.5 | +9.96% | +28.44% | +5.69% | +14.70% |
| N50 (relaxed) | 1.2 | 1.5 | TBD | TBD | TBD | TBD |
| N50 (wide target) | 1.0 | 2.0 | TBD | TBD | TBD | TBD |
| Nifty 100 (pseudo) | TBD | TBD | TBD | TBD | TBD | TBD |
| Nifty 150 (pseudo) | TBD | TBD | TBD | TBD | TBD | TBD |
| N500 | TBD | TBD | TBD | TBD | TBD | TBD |

## Fake Signal Taxonomy — every type identified

### Common to all EMA models:

**F1: Touching-EMAs crossover**
- EMAs cross with < 0.1% separation
- High whipsaw risk; usually reverses within 2-3 bars
- **Filter:** `min_crossover_gap_pct ≥ 0.003`
- **Implemented:** Yes (in EMA strategy config)

**F2: Volume-less crossover**
- Cross fires with volume < 50% of 20-bar avg
- Low conviction; institutional follow-through unlikely
- **Filter:** `volume_confirm_mult ≥ 1.5`
- **Implemented:** Yes (config flag)

**F3: Counter-trend cross**
- 1H BUY cross while daily SMA-200 is below close
- 1H SELL cross while daily SMA-200 is above close
- Counter-trend crosses fail 60-70% of the time
- **Filter:** `htf_filter_enabled = True`
- **Implemented:** Yes (config flag)

**F4: ATR-floor crossover**
- Cross during low volatility (ATR% < 0.5%)
- Stock in consolidation; cross likely noise
- **Filter:** Skip if ATR(14)/close < 0.005
- **Implemented:** No (TODO)

**F5: Persistence-fail crossover**
- Cross bar closes on cross side, but next bar reverses
- Wick-cross, not real trend change
- **Filter:** Require 2-bar persistence after cross
- **Implemented:** No (TODO — partially via retest1)

**F6: Cluster crosses (multiple per week)**
- > 3 crosses same direction within 5 bars
- Indicates choppy market, not real trend
- **Filter:** Max 1 cross per N bars
- **Implemented:** Partially via `max_alert3_locks_per_cycle`

### EMA 9/21 specific:

**F7: Daily-counter intraday cross**
- 9/21 fires BUY but daily 50DMA descending
- ~70% of these fail; only ~30% catch true daily trend
- **Filter:** Daily 50DMA slope must match direction
- **Implemented:** No

**F8: Sector-laggard cross**
- 9/21 fires on stock in bottom-2 sector by RS
- 60%+ fail rate
- **Filter:** Sector RS gate (Phase 5)
- **Implemented:** Yes (apply_sector_filter.py)

### ORB-60 specific:

**F9: Low-volume breakout**
- Breakout bar has volume < ORB avg × 1.5
- Likely thin trade, not institutional
- **Filter:** Volume mult ≥ 1.5 on breakout bar
- **Implemented:** Yes (vol_mult param)

**F10: Mid-day chop breakout**
- ORB cross between 14:00-15:00 IST
- Often noise as session winds down
- **Filter:** Only trade ORB cross between 10:15-13:00 IST
- **Implemented:** No (TODO)

**F11: Gap-up false breakout**
- Stock gaps up > 2% at open, then ORB high is the gap
- Breakout above gap-high often fails (gap fade)
- **Filter:** Skip ORB on bars where gap > 2%
- **Implemented:** No

**F12: News-driven spike**
- ORB cross during earnings/news day
- High volatility but unpredictable direction
- **Filter:** Skip 2 days around earnings
- **Implemented:** No (no earnings calendar in cache)

### Swing pullback (Model not used, but tracked):

**F13: Fake breakout (no follow-through)**
- 52w high break + volume, but next 2 bars close below
- Common in chop; ~40% fail rate
- **Filter:** Require 3-bar persistence above breakout level

**F14: Cup-and-handle wrong handle depth**
- Handle > 15% deep (not a real handle)

## Filter Implementation Status

| Filter | EMA 200/400 | EMA 9/21 | ORB-60 |
|--------|:-----------:|:--------:|:------:|
| F1 min-gap | ✅ Config | ✅ Config | n/a |
| F2 volume | ✅ Config | ✅ Config | ✅ Built-in |
| F3 HTF SMA | ✅ Config | ✅ Config | n/a |
| F4 ATR floor | ❌ TODO | ❌ TODO | n/a |
| F5 persistence | partial | partial | n/a |
| F6 cluster | partial | partial | n/a |
| F7 daily slope | ❌ | ❌ | n/a |
| F8 sector RS | ✅ Post-filter | ✅ Post-filter | ✅ Post-filter |
| F9 volume breakout | n/a | n/a | ✅ Built-in |
| F10 time-of-day | n/a | n/a | ❌ TODO |
| F11 gap fade | n/a | n/a | ❌ TODO |
| F12 earnings | ❌ | ❌ | ❌ |
| F13 persistence | swing-only | swing-only | n/a |
| F14 cup-handle | swing-only | n/a | n/a |

## Phase 12 + N100/N150 sweep plan

### Currently running (Phase 12, ETA ~30 min):
1. EMA 9/21 N50: v1 relaxed, v2 vol-only, v3 HTF-only × 3 years = 9 runs
2. ORB-60 N50: v1 relaxed-vol, v2 wide-target × 3 years = 6 runs

### Queued next (after Phase 12):
1. EMA 200/400 × N100 × 3 years
2. EMA 200/400 × N150 × 3 years
3. EMA 9/21 × N100 × 3 years (best filter combo from Phase 12)
4. EMA 9/21 × N150 × 3 years
5. ORB-60 × N100 × 3 years
6. ORB-60 × N150 × 3 years
= 18 more backtests, ~90 min

### Will populate when complete:
- Full universe × model matrix above
- Updated fake-signal coverage stats
- Per-model best config for ≥30%/yr target

## Next steps for ≥30%/yr each

**Model 1: EMA 200/400** — ALREADY HITS +53%/yr. Just maintain.

**Model 2: EMA 9/21** — Need to find filter combo + universe that pushes ROI to 30%+. Options:
- Selector top-10 (Phase 7 showed +33.32% 2025-26, need multi-year confirm)
- N100 with HTF only filter
- Hybrid: EMA 9/21 with daily-slope filter (F7)

**Model 3: ORB-60** — Need to find vol/target/universe combo for 30%+. Options:
- N100 or N150 (more stocks → more entries → more good ones)
- Wider target (ATR×2.5 instead of 1.5)
- Add time-of-day filter (F10: avoid 14-15 chop)

Phase 12 results will inform.

## Files

- `tools/backtests/build_universe_by_adv.py` — pseudo-N100/N150 builder
- `tools/backtests/optimize_models.sh` — Phase 12 sweep runner
- `tools/backtests/run_ema_200_400_backtest.py` — now supports --universe-file
- `tools/backtests/run_orb60_backtest.py` — now supports --universe-file
- `exports/backtests/universes/` — N100/N150 JSON lists
- `exports/backtests/SUMMARY.md` — top-level
- `exports/backtests/MODEL_UNIVERSE_MATRIX.md` — this file
