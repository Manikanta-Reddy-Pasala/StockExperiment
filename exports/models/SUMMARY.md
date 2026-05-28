# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12** · ₹10L per model · Fyers data (10-yr re-pull 2026-05-28; daily history now spans 2016-04-11 → present)

4 equity models. Each has ONE canonical version. Last refreshed **2026-05-28** after the n100 LOOKBACK 30→15 switch (6yr walk-forward validated) and the 10-year backfill.

(finnifty_ic_otm4_w300_lots5 options model discarded — NIFTY/FinNifty IC abandoned; use equity momentum instead.)

> **n100 canonical numbers updated 2026-05-27 (LB=30 → LB=15 trading days).** The new lookback
> won 4 of 6 walk-forward years AND the full-period CAGR+DD over 2020-2026 (LB15 +151.7%/45.7%DD
> vs LB30 +129.0%/57.3%DD). Previous canonical (LB=30) was +125.13% CAGR / 28.21% DD / Calmar 4.44.
> See `momentum_n100_top5_max1/README.md` > "Change history".
>
> **10-Year Backtest Appendix** (2016-05 → 2026-05) lives in the repo root `/README.md`. 10yr CAGR
> is materially lower for all 4 models — 2017-2019 was regime-hostile across the board (compound
> n100 −25%, pseudo −76%, n20 −23%, midcap −28% over those 3 years).

## Code architecture (KISS / separation of concerns)

Each model is 3 layers; backtest and live share the decision so they cannot drift:

| Layer | Where | Shared? |
|---|---|---|
| SELECTION — universe + filters + ranking | each model's `backtest.py` / `live_signal.py` | per-model config |
| RULE — entry/exit decision | `tools/shared/rotation_strategy.py` (n100/pseudo/n20), `tools/shared/breakout_strategy.py` (midcap) | ✅ backtest + live |
| EXECUTION — backtest accounting/metrics | `tools/shared/backtest_engine.py` (rotation models) | ✅ |
| EXECUTION — live orders | `tools/live/fyers_executor.py` | ✅ all models |

Guarded by `tests/test_rotation_parity.py` + `tests/test_breakout_parity.py` (26 cases):
the rule a backtest tests is the exact code that trades live. All backtests reproduce these
headline numbers to the rupee after the refactor.

## Headline — deploy ₹10L per model (3-year backtest)

| # | Model | Universe | Rebalance | Final NAV | CAGR | Max DD | LIVE |
|--:|---|---|---|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Real NSE Nifty 100 (LB=15td + mid-month) | Monthly + mid-month | ₹22,814,268 | **+184.36%** | **14.89%** | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Top-100 ADV from N500 − Small + uptrend + MAX_PRICE≤₹3K | Monthly | ₹15,361,000 | **+149.15%** | **16.17%** | ✅ |
| 3 | `midcap_narrow_60d_breakout` | Top-100 ADV from N500 − Large | Event-driven | ₹14,134,367 | **+141.73%** | **8.12%** | ✅ |
| 4 | `n20_daily_large_only` | Top-20 ADV + uptrend + NSE Nifty 100 | Daily | ₹13,655,640 | **+139.55%** | 25.66% | ✅ |

### 10-year backtest (2016-05-15 → 2026-05-12, canonical configs)

| # | Model | CAGR | Max DD | Calmar | Trades | WR |
|--:|---|---:|---:|---:|---:|---:|
| 1 | `momentum_n100_top5_max1` (LB=15 + mid-month — LIVE) | **+67.14%** | 60.89% | 1.10 | 201 | 54.7% |
| 2 | `n20_daily_large_only` | +45.28% | 53.54% | 0.85 | 519 | 47.1% |
| 3 | `midcap_narrow_60d_breakout` | +21.00% | 53.14% | 0.40 | 29 | 65.5% |
| 4 | `momentum_pseudo_n100_adv` | +10.87% | **88.04%** | 0.12 | 97 | 59.8% |

> Previous 10yr appendix showed n100 +43.31% / 60% DD — that was the WITHOUT-mid-month canonical default. The WITH-mid-month variant (what `live_signal.py` actually runs) is +67.14% / 60.89% / Calmar 1.10. Pseudo / n20 / midcap unchanged.

### n100 lookback sweep (2026-05-28 research — NOT yet live)

The 2026-05-27 switch to LB=15 was based on a **6-year** walk-forward (2020-2026, bull-only). With full 10-year data including the regime-hostile 2017-2019, **LB=30 outperforms LB=15** on every 10yr metric:

| n100 LOOKBACK | 10yr CAGR | 10yr Max DD | 10yr Calmar | 3yr CAGR | 3yr Max DD |
|---:|---:|---:|---:|---:|---:|
| 15 (LIVE) | +67.14% | 60.89% | 1.10 | **+184.36%** | **14.89%** |
| 20 | +58.37% | 53.17% | 1.10 | +118.66% | 15.93% |
| 25 | +46.64% | 61.70% | 0.76 | +129.25% | 22.24% |
| **30** | **+91.27%** | 57.30% | **1.59** | +125.13% | 28.21% |
| 35 | +73.91% | **46.48%** | 1.59 | +86.23% | 21.90% |
| 40 | +64.00% | 58.22% | 1.10 | +98.76% | 19.39% |
| 45 | +61.69% | 43.23% | 1.43 | +76.84% | 19.39% |

**Findings**:
- **LB=30 is best 10yr** (+91.27% CAGR vs LB=15's +67.14% = +24pp uplift, AND tighter DD 57.30 vs 60.89).
- **LB=15 is best 3yr** (+184.36% — wins the 2023-2025 bull window decisively over LB=30's +125.13%).
- Trade-off is regime-recency vs decade-robustness. Live currently runs LB=15 (recency-optimized). NOT yet changed.

### Other variants tested 2026-05-28 (research only)

| Model | Variant | 10yr CAGR | 10yr DD | Calmar | Notes |
|---|---|---:|---:|---:|---|
| pseudo | top-5 equal-weight | +16.27% | 52.15% | 0.31 | Best pseudo 10yr; 2017 cut from -60% to -10% (multi-position diversifies away single-stock concentration). 3yr cost: +149 → +57. |
| pseudo | TIER L=400/M=200/S=100 ₹Cr ADV gate | +14.99% | 75.17% | 0.20 | Per-tier liquidity filter; 2017 cut to -29%. 3yr cost: +149 → +137. |
| n20 | TIER L=200/M=100/S=50 ADV gate | +47.85% | 51.34% | 0.93 | Small free 10yr win; 3yr unchanged. |
| n100 | top-3 equal-weight + mid-month | +40.33% | 36.31% | 1.11 | Best Calmar; DD halved vs canonical; CAGR lower. |
| n100 | top-3 + LB=30 + mid-month (combo) | not yet run | | | Potential further sweet spot |

**60-150% 10yr target reality check**:
- Only `momentum_n100_top5_max1` LB=30 hits the band (+91% inside 60-150).
- Pseudo / n20 / midcap CANNOT clear 60% over 10yr without leverage or fundamentally new architecture (sector rotation, factor overlays, multi-factor scoring).
- Indian momentum literature (Capitalmind premium service) caps at ~35% CAGR / decade. n100 LB=30's 91% is exceptional and partly survivorship-inflated.
- Leverage estimate on n100 LB=30: 1.25× = ~118% / DD 75; 1.5× = ~137% / DD 87. Brings 60-150 target inside reach but DD scales linearly.

10yr numbers carry **survivorship bias of ~5-15pp** (universes are today's CSVs; IRFC / MAZDOCK / ETERNAL / LIC etc. didn't trade pre-2020). DD figures stay reliable. See repo root `/README.md` > "10-Year Backtest Appendix" for per-year breakdown.

## Unique stock-filtering approach per model

| Model | Filter mechanism | Why unique |
|---|---|---|
| `momentum_n100_top5_max1` | NSE Nifty 100 official list (no derived filter) | Only model using REAL NSE constituents — large-cap pure baseline |
| `momentum_pseudo_n100_adv` | Top-100 ADV from N500, drop Small, uptrend, price ≤ ₹3,000 | Liquidity + trend + price gate; only model retaining MAX_PRICE filter (justified by share-count floor heuristic) |
| `midcap_narrow_60d_breakout` | Top-100 ADV from N500, drop Large | Highest-liquidity mid/small breakouts; event-driven |
| `n20_daily_large_only` | Top-20 ADV + close>200d SMA + NSE Nifty 100 | Smallest universe (20); strictest gate; daily rebuild |

## Composite ranking — risk-adjusted Calmar (3-year, CAGR / Max DD)

| Rank | Model | CAGR | MaxDD | Calmar |
|--:|---|---:|---:|---:|
| 1 | midcap_narrow_60d_breakout | +141.73% | 8.12% | **17.46** |
| 2 | momentum_n100_top5_max1 | +184.36% | 14.89% | **12.38** |
| 3 | momentum_pseudo_n100_adv | +149.15% | 16.17% | **9.22** |
| 4 | n20_daily_large_only | +139.55% | 25.66% | **5.44** |

> n100's previous ranking (#4, Calmar 4.44 from +125.13% / 28.21%) reflected the LOOKBACK=30 + mid-month config. The 2026-05-27 switch to LB=15td bumped it to **#2** (12.38 Calmar).

## MAX_PRICE filter decision history (2026-05-17)

Initially applied MAX_PRICE filter to all 3 momentum models. Reviewed and removed on N100 + N20 because threshold (₹3K / ₹2.5K) was curve-fit on backtest losses (in-sample bias) — replicates only if next-3-yr losers happen to sit in same price buckets.

**Kept on PSEUDO** as a defensible position-sizing heuristic: with ₹30K live capital, any share priced > ₹3,000 leaves <10 shares — i.e. 1 share = >10% of capital, which is excessive per-trade concentration. The filter is observable from current price (no future data) and applies identically live.

**Removed on N100 (LIVE)** and **N20** to give honest unfiltered backtest numbers. CAGR drops back to baseline but no hidden tuning.

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity — real NSE universe, highest CAGR + strong Calmar** | `momentum_n100_top5_max1` (LIVE, +184% CAGR / Calmar 12.38 / 3yr) |
| **Live equity — best risk-adjusted Calmar overall** | `midcap_narrow_60d_breakout` (LIVE, Calmar 17.46 / 3yr) |
| **Live equity — yearly-PIT universe + highest WR** | `momentum_pseudo_n100_adv` (LIVE, Calmar 9.22 / WR 88.9%) |
| **Live equity — high-churn daily rotation** | `n20_daily_large_only` (LIVE, +45% CAGR / 10yr) |

## Stock Overlap Across Models

Stocks that appear in trade ledgers of 2+ equity models (multi-model conviction signal).

### Large-cap (NSE Nifty 100)

| Stock | n100 | pseudo | midcap | n20 | Total models |
|---|:-:|:-:|:-:|:-:|:-:|
| ADANIPOWER | ✓ | ✓ |  | ✓ | 3 |
| ETERNAL | ✓ | ✓ |  | ✓ | 3 |
| IRFC | ✓ | ✓ |  | ✓ | 3 |
| MAZDOCK | ✓ | ✓ |  | ✓ | 3 |
| PFC | ✓ | ✓ |  | ✓ | 3 |
| SHRIRAMFIN | ✓ | ✓ |  | ✓ | 3 |
| ADANIGREEN | ✓ | ✓ |  |  | 2 |
| BAJFINANCE |  | ✓ |  | ✓ | 2 |
| HAL | ✓ |  |  | ✓ | 2 |
| HINDZINC | ✓ |  |  | ✓ | 2 |
| ONGC |  | ✓ |  | ✓ | 2 |
| RECLTD |  | ✓ |  | ✓ | 2 |
| TRENT | ✓ |  |  | ✓ | 2 |

### Mid-cap (NSE Nifty Midcap 150)

| Stock | n100 | pseudo | midcap | n20 | Total models |
|---|:-:|:-:|:-:|:-:|:-:|
| BSE |  | ✓ | ✓ |  | 2 |
| COCHINSHIP |  | ✓ | ✓ |  | 2 |
| INDUSTOWER |  | ✓ | ✓ |  | 2 |

### Small-cap (NSE Nifty Smallcap 250)

_None._

### Other / outside top-500

_None._

**Interpretation**: stocks appearing in 3+ models are high-conviction across strategies. Used for portfolio concentration decisions.

## Data integrity

- All data from Fyers production source (cont_flag=1 split-adjusted)
- 498/504 N500 stocks covered (4 DUMMYVEDL placeholders + 2 symbol mismatch = legitimately missing)
- 1 known anomaly: ABFRL 2025-05-22 real demerger (-67%, not data bug)
- yfinance NEVER used (project rule)

## Files

```
exports/models/
├── SUMMARY.md                          ← this file
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
└── n20_daily_large_only/
    ├── SUMMARY.md
    └── TRADE_LEDGER.md
```
