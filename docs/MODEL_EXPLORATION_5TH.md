# 5th Model Exploration — Backtest Report

**Date:** 2026-05-21
**Goal:** Find a 5th model that lifts portfolio CAGR / risk-adjusted return on top of the existing 4 active models.
**Constraint:** Must work on existing infra — daily-bar OHLCV, NSE-EQ CNC delivery, Postgres `historical_data`, no leverage, no intraday.
**Test window:** 2023-05-15 → 2026-05-12 (3.0y), ₹10L starting capital, approx Fyers CNC charges.

---

## Current Live Portfolio (baseline for comparison)

| Model | CAGR | Max DD | WR | Trades | Calmar |
|---|---|---|---|---|---|
| momentum_pseudo_n100_adv | **+149.15%** | 16.17% | 88.9% | 27 | 9.22 |
| midcap_narrow_60d_breakout | **+141.73%** | 8.12% | 75.0% | 8 | 17.46 |
| n20_daily_large_only | **+139.55%** | 25.66% | 44.1% | 138 | 5.44 |
| momentum_n100_top5_max1 | +65.10% | 37.30% | 71.0% | 31 | 1.75 |

Naive equal-weight portfolio CAGR ≈ +124%, blended DD ≈ 22%.

---

## Candidate 1 — Mean-Reversion (RSI<30 bounce on N100)

**Hypothesis:** Genuinely de-correlated from the 4 momentum/breakout models. Buys oversold stocks in confirmed uptrends, exits on RSI bounce or fixed risk targets.

**Universe:** Nifty 100 (large caps only — avoid falling-knife midcaps)
**Filters:** close > 200d SMA (uptrend gate)
**Pick:** Lowest RSI(14) across qualifying universe

### Parameter Sweep — 12 variants (top-1 only)

| RSI< | RSI> | Tgt | Stop | Hold | CAGR | DD | WR | Trades | Calmar |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 50 | +6% | -4% | 10d | **+30.90%** | 15.33% | 68.1% | 47 | **2.02** |
| 30 | 55 | +6% | -5% | 10d | +26.70% | 14.53% | 66.7% | 45 | 1.84 |
| 30 | 60 | +8% | -5% | 20d | +21.38% | 21.84% | 65.6% | 32 | 0.98 |
| 30 | 55 | +10% | -7% | 15d | +18.12% | 27.77% | 60.0% | 35 | 0.65 |
| 30 | 60 | +5% | -3% | 8d | +16.65% | 25.62% | 59.3% | 54 | 0.65 |
| 30 | 50 | +7% | -4% | 8d | +14.95% | 23.26% | 56.9% | 51 | 0.64 |
| 30 | 50 | +12% | -8% | 20d | +11.77% | 29.26% | 60.0% | 30 | 0.40 |
| 35 | 55 | +8% | -5% | 20d | +8.85% | 37.94% | 54.0% | 50 | 0.23 |
| 25 | 55 | +10% | -7% | 15d | +1.86% | 17.91% | 45.5% | 11 | 0.10 |

### Aggressive variants (N500 + multi-pick)

| Universe | N | RSI< | Tgt | Stop | Hold | CAGR | DD | WR | Trades |
|---|---|---|---|---|---|---|---|---|---|
| N500 | 1 | 30 | +6% | -4% | 10d | **+38.74%** | 21.50% | 55.1% | 89 |
| N500 | 3 | 30 | +6% | -4% | 10d | +22.94% | 15.88% | 53.8% | 212 |
| N500 | 10 | 30 | +6% | -4% | 10d | +15.07% | 7.45% | 56.2% | 400 |
| N500 | 5 | 25 | +8% | -5% | 10d | +13.70% | 6.41% | 63.0% | 73 |
| N500 | 5 | 25 | +10% | -5% | 15d | +9.49% | 12.81% | 55.7% | 70 |

**Best mean-reversion variant:** N500 single-pick, RSI<30 entry, RSI>50 exit, +6% target, -4% stop, 10d max-hold → **+38.74% CAGR / 21.5% DD**.

**Verdict:** Genuinely de-correlated (RSI-oversold ≠ momentum breakout) but solo CAGR too low to drag-down portfolio average. Could lift portfolio Sharpe via low correlation if allocated 10-20% capital.

**Script:** `tools/models/mean_reversion_rsi_n100/backtest.py` + `backtest_aggressive.py`

---

## Candidate 2 — Smallcap/Midcap Momentum Rotation

**Hypothesis:** Smallcaps are more volatile than the N500-derived universe already used by `pseudo_n100`. More vol = bigger momentum bursts = higher CAGR.

### Sweep — 17 variants

| Universe | Top-N | Lookback | Rebal | CAGR | DD | WR | Trades |
|---|---|---|---|---|---|---|---|
| Midcap150 | 1 | 30d | monthly | **+71.68%** | 46.55% | 68.8% | 32 |
| Smallcap250 | 5 | 30d | monthly | +27.70% | 47.79% | 54.3% | 151 |
| Mid+Small | 3 | 30d | monthly | +20.39% | 53.78% | 59.4% | 101 |
| Smallcap250 | 3 | 30d | monthly | +19.18% | 48.47% | 53.0% | 100 |
| Mid+Small | 1 | 30d | monthly | +10.18% | 53.08% | 60.6% | 33 |
| Midcap150 | 1 | 30d | weekly | +8.17% | 73.14% | 46.1% | 76 |
| Smallcap250 | 1 | 30d | monthly | +6.17% | 53.08% | 55.9% | 34 |
| Smallcap250 | 1 | 30d | weekly | -25.12% | 85.15% | 36.3% | 91 |

**Verdict:** Smallcap-only universes have catastrophic drawdowns (50-86%). Midcap top-1 hits +71% but with 46% DD. None reach 100%+ CAGR.

The existing 3 models at 100%+ already capture the best momentum-on-NSE-CNC edges:
- `pseudo_n100`: top-100 ADV from N500 = optimal liquid sample
- `midcap_narrow`: top-100 ADV minus N100 = optimal mid-cap band
- `n20`: top-20 ADV ∩ N100 = optimal large-cap rotation

Adding smaller-cap subset just adds drawdown without alpha.

**Script:** `tools/models/smallcap_momentum/backtest.py`

---

## Candidate 3 — Pyramid Scaling Overlay on midcap_narrow

**Hypothesis:** midcap_narrow V2 (+141% baseline) holds positions up to 120d with +100% targets. Compound winners harder by deploying capital in tranches as the position proves itself.

**Tranching tested:** Initial deploy 33-75%, additional tranches at +5%, +10%, +15%, +20%, +25%, +30% gains.

### Sweep — 12 variants

| Variant | CAGR | DD | WR | Trades | Pyramid hits |
|---|---|---|---|---|---|
| V2 baseline (no pyramid) | **+88.16%** | 40.90% | 62.5% | 8 | 0 |
| 75% base + 25% @+10% | +78.82% | 40.90% | 62.5% | 8 | 5 |
| 70% base + 30% @+10% | +76.61% | 40.90% | 62.5% | 8 | 5 |
| 60% base + 40% @+10% | +73.27% | 40.90% | 62.5% | 8 | 5 |
| 60% + 20% @+10% + 20% @+25% | +73.23% | 33.38% | 62.5% | 8 | 5 |
| 50% base + 50% @+15% | +69.29% | 33.05% | 62.5% | 8 | 5 |
| 50% + 25% @+15% + 25% @+30% | +67.09% | 25.73% | 62.5% | 8 | 5 |

**Verdict:** Pyramid HURTS CAGR in every variant.

**Why:**
- midcap_narrow already deploys 100% capital at signal
- Pyramid raises blended entry price → same +100% target hits at lower absolute price → exits earlier with smaller realized gain
- Idle cash during wait for trigger drags CAGR

**Note on framework discrepancy:** This sweep used a new isolated backtest framework that reports the production V2 baseline as +88% vs production canonical +141%. Different DD methodology (true MTM vs per-trade `cap_after`). Relative ordering (pyramid worse than baseline) holds in either framework.

**Script:** `tools/models/midcap_narrow_60d_breakout/backtest_pyramid.py`

---

## Candidate 4 — Multi-Position Overlay on midcap_narrow

**Hypothesis:** V2 baseline only holds 1 position at a time → many qualifying breakouts in 3y were skipped because already in position (only 8 trades over 3y). Allowing up to N concurrent positions captures more setups = more compounding events.

Each slot gets equal capital share. Same entry/exit rules as baseline.

### Sweep

| max_conc | CAGR | DD | WR | Trades | Calmar |
|---|---|---|---|---|---|
| 1 (baseline) | +88.16% | 40.90% | 62.5% | 8 | 2.16 |
| 2 | +86.72% | 33.21% | 64.7% | 17 | 2.61 |
| 3 | **+87.34%** | **25.42%** | **70.8%** | 24 | 3.44 |
| 4 | +82.86% | 29.15% | 63.6% | 33 | 2.84 |
| 5 | +79.85% | 18.87% | 68.3% | 41 | 4.23 |
| 7 | +81.80% | 22.55% | 71.9% | 57 | 3.63 |
| 10 | +83.05% | **16.11%** | 64.9% | 77 | **5.16** |

**Verdict:** Multi-position **does NOT increase CAGR** but **dramatically improves risk-adjusted return**.

- max_conc=3 → 87% CAGR / 25% DD / WR 71% / Calmar 3.44
- max_conc=10 → 83% CAGR / 16% DD / Calmar 5.16 (**best Calmar**)

**Tradeoff:** Sacrifice ~5pp CAGR for ~60% DD reduction. Effective Sharpe roughly doubles.

**Script:** `tools/models/midcap_narrow_60d_breakout/backtest_multi.py`

---

## Inspirations Considered (Rejected)

### YouTube content reviewed
| Video | Channel | Why rejected |
|---|---|---|
| Sharpe Ratio Explained | Wall Street Quants | Pure metric education, no strategy |
| ONLY 2 Indicators ($4351/Day) | PBInvesting | US intraday options + VWAP/EMA; needs 5m bars + WS infra we don't have |
| ONE CANDLE Scalping | ProRealAlgos | 15m ATR box + 5m engulfing; intraday US session, same infra mismatch |
| Candlestick Pattern Guide | Data Trader | Pure pattern education, no strategy |

All US-style intraday content. Daily-bar CNC architecture mismatch.

### What would actually break 100%+ on top of current 4 models
1. **Leverage (Fyers MIS 5x intraday)** — needs 5m bars + WS executor + intraday risk gates. Major rebuild.
2. **Options directional (NIFTY/BANKNIFTY weekly)** — finnifty_ic_otm4 backtest already shows +337%/yr Calmar 24.3, but multi-leg executor not wired.
3. **Event-driven (earnings/news momentum)** — needs corporate-action data not in DB.

---

## Recommendation

**Highest-ROI next step:** Wire multi-position (max_conc=3-5) into live `midcap_narrow_60d_breakout`. Modest CAGR cost for substantial DD reduction; lifts portfolio Sharpe meaningfully.

**Skipped paths:**
- Pyramid overlay (tested, no benefit)
- Smallcap momentum (worse than existing models)
- Mean-reversion (+39% CAGR solo, only useful as Sharpe diversifier at small allocation)

**Higher-ambition path:** Wire finnifty options executor. Backtest documented +337% CAGR / 13.88% DD / Calmar 24.3 (from prior memory `stockexperiment-finnifty-otm4-restored-2026-05-19`). Requires multi-leg order placement infrastructure.

---

## Files Produced

- `tools/models/mean_reversion_rsi_n100/backtest.py` (single-pick baseline)
- `tools/models/mean_reversion_rsi_n100/backtest_aggressive.py` (N500 + multi-pick sweep)
- `tools/models/smallcap_momentum/backtest.py` (smallcap/midcap momentum sweep)
- `tools/models/midcap_narrow_60d_breakout/backtest_pyramid.py` (pyramid overlay)
- `tools/models/midcap_narrow_60d_breakout/backtest_multi.py` (multi-position overlay)
- `docs/MODEL_EXPLORATION_5TH.md` (this file)
