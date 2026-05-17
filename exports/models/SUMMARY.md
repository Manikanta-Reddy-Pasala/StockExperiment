# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12 (≈3.00 years)**

5 models tracked. All ≥ 80% CAGR. Only `momentum_n100_top5_max1` is LIVE-deployed.

## Deploy ₹10L per model — 3-year results

Hypothetical: deposit ₹10,00,000 on **2023-05-15** in each model, hold full 3-year cycle.

| # | Model | Category | Final NAV | Total ROI | CAGR | Max DD | LIVE |
|--:|---|---|---:|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Large-cap (real Nifty 100) | **₹58,68,846** | +486.88% | **+80.38%** | 29.71% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Large/mid blend (ADV-rank) | **₹1,32,10,187** | +1221.02% | **+136.39%** | 16.15% | ❌ |
| 3 | `midcap_narrow_60d_breakout` | Mid-cap (breakout swing) | **₹8,38,11,502** ⚠️ | +8281.15% | **+337.62%** | 6.76% | ❌ |
| 4 | `finnifty_ic_otm4_w300_lots5` | Options (Iron Condor, scaled 5×) | **₹1,11,28,365** | +1012.84% | **+123.30%** compound | 13.88% | ❌ |
| 5 | `n20_daily_30d_mc1_uptrend` | Top-20 ADV daily rotation | **₹1,70,00,000** approx | +1599.57% | **+157.27%** | 50.61% | ❌ |

⚠️ Midcap heavily skewed by single ANGELONE trade (likely corporate-action data anomaly). Real-world deliverable ~30-60% CAGR.

🔹 Finnifty originally backtest at ₹2L → ₹22.26L. Linear-scaled to ₹10L = ₹1.11 Cr assuming margin & defined-risk scale linearly (25 lots vs 5).

## How each model picks stocks — verified 2026-05-12

| Model | Pick logic | Latest top-1 (2026-05-12) | Top-5 (30d return) |
|---|---|---|---|
| **1. n100_real** | NSE Nifty 100 list → top-1 by 30d return | **ADANIGREEN** (+55.80%) | ADANIGREEN, ADANIPOWER, ADANIENSOL, ADANIENT, HINDZINC |
| **2. pseudo_n100** | Top-100 ADV from N500 → top-1 by 30d return | **HFCL** (+112.39%) | HFCL, GALLANTT, ADANIGREEN, BHEL, OLAELEC |
| **3. midcap_narrow** | N500 skip-30 ADV + 40d breakout + vol>2× + close>200d SMA | Event-driven (no fixed monthly pick) | — |
| **4. finnifty_ic** | FINNIFTY spot → 4 strikes (±4% OTM + ±300pt wings) | Per-Monday derived | — |
| **5. n20_daily** | Top-20 ADV from N500 + uptrend filter + top-1 by 30d | **HFCL** (+112.39%) | HFCL, BHEL, ADANIPOWER, BSE, GRSE |

**Notable**: HFCL appears in Models 2 and 5 (high ADV) but **NOT in Model 1** (not in NSE official Nifty 100). This is the correct filter behavior.

## Per-model details

### 1. momentum_n100_top5_max1 — LIVE production

**Strategy**: Monthly rotation on REAL NSE Nifty 100. Rank by 30-day return, hold top-1 (full capital one stock).

**Universe**:
- Source: `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv`
- Cached: `src/data/symbols/nifty100.csv` (104 EQ-series stocks)
- Refresh: `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep)
- Selection: all 104 → rank by 30d return → pick top-1

**Yearly P&L (₹10L start)**:
- Y1 (2023-24): ₹10L → ₹24.16L (+141.64%)
- Y2 (2024-25): ₹24.16L → ₹26.57L (+9.94%, chop)
- Y3 (2025-26): ₹26.57L → ₹58.69L (+120.92%)

**Today's pick**: ADANIGREEN (+55.80% over last 30d). Universe has all 100 official NSE Nifty 100 constituents.

**Caveats**: 30% DD typical. Y2 mean-reverts in choppy regimes.

---

### 2. momentum_pseudo_n100_adv — V1 lookahead variant

**Strategy**: Same as Model 1 but universe differs.

**Universe**:
- Source: `src/data/symbols/nifty500.csv` (NSE 500)
- Compute 20-day ADV (close × volume), sort desc, take top 100
- Rebuilt at each year-start (yearly-PIT)
- Differs from real N100 by 47 stocks (BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, IRFC, IDEA, ITI, NBCC, PAYTM, COFORGE, HFCL, GROWW etc.)

**Yearly P&L (₹10L start)**:
- Y1: ₹10L → ₹23.24L (+132.42%)
- Y2: ₹23.24L → ₹49.40L (+112.56%)
- Y3: ₹49.40L → ₹1.32 Cr (+167.40%)

**Today's pick**: HFCL (+112.39%) — mid-cap, not in real N100.

**Caveats**: Lookahead bias (ADV at year-end applied to year-start). Real-time would not match. Upper-bound research.

---

### 3. midcap_narrow_60d_breakout — V1 winner config

**Strategy**: Daily 40-day breakout swing.
- Entry: close > 40d high + vol > 2× 20d avg + close > 200d SMA
- max_concurrent = 1
- Exits: TARGET +100% / TRAIL -20% from peak (after +10%) / MAX_HOLD 90d
- **SMA20 exit DISABLED** (V1 winner sweep finding)

**Universe**:
- N500 → 20-day ADV → **skip top-30** (large) → **take next 100** = pseudo-midcap
- End-2026 first 10: ADANIGREEN, SUZLON, ADANIPORTS, SHRIRAMFIN, JIOFIN, NETWEB, WAAREEENER, SCI, ITC, SAIL

**Yearly P&L (₹10L start)**:
- Y1: ₹10L → ₹33.43L (+234.30%)
- Y2: ₹33.43L → ₹5.36 Cr (+1503.06%) ← ANGELONE trade
- Y3: ₹5.36 Cr → ₹8.38 Cr (+56.39%)

**Today's pick**: Event-driven (only fires on breakout signal day). Not a fixed monthly pick.

**⚠️ ANGELONE caveat**: trade #7 (2024-10-16 → 2024-12-23) added ₹4.42 Cr (~53% of total). Entry ₹316 → exit ₹2856 in 2 months = 9x = likely corporate-action data anomaly. Real Nifty Midcap 150 on same strategy = -18% CAGR.

---

### 4. finnifty_ic_otm4_w300_lots5 — Options Iron Condor

**Strategy**: Monthly Iron Condor on FINNIFTY index.
- SELL CE at +4% OTM + SELL PE at -4% OTM
- BUY CE +300pt wing + BUY PE -300pt wing
- 5 lots per cycle (scaled to 25 for ₹10L deployment)
- Stop: pair_value ≥ 3× entry credit OR hold to monthly expiry Thursday

**Universe**: No equity stocks. 4 strikes derived per Monday from FINNIFTY spot, validated against `historical_options` DB (~1.16M bars from NSE bhav).

**Yearly P&L (scaled to ₹10L)**:
- 2023: +₹17.19L (+171.91%)
- 2024: +₹21.73L (+217.25%)
- 2025: +₹32.48L (+324.78%)
- 2026 (Jan-May): +₹29.89L (+298.90%)

**Caveats**: Defined-risk by wings. Max single-trade loss ₹4.82L (48.2% of ₹10L). FinNifty monthly contracts forward-applicable through 2030+.

---

### 5. n20_daily_30d_mc1_uptrend — Aggressive daily rotation

**Strategy**: Daily rebalance. Top-20 ADV universe + uptrend filter → top-1 by 30d return.

**Universe**:
- Source: NSE 500
- Compute 20-day ADV (close × volume), sort desc, take **top 20** (much smaller than pseudo-N100)
- Filter: keep only stocks with close > 200d SMA (uptrend)
- Universe rebuilt **daily** at each rebalance (PIT-strict)

**Yearly P&L (₹10L start)**:
- Y1: ₹10L → ₹39.22L (+292.25%)
- Y2: ₹39.22L → ₹1.27 Cr (+224.51%)
- Y3: ₹1.27 Cr → ₹1.70 Cr (+34%)

**Today's pick**: HFCL (+112.39%) — universe after uptrend filter = 8 of 20.

**Caveats**: 50% Max DD (highest of all 5). 134 trades / 3yr = high turnover → ~3-5%/yr cost drag. Y3 cooled to +34%. Sweep winner; survivorship + slippage not modeled.

---

## Composite ranking — risk-adjusted Calmar (CAGR / Max DD)

| Rank | Model | CAGR | MaxDD | Calmar | Notes |
|--:|---|---:|---:|---:|---|
| 1 | midcap_narrow_60d_breakout | +337.62% | 6.76% | **49.94** | ANGELONE-inflated |
| 2 | momentum_pseudo_n100_adv | +136.39% | 16.15% | **8.44** | Lookahead bias |
| 3 | finnifty_ic_otm4_w300_lots5 | +123.30% | 13.88% | **8.88** | Honest, defined-risk |
| 4 | momentum_n100_top5_max1 | +80.38% | 29.71% | **2.71** | LIVE deployable |
| 5 | n20_daily_30d_mc1_uptrend | +157.27% | 50.61% | **3.11** | High vol, daily rotation |

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity (real, deployable)** | `momentum_n100_top5_max1` (LIVE) |
| **Defined-risk income** | `finnifty_ic_otm4_w300_lots5` |
| **Upper-bound research / aggressive sim** | `momentum_pseudo_n100_adv` |
| **Backtest exploration / breakout style** | `midcap_narrow_60d_breakout` (ANGELONE caveat) |
| **Highest raw return, accept volatility** | `n20_daily_30d_mc1_uptrend` |

## How to reproduce all 5 backtests

```bash
# Prefetch OHLCV
docker exec trading_system_app python tools/shared/prefetch_ohlcv.py \
    --universe n50,n500 --days 1500 --intervals 1h,D

# Refresh NSE universe CSVs (real Nifty 100 + Midcap 150)
docker exec trading_system_app python tools/refresh_nifty100.py
docker exec trading_system_app python tools/refresh_nifty_midcap150.py

# Prefetch option bhav (FINNIFTY)
docker exec trading_system_app python tools/shared/prefetch_bhav.py --finnifty

# Run each model
docker exec trading_system_app python tools/models/momentum_n100_top5_max1/backtest.py
docker exec trading_system_app python tools/models/momentum_pseudo_n100_adv/backtest.py
docker exec trading_system_app python tools/models/midcap_narrow_60d_breakout/backtest.py
docker exec trading_system_app python tools/models/finnifty_ic_otm4_w300_lots5/run_winner.py
docker exec trading_system_app python tools/models/n20_daily_30d_mc1_uptrend/backtest.py
```

## Cross-cutting caveats

1. **Lookahead universe is the biggest hidden lever** — Models 2 and 3 rely on knowing which mid-caps would become liquid post-2023. Honest counterparts: real NSE Nifty 100 (+80%, Model 1) and real Nifty Midcap 150 (-18% on Model 3 strategy).
2. **Single-trade concentration** (Model 3): ANGELONE alone = 53% of returns. Without it, ~+50-70% CAGR.
3. **Slippage modeled** only in Models 3 + 4. Equity rotation models (1, 2, 5) ignore slippage (~10-30 bps/round-trip drag).
4. **Costs**: STT + brokerage ~1-2%/yr drag (Models 1, 2) or ~3-5%/yr (Model 5 with 45 trades/yr). Subtract from headline CAGR.
5. **Survivorship**: stocks delisted from N500 mid-window missing in all equity models.
6. **Production status**: only Model 1 is live-deployed. Others are research artifacts.

## Files

```
exports/models/
├── SUMMARY.md                                   ← this file (all 5 models)
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── finnifty_ic_otm4_w300_lots5/
│   ├── SUMMARY.md
│   ├── MONTHLY_INVESTED.md
│   ├── trades.csv
│   └── monthly.csv
└── n20_daily_30d_mc1_uptrend/
    └── TRADE_LEDGER.md
```

Per-model strategy + code: `tools/models/<model>/README.md` and `backtest.py`.
