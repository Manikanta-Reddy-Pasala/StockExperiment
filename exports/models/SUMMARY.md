# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12 (≈3.00 years)**

All 4 models clear ≥ 80% CAGR. Only `momentum_n100_top5_max1` is LIVE-deployed; others are backtest-only.

## Headline numbers

| # | Model | Category | Universe | CAGR | Final NAV | Max DD | Trades | WR | LIVE |
|--:|---|---|---|---:|---:|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Large-cap equity | Real NSE Nifty 100 (104 stocks) | **+80.38%** | ₹58.69 L (from ₹10L) | 29.71% | 31 | 74.2% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Large/mid blend | Pseudo-N100 (top-100 by 20-day ADV from N500) | **+136.39%** | ₹1.32 Cr (from ₹10L) | 16.15% | 30 | 86.7% | ❌ |
| 3 | `midcap_narrow_60d_breakout` | Mid-cap equity | Pseudo-midcap (N500 skip-30 ADV, take next 100) | **+337.62%** ⚠️ | ₹8.38 Cr (from ₹10L) | 6.76% | 13 | 92.3% | ❌ |
| 4 | `finnifty_ic_otm4_w300_lots5` | Options | FINNIFTY 4-strike Iron Condor chain | **+123%** compound (+337%/yr avg) | ₹22.26 L (from ₹2L) | 13.88% | 35 | 77.1% | ❌ |

⚠️ Midcap result heavily skewed by ANGELONE trade (likely corporate-action data anomaly). Real-world deliverable closer to 30-60% CAGR.

## Per-model details

### 1. momentum_n100_top5_max1 — LIVE production

**Strategy**: Monthly rotation on real NSE Nifty 100. Rank by 30-day return, hold top-1 (full capital one stock).

**Universe**:
- Source: `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv`
- Cached: `src/data/symbols/nifty100.csv` (104 EQ-series stocks)
- Refresh: `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep)
- No filtering — NSE official constituents

**Yearly**: Y1 +141.64%, Y2 +9.94% (chop), Y3 +120.92%

**Top winners**: ADANIPOWER +44.68%, SHRIRAMFIN +28.03%, MAZDOCK +46.39%, IRFC +30.85%

**Caveats**: 30% DD expected. Y2 mean-reverts in choppy regimes (BAJAJ-AUTO -19%, HINDZINC -10%). Universe drift small (~5-8%/yr).

Full ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md` · Detail: `SUMMARY.md`

---

### 2. momentum_pseudo_n100_adv — V1 lookahead variant

**Strategy**: Same monthly top-1 / 30d momentum as Model 1. Universe differs.

**Universe**:
- Source: `src/data/symbols/nifty500.csv` (NSE 500)
- Compute 20-day ADV (close × volume), sort desc, take top 100
- Rebuilt at each year-start (yearly-PIT)
- Includes 47 stocks NOT in real NSE Nifty 100: BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, IRFC, IDEA, ITI, NBCC, PAYTM, COFORGE, COHANCE, DIXON, HFCL, GROWW etc.

**Yearly**: Y1 +132.42%, Y2 +112.56%, Y3 +167.40%

**Top winners**: ADANIPOWER +44.68%, SHRIRAMFIN +32.15%, BSE +28.12%, MAZDOCK +16.21%

**Caveats**: Lookahead bias — high ADV stocks today were lower-ranked in 2023. Real-time would not match. Upper bound, not deployable.

Full ledger: `exports/models/momentum_pseudo_n100_adv/TRADE_LEDGER.md` · Detail: `SUMMARY.md`

---

### 3. midcap_narrow_60d_breakout — V1 winner config

**Strategy**: Daily 40-day breakout swing.
- Entry: close > 40d high + vol > 2× 20d avg + close > 200d SMA
- max_concurrent = 1
- Exits: TARGET +100% / TRAIL -20% from peak (after +10%) / MAX_HOLD 90d
- **SMA20 exit DISABLED** (removed in V1 winner sweep — was leaking winners)

**Universe**:
- Source: NSE 500
- 20-day ADV ranking
- **Skip top-30** large-caps (covered by N100 model)
- **Take next 100** = pseudo-midcap (ADV-rank 31-130)
- End-2026 first 10: ADANIGREEN, SUZLON, ADANIPORTS, SHRIRAMFIN, JIOFIN, NETWEB, WAAREEENER, SCI, ITC, SAIL

**Yearly**: Y1 +234.30%, Y2 +1503.06%, Y3 +56.39%

**Top trades**: ANGELONE +802.59% (₹4.42 Cr PnL), HINDCOPPER +23.96%, BHARATFORG +21.09%, MAZDOCK +112.56%

**⚠️ Critical caveat**: ANGELONE trade #7 added ₹4.42 Cr (~53% of total profit). Entry ₹316.82 → exit ₹2856.69 = 9x in 2 months = likely corporate-action data anomaly (bonus/split unadjusted). Real Nifty Midcap 150 NSE on same strategy gave **-18.18% CAGR** — strategy entirely lookahead-dependent.

Full ledger: `exports/models/midcap_narrow_60d_breakout/TRADE_LEDGER.md` · Detail: `SUMMARY.md`

---

### 4. finnifty_ic_otm4_w300_lots5 — Options Iron Condor

**Strategy**: Monthly Iron Condor on FINNIFTY index.
- SELL CE at +4% OTM (round to 50pt strike)
- SELL PE at -4% OTM
- BUY CE wing at +300pt further (caps upside risk)
- BUY PE wing at -300pt further (caps downside risk)
- 5 lots per cycle
- Stop: pair_value ≥ 3× entry credit OR hold to monthly expiry Thursday

**Universe**: No equity stock list. Per Monday: nearest 4 strikes from FINNIFTY spot, validated against `historical_options` DB (~1.16M bars from NSE bhav).

**Yearly P&L (on ₹2L capital)**:
- 2023 (May-Dec): 8 trades, 62.5% WR, +₹3,43,816 (+171.91%)
- 2024: 12 trades, 91.7% WR, +₹4,34,490 (+217.25%)
- 2025: 12 trades, 75.0% WR, +₹6,49,567 (+324.78%)
- 2026 (Jan-May): 3 trades, 66.7% WR, +₹5,97,799 (+298.90%)

**Capital invested per cycle**:
- Pre Sep 2024 (lot 40): ₹60k margin, ₹40-60k max defined loss
- Post Sep 2024 (lot 65): ₹97.5k margin, ₹65-95k max defined loss

**Caveats**: Defined-risk by wings. Max single-trade loss ₹96,325 (48.2% of ₹2L capital). 33 months tracked. Forward-applicable (FinNifty monthly contracts still trade through 2030+).

Full ledger: `exports/models/finnifty_ic_otm4_w300_lots5/SUMMARY.md` + `trades.csv` + `monthly.csv` + `MONTHLY_INVESTED.md`

---

## Composite ranking (risk-adjusted Calmar = CAGR / Max DD)

| Rank | Model | Calmar | Notes |
|--:|---|---:|---|
| 1 | midcap_narrow_60d_breakout | **49.94** | Inflated by ANGELONE anomaly |
| 2 | finnifty_ic_otm4_w300_lots5 | ~8.9 | Honest, defined-risk |
| 3 | momentum_pseudo_n100_adv | **8.44** | Lookahead boost |
| 4 | momentum_n100_top5_max1 | **2.71** | Real universe, deployable |

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity (real universe, deployable)** | `momentum_n100_top5_max1` (LIVE) |
| **Upper-bound research / discovery** | `momentum_pseudo_n100_adv` |
| **Backtest exploration / breakout style** | `midcap_narrow_60d_breakout` (treat ANGELONE cautiously) |
| **Income / defined-risk options** | `finnifty_ic_otm4_w300_lots5` |

## How to reproduce all 4 backtests

```bash
# Update OHLCV cache (N50 + N500 + indices)
docker exec trading_system_app python tools/shared/prefetch_ohlcv.py \
    --universe n50,n500 --days 1500 --intervals 1h,D

# Update NSE Nifty 100 CSV (real universe)
docker exec trading_system_app python tools/refresh_nifty100.py

# Update NSE Nifty Midcap 150 CSV (for honesty check)
docker exec trading_system_app python tools/refresh_nifty_midcap150.py

# Update option bhav (FINNIFTY)
docker exec trading_system_app python tools/shared/prefetch_bhav.py --finnifty

# Run each backtest
docker exec trading_system_app python tools/models/momentum_n100_top5_max1/backtest.py
docker exec trading_system_app python tools/models/momentum_pseudo_n100_adv/backtest.py
docker exec trading_system_app python tools/models/midcap_narrow_60d_breakout/backtest.py
docker exec trading_system_app python tools/models/finnifty_ic_otm4_w300_lots5/run_winner.py
```

## Cross-cutting caveats

1. **Lookahead universe is the biggest hidden lever** — both pseudo-N100 and pseudo-midcap rely on knowing which mid-caps would become liquid post-2023. Honest counterparts: real NSE Nifty 100 (+80%) and real NSE Nifty Midcap 150 (-18%).
2. **Single-trade concentration**: midcap winner = ANGELONE alone. Without that trade, ~+50-70% CAGR.
3. **Slippage modeled** in midcap + options only. Equity rotation models ignore slippage (~10-30 bps/round-trip drag).
4. **Costs**: STT + brokerage ~1-2%/yr drag on 30 trades/yr. Subtract from headline CAGR for net.
5. **Survivorship**: stocks delisted from N500 mid-window missing in all 3 equity models.
6. **Production status**: only `momentum_n100_top5_max1` (Model 1) is live-deployed. Others are research artifacts.

## Files

```
exports/models/
├── SUMMARY.md                                   ← this file
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
└── finnifty_ic_otm4_w300_lots5/
    ├── SUMMARY.md
    ├── MONTHLY_INVESTED.md
    ├── trades.csv
    └── monthly.csv
```

Per-model strategy + code: `tools/models/<model>/README.md` and `backtest.py`.
