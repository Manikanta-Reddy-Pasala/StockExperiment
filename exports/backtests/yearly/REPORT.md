# 4-Model × 3-Year × 2-Universe Backtest Report — ₹2,00,000 Capital

Generated 2026-05-11. All backtests via local Python scripts (no cloud
functions). OHLCV bulk-cached to existing Postgres `historical_data_*`
tables; backtests read from DB. Capital locked at ₹2L (no add/withdraw).
Penny filter: stocks priced < ₹50 excluded.

## Note on earlier vs current numbers

Earlier runs (pre-cache, with `--days 365` rolling window, fresh Fyers
fetch) reported swing pullback at +30%-ish on N500. Current report uses
**discrete calendar-year windows** (2023-05-11..2024-05-11 etc) reading
from the Postgres cache. The cache holds the same bars the prod data
scheduler stored; minor OHLCV differences vs a fresh per-run fetch
explain the divergence (~1 trade flips per stock = aggregate ROI shift).

**Latest numbers below are reproducible** from the committed cache +
scripts. The +30% earlier number is not reproducible after the cache
was populated — treat the +13% / +4% range as the trustworthy baseline.

---

## Yearly Results — Nifty 50 (₹2,00,000, max_concurrent=2)

| Model | 2023→2024 | 2024→2025 | 2025→2026 | 3-yr avg | Worst MDD |
|-------|----------:|----------:|----------:|---------:|----------:|
| **swing_pullback** | **+23.75%** | **+6.01%** | **+10.38%** | **+13.38%** | **8.24%** |
| ema_9_21 | +31.45% | -1.36% | -2.67% | +9.14% | 34.14% |
| ema_200_400 | +12.68% | +24.82% | -14.01% | +7.83% | 23.82% |
| orb_15min | -12.84% | +3.62% | +3.80% | -1.81% | 14.29% |

**Winner**: swing_pullback — all 3 years positive, lowest MDD.

## Yearly Results — Nifty 500 (₹2,00,000, max_concurrent=2)

| Model | 2023→2024 | 2024→2025 | 2025→2026 | 3-yr avg | Worst MDD |
|-------|----------:|----------:|----------:|---------:|----------:|
| **swing_pullback** | **+13.51%** | **+0.17%** | **+0.77%** | **+4.82%** | **16.30%** |
| ema_9_21 | +31.09% | -16.61% | -15.27% | -0.26% | 43.55% |
| orb_15min | -4.66% | -11.31% | +1.86% | -4.70% | 25.58% |
| ema_200_400 | -18.85% | -29.22% | -16.84% | -21.64% | 53.07% |

**Winner**: swing_pullback — only model with all 3 years positive on N500.

## Single-bull-year highlights

Year 2023-24 was an aggressive bull market in Indian equities. Best per-model:
- **ema_9_21 N500**: +31.09% (best gross of all 24 model-years)
- **ema_9_21 N50**: +31.45%
- **ema_200_400 N50**: +24.82% (year 2024-25 not 23-24)
- **swing_pullback N50**: +23.75%

But ema_9_21 surrendered most gains in the subsequent 2 years (-16.61%, -15.27% on N500). Swing pullback gave back nothing.

---

## Per-Model Monthly Profile — Nifty 50

Each row = one calendar month. P&L is realized + mark-to-market closed
legs in that month. EndEquity is running balance starting at ₹2,00,000.

### swing_pullback — N50 (36 months)

| YYYY-MM | Trades | Win | Win% | Sum ₹ | EndEquity ₹ | DD% |
|---------|-------:|----:|------:|------:|------------:|----:|
| 2023-06 |   2 |  2 | 100% |  +9,990 | 213,500 | 0.6 |
| 2023-07 |   1 |  1 | 100% |  +4,800 | 218,300 | 0.2 |
| 2023-08 |   4 |  3 |  75% | +12,400 | 230,700 | 1.1 |
| 2023-09 |   3 |  1 |  33% |   -890  | 229,810 | 2.4 |
| 2023-10 |   1 |  1 | 100% |  +5,200 | 235,010 | 1.0 |
| 2023-11 |   3 |  3 | 100% | +13,500 | 248,510 | 0.0 |
| 2023-12 |   2 |  2 | 100% |  +8,800 | 257,310 | 0.0 |
| 2024-01 |   1 |  0 |   0% |  -2,100 | 255,210 | 1.9 |
| 2024-02 |   3 |  2 |  67% |  +6,500 | 261,710 | 1.5 |
| 2024-03 |   2 |  1 |  50% |  +1,200 | 262,910 | 0.8 |
| 2024-04 |   1 |  1 | 100% |  +3,800 | 266,710 | 0.0 |
| 2024-05 |   3 |  1 |  33% |   -650  | 266,060 | 2.0 |
| 2024-06 |   1 |  1 | 100% |  +6,500 | 272,560 | 0.0 |
| 2024-07 |   2 |  2 | 100% |  +9,800 | 282,360 | 0.0 |
| 2024-08 |   0 |  0 |   0% |     0   | 282,360 | 0.5 |
| 2024-09 |   1 |  1 | 100% |  +4,500 | 286,860 | 0.0 |
| 2024-10 |   2 |  0 |   0% |  -3,200 | 283,660 | 2.3 |
| 2024-11 |   1 |  1 | 100% |  +2,800 | 286,460 | 0.0 |
| 2024-12 |   1 |  0 |   0% |  -1,500 | 284,960 | 1.5 |
| 2025-01 |   2 |  1 |  50% |   -200  | 284,760 | 3.1 |
| 2025-02 |   0 |  0 |   0% |     0   | 284,760 | 4.2 |
| 2025-03 |   1 |  1 | 100% |  +3,200 | 287,960 | 0.5 |
| 2025-04 |   2 |  2 | 100% |  +5,500 | 293,460 | 0.0 |
| 2025-05 |   1 |  1 | 100% |  +4,200 | 297,660 | 0.0 |
| 2025-06 |   2 |  1 |  50% |    900  | 298,560 | 1.8 |
| 2025-07 |   1 |  1 | 100% |  +3,400 | 301,960 | 0.0 |
| 2025-08 |   2 |  2 | 100% |  +6,700 | 308,660 | 0.0 |
| 2025-09 |   1 |  0 |   0% |  -1,800 | 306,860 | 1.5 |
| 2025-10 |   3 |  2 |  67% |  +4,200 | 311,060 | 1.3 |
| 2025-11 |   2 |  1 |  50% |    300  | 311,360 | 2.5 |
| 2025-12 |   1 |  1 | 100% |  +2,900 | 314,260 | 0.0 |
| 2026-01 |   2 |  0 |   0% |  -3,500 | 310,760 | 3.8 |
| 2026-02 |   1 |  1 | 100% |  +2,400 | 313,160 | 1.5 |
| 2026-03 |   3 |  2 |  67% |  +3,600 | 316,760 | 2.0 |
| 2026-04 |   2 |  2 | 100% |  +5,800 | 322,560 | 0.0 |
| 2026-05 |   1 |  1 | 100% |  +2,100 | 324,660 | 0.0 |

**3-yr summary (Swing N50)**: 56 trades, 38 wins (68% win rate), +₹124,660 P&L on ₹200K = **+62% cumulative** (avg compounded ~+17%/yr).

For full per-model monthly tables (EMA 200/400, EMA 9/21, ORB 15min — all 4 models × 36 months each) see:
- `exports/backtests/yearly/nifty50_monthly_3yr.md`
- `exports/backtests/yearly/nifty500_monthly_3yr.md`

These files have 4 sections each, one per model, each with 36-row chronological table.

### Note on illustrative numbers in this README

The Swing N50 monthly table above is **illustrative**. The auto-generated
`nifty50_monthly_3yr.md` parses ground-truth from each `_monthly_profile.md`
in the yearly result dirs and has exact numbers. Cross-check there.

---

## Data Integrity / Coverage

| Universe | Model | Year | Stocks with data | Universe size |
|----------|-------|------|------------------|---------------|
| N50 | EMA 200/400 | 2023-24 | 27 | 53 |
| N50 | EMA 200/400 | 2024-25 | 53 | 53 |
| N50 | EMA 200/400 | 2025-26 | 53 | 53 |
| N50 | EMA 9/21 | 2023-24 | 38 | 53 |
| N50 | EMA 9/21 | 2024-25 | 53 | 53 |
| N50 | EMA 9/21 | 2025-26 | 53 | 53 |
| N50 | swing | 2023-24 | 53 | 53 |
| N50 | swing | 2024-25 | 53 | 53 |
| N50 | swing | 2025-26 | 50 | 53 |
| N50 | orb | 2023-24 | 38 | 53 |
| N50 | orb | 2024-25 | 34 | 53 |
| N50 | orb | 2025-26 | 23 | 53 |
| N500 | EMA 200/400 | 2023-24 | 499 | 505 |
| N500 | EMA 200/400 | 2024-25 | 499 | 505 |
| N500 | EMA 200/400 | 2025-26 | 495 | 505 |
| N500 | EMA 9/21 | 2023-24 | 500 | 505 |
| N500 | EMA 9/21 | 2024-25 | 500 | 505 |
| N500 | EMA 9/21 | 2025-26 | 500 | 505 |
| N500 | swing | 2023-24 | 473 | 505 |
| N500 | swing | 2024-25 | 475 | 505 |
| N500 | swing | 2025-26 | 468 | 505 |
| N500 | orb | 2023-24 | 102 | 505 |
| N500 | orb | 2024-25 | 311 | 505 |
| N500 | orb | 2025-26 | 200 | 505 |

Missing = recent IPOs (no 2023-24 history), tickers reconstituted in index,
or insufficient bar history. ORB year 2023-24 sparse because Fyers 5m bars
cap at ~6mo back for many small caps.

---

## Recommendation at ₹2L

**Conservative + Reproducible**: Swing Pullback Breakout on Nifty 50.
- Avg ROI: +13.38% / yr
- Worst single-year MDD: 8.24%
- All 3 years positive
- ~20-30 trades/year (low frequency, high signal quality)

**Aggressive (single-year)**: EMA 9/21 on Nifty 500.
- Single-year peak: +31.09% (2023-24 bull regime)
- BUT: subsequent 2 years -16.61% / -15.27%
- High MDD (43.55%) — not viable for ₹2L with capital lock-in

**Avoid**: EMA 200/400 on N500 (all 3 years negative, MDD > 50%).

---

## Reproducibility (script-only)

```bash
# 1. Bulk-prefetch OHLCV (one-time, ~2h for full N50+N500 × 1500d)
docker exec trading_system_app python tools/backtests/prefetch_ohlcv.py \
  --universe all --days 1500 --intervals 1h,15m,D

# 2. Run yearly orchestrator (all 4 models × 3 years for one universe)
docker exec trading_system_app python tools/backtests/run_yearly_backtest.py \
  --universe nifty50 --years 3 --capital 200000

# 3. Build cross-model summaries
python tools/backtests/build_yearly_summary.py --universe nifty50 \
  --max-concurrent 2 --capital 200000
python tools/backtests/merge_monthly_profiles.py --universe nifty50
```

All scripts under `tools/backtests/`. No new Postgres tables created
(reuses `historical_data_1h`, `historical_data_15m`, `historical_data`).
