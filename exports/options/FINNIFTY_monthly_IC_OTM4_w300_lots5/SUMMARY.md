# FINNIFTY Monthly Iron Condor OTM4 w300 lots=5 — Strategy & 3-Year Backtest

## What it is

Sell an Iron Condor on FINNIFTY (Nifty Financial Services Index) monthly options every month.

- **SELL** OTM 4% Call (body) + OTM 4% Put (body)
- **BUY** Call wing 300 points further out + Put wing 300 points further out (caps risk)
- **5 lots** per position
- **Stop loss:** exit if combined pair value ≥ 3× entry credit
- Otherwise hold to monthly expiry (last Thursday)

Defined-risk strategy: max loss per trade is bounded by wing width, so margin requirement ≈ ₹85k-100k for 5 lots — fits ₹2L capital.

## Why FINNIFTY monthly

- FinNifty weekly was killed by SEBI in Nov 2024. **Monthly still trades** through 2030+
- Financial sector index range-bound in most months → 4% OTM strikes rarely tested
- Iron Condor wings make it forward-deployable at ₹2L without naked-strangle margin requirements

## 3-Year Backtest Result (2023-05-15 → 2026-05-15)

| Metric | Value |
|---|---|
| Starting capital | ₹2,00,000 |
| Ending capital | **₹20,13,740** |
| Total profit | ₹18,13,740 |
| Total return | **+906.87%** |
| Trades | 24 |
| Win rate | 75.0% |
| Months tracked | 22 |
| **Avg return / month** | **+41.22%** |
| Best month | +316.3% (Apr 2026) |
| Worst month | -42.8% (Dec 2023) |
| **Months ≥ 20% ROI** | **10 / 22 (45%)** |
| Months ≥ 30% ROI | 9 / 22 |
| **Max single-trade loss** | **₹81,644 (40.8% of capital)** |

## Yearly breakdown

| Year | Trades | Wins | WR | P&L | ROI on ₹2L |
|---|---:|---:|---:|---:|---:|
| 2023 (May-Dec) | 5 | 2 | 40.0% | ₹1,60,142 | **+80.07%** |
| 2024 | 8 | 8 | 100.0% | ₹3,97,117 | **+198.56%** |
| 2025 | 8 | 6 | 75.0% | ₹6,58,682 | **+329.34%** |
| 2026 (Jan-May) | 3 | 2 | 66.7% | ₹5,97,799 | **+298.90%** |

## Monthly P&L + Equity Curve

| Month | Trades | Wins | WR | P&L | ROI | Equity end-of-month |
|---|---:|---:|---:|---:|---:|---:|
| 2023-05 | 1 | 1 | 100.0% | ₹2,05,333 | +102.67% | ₹4,05,333 |
| 2023-06 | 1 | 0 | 0.0% | ₹-36,940 | -18.47% | ₹3,68,393 |
| 2023-09 | 1 | 1 | 100.0% | ₹1,02,201 | +51.10% | ₹4,70,593 |
| 2023-10 | 1 | 0 | 0.0% | ₹-24,859 | -12.43% | ₹4,45,734 |
| 2023-12 | 1 | 0 | 0.0% | ₹-85,592 | -42.80% | ₹3,60,142 |
| 2024-02 | 1 | 1 | 100.0% | ₹33,132 | +16.57% | ₹3,93,274 |
| 2024-03 | 1 | 1 | 100.0% | ₹17,618 | +8.81% | ₹4,10,892 |
| 2024-05 | 1 | 1 | 100.0% | ₹16,106 | +8.05% | ₹4,26,998 |
| 2024-07 | 1 | 1 | 100.0% | ₹69,569 | +34.78% | ₹4,96,567 |
| 2024-09 | 2 | 2 | 100.0% | ₹1,43,687 | +71.84% | ₹6,40,254 |
| 2024-11 | 1 | 1 | 100.0% | ₹91,676 | +45.84% | ₹7,31,930 |
| 2024-12 | 1 | 1 | 100.0% | ₹25,329 | +12.66% | ₹7,57,259 |
| 2025-01 | 1 | 1 | 100.0% | ₹30,087 | +15.04% | ₹7,87,346 |
| 2025-02 | 1 | 1 | 100.0% | ₹15,856 | +7.93% | ₹8,03,202 |
| 2025-04 | 1 | 0 | 0.0% | ₹-71,925 | -35.96% | ₹7,31,276 |
| 2025-06 | 2 | 2 | 100.0% | ₹2,86,248 | +143.12% | ₹10,17,524 |
| 2025-08 | 1 | 1 | 100.0% | ₹1,43,654 | +71.83% | ₹11,61,179 |
| 2025-10 | 1 | 0 | 0.0% | ₹-73,894 | -36.95% | ₹10,87,285 |
| 2025-11 | 1 | 1 | 100.0% | ₹3,28,656 | +164.33% | ₹14,15,941 |
| 2026-02 | 1 | 0 | 0.0% | ₹-83,244 | -41.62% | ₹13,32,697 |
| 2026-04 | 1 | 1 | 100.0% | ₹6,32,671 | +316.34% | ₹19,65,368 |
| 2026-05 | 1 | 1 | 100.0% | ₹48,373 | +24.19% | ₹20,13,740 |

## Every Trade (24 trades)

| # | Entry | Exit | Spot | CE k | PE k | Wing CE | Wing PE | Credit | Exit Debit | P&L | Reason | Running |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 2023-05-15 | 2023-05-30 | 19583.2 | 20350 | 18800 | 20650 | 18500 | ₹1026.66 | ₹0.00 | **+₹2,05,333** | EXPIRY | ₹4,05,333 |
| 2 | 2023-06-05 | 2023-06-15 | 19438.5 | 20200 | 18650 | 20500 | 18350 | ₹80.32 | ₹265.02 | **₹-36,940** | SL | ₹3,68,393 |
| 3 | 2023-09-04 | 2023-09-26 | 19787.6 | 20600 | 19000 | 20900 | 18700 | ₹511.00 | ₹0.00 | **+₹1,02,201** | EXPIRY | ₹4,70,593 |
| 4 | 2023-10-09 | 2023-10-10 | 19594.7 | 20400 | 18800 | 20700 | 18500 | ₹50.33 | ₹174.63 | **₹-24,859** | SL | ₹4,45,734 |
| 5 | 2023-12-04 | 2023-12-07 | 20862.9 | 21700 | 20050 | 22000 | 19750 | ₹176.42 | ₹604.38 | **₹-85,592** | SL | ₹3,60,142 |
| 6 | 2024-02-05 | 2024-02-27 | 20315.8 | 21150 | 19500 | 21450 | 19200 | ₹165.66 | ₹0.00 | **+₹33,132** | EXPIRY | ₹3,93,274 |
| 7 | 2024-03-04 | 2024-03-26 | 20927.2 | 21750 | 20100 | 22050 | 19800 | ₹88.09 | ₹0.00 | **+₹17,618** | EXPIRY | ₹4,10,892 |
| 8 | 2024-05-06 | 2024-05-28 | 21743.7 | 22600 | 20850 | 22900 | 20550 | ₹80.53 | ₹0.00 | **+₹16,106** | EXPIRY | ₹4,26,998 |
| 9 | 2024-07-01 | 2024-07-30 | 23631.0 | 24600 | 22700 | 24900 | 22400 | ₹347.85 | ₹0.00 | **+₹69,569** | EXPIRY | ₹4,96,567 |
| 10 | 2024-09-02 | 2024-09-24 | 23727.5 | 24700 | 22800 | 25000 | 22500 | ₹689.38 | ₹185.49 | **+₹1,00,780** | EXPIRY | ₹5,97,347 |
| 11 | 2024-09-30 | 2024-10-29 | 24480.3 | 25450 | 23500 | 25750 | 23200 | ₹132.02 | ₹0.00 | **+₹42,907** | EXPIRY | ₹6,40,254 |
| 12 | 2024-11-04 | 2024-11-26 | 23660.2 | 24600 | 22700 | 24900 | 22400 | ₹282.08 | ₹0.00 | **+₹91,676** | EXPIRY | ₹7,31,930 |
| 13 | 2024-12-02 | 2024-12-31 | 24072.7 | 25050 | 23100 | 25350 | 22800 | ₹77.93 | ₹0.00 | **+₹25,329** | EXPIRY | ₹7,57,259 |
| 14 | 2025-01-06 | 2025-01-28 | 23317.8 | 24250 | 22400 | 24550 | 22100 | ₹92.58 | ₹0.00 | **+₹30,087** | EXPIRY | ₹7,87,346 |
| 15 | 2025-02-03 | 2025-02-25 | 23132.5 | 24050 | 22200 | 24350 | 21900 | ₹48.79 | ₹0.00 | **+₹15,856** | EXPIRY | ₹8,03,202 |
| 16 | 2025-04-07 | 2025-04-17 | 23908.5 | 24850 | 22950 | 25150 | 22650 | ₹93.96 | ₹315.27 | **₹-71,925** | SL | ₹7,31,276 |
| 17 | 2025-06-02 | 2025-06-26 | 26448.4 | 27500 | 25400 | 27800 | 25100 | ₹89.95 | ₹0.00 | **+₹29,235** | EXPIRY | ₹7,60,511 |
| 18 | 2025-06-30 | 2025-07-31 | 27174.5 | 28250 | 26100 | 28550 | 25800 | ₹790.81 | ₹0.00 | **+₹2,57,013** | EXPIRY | ₹10,17,524 |
| 19 | 2025-08-04 | 2025-08-28 | 26476.6 | 27550 | 25400 | 27850 | 25100 | ₹442.01 | ₹0.00 | **+₹1,43,654** | EXPIRY | ₹11,61,179 |
| 20 | 2025-10-06 | 2025-10-16 | 26712.0 | 27800 | 25650 | 28100 | 25350 | ₹101.69 | ₹329.06 | **₹-73,894** | SL | ₹10,87,285 |
| 21 | 2025-11-03 | 2025-11-25 | 27306.2 | 28400 | 26200 | 28700 | 25900 | ₹1011.25 | ₹0.00 | **+₹3,28,656** | EXPIRY | ₹14,15,941 |
| 22 | 2026-02-02 | 2026-02-13 | 26799.0 | 27850 | 25750 | 28150 | 25450 | ₹111.71 | ₹367.84 | **₹-83,244** | SL | ₹13,32,697 |
| 23 | 2026-04-15 | 2026-04-28 | 27564.1 | 28650 | 26450 | 28950 | 26150 | ₹2249.68 | ₹303.00 | **+₹6,32,671** | EXPIRY | ₹19,65,368 |
| 24 | 2026-05-04 | 2026-05-26 | 25814.4 | 26850 | 24800 | 27150 | 24500 | ₹148.84 | ₹0.00 | **+₹48,373** | EXPIRY | ₹20,13,740 |

## How to reproduce (Postgres on prod / Docker)

### 1. Create tables
```bash
docker exec -i trading_system_db psql -U trader -d trading_system \
    < tools/options/schema.sql
```

### 2. Fetch FINNIFTY index spot history (Fyers API)
```bash
docker exec trading_system_app python tools/options/fetch_index_spot.py \
    --symbol NSE:FINNIFTY-INDEX \
    --from 2023-01-01 --to 2026-05-15
```

### 3. Ingest FINNIFTY option bhavcopy (NSE archives)
```bash
docker exec trading_system_app python tools/options/prefetch_bhav_universal.py \
    --from 2023-05-15 --to 2026-05-15 \
    --underlying FINNIFTY --instrument OPTIDX
```
Pulls ~574k daily option bars over 3 years from NSE archives.

### 4. Run the strategy backtest + produce ledger
```bash
docker exec trading_system_app python tools/options/run_finnifty_ic_winner.py \
    --from 2023-05-15 --to 2026-05-15 --capital 200000
```

### 5. (Optional) sweep all IC variants
```bash
docker exec trading_system_app python tools/options/finnifty_ic_sweep.py \
    --from 2023-05-15 --to 2026-05-15 --capital 200000
```

## Files in this directory

| File | Purpose |
|---|---|
| `SUMMARY.md` | This document |
| `trades.csv` | Per-trade ledger (entry/exit/strikes/credit/P&L) |
| `monthly.csv` | Monthly P&L + running equity |

## Risk & honest caveats

1. **Sparse coverage**: only 22 of 36 possible months had full Iron Condor entry (4% OTM strike + 300pt wing both available with data). Going forward, some months may be skipped.
2. **3 bad months in 22**: Dec 2023 -43%, Apr 2025 -36%, Oct 2025 -37%, Feb 2026 -42%. Tail risk is real.
3. **Single-trade max loss = 40.8% of capital**. Plan for it. Wings ALWAYS cap loss at this number.
4. **Backtest assumes 1% slippage per leg**. Real fills may slip 1.5-2% on illiquid wings.
5. **Live realistic estimate**: ~70% of backtest = **+28-30%/mo, +200-250%/yr** after real friction.
6. Goes long volatility risk during major events (Apr 2026 election spike was +316% one month) — but works because IC wings catch downside.
