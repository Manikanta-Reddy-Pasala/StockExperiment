# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (34996 bars)
- **Last close:** 11220.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 37
- **Target hits / Stop hits / Partials:** 13 / 37 / 26
- **Avg / median % per leg:** 0.38% / 0.38%
- **Sum % (uncompounded):** 28.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 27 | 55.1% | 10 | 22 | 17 | 0.43% | 21.2% |
| BUY @ 2nd Alert (retest1) | 49 | 27 | 55.1% | 10 | 22 | 17 | 0.43% | 21.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 12 | 44.4% | 3 | 15 | 9 | 0.28% | 7.5% |
| SELL @ 2nd Alert (retest1) | 27 | 12 | 44.4% | 3 | 15 | 9 | 0.28% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 39 | 51.3% | 13 | 37 | 26 | 0.38% | 28.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:40:00 | 7958.00 | 8025.66 | 0.00 | ORB-short ORB[8005.05,8095.00] vol=1.6x ATR=25.58 |
| Stop hit — per-position SL triggered | 2024-05-15 11:50:00 | 7983.58 | 8005.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:10:00 | 9080.00 | 8974.01 | 0.00 | ORB-long ORB[8900.00,9027.80] vol=3.2x ATR=43.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:25:00 | 9144.86 | 9022.59 | 0.00 | T1 1.5R @ 9144.86 |
| Stop hit — per-position SL triggered | 2024-05-21 10:30:00 | 9080.00 | 9025.86 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 9347.55 | 9290.52 | 0.00 | ORB-long ORB[9224.00,9312.45] vol=3.1x ATR=28.25 |
| Stop hit — per-position SL triggered | 2024-05-24 10:50:00 | 9319.30 | 9295.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 9162.80 | 9202.52 | 0.00 | ORB-short ORB[9180.00,9270.00] vol=1.6x ATR=30.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:45:00 | 9117.38 | 9177.73 | 0.00 | T1 1.5R @ 9117.38 |
| Target hit | 2024-05-28 15:20:00 | 9089.95 | 9137.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 9421.00 | 9395.59 | 0.00 | ORB-long ORB[9325.40,9416.95] vol=4.0x ATR=31.63 |
| Stop hit — per-position SL triggered | 2024-05-31 09:40:00 | 9389.37 | 9397.17 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 10282.75 | 10221.80 | 0.00 | ORB-long ORB[10160.05,10227.50] vol=2.8x ATR=26.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:15:00 | 10322.12 | 10245.34 | 0.00 | T1 1.5R @ 10322.12 |
| Stop hit — per-position SL triggered | 2024-06-12 10:55:00 | 10282.75 | 10259.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 10488.05 | 10437.28 | 0.00 | ORB-long ORB[10354.40,10466.05] vol=1.7x ATR=27.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:40:00 | 10528.67 | 10452.30 | 0.00 | T1 1.5R @ 10528.67 |
| Target hit | 2024-06-13 15:20:00 | 10876.40 | 10650.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 11181.35 | 11021.46 | 0.00 | ORB-long ORB[10864.85,11017.95] vol=4.6x ATR=55.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:45:00 | 11264.30 | 11129.99 | 0.00 | T1 1.5R @ 11264.30 |
| Target hit | 2024-06-14 13:15:00 | 11246.40 | 11276.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:00:00 | 11514.20 | 11625.12 | 0.00 | ORB-short ORB[11600.00,11723.00] vol=1.6x ATR=38.14 |
| Stop hit — per-position SL triggered | 2024-06-26 10:20:00 | 11552.34 | 11597.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:50:00 | 11565.00 | 11487.46 | 0.00 | ORB-long ORB[11372.15,11495.00] vol=1.9x ATR=31.22 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 11533.78 | 11499.07 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:00:00 | 11801.40 | 11732.97 | 0.00 | ORB-long ORB[11650.05,11769.90] vol=2.9x ATR=37.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 13:35:00 | 11858.37 | 11770.90 | 0.00 | T1 1.5R @ 11858.37 |
| Target hit | 2024-06-28 15:20:00 | 11964.75 | 11923.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:40:00 | 12668.00 | 12775.48 | 0.00 | ORB-short ORB[12700.10,12839.00] vol=3.3x ATR=40.92 |
| Stop hit — per-position SL triggered | 2024-07-04 10:50:00 | 12708.92 | 12770.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 12415.45 | 12517.32 | 0.00 | ORB-short ORB[12440.00,12580.00] vol=1.6x ATR=35.99 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 12451.44 | 12516.14 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 12344.45 | 12495.36 | 0.00 | ORB-short ORB[12490.00,12640.85] vol=1.5x ATR=35.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 12291.12 | 12457.67 | 0.00 | T1 1.5R @ 12291.12 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 12344.45 | 12373.34 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 12577.30 | 12635.12 | 0.00 | ORB-short ORB[12610.00,12689.40] vol=1.9x ATR=33.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:50:00 | 12527.06 | 12600.61 | 0.00 | T1 1.5R @ 12527.06 |
| Stop hit — per-position SL triggered | 2024-07-12 10:55:00 | 12577.30 | 12580.51 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 11276.35 | 11330.68 | 0.00 | ORB-short ORB[11290.00,11414.40] vol=1.7x ATR=45.49 |
| Stop hit — per-position SL triggered | 2024-07-29 09:40:00 | 11321.84 | 11328.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 11748.55 | 11674.42 | 0.00 | ORB-long ORB[11553.75,11700.00] vol=3.5x ATR=47.67 |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 11700.88 | 11714.61 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 12013.70 | 12077.91 | 0.00 | ORB-short ORB[12030.00,12149.00] vol=1.9x ATR=47.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:35:00 | 11942.62 | 12065.20 | 0.00 | T1 1.5R @ 11942.62 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 12013.70 | 12052.90 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 12226.90 | 12164.77 | 0.00 | ORB-long ORB[12025.20,12207.80] vol=1.7x ATR=46.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:10:00 | 12296.12 | 12202.69 | 0.00 | T1 1.5R @ 12296.12 |
| Target hit | 2024-08-16 15:20:00 | 12330.00 | 12284.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 11:10:00 | 12659.55 | 12545.09 | 0.00 | ORB-long ORB[12501.00,12640.00] vol=2.1x ATR=40.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 12:35:00 | 12720.59 | 12582.97 | 0.00 | T1 1.5R @ 12720.59 |
| Target hit | 2024-08-19 15:20:00 | 12780.70 | 12669.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 13174.15 | 13246.83 | 0.00 | ORB-short ORB[13200.40,13333.95] vol=1.8x ATR=42.06 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 13216.21 | 13242.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:55:00 | 12625.00 | 12561.56 | 0.00 | ORB-long ORB[12480.00,12614.00] vol=1.8x ATR=39.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:00:00 | 12684.27 | 12595.37 | 0.00 | T1 1.5R @ 12684.27 |
| Target hit | 2024-09-11 14:15:00 | 12768.00 | 12781.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 14396.45 | 14339.14 | 0.00 | ORB-long ORB[14232.60,14373.50] vol=1.7x ATR=41.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:40:00 | 14458.99 | 14361.07 | 0.00 | T1 1.5R @ 14458.99 |
| Stop hit — per-position SL triggered | 2024-09-24 10:00:00 | 14396.45 | 14392.00 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 14050.85 | 13976.62 | 0.00 | ORB-long ORB[13830.00,14025.00] vol=2.1x ATR=48.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:45:00 | 14122.87 | 14027.67 | 0.00 | T1 1.5R @ 14122.87 |
| Target hit | 2024-10-01 12:10:00 | 14099.40 | 14108.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 13514.40 | 13690.67 | 0.00 | ORB-short ORB[13662.35,13781.00] vol=1.8x ATR=57.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:45:00 | 13428.27 | 13662.63 | 0.00 | T1 1.5R @ 13428.27 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 13514.40 | 13644.66 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-10-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:00:00 | 13714.25 | 13601.14 | 0.00 | ORB-long ORB[13465.65,13612.10] vol=2.2x ATR=61.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:05:00 | 13805.85 | 13666.46 | 0.00 | T1 1.5R @ 13805.85 |
| Target hit | 2024-10-08 15:20:00 | 14541.30 | 14269.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 14974.00 | 14915.16 | 0.00 | ORB-long ORB[14837.35,14948.80] vol=1.6x ATR=43.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:25:00 | 15038.99 | 14951.18 | 0.00 | T1 1.5R @ 15038.99 |
| Stop hit — per-position SL triggered | 2024-10-11 10:40:00 | 14974.00 | 14956.83 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 15414.00 | 15340.14 | 0.00 | ORB-long ORB[15248.75,15397.00] vol=2.0x ATR=41.71 |
| Stop hit — per-position SL triggered | 2024-10-15 09:40:00 | 15372.29 | 15345.54 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:55:00 | 15452.95 | 15369.94 | 0.00 | ORB-long ORB[15285.00,15447.95] vol=1.6x ATR=41.63 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 15411.32 | 15381.78 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 15147.60 | 15263.12 | 0.00 | ORB-short ORB[15261.20,15393.50] vol=2.7x ATR=40.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:55:00 | 15087.34 | 15206.34 | 0.00 | T1 1.5R @ 15087.34 |
| Stop hit — per-position SL triggered | 2024-10-17 10:30:00 | 15147.60 | 15157.70 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 15210.95 | 15342.43 | 0.00 | ORB-short ORB[15322.00,15460.00] vol=1.7x ATR=53.10 |
| Stop hit — per-position SL triggered | 2024-10-22 10:50:00 | 15264.05 | 15323.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:15:00 | 15430.00 | 15330.03 | 0.00 | ORB-long ORB[15210.10,15399.00] vol=1.6x ATR=64.84 |
| Stop hit — per-position SL triggered | 2024-10-24 10:55:00 | 15365.16 | 15367.44 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 15920.00 | 15754.05 | 0.00 | ORB-long ORB[15583.50,15741.65] vol=3.9x ATR=58.58 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 15861.42 | 15776.37 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-11-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 11:00:00 | 15408.05 | 15488.41 | 0.00 | ORB-short ORB[15450.60,15645.00] vol=1.5x ATR=53.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:15:00 | 15327.96 | 15480.30 | 0.00 | T1 1.5R @ 15327.96 |
| Target hit | 2024-11-12 15:20:00 | 15003.50 | 15250.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:35:00 | 15187.05 | 15024.76 | 0.00 | ORB-long ORB[14850.95,15038.95] vol=3.3x ATR=62.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:45:00 | 15281.06 | 15131.58 | 0.00 | T1 1.5R @ 15281.06 |
| Stop hit — per-position SL triggered | 2024-11-19 09:50:00 | 15187.05 | 15146.48 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 15758.90 | 15700.17 | 0.00 | ORB-long ORB[15591.25,15746.50] vol=1.6x ATR=46.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:40:00 | 15827.98 | 15773.81 | 0.00 | T1 1.5R @ 15827.98 |
| Target hit | 2024-11-28 10:30:00 | 15826.15 | 15849.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 17263.85 | 17413.67 | 0.00 | ORB-short ORB[17295.50,17530.00] vol=1.9x ATR=46.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:25:00 | 17193.92 | 17385.18 | 0.00 | T1 1.5R @ 17193.92 |
| Stop hit — per-position SL triggered | 2024-12-05 13:50:00 | 17263.85 | 17326.20 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 17765.00 | 17686.68 | 0.00 | ORB-long ORB[17512.85,17740.80] vol=2.6x ATR=44.82 |
| Stop hit — per-position SL triggered | 2024-12-12 10:55:00 | 17720.18 | 17723.61 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 10:00:00 | 17690.00 | 17617.82 | 0.00 | ORB-long ORB[17575.00,17669.55] vol=1.5x ATR=39.82 |
| Stop hit — per-position SL triggered | 2024-12-13 10:05:00 | 17650.18 | 17619.63 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 11:10:00 | 18471.00 | 18638.56 | 0.00 | ORB-short ORB[18500.00,18777.00] vol=1.8x ATR=47.19 |
| Stop hit — per-position SL triggered | 2024-12-19 11:30:00 | 18518.19 | 18622.63 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:15:00 | 18091.45 | 18003.00 | 0.00 | ORB-long ORB[17900.00,18080.00] vol=1.7x ATR=47.46 |
| Stop hit — per-position SL triggered | 2024-12-24 10:30:00 | 18043.99 | 18010.56 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:30:00 | 16779.00 | 16886.64 | 0.00 | ORB-short ORB[16825.00,17040.00] vol=2.1x ATR=84.36 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 16863.36 | 16864.32 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:30:00 | 16266.45 | 16375.12 | 0.00 | ORB-short ORB[16320.00,16500.00] vol=1.8x ATR=83.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:00:00 | 16141.02 | 16352.35 | 0.00 | T1 1.5R @ 16141.02 |
| Target hit | 2025-01-13 15:20:00 | 15856.45 | 16121.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 15215.40 | 15054.41 | 0.00 | ORB-long ORB[14880.05,15057.00] vol=2.5x ATR=68.02 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 15147.38 | 15079.42 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 14355.00 | 14270.98 | 0.00 | ORB-long ORB[14146.90,14319.10] vol=1.6x ATR=57.30 |
| Stop hit — per-position SL triggered | 2025-03-05 09:55:00 | 14297.70 | 14276.71 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:05:00 | 13642.10 | 13581.05 | 0.00 | ORB-long ORB[13420.00,13612.50] vol=3.0x ATR=53.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:15:00 | 13722.80 | 13593.50 | 0.00 | T1 1.5R @ 13722.80 |
| Target hit | 2025-03-19 15:20:00 | 13879.85 | 13767.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:05:00 | 13538.95 | 13456.99 | 0.00 | ORB-long ORB[13290.00,13490.00] vol=1.7x ATR=47.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:15:00 | 13610.86 | 13476.92 | 0.00 | T1 1.5R @ 13610.86 |
| Stop hit — per-position SL triggered | 2025-03-27 12:05:00 | 13538.95 | 13501.39 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 13134.95 | 13027.24 | 0.00 | ORB-long ORB[12899.75,13092.65] vol=2.4x ATR=59.18 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 13075.77 | 13037.54 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:10:00 | 16544.00 | 16454.50 | 0.00 | ORB-long ORB[16368.00,16520.00] vol=1.6x ATR=45.11 |
| Stop hit — per-position SL triggered | 2025-05-05 11:30:00 | 16498.89 | 16460.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:10:00 | 16276.00 | 16183.96 | 0.00 | ORB-long ORB[16083.00,16257.00] vol=2.2x ATR=56.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:25:00 | 16360.59 | 16215.33 | 0.00 | T1 1.5R @ 16360.59 |
| Stop hit — per-position SL triggered | 2025-05-08 10:30:00 | 16276.00 | 16220.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:40:00 | 7958.00 | 2024-05-15 11:50:00 | 7983.58 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-21 10:10:00 | 9080.00 | 2024-05-21 10:25:00 | 9144.86 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-05-21 10:10:00 | 9080.00 | 2024-05-21 10:30:00 | 9080.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-24 10:40:00 | 9347.55 | 2024-05-24 10:50:00 | 9319.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-28 09:35:00 | 9162.80 | 2024-05-28 11:45:00 | 9117.38 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-28 09:35:00 | 9162.80 | 2024-05-28 15:20:00 | 9089.95 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2024-05-31 09:30:00 | 9421.00 | 2024-05-31 09:40:00 | 9389.37 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-12 09:50:00 | 10282.75 | 2024-06-12 10:15:00 | 10322.12 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-12 09:50:00 | 10282.75 | 2024-06-12 10:55:00 | 10282.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 11:00:00 | 10488.05 | 2024-06-13 11:40:00 | 10528.67 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-13 11:00:00 | 10488.05 | 2024-06-13 15:20:00 | 10876.40 | TARGET_HIT | 0.50 | 3.70% |
| BUY | retest1 | 2024-06-14 09:30:00 | 11181.35 | 2024-06-14 09:45:00 | 11264.30 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-06-14 09:30:00 | 11181.35 | 2024-06-14 13:15:00 | 11246.40 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2024-06-26 10:00:00 | 11514.20 | 2024-06-26 10:20:00 | 11552.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-27 10:50:00 | 11565.00 | 2024-06-27 11:15:00 | 11533.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-28 11:00:00 | 11801.40 | 2024-06-28 13:35:00 | 11858.37 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-28 11:00:00 | 11801.40 | 2024-06-28 15:20:00 | 11964.75 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2024-07-04 10:40:00 | 12668.00 | 2024-07-04 10:50:00 | 12708.92 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-08 11:10:00 | 12415.45 | 2024-07-08 11:15:00 | 12451.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-10 10:05:00 | 12344.45 | 2024-07-10 10:15:00 | 12291.12 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-10 10:05:00 | 12344.45 | 2024-07-10 10:55:00 | 12344.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 09:30:00 | 12577.30 | 2024-07-12 09:50:00 | 12527.06 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-12 09:30:00 | 12577.30 | 2024-07-12 10:55:00 | 12577.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-29 09:30:00 | 11276.35 | 2024-07-29 09:40:00 | 11321.84 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-09 09:30:00 | 11748.55 | 2024-08-09 10:15:00 | 11700.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-14 09:30:00 | 12013.70 | 2024-08-14 09:35:00 | 11942.62 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-08-14 09:30:00 | 12013.70 | 2024-08-14 09:45:00 | 12013.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:35:00 | 12226.90 | 2024-08-16 10:10:00 | 12296.12 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-16 09:35:00 | 12226.90 | 2024-08-16 15:20:00 | 12330.00 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-08-19 11:10:00 | 12659.55 | 2024-08-19 12:35:00 | 12720.59 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-19 11:10:00 | 12659.55 | 2024-08-19 15:20:00 | 12780.70 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-08-28 09:30:00 | 13174.15 | 2024-08-28 09:35:00 | 13216.21 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-11 09:55:00 | 12625.00 | 2024-09-11 11:00:00 | 12684.27 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-11 09:55:00 | 12625.00 | 2024-09-11 14:15:00 | 12768.00 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2024-09-24 09:35:00 | 14396.45 | 2024-09-24 09:40:00 | 14458.99 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-24 09:35:00 | 14396.45 | 2024-09-24 10:00:00 | 14396.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 09:30:00 | 14050.85 | 2024-10-01 09:45:00 | 14122.87 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-01 09:30:00 | 14050.85 | 2024-10-01 12:10:00 | 14099.40 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-07 10:35:00 | 13514.40 | 2024-10-07 10:45:00 | 13428.27 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-10-07 10:35:00 | 13514.40 | 2024-10-07 10:55:00 | 13514.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 10:00:00 | 13714.25 | 2024-10-08 10:05:00 | 13805.85 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-10-08 10:00:00 | 13714.25 | 2024-10-08 15:20:00 | 14541.30 | TARGET_HIT | 0.50 | 6.03% |
| BUY | retest1 | 2024-10-11 09:35:00 | 14974.00 | 2024-10-11 10:25:00 | 15038.99 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-10-11 09:35:00 | 14974.00 | 2024-10-11 10:40:00 | 14974.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 09:35:00 | 15414.00 | 2024-10-15 09:40:00 | 15372.29 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-16 09:55:00 | 15452.95 | 2024-10-16 10:35:00 | 15411.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-17 09:40:00 | 15147.60 | 2024-10-17 09:55:00 | 15087.34 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-17 09:40:00 | 15147.60 | 2024-10-17 10:30:00 | 15147.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:30:00 | 15210.95 | 2024-10-22 10:50:00 | 15264.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-24 10:15:00 | 15430.00 | 2024-10-24 10:55:00 | 15365.16 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-11-08 09:45:00 | 15920.00 | 2024-11-08 09:50:00 | 15861.42 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-12 11:00:00 | 15408.05 | 2024-11-12 11:15:00 | 15327.96 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-12 11:00:00 | 15408.05 | 2024-11-12 15:20:00 | 15003.50 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2024-11-19 09:35:00 | 15187.05 | 2024-11-19 09:45:00 | 15281.06 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-19 09:35:00 | 15187.05 | 2024-11-19 09:50:00 | 15187.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 09:30:00 | 15758.90 | 2024-11-28 09:40:00 | 15827.98 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-11-28 09:30:00 | 15758.90 | 2024-11-28 10:30:00 | 15826.15 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-05 10:55:00 | 17263.85 | 2024-12-05 11:25:00 | 17193.92 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-05 10:55:00 | 17263.85 | 2024-12-05 13:50:00 | 17263.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 09:40:00 | 17765.00 | 2024-12-12 10:55:00 | 17720.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-13 10:00:00 | 17690.00 | 2024-12-13 10:05:00 | 17650.18 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-19 11:10:00 | 18471.00 | 2024-12-19 11:30:00 | 18518.19 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-24 10:15:00 | 18091.45 | 2024-12-24 10:30:00 | 18043.99 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-10 09:30:00 | 16779.00 | 2025-01-10 09:40:00 | 16863.36 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-13 10:30:00 | 16266.45 | 2025-01-13 11:00:00 | 16141.02 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2025-01-13 10:30:00 | 16266.45 | 2025-01-13 15:20:00 | 15856.45 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2025-02-05 09:35:00 | 15215.40 | 2025-02-05 09:45:00 | 15147.38 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-05 09:45:00 | 14355.00 | 2025-03-05 09:55:00 | 14297.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-19 10:05:00 | 13642.10 | 2025-03-19 10:15:00 | 13722.80 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-19 10:05:00 | 13642.10 | 2025-03-19 15:20:00 | 13879.85 | TARGET_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2025-03-27 11:05:00 | 13538.95 | 2025-03-27 11:15:00 | 13610.86 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-27 11:05:00 | 13538.95 | 2025-03-27 12:05:00 | 13538.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 09:30:00 | 13134.95 | 2025-04-02 09:35:00 | 13075.77 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-05-05 11:10:00 | 16544.00 | 2025-05-05 11:30:00 | 16498.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-08 10:10:00 | 16276.00 | 2025-05-08 10:25:00 | 16360.59 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-05-08 10:10:00 | 16276.00 | 2025-05-08 10:30:00 | 16276.00 | STOP_HIT | 0.50 | 0.00% |
