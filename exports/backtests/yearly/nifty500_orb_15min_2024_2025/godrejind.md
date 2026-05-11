# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1202.00
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 14 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 43
- **Target hits / Stop hits / Partials:** 14 / 43 / 24
- **Avg / median % per leg:** 0.34% / 0.00%
- **Sum % (uncompounded):** 27.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 17 | 50.0% | 7 | 17 | 10 | 0.49% | 16.7% |
| BUY @ 2nd Alert (retest1) | 34 | 17 | 50.0% | 7 | 17 | 10 | 0.49% | 16.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 21 | 44.7% | 7 | 26 | 14 | 0.22% | 10.6% |
| SELL @ 2nd Alert (retest1) | 47 | 21 | 44.7% | 7 | 26 | 14 | 0.22% | 10.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 38 | 46.9% | 14 | 43 | 24 | 0.34% | 27.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 823.95 | 827.44 | 0.00 | ORB-short ORB[824.60,832.90] vol=2.8x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-05-14 09:40:00 | 827.20 | 827.28 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 799.20 | 801.99 | 0.00 | ORB-short ORB[800.25,804.50] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 802.08 | 801.97 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:10:00 | 802.50 | 800.63 | 0.00 | ORB-long ORB[795.75,800.45] vol=2.0x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-06-06 12:00:00 | 798.54 | 801.22 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 808.10 | 803.28 | 0.00 | ORB-long ORB[797.30,806.75] vol=2.7x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:35:00 | 811.73 | 806.88 | 0.00 | T1 1.5R @ 811.73 |
| Target hit | 2024-06-13 11:45:00 | 829.35 | 830.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:15:00 | 815.55 | 823.37 | 0.00 | ORB-short ORB[821.95,832.00] vol=2.1x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-06-21 10:20:00 | 818.01 | 823.03 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:55:00 | 851.50 | 843.70 | 0.00 | ORB-long ORB[828.80,837.85] vol=4.8x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:00:00 | 856.56 | 847.74 | 0.00 | T1 1.5R @ 856.56 |
| Stop hit — per-position SL triggered | 2024-06-25 10:05:00 | 851.50 | 849.79 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 895.95 | 891.57 | 0.00 | ORB-long ORB[887.00,894.50] vol=2.0x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-07-05 11:20:00 | 893.44 | 892.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:45:00 | 905.60 | 901.70 | 0.00 | ORB-long ORB[898.50,905.00] vol=2.4x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-07-15 10:00:00 | 903.43 | 902.21 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:55:00 | 881.20 | 884.26 | 0.00 | ORB-short ORB[883.00,887.90] vol=2.9x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 877.48 | 881.22 | 0.00 | T1 1.5R @ 877.48 |
| Stop hit — per-position SL triggered | 2024-07-19 10:35:00 | 881.20 | 880.86 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:40:00 | 896.30 | 891.95 | 0.00 | ORB-long ORB[884.60,891.00] vol=2.4x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:50:00 | 900.35 | 893.83 | 0.00 | T1 1.5R @ 900.35 |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 896.30 | 899.86 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:55:00 | 880.10 | 883.75 | 0.00 | ORB-short ORB[883.00,888.95] vol=2.5x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-07-25 10:40:00 | 883.58 | 883.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:05:00 | 898.15 | 893.39 | 0.00 | ORB-long ORB[889.00,893.00] vol=2.9x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:10:00 | 901.89 | 894.41 | 0.00 | T1 1.5R @ 901.89 |
| Stop hit — per-position SL triggered | 2024-07-26 10:20:00 | 898.15 | 895.97 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:10:00 | 914.50 | 910.46 | 0.00 | ORB-long ORB[904.00,910.75] vol=3.5x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-07-30 11:55:00 | 911.26 | 913.41 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 11:00:00 | 909.00 | 913.68 | 0.00 | ORB-short ORB[910.15,920.00] vol=1.5x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:40:00 | 905.41 | 912.22 | 0.00 | T1 1.5R @ 905.41 |
| Stop hit — per-position SL triggered | 2024-07-31 13:45:00 | 909.00 | 906.30 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:10:00 | 925.05 | 919.45 | 0.00 | ORB-long ORB[908.80,921.90] vol=5.2x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:15:00 | 930.62 | 922.51 | 0.00 | T1 1.5R @ 930.62 |
| Target hit | 2024-08-01 13:25:00 | 945.40 | 945.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2024-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:25:00 | 875.20 | 878.43 | 0.00 | ORB-short ORB[875.30,882.60] vol=3.5x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-08-09 10:45:00 | 877.93 | 877.35 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 900.50 | 904.18 | 0.00 | ORB-short ORB[902.30,911.80] vol=5.8x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:55:00 | 894.65 | 902.82 | 0.00 | T1 1.5R @ 894.65 |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 900.50 | 901.07 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 902.85 | 905.77 | 0.00 | ORB-short ORB[907.10,912.40] vol=2.9x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-08-19 10:55:00 | 905.41 | 905.86 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:00:00 | 928.00 | 929.99 | 0.00 | ORB-short ORB[928.95,936.45] vol=1.5x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-08-21 10:40:00 | 931.04 | 930.03 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:50:00 | 942.85 | 939.05 | 0.00 | ORB-long ORB[934.65,942.00] vol=4.3x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-08-23 11:00:00 | 940.79 | 939.54 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 994.30 | 997.97 | 0.00 | ORB-short ORB[996.00,1006.30] vol=2.2x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-08-29 10:30:00 | 997.58 | 997.73 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:00:00 | 1210.90 | 1206.92 | 0.00 | ORB-long ORB[1192.05,1208.90] vol=2.1x ATR=7.34 |
| Stop hit — per-position SL triggered | 2024-09-05 10:10:00 | 1203.56 | 1207.30 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:05:00 | 1237.00 | 1222.18 | 0.00 | ORB-long ORB[1206.70,1223.50] vol=5.4x ATR=6.53 |
| Stop hit — per-position SL triggered | 2024-09-11 10:10:00 | 1230.47 | 1227.50 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:05:00 | 1206.05 | 1222.67 | 0.00 | ORB-short ORB[1220.25,1236.55] vol=4.4x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-09-18 10:10:00 | 1212.02 | 1221.74 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:55:00 | 1201.65 | 1205.42 | 0.00 | ORB-short ORB[1202.80,1215.60] vol=1.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-09-19 10:40:00 | 1205.85 | 1204.70 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:40:00 | 1200.90 | 1205.32 | 0.00 | ORB-short ORB[1204.10,1215.70] vol=1.5x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-09-20 11:10:00 | 1204.20 | 1204.53 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1249.50 | 1240.12 | 0.00 | ORB-long ORB[1225.00,1236.45] vol=9.8x ATR=4.14 |
| Stop hit — per-position SL triggered | 2024-09-26 09:35:00 | 1245.36 | 1245.83 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:55:00 | 1222.70 | 1233.06 | 0.00 | ORB-short ORB[1230.05,1242.00] vol=1.7x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 11:15:00 | 1217.37 | 1231.18 | 0.00 | T1 1.5R @ 1217.37 |
| Target hit | 2024-10-01 15:20:00 | 1194.40 | 1201.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 1122.75 | 1126.81 | 0.00 | ORB-short ORB[1124.05,1134.10] vol=2.7x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:40:00 | 1117.61 | 1125.51 | 0.00 | T1 1.5R @ 1117.61 |
| Target hit | 2024-10-10 15:20:00 | 1101.25 | 1113.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 1041.35 | 1043.85 | 0.00 | ORB-short ORB[1047.90,1059.40] vol=2.0x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:05:00 | 1034.52 | 1043.05 | 0.00 | T1 1.5R @ 1034.52 |
| Stop hit — per-position SL triggered | 2024-10-17 13:25:00 | 1041.35 | 1038.80 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 1026.00 | 1032.48 | 0.00 | ORB-short ORB[1031.40,1044.70] vol=2.6x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:10:00 | 1019.09 | 1025.40 | 0.00 | T1 1.5R @ 1019.09 |
| Stop hit — per-position SL triggered | 2024-10-21 12:40:00 | 1026.00 | 1024.88 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:05:00 | 1034.40 | 1027.85 | 0.00 | ORB-long ORB[1019.00,1028.90] vol=6.4x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-10-31 11:25:00 | 1030.85 | 1028.47 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:05:00 | 1025.85 | 1031.41 | 0.00 | ORB-short ORB[1032.05,1044.25] vol=2.0x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:55:00 | 1019.74 | 1027.43 | 0.00 | T1 1.5R @ 1019.74 |
| Stop hit — per-position SL triggered | 2024-11-04 12:00:00 | 1025.85 | 1024.44 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:45:00 | 976.55 | 979.46 | 0.00 | ORB-short ORB[978.20,984.65] vol=2.9x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:55:00 | 972.21 | 977.94 | 0.00 | T1 1.5R @ 972.21 |
| Target hit | 2024-11-12 11:50:00 | 975.60 | 974.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 932.60 | 940.09 | 0.00 | ORB-short ORB[934.15,948.00] vol=1.8x ATR=5.32 |
| Stop hit — per-position SL triggered | 2024-11-13 09:55:00 | 937.92 | 937.03 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 10:35:00 | 955.60 | 947.50 | 0.00 | ORB-long ORB[934.15,942.20] vol=1.8x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 10:45:00 | 963.42 | 950.63 | 0.00 | T1 1.5R @ 963.42 |
| Target hit | 2024-11-18 15:20:00 | 1026.00 | 996.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-11-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:50:00 | 1018.95 | 1011.80 | 0.00 | ORB-long ORB[1001.20,1013.95] vol=3.1x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-11-22 11:25:00 | 1015.25 | 1012.18 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:10:00 | 1064.40 | 1070.64 | 0.00 | ORB-short ORB[1070.55,1079.05] vol=1.6x ATR=3.43 |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 1067.83 | 1069.90 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:10:00 | 1072.25 | 1075.61 | 0.00 | ORB-short ORB[1074.30,1088.95] vol=3.4x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:15:00 | 1067.86 | 1075.24 | 0.00 | T1 1.5R @ 1067.86 |
| Target hit | 2024-12-06 15:20:00 | 1065.75 | 1071.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1091.30 | 1081.21 | 0.00 | ORB-long ORB[1072.10,1088.15] vol=2.4x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:15:00 | 1097.76 | 1087.05 | 0.00 | T1 1.5R @ 1097.76 |
| Target hit | 2024-12-16 15:20:00 | 1100.75 | 1093.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-12-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 11:00:00 | 1125.00 | 1131.97 | 0.00 | ORB-short ORB[1129.05,1144.00] vol=2.7x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:40:00 | 1118.07 | 1125.28 | 0.00 | T1 1.5R @ 1118.07 |
| Target hit | 2024-12-20 14:40:00 | 1122.85 | 1122.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 1138.50 | 1133.90 | 0.00 | ORB-long ORB[1126.00,1134.90] vol=1.6x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-12-30 10:50:00 | 1135.40 | 1134.05 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 1167.30 | 1165.15 | 0.00 | ORB-long ORB[1153.00,1164.55] vol=2.5x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:00:00 | 1174.88 | 1167.62 | 0.00 | T1 1.5R @ 1174.88 |
| Target hit | 2025-01-01 12:20:00 | 1181.55 | 1182.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1186.35 | 1172.28 | 0.00 | ORB-long ORB[1164.50,1180.70] vol=8.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-01-02 11:50:00 | 1182.11 | 1177.80 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 943.95 | 950.58 | 0.00 | ORB-short ORB[949.05,960.95] vol=4.4x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-01-15 09:40:00 | 947.54 | 949.21 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 957.90 | 964.87 | 0.00 | ORB-short ORB[959.60,972.40] vol=3.9x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:30:00 | 954.17 | 963.79 | 0.00 | T1 1.5R @ 954.17 |
| Target hit | 2025-01-16 15:20:00 | 933.65 | 942.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 920.80 | 925.48 | 0.00 | ORB-short ORB[921.25,934.70] vol=2.3x ATR=4.02 |
| Stop hit — per-position SL triggered | 2025-01-20 09:35:00 | 924.82 | 925.39 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:05:00 | 900.00 | 913.30 | 0.00 | ORB-short ORB[918.05,926.40] vol=5.8x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-01-21 11:20:00 | 903.47 | 911.79 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 891.20 | 881.42 | 0.00 | ORB-long ORB[872.05,877.85] vol=2.3x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:20:00 | 897.36 | 885.67 | 0.00 | T1 1.5R @ 897.36 |
| Target hit | 2025-01-23 11:45:00 | 893.40 | 893.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-01-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:40:00 | 868.15 | 875.38 | 0.00 | ORB-short ORB[884.00,894.50] vol=9.3x ATR=3.45 |
| Stop hit — per-position SL triggered | 2025-01-24 11:50:00 | 871.60 | 874.01 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 890.15 | 896.29 | 0.00 | ORB-short ORB[895.05,903.00] vol=2.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 892.24 | 896.17 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:35:00 | 869.75 | 874.36 | 0.00 | ORB-short ORB[871.50,883.50] vol=3.4x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:10:00 | 864.07 | 870.72 | 0.00 | T1 1.5R @ 864.07 |
| Target hit | 2025-02-10 15:20:00 | 853.85 | 859.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:50:00 | 1155.90 | 1144.53 | 0.00 | ORB-long ORB[1133.10,1148.00] vol=5.3x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-03-18 11:05:00 | 1151.23 | 1145.82 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 11:00:00 | 1127.85 | 1134.48 | 0.00 | ORB-short ORB[1128.05,1142.95] vol=2.0x ATR=4.16 |
| Stop hit — per-position SL triggered | 2025-03-19 11:35:00 | 1132.01 | 1134.01 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 1204.85 | 1199.66 | 0.00 | ORB-long ORB[1188.00,1204.00] vol=5.7x ATR=4.77 |
| Stop hit — per-position SL triggered | 2025-03-21 11:15:00 | 1200.08 | 1200.00 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 09:30:00 | 1071.10 | 1078.90 | 0.00 | ORB-short ORB[1073.30,1089.50] vol=2.5x ATR=8.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-08 10:35:00 | 1058.27 | 1074.12 | 0.00 | T1 1.5R @ 1058.27 |
| Stop hit — per-position SL triggered | 2025-04-08 12:30:00 | 1071.10 | 1068.34 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:15:00 | 1139.80 | 1129.50 | 0.00 | ORB-long ORB[1122.10,1133.40] vol=2.2x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:25:00 | 1145.43 | 1135.26 | 0.00 | T1 1.5R @ 1145.43 |
| Target hit | 2025-04-22 12:05:00 | 1158.20 | 1158.37 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:35:00 | 823.95 | 2024-05-14 09:40:00 | 827.20 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-16 09:35:00 | 799.20 | 2024-05-16 09:40:00 | 802.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-06 10:10:00 | 802.50 | 2024-06-06 12:00:00 | 798.54 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-13 09:30:00 | 808.10 | 2024-06-13 09:35:00 | 811.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-13 09:30:00 | 808.10 | 2024-06-13 11:45:00 | 829.35 | TARGET_HIT | 0.50 | 2.63% |
| SELL | retest1 | 2024-06-21 10:15:00 | 815.55 | 2024-06-21 10:20:00 | 818.01 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-25 09:55:00 | 851.50 | 2024-06-25 10:00:00 | 856.56 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-25 09:55:00 | 851.50 | 2024-06-25 10:05:00 | 851.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 10:45:00 | 895.95 | 2024-07-05 11:20:00 | 893.44 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-15 09:45:00 | 905.60 | 2024-07-15 10:00:00 | 903.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-19 09:55:00 | 881.20 | 2024-07-19 10:10:00 | 877.48 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-19 09:55:00 | 881.20 | 2024-07-19 10:35:00 | 881.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 09:40:00 | 896.30 | 2024-07-23 09:50:00 | 900.35 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-23 09:40:00 | 896.30 | 2024-07-23 12:15:00 | 896.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-25 09:55:00 | 880.10 | 2024-07-25 10:40:00 | 883.58 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-26 10:05:00 | 898.15 | 2024-07-26 10:10:00 | 901.89 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-26 10:05:00 | 898.15 | 2024-07-26 10:20:00 | 898.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 10:10:00 | 914.50 | 2024-07-30 11:55:00 | 911.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-31 11:00:00 | 909.00 | 2024-07-31 11:40:00 | 905.41 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-31 11:00:00 | 909.00 | 2024-07-31 13:45:00 | 909.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 10:10:00 | 925.05 | 2024-08-01 10:15:00 | 930.62 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-01 10:10:00 | 925.05 | 2024-08-01 13:25:00 | 945.40 | TARGET_HIT | 0.50 | 2.20% |
| SELL | retest1 | 2024-08-09 10:25:00 | 875.20 | 2024-08-09 10:45:00 | 877.93 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-13 09:35:00 | 900.50 | 2024-08-13 09:55:00 | 894.65 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-08-13 09:35:00 | 900.50 | 2024-08-13 11:15:00 | 900.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-19 10:45:00 | 902.85 | 2024-08-19 10:55:00 | 905.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-21 10:00:00 | 928.00 | 2024-08-21 10:40:00 | 931.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-23 10:50:00 | 942.85 | 2024-08-23 11:00:00 | 940.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-29 10:20:00 | 994.30 | 2024-08-29 10:30:00 | 997.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-05 10:00:00 | 1210.90 | 2024-09-05 10:10:00 | 1203.56 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-09-11 10:05:00 | 1237.00 | 2024-09-11 10:10:00 | 1230.47 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-09-18 10:05:00 | 1206.05 | 2024-09-18 10:10:00 | 1212.02 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-09-19 09:55:00 | 1201.65 | 2024-09-19 10:40:00 | 1205.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-20 10:40:00 | 1200.90 | 2024-09-20 11:10:00 | 1204.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-26 09:30:00 | 1249.50 | 2024-09-26 09:35:00 | 1245.36 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-01 10:55:00 | 1222.70 | 2024-10-01 11:15:00 | 1217.37 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-01 10:55:00 | 1222.70 | 2024-10-01 15:20:00 | 1194.40 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2024-10-10 11:10:00 | 1122.75 | 2024-10-10 11:40:00 | 1117.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-10 11:10:00 | 1122.75 | 2024-10-10 15:20:00 | 1101.25 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2024-10-17 10:55:00 | 1041.35 | 2024-10-17 11:05:00 | 1034.52 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-17 10:55:00 | 1041.35 | 2024-10-17 13:25:00 | 1041.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:35:00 | 1026.00 | 2024-10-21 12:10:00 | 1019.09 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-21 09:35:00 | 1026.00 | 2024-10-21 12:40:00 | 1026.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 11:05:00 | 1034.40 | 2024-10-31 11:25:00 | 1030.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-04 10:05:00 | 1025.85 | 2024-11-04 10:55:00 | 1019.74 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-04 10:05:00 | 1025.85 | 2024-11-04 12:00:00 | 1025.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 09:45:00 | 976.55 | 2024-11-12 09:55:00 | 972.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-12 09:45:00 | 976.55 | 2024-11-12 11:50:00 | 975.60 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-11-13 09:30:00 | 932.60 | 2024-11-13 09:55:00 | 937.92 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-11-18 10:35:00 | 955.60 | 2024-11-18 10:45:00 | 963.42 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-11-18 10:35:00 | 955.60 | 2024-11-18 15:20:00 | 1026.00 | TARGET_HIT | 0.50 | 7.37% |
| BUY | retest1 | 2024-11-22 10:50:00 | 1018.95 | 2024-11-22 11:25:00 | 1015.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-05 10:10:00 | 1064.40 | 2024-12-05 10:15:00 | 1067.83 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-06 11:10:00 | 1072.25 | 2024-12-06 11:15:00 | 1067.86 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-06 11:10:00 | 1072.25 | 2024-12-06 15:20:00 | 1065.75 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-16 10:15:00 | 1091.30 | 2024-12-16 12:15:00 | 1097.76 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-12-16 10:15:00 | 1091.30 | 2024-12-16 15:20:00 | 1100.75 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-12-20 11:00:00 | 1125.00 | 2024-12-20 12:40:00 | 1118.07 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-20 11:00:00 | 1125.00 | 2024-12-20 14:40:00 | 1122.85 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2024-12-30 10:35:00 | 1138.50 | 2024-12-30 10:50:00 | 1135.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-01 10:50:00 | 1167.30 | 2025-01-01 11:00:00 | 1174.88 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-01 10:50:00 | 1167.30 | 2025-01-01 12:20:00 | 1181.55 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-01-02 11:15:00 | 1186.35 | 2025-01-02 11:50:00 | 1182.11 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-15 09:30:00 | 943.95 | 2025-01-15 09:40:00 | 947.54 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-16 11:05:00 | 957.90 | 2025-01-16 11:30:00 | 954.17 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-16 11:05:00 | 957.90 | 2025-01-16 15:20:00 | 933.65 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2025-01-20 09:30:00 | 920.80 | 2025-01-20 09:35:00 | 924.82 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-21 11:05:00 | 900.00 | 2025-01-21 11:20:00 | 903.47 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-23 09:55:00 | 891.20 | 2025-01-23 10:20:00 | 897.36 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-23 09:55:00 | 891.20 | 2025-01-23 11:45:00 | 893.40 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-01-24 10:40:00 | 868.15 | 2025-01-24 11:50:00 | 871.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-01 11:00:00 | 890.15 | 2025-02-01 11:10:00 | 892.24 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-10 09:35:00 | 869.75 | 2025-02-10 10:10:00 | 864.07 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-02-10 09:35:00 | 869.75 | 2025-02-10 15:20:00 | 853.85 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2025-03-18 10:50:00 | 1155.90 | 2025-03-18 11:05:00 | 1151.23 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-19 11:00:00 | 1127.85 | 2025-03-19 11:35:00 | 1132.01 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1204.85 | 2025-03-21 11:15:00 | 1200.08 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-08 09:30:00 | 1071.10 | 2025-04-08 10:35:00 | 1058.27 | PARTIAL | 0.50 | 1.20% |
| SELL | retest1 | 2025-04-08 09:30:00 | 1071.10 | 2025-04-08 12:30:00 | 1071.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 10:15:00 | 1139.80 | 2025-04-22 10:25:00 | 1145.43 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-22 10:15:00 | 1139.80 | 2025-04-22 12:05:00 | 1158.20 | TARGET_HIT | 0.50 | 1.61% |
