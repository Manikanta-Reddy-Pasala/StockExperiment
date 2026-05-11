# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1237.00
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
| ENTRY1 | 68 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 11 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 57
- **Target hits / Stop hits / Partials:** 11 / 57 / 25
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 15.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.25% | 9.8% |
| BUY @ 2nd Alert (retest1) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.25% | 9.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 19 | 35.8% | 6 | 34 | 13 | 0.12% | 6.2% |
| SELL @ 2nd Alert (retest1) | 53 | 19 | 35.8% | 6 | 34 | 13 | 0.12% | 6.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 93 | 36 | 38.7% | 11 | 57 | 25 | 0.17% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:40:00 | 989.95 | 986.70 | 0.00 | ORB-long ORB[985.10,989.85] vol=2.0x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:45:00 | 992.70 | 992.34 | 0.00 | T1 1.5R @ 992.70 |
| Target hit | 2024-05-27 10:55:00 | 991.00 | 992.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:45:00 | 988.70 | 993.25 | 0.00 | ORB-short ORB[995.00,998.00] vol=1.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 14:10:00 | 984.01 | 988.72 | 0.00 | T1 1.5R @ 984.01 |
| Target hit | 2024-05-28 15:05:00 | 988.25 | 988.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-06-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:00:00 | 962.00 | 955.01 | 0.00 | ORB-long ORB[944.70,955.00] vol=6.2x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:05:00 | 967.73 | 959.13 | 0.00 | T1 1.5R @ 967.73 |
| Target hit | 2024-06-07 15:20:00 | 987.90 | 980.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 1004.90 | 1000.34 | 0.00 | ORB-long ORB[995.10,1001.90] vol=4.9x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-06-11 10:05:00 | 1001.84 | 1001.00 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 999.85 | 998.07 | 0.00 | ORB-long ORB[985.40,998.35] vol=2.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-06-12 11:35:00 | 996.99 | 998.36 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:20:00 | 1047.00 | 1041.67 | 0.00 | ORB-long ORB[1030.05,1040.00] vol=4.0x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:25:00 | 1053.55 | 1043.14 | 0.00 | T1 1.5R @ 1053.55 |
| Stop hit — per-position SL triggered | 2024-06-20 11:35:00 | 1047.00 | 1046.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:05:00 | 1030.00 | 1036.04 | 0.00 | ORB-short ORB[1038.00,1048.30] vol=1.8x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:45:00 | 1024.55 | 1032.05 | 0.00 | T1 1.5R @ 1024.55 |
| Stop hit — per-position SL triggered | 2024-06-21 10:55:00 | 1030.00 | 1030.64 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:55:00 | 1016.05 | 1013.48 | 0.00 | ORB-long ORB[1001.80,1013.50] vol=5.2x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-06-25 10:00:00 | 1012.05 | 1013.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1002.45 | 998.08 | 0.00 | ORB-long ORB[992.70,1002.00] vol=4.2x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-06-26 11:20:00 | 999.93 | 998.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 984.75 | 986.87 | 0.00 | ORB-short ORB[986.40,991.35] vol=4.1x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-06-28 11:20:00 | 986.85 | 986.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:00:00 | 1003.50 | 1007.20 | 0.00 | ORB-short ORB[1005.00,1014.95] vol=2.8x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-07-02 10:30:00 | 1006.19 | 1006.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:45:00 | 1010.25 | 1014.89 | 0.00 | ORB-short ORB[1014.80,1024.00] vol=3.2x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:00:00 | 1005.97 | 1013.17 | 0.00 | T1 1.5R @ 1005.97 |
| Target hit | 2024-07-08 15:20:00 | 991.00 | 998.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 997.00 | 992.93 | 0.00 | ORB-long ORB[986.65,993.95] vol=1.5x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-07-09 10:30:00 | 993.96 | 993.71 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 988.80 | 997.08 | 0.00 | ORB-short ORB[993.85,1004.25] vol=2.0x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 992.66 | 994.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 987.85 | 991.57 | 0.00 | ORB-short ORB[990.50,997.85] vol=2.5x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-07-15 09:40:00 | 991.22 | 991.44 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 999.00 | 1005.53 | 0.00 | ORB-short ORB[1002.95,1014.25] vol=2.1x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1003.51 | 1005.10 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:55:00 | 996.40 | 1002.78 | 0.00 | ORB-short ORB[996.45,1010.65] vol=2.0x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:55:00 | 990.56 | 999.18 | 0.00 | T1 1.5R @ 990.56 |
| Target hit | 2024-07-19 15:20:00 | 970.25 | 986.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 09:30:00 | 977.35 | 970.61 | 0.00 | ORB-long ORB[961.00,975.00] vol=1.8x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:50:00 | 984.48 | 976.33 | 0.00 | T1 1.5R @ 984.48 |
| Target hit | 2024-07-22 15:20:00 | 998.00 | 993.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:15:00 | 1003.00 | 996.01 | 0.00 | ORB-long ORB[989.30,1000.00] vol=2.5x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 10:35:00 | 1009.13 | 997.48 | 0.00 | T1 1.5R @ 1009.13 |
| Target hit | 2024-07-23 11:15:00 | 1009.85 | 1011.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 1052.00 | 1044.30 | 0.00 | ORB-long ORB[1032.95,1048.10] vol=3.0x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-07-29 09:35:00 | 1048.29 | 1045.31 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:55:00 | 1068.35 | 1063.99 | 0.00 | ORB-long ORB[1053.15,1063.00] vol=2.3x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-08-01 10:20:00 | 1064.37 | 1064.60 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:35:00 | 1169.25 | 1160.78 | 0.00 | ORB-long ORB[1153.95,1167.90] vol=1.7x ATR=7.05 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 1162.20 | 1161.08 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 1123.45 | 1130.50 | 0.00 | ORB-short ORB[1129.20,1138.25] vol=2.0x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-08-13 09:40:00 | 1128.32 | 1129.73 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 1102.50 | 1104.96 | 0.00 | ORB-short ORB[1104.00,1120.00] vol=1.6x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-08-14 11:45:00 | 1106.63 | 1104.64 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 1152.25 | 1156.11 | 0.00 | ORB-short ORB[1155.00,1164.40] vol=2.2x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:35:00 | 1146.30 | 1154.37 | 0.00 | T1 1.5R @ 1146.30 |
| Stop hit — per-position SL triggered | 2024-08-23 09:40:00 | 1152.25 | 1154.48 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 1132.00 | 1137.56 | 0.00 | ORB-short ORB[1132.20,1145.65] vol=1.5x ATR=4.83 |
| Stop hit — per-position SL triggered | 2024-08-29 09:50:00 | 1136.83 | 1135.82 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 1182.30 | 1167.76 | 0.00 | ORB-long ORB[1152.00,1166.55] vol=3.9x ATR=6.35 |
| Stop hit — per-position SL triggered | 2024-09-02 10:05:00 | 1175.95 | 1171.70 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 1145.85 | 1151.98 | 0.00 | ORB-short ORB[1147.15,1159.30] vol=2.5x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 11:25:00 | 1140.39 | 1151.24 | 0.00 | T1 1.5R @ 1140.39 |
| Stop hit — per-position SL triggered | 2024-09-03 12:55:00 | 1145.85 | 1147.80 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1157.60 | 1150.62 | 0.00 | ORB-long ORB[1137.00,1148.35] vol=4.2x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-09-05 09:35:00 | 1153.01 | 1151.13 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 1120.75 | 1126.14 | 0.00 | ORB-short ORB[1129.00,1140.70] vol=5.4x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-09-06 10:50:00 | 1125.22 | 1124.16 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 1137.80 | 1136.38 | 0.00 | ORB-long ORB[1129.05,1134.85] vol=8.7x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-09-10 09:35:00 | 1132.28 | 1135.59 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:25:00 | 1126.75 | 1126.81 | 0.00 | ORB-short ORB[1128.00,1138.45] vol=4.0x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-09-11 10:35:00 | 1130.55 | 1127.63 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:40:00 | 1098.20 | 1100.46 | 0.00 | ORB-short ORB[1099.35,1105.65] vol=2.9x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-09-12 09:50:00 | 1102.31 | 1100.69 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:20:00 | 1082.10 | 1090.30 | 0.00 | ORB-short ORB[1087.00,1102.75] vol=1.7x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-09-16 10:35:00 | 1086.57 | 1089.91 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 1081.50 | 1086.87 | 0.00 | ORB-short ORB[1090.05,1098.95] vol=5.8x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-09-19 09:55:00 | 1085.71 | 1086.70 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 1083.05 | 1086.97 | 0.00 | ORB-short ORB[1088.00,1102.75] vol=3.2x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 1088.16 | 1086.60 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:10:00 | 1060.90 | 1069.34 | 0.00 | ORB-short ORB[1065.55,1080.20] vol=1.6x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-09-25 10:20:00 | 1064.91 | 1068.94 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:50:00 | 1073.80 | 1077.40 | 0.00 | ORB-short ORB[1076.00,1086.00] vol=1.6x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:15:00 | 1068.45 | 1075.79 | 0.00 | T1 1.5R @ 1068.45 |
| Stop hit — per-position SL triggered | 2024-09-26 10:25:00 | 1073.80 | 1074.46 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:45:00 | 1044.85 | 1042.07 | 0.00 | ORB-long ORB[1035.35,1039.70] vol=1.9x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-10-01 09:50:00 | 1042.02 | 1042.15 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:30:00 | 1028.40 | 1034.35 | 0.00 | ORB-short ORB[1035.00,1048.85] vol=1.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-10-03 10:40:00 | 1032.60 | 1033.97 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:45:00 | 991.20 | 982.80 | 0.00 | ORB-long ORB[970.70,983.00] vol=2.2x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:15:00 | 997.53 | 985.30 | 0.00 | T1 1.5R @ 997.53 |
| Stop hit — per-position SL triggered | 2024-10-08 11:20:00 | 991.20 | 985.92 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:30:00 | 1004.50 | 998.45 | 0.00 | ORB-long ORB[992.35,999.50] vol=2.3x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 13:20:00 | 1011.53 | 1005.30 | 0.00 | T1 1.5R @ 1011.53 |
| Stop hit — per-position SL triggered | 2024-10-09 13:25:00 | 1004.50 | 1005.35 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:40:00 | 1017.95 | 1014.58 | 0.00 | ORB-long ORB[1009.10,1017.05] vol=1.9x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-10-10 10:25:00 | 1014.21 | 1015.70 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 1027.55 | 1018.90 | 0.00 | ORB-long ORB[1003.85,1018.00] vol=6.0x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:00:00 | 1035.74 | 1021.44 | 0.00 | T1 1.5R @ 1035.74 |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 1027.55 | 1023.29 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:10:00 | 1018.35 | 1016.67 | 0.00 | ORB-long ORB[1009.50,1016.10] vol=2.0x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:50:00 | 1022.67 | 1020.59 | 0.00 | T1 1.5R @ 1022.67 |
| Target hit | 2024-10-15 15:20:00 | 1050.20 | 1040.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 1028.60 | 1031.51 | 0.00 | ORB-short ORB[1029.20,1044.15] vol=1.7x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:45:00 | 1021.37 | 1029.52 | 0.00 | T1 1.5R @ 1021.37 |
| Target hit | 2024-10-25 15:20:00 | 995.80 | 1000.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 1003.75 | 1006.89 | 0.00 | ORB-short ORB[1005.80,1015.20] vol=6.4x ATR=4.66 |
| Stop hit — per-position SL triggered | 2024-10-29 10:10:00 | 1008.41 | 1006.12 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 1144.70 | 1156.95 | 0.00 | ORB-short ORB[1153.60,1168.40] vol=1.5x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:00:00 | 1137.34 | 1154.14 | 0.00 | T1 1.5R @ 1137.34 |
| Stop hit — per-position SL triggered | 2024-11-27 14:15:00 | 1144.70 | 1144.68 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 1158.00 | 1153.74 | 0.00 | ORB-long ORB[1137.95,1155.00] vol=7.3x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-12-03 09:45:00 | 1153.01 | 1153.77 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 1113.70 | 1118.16 | 0.00 | ORB-short ORB[1117.00,1126.75] vol=1.8x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 12:00:00 | 1108.78 | 1117.31 | 0.00 | T1 1.5R @ 1108.78 |
| Stop hit — per-position SL triggered | 2024-12-11 15:15:00 | 1113.70 | 1112.52 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 1092.95 | 1097.92 | 0.00 | ORB-short ORB[1098.50,1114.00] vol=3.5x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:55:00 | 1085.82 | 1092.91 | 0.00 | T1 1.5R @ 1085.82 |
| Stop hit — per-position SL triggered | 2024-12-12 12:30:00 | 1092.95 | 1092.42 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:05:00 | 1093.60 | 1086.84 | 0.00 | ORB-long ORB[1082.25,1091.20] vol=2.7x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:30:00 | 1099.39 | 1091.12 | 0.00 | T1 1.5R @ 1099.39 |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 1093.60 | 1093.52 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 1074.35 | 1084.40 | 0.00 | ORB-short ORB[1084.20,1090.50] vol=1.5x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:25:00 | 1069.20 | 1077.94 | 0.00 | T1 1.5R @ 1069.20 |
| Stop hit — per-position SL triggered | 2024-12-20 10:35:00 | 1074.35 | 1077.81 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:05:00 | 1078.15 | 1084.14 | 0.00 | ORB-short ORB[1091.20,1103.90] vol=3.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 1082.28 | 1083.26 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 1084.65 | 1088.16 | 0.00 | ORB-short ORB[1086.30,1099.80] vol=1.7x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-12-27 09:50:00 | 1088.65 | 1088.00 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:45:00 | 1100.05 | 1087.50 | 0.00 | ORB-long ORB[1080.05,1094.75] vol=2.5x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 11:00:00 | 1107.00 | 1093.18 | 0.00 | T1 1.5R @ 1107.00 |
| Stop hit — per-position SL triggered | 2024-12-31 11:05:00 | 1100.05 | 1095.28 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:15:00 | 1112.50 | 1117.08 | 0.00 | ORB-short ORB[1113.70,1127.45] vol=1.5x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-01-02 12:10:00 | 1116.25 | 1116.18 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1040.25 | 1044.28 | 0.00 | ORB-short ORB[1041.55,1053.85] vol=2.7x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-01-15 09:40:00 | 1045.21 | 1043.98 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:05:00 | 1074.65 | 1069.90 | 0.00 | ORB-long ORB[1054.15,1069.80] vol=3.6x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 1070.55 | 1070.60 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:45:00 | 1095.15 | 1106.59 | 0.00 | ORB-short ORB[1099.55,1108.70] vol=1.5x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-02-04 11:00:00 | 1098.85 | 1106.08 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 1036.10 | 1041.63 | 0.00 | ORB-short ORB[1037.45,1050.45] vol=2.1x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 1040.41 | 1041.45 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:35:00 | 1025.00 | 1030.91 | 0.00 | ORB-short ORB[1031.70,1044.45] vol=5.8x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-02-18 10:50:00 | 1028.87 | 1030.83 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:55:00 | 935.15 | 943.69 | 0.00 | ORB-short ORB[946.00,958.10] vol=1.5x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-02-27 10:00:00 | 940.08 | 943.14 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 1002.60 | 996.67 | 0.00 | ORB-long ORB[988.00,997.45] vol=1.8x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:25:00 | 1009.67 | 1001.98 | 0.00 | T1 1.5R @ 1009.67 |
| Stop hit — per-position SL triggered | 2025-03-18 10:50:00 | 1002.60 | 1003.26 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-03-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:00:00 | 1049.90 | 1060.55 | 0.00 | ORB-short ORB[1053.70,1067.95] vol=3.2x ATR=5.31 |
| Target hit | 2025-03-26 15:20:00 | 1044.65 | 1052.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-03-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:05:00 | 1027.35 | 1029.58 | 0.00 | ORB-short ORB[1030.30,1045.50] vol=3.4x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-03-27 10:35:00 | 1032.16 | 1029.35 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 09:30:00 | 1083.40 | 1076.56 | 0.00 | ORB-long ORB[1062.65,1076.15] vol=2.7x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-04-03 09:45:00 | 1078.66 | 1077.53 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 1040.00 | 1050.20 | 0.00 | ORB-short ORB[1055.50,1068.60] vol=2.4x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:05:00 | 1032.88 | 1045.52 | 0.00 | T1 1.5R @ 1032.88 |
| Target hit | 2025-04-25 13:40:00 | 1029.30 | 1027.69 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-27 10:40:00 | 989.95 | 2024-05-27 10:45:00 | 992.70 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-05-27 10:40:00 | 989.95 | 2024-05-27 10:55:00 | 991.00 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-05-28 10:45:00 | 988.70 | 2024-05-28 14:10:00 | 984.01 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-28 10:45:00 | 988.70 | 2024-05-28 15:05:00 | 988.25 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-06-07 11:00:00 | 962.00 | 2024-06-07 11:05:00 | 967.73 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-06-07 11:00:00 | 962.00 | 2024-06-07 15:20:00 | 987.90 | TARGET_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2024-06-11 10:00:00 | 1004.90 | 2024-06-11 10:05:00 | 1001.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-12 11:00:00 | 999.85 | 2024-06-12 11:35:00 | 996.99 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-20 10:20:00 | 1047.00 | 2024-06-20 10:25:00 | 1053.55 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-20 10:20:00 | 1047.00 | 2024-06-20 11:35:00 | 1047.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:05:00 | 1030.00 | 2024-06-21 10:45:00 | 1024.55 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-21 10:05:00 | 1030.00 | 2024-06-21 10:55:00 | 1030.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 09:55:00 | 1016.05 | 2024-06-25 10:00:00 | 1012.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-26 11:15:00 | 1002.45 | 2024-06-26 11:20:00 | 999.93 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-28 11:15:00 | 984.75 | 2024-06-28 11:20:00 | 986.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-02 10:00:00 | 1003.50 | 2024-07-02 10:30:00 | 1006.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-08 10:45:00 | 1010.25 | 2024-07-08 11:00:00 | 1005.97 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-08 10:45:00 | 1010.25 | 2024-07-08 15:20:00 | 991.00 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2024-07-09 10:15:00 | 997.00 | 2024-07-09 10:30:00 | 993.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:10:00 | 988.80 | 2024-07-10 10:15:00 | 992.66 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-15 09:35:00 | 987.85 | 2024-07-15 09:40:00 | 991.22 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-18 09:35:00 | 999.00 | 2024-07-18 09:40:00 | 1003.51 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-19 09:55:00 | 996.40 | 2024-07-19 10:55:00 | 990.56 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-07-19 09:55:00 | 996.40 | 2024-07-19 15:20:00 | 970.25 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2024-07-22 09:30:00 | 977.35 | 2024-07-22 09:50:00 | 984.48 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-22 09:30:00 | 977.35 | 2024-07-22 15:20:00 | 998.00 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2024-07-23 10:15:00 | 1003.00 | 2024-07-23 10:35:00 | 1009.13 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-23 10:15:00 | 1003.00 | 2024-07-23 11:15:00 | 1009.85 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-07-29 09:30:00 | 1052.00 | 2024-07-29 09:35:00 | 1048.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-01 09:55:00 | 1068.35 | 2024-08-01 10:20:00 | 1064.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-09 10:35:00 | 1169.25 | 2024-08-09 10:50:00 | 1162.20 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-08-13 09:30:00 | 1123.45 | 2024-08-13 09:40:00 | 1128.32 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-14 10:55:00 | 1102.50 | 2024-08-14 11:45:00 | 1106.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-23 09:30:00 | 1152.25 | 2024-08-23 09:35:00 | 1146.30 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-23 09:30:00 | 1152.25 | 2024-08-23 09:40:00 | 1152.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 09:30:00 | 1132.00 | 2024-08-29 09:50:00 | 1136.83 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-02 10:00:00 | 1182.30 | 2024-09-02 10:05:00 | 1175.95 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-09-03 11:10:00 | 1145.85 | 2024-09-03 11:25:00 | 1140.39 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-03 11:10:00 | 1145.85 | 2024-09-03 12:55:00 | 1145.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 09:30:00 | 1157.60 | 2024-09-05 09:35:00 | 1153.01 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-06 09:50:00 | 1120.75 | 2024-09-06 10:50:00 | 1125.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-10 09:30:00 | 1137.80 | 2024-09-10 09:35:00 | 1132.28 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-09-11 10:25:00 | 1126.75 | 2024-09-11 10:35:00 | 1130.55 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-12 09:40:00 | 1098.20 | 2024-09-12 09:50:00 | 1102.31 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-16 10:20:00 | 1082.10 | 2024-09-16 10:35:00 | 1086.57 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-19 09:50:00 | 1081.50 | 2024-09-19 09:55:00 | 1085.71 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-24 09:55:00 | 1083.05 | 2024-09-24 10:10:00 | 1088.16 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-09-25 10:10:00 | 1060.90 | 2024-09-25 10:20:00 | 1064.91 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-26 09:50:00 | 1073.80 | 2024-09-26 10:15:00 | 1068.45 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-09-26 09:50:00 | 1073.80 | 2024-09-26 10:25:00 | 1073.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 09:45:00 | 1044.85 | 2024-10-01 09:50:00 | 1042.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-03 10:30:00 | 1028.40 | 2024-10-03 10:40:00 | 1032.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-08 10:45:00 | 991.20 | 2024-10-08 11:15:00 | 997.53 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-08 10:45:00 | 991.20 | 2024-10-08 11:20:00 | 991.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 09:30:00 | 1004.50 | 2024-10-09 13:20:00 | 1011.53 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-10-09 09:30:00 | 1004.50 | 2024-10-09 13:25:00 | 1004.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 09:40:00 | 1017.95 | 2024-10-10 10:25:00 | 1014.21 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-11 09:50:00 | 1027.55 | 2024-10-11 10:00:00 | 1035.74 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-10-11 09:50:00 | 1027.55 | 2024-10-11 10:15:00 | 1027.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 10:10:00 | 1018.35 | 2024-10-15 10:50:00 | 1022.67 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-15 10:10:00 | 1018.35 | 2024-10-15 15:20:00 | 1050.20 | TARGET_HIT | 0.50 | 3.13% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1028.60 | 2024-10-25 09:45:00 | 1021.37 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-10-25 09:35:00 | 1028.60 | 2024-10-25 15:20:00 | 995.80 | TARGET_HIT | 0.50 | 3.19% |
| SELL | retest1 | 2024-10-29 09:35:00 | 1003.75 | 2024-10-29 10:10:00 | 1008.41 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-11-27 10:50:00 | 1144.70 | 2024-11-27 11:00:00 | 1137.34 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-11-27 10:50:00 | 1144.70 | 2024-11-27 14:15:00 | 1144.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 09:40:00 | 1158.00 | 2024-12-03 09:45:00 | 1153.01 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-11 11:05:00 | 1113.70 | 2024-12-11 12:00:00 | 1108.78 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-11 11:05:00 | 1113.70 | 2024-12-11 15:15:00 | 1113.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:40:00 | 1092.95 | 2024-12-12 11:55:00 | 1085.82 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-12-12 09:40:00 | 1092.95 | 2024-12-12 12:30:00 | 1092.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-16 10:05:00 | 1093.60 | 2024-12-16 10:30:00 | 1099.39 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-12-16 10:05:00 | 1093.60 | 2024-12-16 11:15:00 | 1093.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:00:00 | 1074.35 | 2024-12-20 10:25:00 | 1069.20 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-20 10:00:00 | 1074.35 | 2024-12-20 10:35:00 | 1074.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:05:00 | 1078.15 | 2024-12-26 11:00:00 | 1082.28 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-27 09:40:00 | 1084.65 | 2024-12-27 09:50:00 | 1088.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-31 10:45:00 | 1100.05 | 2024-12-31 11:00:00 | 1107.00 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-31 10:45:00 | 1100.05 | 2024-12-31 11:05:00 | 1100.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 10:15:00 | 1112.50 | 2025-01-02 12:10:00 | 1116.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1040.25 | 2025-01-15 09:40:00 | 1045.21 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-01-17 10:05:00 | 1074.65 | 2025-01-17 10:15:00 | 1070.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-04 10:45:00 | 1095.15 | 2025-02-04 11:00:00 | 1098.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-06 09:30:00 | 1036.10 | 2025-02-06 09:40:00 | 1040.41 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-02-18 10:35:00 | 1025.00 | 2025-02-18 10:50:00 | 1028.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-27 09:55:00 | 935.15 | 2025-02-27 10:00:00 | 940.08 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-03-18 09:35:00 | 1002.60 | 2025-03-18 10:25:00 | 1009.67 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-03-18 09:35:00 | 1002.60 | 2025-03-18 10:50:00 | 1002.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-26 11:00:00 | 1049.90 | 2025-03-26 15:20:00 | 1044.65 | TARGET_HIT | 1.00 | 0.50% |
| SELL | retest1 | 2025-03-27 10:05:00 | 1027.35 | 2025-03-27 10:35:00 | 1032.16 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-03 09:30:00 | 1083.40 | 2025-04-03 09:45:00 | 1078.66 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-04-25 09:50:00 | 1040.00 | 2025-04-25 10:05:00 | 1032.88 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-04-25 09:50:00 | 1040.00 | 2025-04-25 13:40:00 | 1029.30 | TARGET_HIT | 0.50 | 1.03% |
