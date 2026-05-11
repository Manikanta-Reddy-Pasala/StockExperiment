# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1695.00
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 15 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 70
- **Target hits / Stop hits / Partials:** 15 / 70 / 37
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 16.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 24 | 41.4% | 7 | 34 | 17 | 0.12% | 6.7% |
| BUY @ 2nd Alert (retest1) | 58 | 24 | 41.4% | 7 | 34 | 17 | 0.12% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 28 | 43.8% | 8 | 36 | 20 | 0.15% | 9.7% |
| SELL @ 2nd Alert (retest1) | 64 | 28 | 43.8% | 8 | 36 | 20 | 0.15% | 9.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 52 | 42.6% | 15 | 70 | 37 | 0.13% | 16.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:45:00 | 1007.50 | 1004.79 | 0.00 | ORB-long ORB[991.60,1006.55] vol=1.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-05-17 09:55:00 | 1004.19 | 1004.96 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:20:00 | 965.00 | 971.84 | 0.00 | ORB-short ORB[973.25,983.60] vol=1.5x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:35:00 | 961.10 | 970.74 | 0.00 | T1 1.5R @ 961.10 |
| Stop hit — per-position SL triggered | 2024-05-23 11:30:00 | 965.00 | 966.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:00:00 | 959.40 | 964.90 | 0.00 | ORB-short ORB[962.15,972.45] vol=1.7x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 11:50:00 | 954.61 | 961.81 | 0.00 | T1 1.5R @ 954.61 |
| Stop hit — per-position SL triggered | 2024-05-27 13:20:00 | 959.40 | 960.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:00:00 | 913.00 | 918.20 | 0.00 | ORB-short ORB[925.00,933.30] vol=1.8x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-05-31 11:10:00 | 915.54 | 918.09 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-03 10:40:00 | 940.65 | 932.86 | 0.00 | ORB-long ORB[926.10,937.95] vol=1.6x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-06-03 11:30:00 | 936.65 | 934.63 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 10:00:00 | 919.65 | 912.34 | 0.00 | ORB-long ORB[899.45,910.00] vol=1.5x ATR=6.43 |
| Stop hit — per-position SL triggered | 2024-06-05 10:05:00 | 913.22 | 912.84 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:25:00 | 935.05 | 937.08 | 0.00 | ORB-short ORB[937.75,947.00] vol=2.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-06-10 10:30:00 | 937.58 | 937.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 11:15:00 | 962.35 | 955.28 | 0.00 | ORB-long ORB[945.55,957.20] vol=4.5x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:40:00 | 965.93 | 957.18 | 0.00 | T1 1.5R @ 965.93 |
| Stop hit — per-position SL triggered | 2024-06-11 11:45:00 | 962.35 | 957.34 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:05:00 | 954.70 | 951.84 | 0.00 | ORB-long ORB[945.30,953.60] vol=1.7x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:35:00 | 957.97 | 952.67 | 0.00 | T1 1.5R @ 957.97 |
| Target hit | 2024-06-12 15:20:00 | 964.35 | 959.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-06-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:30:00 | 997.85 | 993.85 | 0.00 | ORB-long ORB[983.30,996.00] vol=1.7x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-06-14 10:40:00 | 995.08 | 994.02 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:10:00 | 986.35 | 988.94 | 0.00 | ORB-short ORB[990.00,997.95] vol=1.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:15:00 | 981.90 | 988.30 | 0.00 | T1 1.5R @ 981.90 |
| Stop hit — per-position SL triggered | 2024-06-19 11:15:00 | 986.35 | 985.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:20:00 | 979.40 | 984.07 | 0.00 | ORB-short ORB[986.80,990.70] vol=5.3x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:55:00 | 975.44 | 982.55 | 0.00 | T1 1.5R @ 975.44 |
| Stop hit — per-position SL triggered | 2024-06-25 14:50:00 | 979.40 | 977.27 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:55:00 | 972.70 | 977.50 | 0.00 | ORB-short ORB[976.00,980.55] vol=2.8x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-06-26 10:00:00 | 975.81 | 977.47 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:10:00 | 975.00 | 973.21 | 0.00 | ORB-long ORB[966.95,972.05] vol=2.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-06-28 12:35:00 | 972.57 | 973.91 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:35:00 | 1004.20 | 997.87 | 0.00 | ORB-long ORB[988.15,1001.20] vol=3.8x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-07-09 11:25:00 | 1001.10 | 998.76 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 1002.00 | 1011.69 | 0.00 | ORB-short ORB[1012.15,1019.80] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 996.78 | 1008.75 | 0.00 | T1 1.5R @ 996.78 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 1002.00 | 1007.28 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 1024.55 | 1026.46 | 0.00 | ORB-short ORB[1031.30,1039.35] vol=1.5x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:30:00 | 1020.92 | 1025.19 | 0.00 | T1 1.5R @ 1020.92 |
| Target hit | 2024-07-12 13:05:00 | 1021.55 | 1021.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 1028.85 | 1036.46 | 0.00 | ORB-short ORB[1035.65,1047.40] vol=1.6x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-07-16 10:50:00 | 1032.43 | 1034.79 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:45:00 | 1029.20 | 1029.41 | 0.00 | ORB-short ORB[1029.50,1041.75] vol=2.4x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-07-18 09:50:00 | 1032.90 | 1029.68 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:50:00 | 1014.55 | 1025.28 | 0.00 | ORB-short ORB[1027.35,1038.95] vol=4.1x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:55:00 | 1009.80 | 1022.76 | 0.00 | T1 1.5R @ 1009.80 |
| Stop hit — per-position SL triggered | 2024-07-19 11:05:00 | 1014.55 | 1021.39 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 1091.00 | 1085.89 | 0.00 | ORB-long ORB[1078.75,1090.05] vol=2.5x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 12:20:00 | 1096.74 | 1089.22 | 0.00 | T1 1.5R @ 1096.74 |
| Target hit | 2024-07-25 15:20:00 | 1098.85 | 1093.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 1109.50 | 1104.35 | 0.00 | ORB-long ORB[1095.45,1106.95] vol=2.1x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:50:00 | 1116.23 | 1107.70 | 0.00 | T1 1.5R @ 1116.23 |
| Target hit | 2024-07-26 14:15:00 | 1115.35 | 1117.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-07-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:50:00 | 1097.65 | 1089.40 | 0.00 | ORB-long ORB[1083.35,1090.95] vol=2.4x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-07-31 10:55:00 | 1095.13 | 1089.58 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:20:00 | 1072.15 | 1068.10 | 0.00 | ORB-long ORB[1061.95,1069.15] vol=1.9x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:40:00 | 1078.37 | 1071.19 | 0.00 | T1 1.5R @ 1078.37 |
| Stop hit — per-position SL triggered | 2024-08-07 10:50:00 | 1072.15 | 1071.04 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 1089.80 | 1079.15 | 0.00 | ORB-long ORB[1075.00,1083.85] vol=1.7x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:20:00 | 1095.31 | 1081.75 | 0.00 | T1 1.5R @ 1095.31 |
| Stop hit — per-position SL triggered | 2024-08-08 11:25:00 | 1089.80 | 1083.28 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 1019.90 | 1008.23 | 0.00 | ORB-long ORB[994.05,1006.70] vol=1.9x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-08-20 10:45:00 | 1016.29 | 1013.90 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:05:00 | 1031.00 | 1027.90 | 0.00 | ORB-long ORB[1015.40,1029.00] vol=3.1x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:50:00 | 1037.11 | 1029.07 | 0.00 | T1 1.5R @ 1037.11 |
| Target hit | 2024-08-21 15:20:00 | 1047.60 | 1037.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:55:00 | 1053.00 | 1048.91 | 0.00 | ORB-long ORB[1043.80,1051.65] vol=1.9x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-08-22 10:10:00 | 1049.72 | 1049.83 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 1049.65 | 1055.17 | 0.00 | ORB-short ORB[1058.45,1069.95] vol=7.1x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:45:00 | 1043.68 | 1051.05 | 0.00 | T1 1.5R @ 1043.68 |
| Target hit | 2024-08-29 15:20:00 | 1037.25 | 1044.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2024-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:20:00 | 1066.75 | 1051.06 | 0.00 | ORB-long ORB[1037.90,1051.35] vol=2.8x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-08-30 10:35:00 | 1061.28 | 1055.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 1103.15 | 1108.13 | 0.00 | ORB-short ORB[1103.60,1117.90] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-09-03 11:30:00 | 1105.90 | 1107.31 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:55:00 | 1135.95 | 1125.41 | 0.00 | ORB-long ORB[1115.25,1126.70] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-09-10 11:05:00 | 1132.26 | 1126.16 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1151.80 | 1142.69 | 0.00 | ORB-long ORB[1133.75,1144.05] vol=2.7x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-09-11 09:40:00 | 1148.36 | 1145.35 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1145.00 | 1148.67 | 0.00 | ORB-short ORB[1148.35,1155.00] vol=1.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-09-13 11:45:00 | 1147.43 | 1147.54 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:15:00 | 1126.20 | 1135.35 | 0.00 | ORB-short ORB[1134.85,1146.40] vol=6.4x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-09-16 11:25:00 | 1129.12 | 1135.04 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:25:00 | 1141.40 | 1136.14 | 0.00 | ORB-long ORB[1131.00,1139.00] vol=1.5x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:20:00 | 1145.84 | 1140.65 | 0.00 | T1 1.5R @ 1145.84 |
| Stop hit — per-position SL triggered | 2024-09-17 11:45:00 | 1141.40 | 1141.15 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:35:00 | 1171.50 | 1166.25 | 0.00 | ORB-long ORB[1159.00,1166.00] vol=4.2x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:45:00 | 1177.08 | 1168.86 | 0.00 | T1 1.5R @ 1177.08 |
| Stop hit — per-position SL triggered | 2024-09-20 09:50:00 | 1171.50 | 1168.74 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:25:00 | 1185.10 | 1191.82 | 0.00 | ORB-short ORB[1190.10,1201.85] vol=1.9x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:00:00 | 1180.41 | 1190.85 | 0.00 | T1 1.5R @ 1180.41 |
| Stop hit — per-position SL triggered | 2024-09-24 13:30:00 | 1185.10 | 1185.40 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 11:10:00 | 1187.65 | 1180.66 | 0.00 | ORB-long ORB[1173.80,1185.75] vol=2.4x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:30:00 | 1191.84 | 1182.47 | 0.00 | T1 1.5R @ 1191.84 |
| Stop hit — per-position SL triggered | 2024-09-26 12:55:00 | 1187.65 | 1189.39 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:10:00 | 1202.05 | 1195.54 | 0.00 | ORB-long ORB[1185.05,1197.90] vol=1.6x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:40:00 | 1208.26 | 1198.75 | 0.00 | T1 1.5R @ 1208.26 |
| Stop hit — per-position SL triggered | 2024-09-27 10:55:00 | 1202.05 | 1200.34 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:40:00 | 1196.90 | 1187.55 | 0.00 | ORB-long ORB[1174.85,1185.90] vol=2.1x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-09-30 11:35:00 | 1193.27 | 1193.45 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:35:00 | 1187.70 | 1179.43 | 0.00 | ORB-long ORB[1166.80,1180.20] vol=1.9x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-10-04 10:40:00 | 1183.77 | 1179.89 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1146.60 | 1148.28 | 0.00 | ORB-short ORB[1147.75,1164.15] vol=8.6x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:10:00 | 1140.24 | 1148.08 | 0.00 | T1 1.5R @ 1140.24 |
| Stop hit — per-position SL triggered | 2024-10-08 10:35:00 | 1146.60 | 1146.02 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:15:00 | 1174.70 | 1175.52 | 0.00 | ORB-short ORB[1175.30,1189.95] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-10-14 11:30:00 | 1177.96 | 1175.59 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:00:00 | 1179.20 | 1191.74 | 0.00 | ORB-short ORB[1191.55,1202.40] vol=1.6x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-10-17 10:30:00 | 1184.05 | 1189.87 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 11:15:00 | 1191.00 | 1195.00 | 0.00 | ORB-short ORB[1191.05,1201.95] vol=3.4x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:30:00 | 1183.41 | 1189.83 | 0.00 | T1 1.5R @ 1183.41 |
| Target hit | 2024-10-22 15:20:00 | 1172.30 | 1183.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-10-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:05:00 | 1274.15 | 1269.45 | 0.00 | ORB-long ORB[1260.05,1270.70] vol=1.8x ATR=5.53 |
| Stop hit — per-position SL triggered | 2024-10-28 10:10:00 | 1268.62 | 1269.56 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 10:20:00 | 1262.85 | 1266.86 | 0.00 | ORB-short ORB[1264.10,1279.90] vol=1.5x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 13:00:00 | 1256.17 | 1264.43 | 0.00 | T1 1.5R @ 1256.17 |
| Target hit | 2024-10-30 15:20:00 | 1252.25 | 1258.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:10:00 | 1233.10 | 1244.37 | 0.00 | ORB-short ORB[1245.50,1262.30] vol=2.1x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:30:00 | 1226.08 | 1240.71 | 0.00 | T1 1.5R @ 1226.08 |
| Stop hit — per-position SL triggered | 2024-11-06 11:35:00 | 1233.10 | 1235.24 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:15:00 | 1226.95 | 1220.30 | 0.00 | ORB-long ORB[1207.35,1223.95] vol=3.1x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-11-12 10:25:00 | 1223.60 | 1220.89 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 11:10:00 | 1208.80 | 1213.28 | 0.00 | ORB-short ORB[1213.65,1228.25] vol=3.1x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 11:40:00 | 1203.77 | 1211.73 | 0.00 | T1 1.5R @ 1203.77 |
| Stop hit — per-position SL triggered | 2024-11-13 13:35:00 | 1208.80 | 1210.24 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:55:00 | 1161.05 | 1168.35 | 0.00 | ORB-short ORB[1169.55,1176.85] vol=1.5x ATR=3.08 |
| Stop hit — per-position SL triggered | 2024-11-22 10:05:00 | 1164.13 | 1164.89 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 1160.10 | 1163.81 | 0.00 | ORB-short ORB[1161.00,1177.35] vol=2.4x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-12-06 09:35:00 | 1164.37 | 1164.35 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:40:00 | 1170.00 | 1164.45 | 0.00 | ORB-long ORB[1160.00,1167.95] vol=1.5x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 13:20:00 | 1175.53 | 1167.35 | 0.00 | T1 1.5R @ 1175.53 |
| Target hit | 2024-12-09 15:20:00 | 1185.45 | 1174.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2024-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:40:00 | 1153.45 | 1160.50 | 0.00 | ORB-short ORB[1157.65,1165.85] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:20:00 | 1148.71 | 1158.34 | 0.00 | T1 1.5R @ 1148.71 |
| Stop hit — per-position SL triggered | 2024-12-11 12:25:00 | 1153.45 | 1153.05 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:55:00 | 1129.95 | 1131.70 | 0.00 | ORB-short ORB[1132.00,1148.20] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-12-12 11:20:00 | 1133.19 | 1131.57 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:00:00 | 1113.15 | 1117.77 | 0.00 | ORB-short ORB[1114.10,1124.00] vol=3.3x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-12-13 12:00:00 | 1116.84 | 1116.48 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:15:00 | 1120.90 | 1127.99 | 0.00 | ORB-short ORB[1130.45,1143.05] vol=2.3x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-12-20 10:30:00 | 1124.74 | 1126.71 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:45:00 | 1108.30 | 1105.03 | 0.00 | ORB-long ORB[1096.10,1105.40] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-12-26 12:00:00 | 1104.97 | 1106.94 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 1109.60 | 1104.77 | 0.00 | ORB-long ORB[1099.65,1108.25] vol=2.7x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-01-02 10:55:00 | 1106.79 | 1106.50 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 11:05:00 | 1100.05 | 1089.32 | 0.00 | ORB-long ORB[1085.00,1098.10] vol=1.9x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 1096.17 | 1089.95 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:45:00 | 1080.25 | 1081.37 | 0.00 | ORB-short ORB[1081.20,1090.65] vol=4.2x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-01-10 11:00:00 | 1083.16 | 1079.80 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:10:00 | 1075.00 | 1065.31 | 0.00 | ORB-long ORB[1062.50,1073.50] vol=1.9x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-01-14 11:20:00 | 1071.53 | 1065.40 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1041.50 | 1047.17 | 0.00 | ORB-short ORB[1044.35,1056.10] vol=1.6x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:05:00 | 1035.53 | 1042.91 | 0.00 | T1 1.5R @ 1035.53 |
| Target hit | 2025-01-15 14:20:00 | 1030.70 | 1030.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1065.55 | 1071.93 | 0.00 | ORB-short ORB[1067.60,1079.45] vol=2.3x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-01-21 10:25:00 | 1068.39 | 1071.09 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 1039.80 | 1042.59 | 0.00 | ORB-short ORB[1041.00,1056.00] vol=2.0x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-01-28 10:35:00 | 1043.43 | 1040.80 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:05:00 | 1153.40 | 1140.32 | 0.00 | ORB-long ORB[1117.35,1125.85] vol=2.0x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:10:00 | 1159.54 | 1146.76 | 0.00 | T1 1.5R @ 1159.54 |
| Stop hit — per-position SL triggered | 2025-02-01 10:15:00 | 1153.40 | 1147.83 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 1111.90 | 1106.81 | 0.00 | ORB-long ORB[1097.30,1107.95] vol=2.7x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:20:00 | 1116.48 | 1107.91 | 0.00 | T1 1.5R @ 1116.48 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1111.90 | 1110.26 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 11:10:00 | 1100.00 | 1103.40 | 0.00 | ORB-short ORB[1109.30,1118.95] vol=15.4x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:30:00 | 1095.23 | 1103.01 | 0.00 | T1 1.5R @ 1095.23 |
| Target hit | 2025-02-10 15:20:00 | 1090.55 | 1099.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:40:00 | 1072.30 | 1078.47 | 0.00 | ORB-short ORB[1073.65,1086.65] vol=2.0x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:00:00 | 1066.23 | 1074.03 | 0.00 | T1 1.5R @ 1066.23 |
| Target hit | 2025-02-11 13:40:00 | 1064.90 | 1064.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:15:00 | 1065.90 | 1077.80 | 0.00 | ORB-short ORB[1077.80,1092.60] vol=1.9x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-02-14 12:35:00 | 1069.06 | 1073.90 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:35:00 | 1066.15 | 1058.41 | 0.00 | ORB-long ORB[1047.10,1061.80] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:00:00 | 1071.37 | 1060.98 | 0.00 | T1 1.5R @ 1071.37 |
| Stop hit — per-position SL triggered | 2025-02-19 10:10:00 | 1066.15 | 1061.85 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:35:00 | 1007.40 | 1013.80 | 0.00 | ORB-short ORB[1012.00,1026.10] vol=3.3x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-02-28 10:15:00 | 1011.59 | 1012.72 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-03-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 09:55:00 | 989.10 | 992.37 | 0.00 | ORB-short ORB[992.45,1000.60] vol=3.2x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:05:00 | 983.04 | 992.03 | 0.00 | T1 1.5R @ 983.04 |
| Stop hit — per-position SL triggered | 2025-03-03 10:25:00 | 989.10 | 990.93 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:55:00 | 1040.40 | 1037.22 | 0.00 | ORB-long ORB[1026.10,1038.70] vol=2.9x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-03-10 10:00:00 | 1037.06 | 1037.40 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:50:00 | 1069.50 | 1075.15 | 0.00 | ORB-short ORB[1073.85,1089.80] vol=4.3x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 1064.48 | 1074.81 | 0.00 | T1 1.5R @ 1064.48 |
| Target hit | 2025-03-12 15:20:00 | 1061.05 | 1068.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2025-03-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:40:00 | 1095.80 | 1086.81 | 0.00 | ORB-long ORB[1073.50,1082.90] vol=3.1x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-03-19 10:55:00 | 1092.73 | 1089.48 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:55:00 | 1106.25 | 1103.60 | 0.00 | ORB-long ORB[1100.00,1105.80] vol=1.6x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:15:00 | 1110.80 | 1105.15 | 0.00 | T1 1.5R @ 1110.80 |
| Target hit | 2025-03-20 15:00:00 | 1120.40 | 1123.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 79 — SELL (started 2025-04-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 10:05:00 | 1132.55 | 1136.86 | 0.00 | ORB-short ORB[1132.65,1147.65] vol=2.0x ATR=4.03 |
| Stop hit — per-position SL triggered | 2025-04-02 10:35:00 | 1136.58 | 1135.82 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 10:45:00 | 1154.50 | 1146.85 | 0.00 | ORB-long ORB[1135.90,1146.95] vol=1.7x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:05:00 | 1161.33 | 1149.35 | 0.00 | T1 1.5R @ 1161.33 |
| Target hit | 2025-04-04 13:20:00 | 1156.95 | 1159.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — BUY (started 2025-04-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:50:00 | 1187.40 | 1177.89 | 0.00 | ORB-long ORB[1162.90,1179.55] vol=1.5x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-04-11 09:55:00 | 1183.26 | 1178.30 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:45:00 | 1205.50 | 1200.14 | 0.00 | ORB-long ORB[1191.30,1203.00] vol=2.7x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-04-16 11:35:00 | 1201.98 | 1201.67 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:40:00 | 1199.90 | 1202.64 | 0.00 | ORB-short ORB[1200.00,1212.00] vol=12.8x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-04-17 10:45:00 | 1202.94 | 1202.26 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 10:45:00 | 1297.10 | 1300.75 | 0.00 | ORB-short ORB[1303.00,1313.20] vol=1.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-05-05 11:00:00 | 1300.75 | 1300.22 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:15:00 | 1267.10 | 1277.26 | 0.00 | ORB-short ORB[1275.50,1287.20] vol=1.6x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-05-06 11:20:00 | 1270.73 | 1277.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 09:45:00 | 1007.50 | 2024-05-17 09:55:00 | 1004.19 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-23 10:20:00 | 965.00 | 2024-05-23 10:35:00 | 961.10 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-23 10:20:00 | 965.00 | 2024-05-23 11:30:00 | 965.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 10:00:00 | 959.40 | 2024-05-27 11:50:00 | 954.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-27 10:00:00 | 959.40 | 2024-05-27 13:20:00 | 959.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 11:00:00 | 913.00 | 2024-05-31 11:10:00 | 915.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-03 10:40:00 | 940.65 | 2024-06-03 11:30:00 | 936.65 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-05 10:00:00 | 919.65 | 2024-06-05 10:05:00 | 913.22 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest1 | 2024-06-10 10:25:00 | 935.05 | 2024-06-10 10:30:00 | 937.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-11 11:15:00 | 962.35 | 2024-06-11 11:40:00 | 965.93 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-11 11:15:00 | 962.35 | 2024-06-11 11:45:00 | 962.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 11:05:00 | 954.70 | 2024-06-12 11:35:00 | 957.97 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-12 11:05:00 | 954.70 | 2024-06-12 15:20:00 | 964.35 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-06-14 10:30:00 | 997.85 | 2024-06-14 10:40:00 | 995.08 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-19 10:10:00 | 986.35 | 2024-06-19 10:15:00 | 981.90 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-19 10:10:00 | 986.35 | 2024-06-19 11:15:00 | 986.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 10:20:00 | 979.40 | 2024-06-25 10:55:00 | 975.44 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-25 10:20:00 | 979.40 | 2024-06-25 14:50:00 | 979.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 09:55:00 | 972.70 | 2024-06-26 10:00:00 | 975.81 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-28 11:10:00 | 975.00 | 2024-06-28 12:35:00 | 972.57 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-09 10:35:00 | 1004.20 | 2024-07-09 11:25:00 | 1001.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:20:00 | 1002.00 | 2024-07-10 10:35:00 | 996.78 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-10 10:20:00 | 1002.00 | 2024-07-10 10:55:00 | 1002.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:50:00 | 1024.55 | 2024-07-12 11:30:00 | 1020.92 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-12 10:50:00 | 1024.55 | 2024-07-12 13:05:00 | 1021.55 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-07-16 10:30:00 | 1028.85 | 2024-07-16 10:50:00 | 1032.43 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-18 09:45:00 | 1029.20 | 2024-07-18 09:50:00 | 1032.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-19 10:50:00 | 1014.55 | 2024-07-19 10:55:00 | 1009.80 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-19 10:50:00 | 1014.55 | 2024-07-19 11:05:00 | 1014.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 11:05:00 | 1091.00 | 2024-07-25 12:20:00 | 1096.74 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-25 11:05:00 | 1091.00 | 2024-07-25 15:20:00 | 1098.85 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-07-26 09:35:00 | 1109.50 | 2024-07-26 09:50:00 | 1116.23 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-26 09:35:00 | 1109.50 | 2024-07-26 14:15:00 | 1115.35 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-31 10:50:00 | 1097.65 | 2024-07-31 10:55:00 | 1095.13 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-07 10:20:00 | 1072.15 | 2024-08-07 10:40:00 | 1078.37 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-08-07 10:20:00 | 1072.15 | 2024-08-07 10:50:00 | 1072.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:55:00 | 1089.80 | 2024-08-08 11:20:00 | 1095.31 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-08 10:55:00 | 1089.80 | 2024-08-08 11:25:00 | 1089.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:15:00 | 1019.90 | 2024-08-20 10:45:00 | 1016.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-21 10:05:00 | 1031.00 | 2024-08-21 11:50:00 | 1037.11 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-21 10:05:00 | 1031.00 | 2024-08-21 15:20:00 | 1047.60 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2024-08-22 09:55:00 | 1053.00 | 2024-08-22 10:10:00 | 1049.72 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-29 10:20:00 | 1049.65 | 2024-08-29 10:45:00 | 1043.68 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-29 10:20:00 | 1049.65 | 2024-08-29 15:20:00 | 1037.25 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2024-08-30 10:20:00 | 1066.75 | 2024-08-30 10:35:00 | 1061.28 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-09-03 11:10:00 | 1103.15 | 2024-09-03 11:30:00 | 1105.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-10 10:55:00 | 1135.95 | 2024-09-10 11:05:00 | 1132.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-11 09:35:00 | 1151.80 | 2024-09-11 09:40:00 | 1148.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-13 11:15:00 | 1145.00 | 2024-09-13 11:45:00 | 1147.43 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-16 11:15:00 | 1126.20 | 2024-09-16 11:25:00 | 1129.12 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-17 10:25:00 | 1141.40 | 2024-09-17 11:20:00 | 1145.84 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-17 10:25:00 | 1141.40 | 2024-09-17 11:45:00 | 1141.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 09:35:00 | 1171.50 | 2024-09-20 09:45:00 | 1177.08 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-20 09:35:00 | 1171.50 | 2024-09-20 09:50:00 | 1171.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 10:25:00 | 1185.10 | 2024-09-24 11:00:00 | 1180.41 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-24 10:25:00 | 1185.10 | 2024-09-24 13:30:00 | 1185.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 11:10:00 | 1187.65 | 2024-09-26 11:30:00 | 1191.84 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-26 11:10:00 | 1187.65 | 2024-09-26 12:55:00 | 1187.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:10:00 | 1202.05 | 2024-09-27 10:40:00 | 1208.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-27 10:10:00 | 1202.05 | 2024-09-27 10:55:00 | 1202.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-30 10:40:00 | 1196.90 | 2024-09-30 11:35:00 | 1193.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-04 10:35:00 | 1187.70 | 2024-10-04 10:40:00 | 1183.77 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-08 10:00:00 | 1146.60 | 2024-10-08 10:10:00 | 1140.24 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-08 10:00:00 | 1146.60 | 2024-10-08 10:35:00 | 1146.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 11:15:00 | 1174.70 | 2024-10-14 11:30:00 | 1177.96 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-17 10:00:00 | 1179.20 | 2024-10-17 10:30:00 | 1184.05 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-22 11:15:00 | 1191.00 | 2024-10-22 12:30:00 | 1183.41 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-10-22 11:15:00 | 1191.00 | 2024-10-22 15:20:00 | 1172.30 | TARGET_HIT | 0.50 | 1.57% |
| BUY | retest1 | 2024-10-28 10:05:00 | 1274.15 | 2024-10-28 10:10:00 | 1268.62 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-30 10:20:00 | 1262.85 | 2024-10-30 13:00:00 | 1256.17 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-30 10:20:00 | 1262.85 | 2024-10-30 15:20:00 | 1252.25 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2024-11-06 10:10:00 | 1233.10 | 2024-11-06 10:30:00 | 1226.08 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-11-06 10:10:00 | 1233.10 | 2024-11-06 11:35:00 | 1233.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 10:15:00 | 1226.95 | 2024-11-12 10:25:00 | 1223.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-13 11:10:00 | 1208.80 | 2024-11-13 11:40:00 | 1203.77 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-13 11:10:00 | 1208.80 | 2024-11-13 13:35:00 | 1208.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-22 09:55:00 | 1161.05 | 2024-11-22 10:05:00 | 1164.13 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-06 09:30:00 | 1160.10 | 2024-12-06 09:35:00 | 1164.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-09 10:40:00 | 1170.00 | 2024-12-09 13:20:00 | 1175.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-09 10:40:00 | 1170.00 | 2024-12-09 15:20:00 | 1185.45 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2024-12-11 10:40:00 | 1153.45 | 2024-12-11 11:20:00 | 1148.71 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-11 10:40:00 | 1153.45 | 2024-12-11 12:25:00 | 1153.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:55:00 | 1129.95 | 2024-12-12 11:20:00 | 1133.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-13 11:00:00 | 1113.15 | 2024-12-13 12:00:00 | 1116.84 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-20 10:15:00 | 1120.90 | 2024-12-20 10:30:00 | 1124.74 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-26 09:45:00 | 1108.30 | 2024-12-26 12:00:00 | 1104.97 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-02 10:05:00 | 1109.60 | 2025-01-02 10:55:00 | 1106.79 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-07 11:05:00 | 1100.05 | 2025-01-07 11:15:00 | 1096.17 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-10 09:45:00 | 1080.25 | 2025-01-10 11:00:00 | 1083.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-14 11:10:00 | 1075.00 | 2025-01-14 11:20:00 | 1071.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1041.50 | 2025-01-15 10:05:00 | 1035.53 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1041.50 | 2025-01-15 14:20:00 | 1030.70 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2025-01-21 10:15:00 | 1065.55 | 2025-01-21 10:25:00 | 1068.39 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-28 09:30:00 | 1039.80 | 2025-01-28 10:35:00 | 1043.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-01 10:05:00 | 1153.40 | 2025-02-01 10:10:00 | 1159.54 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-02-01 10:05:00 | 1153.40 | 2025-02-01 10:15:00 | 1153.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 11:10:00 | 1111.90 | 2025-02-07 11:20:00 | 1116.48 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-02-07 11:10:00 | 1111.90 | 2025-02-07 12:15:00 | 1111.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 11:10:00 | 1100.00 | 2025-02-10 11:30:00 | 1095.23 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-02-10 11:10:00 | 1100.00 | 2025-02-10 15:20:00 | 1090.55 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-02-11 09:40:00 | 1072.30 | 2025-02-11 10:00:00 | 1066.23 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-02-11 09:40:00 | 1072.30 | 2025-02-11 13:40:00 | 1064.90 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-02-14 11:15:00 | 1065.90 | 2025-02-14 12:35:00 | 1069.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-19 09:35:00 | 1066.15 | 2025-02-19 10:00:00 | 1071.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-19 09:35:00 | 1066.15 | 2025-02-19 10:10:00 | 1066.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-28 09:35:00 | 1007.40 | 2025-02-28 10:15:00 | 1011.59 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-03 09:55:00 | 989.10 | 2025-03-03 10:05:00 | 983.04 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-03-03 09:55:00 | 989.10 | 2025-03-03 10:25:00 | 989.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 09:55:00 | 1040.40 | 2025-03-10 10:00:00 | 1037.06 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-12 10:50:00 | 1069.50 | 2025-03-12 11:25:00 | 1064.48 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-03-12 10:50:00 | 1069.50 | 2025-03-12 15:20:00 | 1061.05 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-03-19 10:40:00 | 1095.80 | 2025-03-19 10:55:00 | 1092.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-20 09:55:00 | 1106.25 | 2025-03-20 10:15:00 | 1110.80 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-20 09:55:00 | 1106.25 | 2025-03-20 15:00:00 | 1120.40 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2025-04-02 10:05:00 | 1132.55 | 2025-04-02 10:35:00 | 1136.58 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-04 10:45:00 | 1154.50 | 2025-04-04 11:05:00 | 1161.33 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-04 10:45:00 | 1154.50 | 2025-04-04 13:20:00 | 1156.95 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-04-11 09:50:00 | 1187.40 | 2025-04-11 09:55:00 | 1183.26 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-16 10:45:00 | 1205.50 | 2025-04-16 11:35:00 | 1201.98 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-17 10:40:00 | 1199.90 | 2025-04-17 10:45:00 | 1202.94 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-05 10:45:00 | 1297.10 | 2025-05-05 11:00:00 | 1300.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-06 11:15:00 | 1267.10 | 2025-05-06 11:20:00 | 1270.73 | STOP_HIT | 1.00 | -0.29% |
