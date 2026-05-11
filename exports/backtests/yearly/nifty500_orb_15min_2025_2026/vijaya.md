# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1275.00
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 17 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 48
- **Target hits / Stop hits / Partials:** 17 / 48 / 33
- **Avg / median % per leg:** 0.18% / 0.16%
- **Sum % (uncompounded):** 17.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 21 | 45.7% | 7 | 25 | 14 | 0.11% | 5.0% |
| BUY @ 2nd Alert (retest1) | 46 | 21 | 45.7% | 7 | 25 | 14 | 0.11% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 29 | 55.8% | 10 | 23 | 19 | 0.25% | 12.8% |
| SELL @ 2nd Alert (retest1) | 52 | 29 | 55.8% | 10 | 23 | 19 | 0.25% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 50 | 51.0% | 17 | 48 | 33 | 0.18% | 17.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 10:10:00 | 924.60 | 927.20 | 0.00 | ORB-short ORB[925.00,936.10] vol=2.0x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:15:00 | 921.49 | 927.11 | 0.00 | T1 1.5R @ 921.49 |
| Target hit | 2025-05-23 15:20:00 | 909.50 | 918.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:40:00 | 932.90 | 923.04 | 0.00 | ORB-long ORB[911.50,923.90] vol=2.8x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:45:00 | 938.41 | 929.63 | 0.00 | T1 1.5R @ 938.41 |
| Stop hit — per-position SL triggered | 2025-05-26 10:00:00 | 932.90 | 934.26 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:30:00 | 940.00 | 932.75 | 0.00 | ORB-long ORB[923.50,936.60] vol=1.7x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:55:00 | 944.33 | 935.92 | 0.00 | T1 1.5R @ 944.33 |
| Target hit | 2025-05-27 15:20:00 | 955.50 | 952.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 943.20 | 947.13 | 0.00 | ORB-short ORB[944.00,956.90] vol=1.6x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:45:00 | 939.66 | 945.97 | 0.00 | T1 1.5R @ 939.66 |
| Stop hit — per-position SL triggered | 2025-05-29 10:05:00 | 943.20 | 945.20 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:50:00 | 976.50 | 972.86 | 0.00 | ORB-long ORB[964.20,973.80] vol=6.8x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-06-05 09:55:00 | 973.86 | 972.97 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:50:00 | 960.10 | 964.72 | 0.00 | ORB-short ORB[962.20,971.75] vol=3.9x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 12:10:00 | 956.46 | 963.73 | 0.00 | T1 1.5R @ 956.46 |
| Target hit | 2025-06-10 15:20:00 | 951.65 | 956.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:05:00 | 967.15 | 957.49 | 0.00 | ORB-long ORB[949.60,960.00] vol=10.1x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-06-12 11:10:00 | 964.32 | 958.63 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 943.50 | 949.80 | 0.00 | ORB-short ORB[947.40,958.05] vol=2.3x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-06-16 10:05:00 | 946.11 | 946.87 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:55:00 | 946.00 | 947.87 | 0.00 | ORB-short ORB[947.10,955.65] vol=1.7x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 943.15 | 947.63 | 0.00 | T1 1.5R @ 943.15 |
| Stop hit — per-position SL triggered | 2025-06-19 12:20:00 | 946.00 | 944.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 09:55:00 | 931.00 | 932.40 | 0.00 | ORB-short ORB[932.20,942.90] vol=5.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-06-24 11:05:00 | 933.64 | 932.11 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:55:00 | 976.45 | 971.16 | 0.00 | ORB-long ORB[960.30,973.00] vol=2.3x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-06-26 10:00:00 | 973.47 | 971.31 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:40:00 | 1010.35 | 1000.94 | 0.00 | ORB-long ORB[993.15,1005.95] vol=5.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-07-04 10:50:00 | 1006.96 | 1001.94 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:40:00 | 1014.10 | 1009.87 | 0.00 | ORB-long ORB[1001.50,1011.60] vol=1.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-07-09 09:45:00 | 1011.35 | 1010.16 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 1032.65 | 1024.70 | 0.00 | ORB-long ORB[1017.95,1026.90] vol=1.6x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-07-16 09:50:00 | 1029.02 | 1026.51 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:35:00 | 1041.60 | 1034.59 | 0.00 | ORB-long ORB[1026.35,1037.00] vol=5.5x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:45:00 | 1046.60 | 1036.65 | 0.00 | T1 1.5R @ 1046.60 |
| Stop hit — per-position SL triggered | 2025-07-17 10:50:00 | 1041.60 | 1036.83 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:50:00 | 1025.25 | 1018.05 | 0.00 | ORB-long ORB[1011.30,1021.00] vol=1.8x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-07-22 09:55:00 | 1022.06 | 1018.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 1059.90 | 1065.87 | 0.00 | ORB-short ORB[1061.40,1073.10] vol=1.6x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:35:00 | 1053.37 | 1062.34 | 0.00 | T1 1.5R @ 1053.37 |
| Stop hit — per-position SL triggered | 2025-08-06 10:00:00 | 1059.90 | 1060.50 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:45:00 | 1043.60 | 1048.08 | 0.00 | ORB-short ORB[1049.00,1058.00] vol=1.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 1039.80 | 1046.12 | 0.00 | T1 1.5R @ 1039.80 |
| Stop hit — per-position SL triggered | 2025-08-11 12:35:00 | 1043.60 | 1045.05 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:15:00 | 1064.30 | 1057.53 | 0.00 | ORB-long ORB[1050.50,1060.20] vol=2.0x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:25:00 | 1068.08 | 1059.22 | 0.00 | T1 1.5R @ 1068.08 |
| Stop hit — per-position SL triggered | 2025-08-20 11:10:00 | 1064.30 | 1061.31 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:55:00 | 997.00 | 993.35 | 0.00 | ORB-long ORB[985.00,996.10] vol=4.9x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:05:00 | 1001.22 | 993.64 | 0.00 | T1 1.5R @ 1001.22 |
| Stop hit — per-position SL triggered | 2025-08-29 13:00:00 | 997.00 | 996.15 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1036.70 | 1027.46 | 0.00 | ORB-long ORB[1008.20,1020.00] vol=3.1x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-09-01 11:10:00 | 1032.76 | 1028.82 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:45:00 | 1102.10 | 1094.94 | 0.00 | ORB-long ORB[1084.70,1094.20] vol=1.5x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:05:00 | 1107.73 | 1096.73 | 0.00 | T1 1.5R @ 1107.73 |
| Target hit | 2025-09-05 13:25:00 | 1105.10 | 1105.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — SELL (started 2025-09-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 11:05:00 | 1106.80 | 1113.42 | 0.00 | ORB-short ORB[1106.90,1120.40] vol=1.5x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 11:15:00 | 1102.56 | 1112.92 | 0.00 | T1 1.5R @ 1102.56 |
| Stop hit — per-position SL triggered | 2025-09-08 11:25:00 | 1106.80 | 1111.75 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:05:00 | 1086.10 | 1097.33 | 0.00 | ORB-short ORB[1095.00,1108.70] vol=2.2x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-09-09 11:10:00 | 1088.92 | 1096.96 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1045.50 | 1051.65 | 0.00 | ORB-short ORB[1049.00,1062.60] vol=2.6x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-09-16 11:40:00 | 1047.51 | 1049.68 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:40:00 | 1042.90 | 1048.81 | 0.00 | ORB-short ORB[1046.00,1061.40] vol=1.9x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-09-17 11:00:00 | 1045.36 | 1047.95 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1042.50 | 1045.08 | 0.00 | ORB-short ORB[1043.70,1052.50] vol=3.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:10:00 | 1039.41 | 1043.68 | 0.00 | T1 1.5R @ 1039.41 |
| Stop hit — per-position SL triggered | 2025-09-18 11:50:00 | 1042.50 | 1042.63 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:25:00 | 1040.00 | 1040.91 | 0.00 | ORB-short ORB[1042.40,1050.70] vol=2.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-09-23 11:35:00 | 1042.38 | 1040.29 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 1063.50 | 1056.20 | 0.00 | ORB-long ORB[1045.40,1054.20] vol=1.9x ATR=4.82 |
| Stop hit — per-position SL triggered | 2025-09-24 09:40:00 | 1058.68 | 1057.03 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 09:50:00 | 1008.00 | 1011.65 | 0.00 | ORB-short ORB[1013.10,1024.90] vol=2.2x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:05:00 | 1001.87 | 1007.38 | 0.00 | T1 1.5R @ 1001.87 |
| Target hit | 2025-09-30 15:20:00 | 999.50 | 1001.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 999.55 | 1004.64 | 0.00 | ORB-short ORB[1000.35,1010.00] vol=2.5x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:25:00 | 995.71 | 1003.11 | 0.00 | T1 1.5R @ 995.71 |
| Target hit | 2025-10-08 15:20:00 | 992.50 | 997.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:45:00 | 984.55 | 987.28 | 0.00 | ORB-short ORB[984.90,994.55] vol=3.9x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-10-16 12:40:00 | 986.59 | 986.55 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:50:00 | 984.00 | 986.20 | 0.00 | ORB-short ORB[987.10,992.60] vol=6.4x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-10-24 10:55:00 | 985.75 | 986.15 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1002.30 | 997.99 | 0.00 | ORB-long ORB[989.85,999.00] vol=4.5x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-10-27 10:10:00 | 998.96 | 1000.09 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:30:00 | 1004.10 | 1000.14 | 0.00 | ORB-long ORB[996.05,1001.60] vol=2.8x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-10-28 10:40:00 | 1001.59 | 1000.19 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 11:15:00 | 1006.00 | 1002.52 | 0.00 | ORB-long ORB[999.90,1004.85] vol=10.8x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:25:00 | 1008.36 | 1005.47 | 0.00 | T1 1.5R @ 1008.36 |
| Target hit | 2025-10-29 12:40:00 | 1009.15 | 1010.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 983.60 | 987.12 | 0.00 | ORB-short ORB[984.05,996.00] vol=2.3x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 986.26 | 985.14 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:30:00 | 1044.15 | 1033.95 | 0.00 | ORB-long ORB[1019.70,1030.45] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-11-04 10:35:00 | 1040.49 | 1034.43 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1020.60 | 1015.08 | 0.00 | ORB-long ORB[1014.05,1018.30] vol=4.3x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 13:40:00 | 1025.09 | 1019.36 | 0.00 | T1 1.5R @ 1025.09 |
| Stop hit — per-position SL triggered | 2025-11-12 14:10:00 | 1020.60 | 1019.44 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:50:00 | 1016.00 | 1018.90 | 0.00 | ORB-short ORB[1019.00,1023.05] vol=1.6x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:00:00 | 1012.83 | 1017.78 | 0.00 | T1 1.5R @ 1012.83 |
| Stop hit — per-position SL triggered | 2025-11-13 10:30:00 | 1016.00 | 1018.58 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 1051.15 | 1047.52 | 0.00 | ORB-long ORB[1038.00,1050.85] vol=1.6x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:55:00 | 1057.61 | 1052.01 | 0.00 | T1 1.5R @ 1057.61 |
| Target hit | 2025-11-14 11:40:00 | 1053.90 | 1054.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2025-11-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:55:00 | 1048.00 | 1052.59 | 0.00 | ORB-short ORB[1052.30,1064.00] vol=2.4x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 12:05:00 | 1042.85 | 1051.65 | 0.00 | T1 1.5R @ 1042.85 |
| Target hit | 2025-11-17 15:20:00 | 1039.70 | 1042.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-11-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:30:00 | 1008.80 | 1011.92 | 0.00 | ORB-short ORB[1011.00,1019.85] vol=2.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-11-21 11:00:00 | 1011.21 | 1011.71 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:20:00 | 990.00 | 996.71 | 0.00 | ORB-short ORB[997.75,1007.00] vol=3.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 993.20 | 993.21 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:15:00 | 1011.35 | 1009.47 | 0.00 | ORB-long ORB[1002.20,1011.00] vol=2.7x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-12-02 12:45:00 | 1008.38 | 1010.16 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1000.05 | 1004.20 | 0.00 | ORB-short ORB[1003.95,1011.40] vol=1.5x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:30:00 | 995.69 | 1001.00 | 0.00 | T1 1.5R @ 995.69 |
| Stop hit — per-position SL triggered | 2025-12-03 12:45:00 | 1000.05 | 1000.97 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:25:00 | 1018.35 | 1010.53 | 0.00 | ORB-long ORB[1003.05,1010.00] vol=1.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-12-04 10:30:00 | 1015.75 | 1013.37 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 10:40:00 | 1005.55 | 1005.87 | 0.00 | ORB-short ORB[1006.50,1021.10] vol=2.8x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:25:00 | 1001.97 | 1004.93 | 0.00 | T1 1.5R @ 1001.97 |
| Target hit | 2025-12-11 15:20:00 | 999.90 | 1002.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-12-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:05:00 | 978.60 | 981.90 | 0.00 | ORB-short ORB[982.45,993.95] vol=10.1x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 11:35:00 | 973.29 | 980.84 | 0.00 | T1 1.5R @ 973.29 |
| Stop hit — per-position SL triggered | 2025-12-15 13:00:00 | 978.60 | 979.07 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:50:00 | 1016.60 | 1009.76 | 0.00 | ORB-long ORB[999.10,1014.00] vol=2.1x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-12-23 10:45:00 | 1012.76 | 1014.12 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 1025.85 | 1019.47 | 0.00 | ORB-long ORB[1012.20,1022.45] vol=3.2x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:05:00 | 1030.05 | 1022.88 | 0.00 | T1 1.5R @ 1030.05 |
| Target hit | 2025-12-24 15:20:00 | 1030.00 | 1029.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 1020.45 | 1026.41 | 0.00 | ORB-short ORB[1021.00,1033.70] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-12-30 11:00:00 | 1022.95 | 1027.97 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 1055.65 | 1049.73 | 0.00 | ORB-long ORB[1041.60,1055.00] vol=5.0x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:15:00 | 1059.45 | 1051.79 | 0.00 | T1 1.5R @ 1059.45 |
| Target hit | 2025-12-31 13:20:00 | 1056.20 | 1056.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 1052.80 | 1056.40 | 0.00 | ORB-short ORB[1060.00,1069.50] vol=2.4x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 1054.54 | 1056.21 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1047.90 | 1040.35 | 0.00 | ORB-long ORB[1034.40,1042.90] vol=3.8x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:30:00 | 1052.34 | 1043.63 | 0.00 | T1 1.5R @ 1052.34 |
| Target hit | 2026-01-02 15:20:00 | 1058.60 | 1053.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2026-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:05:00 | 1039.00 | 1043.17 | 0.00 | ORB-short ORB[1042.30,1046.40] vol=1.8x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:15:00 | 1036.52 | 1043.09 | 0.00 | T1 1.5R @ 1036.52 |
| Target hit | 2026-01-07 13:10:00 | 1028.70 | 1028.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — SELL (started 2026-01-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:45:00 | 958.70 | 964.46 | 0.00 | ORB-short ORB[960.20,968.50] vol=5.7x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-01-28 10:50:00 | 961.83 | 963.97 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:15:00 | 938.10 | 946.51 | 0.00 | ORB-short ORB[946.60,953.00] vol=1.6x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:45:00 | 934.10 | 943.29 | 0.00 | T1 1.5R @ 934.10 |
| Target hit | 2026-01-29 12:35:00 | 930.00 | 929.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — BUY (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 1025.55 | 1022.99 | 0.00 | ORB-long ORB[1015.55,1025.30] vol=2.4x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 1022.42 | 1022.97 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1006.35 | 1010.01 | 0.00 | ORB-short ORB[1013.20,1022.20] vol=4.0x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:00:00 | 1002.62 | 1008.96 | 0.00 | T1 1.5R @ 1002.62 |
| Target hit | 2026-02-18 14:20:00 | 1004.70 | 1002.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1002.05 | 1004.30 | 0.00 | ORB-short ORB[1002.45,1012.10] vol=1.5x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:20:00 | 996.96 | 1001.87 | 0.00 | T1 1.5R @ 996.96 |
| Target hit | 2026-02-24 14:50:00 | 991.55 | 990.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 955.90 | 951.05 | 0.00 | ORB-long ORB[943.10,954.80] vol=3.3x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-11 10:20:00 | 953.20 | 951.69 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-04-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:10:00 | 1010.00 | 993.83 | 0.00 | ORB-long ORB[982.20,994.25] vol=2.3x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 1006.26 | 998.28 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-04-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:20:00 | 1036.70 | 1025.91 | 0.00 | ORB-long ORB[1014.30,1029.00] vol=3.2x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:30:00 | 1043.88 | 1028.83 | 0.00 | T1 1.5R @ 1043.88 |
| Stop hit — per-position SL triggered | 2026-04-17 11:40:00 | 1036.70 | 1036.57 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 1151.90 | 1144.64 | 0.00 | ORB-long ORB[1136.20,1150.80] vol=1.7x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:00:00 | 1160.86 | 1146.50 | 0.00 | T1 1.5R @ 1160.86 |
| Stop hit — per-position SL triggered | 2026-05-04 12:30:00 | 1151.90 | 1153.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-23 10:10:00 | 924.60 | 2025-05-23 10:15:00 | 921.49 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-23 10:10:00 | 924.60 | 2025-05-23 15:20:00 | 909.50 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2025-05-26 09:40:00 | 932.90 | 2025-05-26 09:45:00 | 938.41 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-05-26 09:40:00 | 932.90 | 2025-05-26 10:00:00 | 932.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-27 10:30:00 | 940.00 | 2025-05-27 10:55:00 | 944.33 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-05-27 10:30:00 | 940.00 | 2025-05-27 15:20:00 | 955.50 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2025-05-29 09:35:00 | 943.20 | 2025-05-29 09:45:00 | 939.66 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-29 09:35:00 | 943.20 | 2025-05-29 10:05:00 | 943.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:50:00 | 976.50 | 2025-06-05 09:55:00 | 973.86 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-10 10:50:00 | 960.10 | 2025-06-10 12:10:00 | 956.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-10 10:50:00 | 960.10 | 2025-06-10 15:20:00 | 951.65 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2025-06-12 11:05:00 | 967.15 | 2025-06-12 11:10:00 | 964.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-16 09:30:00 | 943.50 | 2025-06-16 10:05:00 | 946.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-19 10:55:00 | 946.00 | 2025-06-19 11:15:00 | 943.15 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-06-19 10:55:00 | 946.00 | 2025-06-19 12:20:00 | 946.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-24 09:55:00 | 931.00 | 2025-06-24 11:05:00 | 933.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-26 09:55:00 | 976.45 | 2025-06-26 10:00:00 | 973.47 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-04 10:40:00 | 1010.35 | 2025-07-04 10:50:00 | 1006.96 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-09 09:40:00 | 1014.10 | 2025-07-09 09:45:00 | 1011.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-16 09:40:00 | 1032.65 | 2025-07-16 09:50:00 | 1029.02 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-17 10:35:00 | 1041.60 | 2025-07-17 10:45:00 | 1046.60 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-17 10:35:00 | 1041.60 | 2025-07-17 10:50:00 | 1041.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 09:50:00 | 1025.25 | 2025-07-22 09:55:00 | 1022.06 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-06 09:30:00 | 1059.90 | 2025-08-06 09:35:00 | 1053.37 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-08-06 09:30:00 | 1059.90 | 2025-08-06 10:00:00 | 1059.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-11 10:45:00 | 1043.60 | 2025-08-11 11:40:00 | 1039.80 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-11 10:45:00 | 1043.60 | 2025-08-11 12:35:00 | 1043.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:15:00 | 1064.30 | 2025-08-20 10:25:00 | 1068.08 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-08-20 10:15:00 | 1064.30 | 2025-08-20 11:10:00 | 1064.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-29 10:55:00 | 997.00 | 2025-08-29 11:05:00 | 1001.22 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-29 10:55:00 | 997.00 | 2025-08-29 13:00:00 | 997.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 10:45:00 | 1036.70 | 2025-09-01 11:10:00 | 1032.76 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-05 09:45:00 | 1102.10 | 2025-09-05 10:05:00 | 1107.73 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-05 09:45:00 | 1102.10 | 2025-09-05 13:25:00 | 1105.10 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-08 11:05:00 | 1106.80 | 2025-09-08 11:15:00 | 1102.56 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-08 11:05:00 | 1106.80 | 2025-09-08 11:25:00 | 1106.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 11:05:00 | 1086.10 | 2025-09-09 11:10:00 | 1088.92 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-16 11:15:00 | 1045.50 | 2025-09-16 11:40:00 | 1047.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-17 10:40:00 | 1042.90 | 2025-09-17 11:00:00 | 1045.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-18 11:00:00 | 1042.50 | 2025-09-18 11:10:00 | 1039.41 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-09-18 11:00:00 | 1042.50 | 2025-09-18 11:50:00 | 1042.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 10:25:00 | 1040.00 | 2025-09-23 11:35:00 | 1042.38 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-24 09:35:00 | 1063.50 | 2025-09-24 09:40:00 | 1058.68 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-09-30 09:50:00 | 1008.00 | 2025-09-30 10:05:00 | 1001.87 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-09-30 09:50:00 | 1008.00 | 2025-09-30 15:20:00 | 999.50 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-10-08 10:15:00 | 999.55 | 2025-10-08 10:25:00 | 995.71 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-08 10:15:00 | 999.55 | 2025-10-08 15:20:00 | 992.50 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-10-16 10:45:00 | 984.55 | 2025-10-16 12:40:00 | 986.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-24 10:50:00 | 984.00 | 2025-10-24 10:55:00 | 985.75 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-27 09:30:00 | 1002.30 | 2025-10-27 10:10:00 | 998.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-28 10:30:00 | 1004.10 | 2025-10-28 10:40:00 | 1001.59 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-29 11:15:00 | 1006.00 | 2025-10-29 11:25:00 | 1008.36 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-10-29 11:15:00 | 1006.00 | 2025-10-29 12:40:00 | 1009.15 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-31 09:50:00 | 983.60 | 2025-10-31 10:45:00 | 986.26 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-04 10:30:00 | 1044.15 | 2025-11-04 10:35:00 | 1040.49 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-12 10:00:00 | 1020.60 | 2025-11-12 13:40:00 | 1025.09 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-12 10:00:00 | 1020.60 | 2025-11-12 14:10:00 | 1020.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 09:50:00 | 1016.00 | 2025-11-13 10:00:00 | 1012.83 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-13 09:50:00 | 1016.00 | 2025-11-13 10:30:00 | 1016.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 09:30:00 | 1051.15 | 2025-11-14 09:55:00 | 1057.61 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-11-14 09:30:00 | 1051.15 | 2025-11-14 11:40:00 | 1053.90 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-17 10:55:00 | 1048.00 | 2025-11-17 12:05:00 | 1042.85 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-17 10:55:00 | 1048.00 | 2025-11-17 15:20:00 | 1039.70 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-11-21 10:30:00 | 1008.80 | 2025-11-21 11:00:00 | 1011.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-28 10:20:00 | 990.00 | 2025-11-28 11:15:00 | 993.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-02 10:15:00 | 1011.35 | 2025-12-02 12:45:00 | 1008.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1000.05 | 2025-12-03 12:30:00 | 995.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1000.05 | 2025-12-03 12:45:00 | 1000.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 10:25:00 | 1018.35 | 2025-12-04 10:30:00 | 1015.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-11 10:40:00 | 1005.55 | 2025-12-11 12:25:00 | 1001.97 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-11 10:40:00 | 1005.55 | 2025-12-11 15:20:00 | 999.90 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-12-15 10:05:00 | 978.60 | 2025-12-15 11:35:00 | 973.29 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-15 10:05:00 | 978.60 | 2025-12-15 13:00:00 | 978.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 09:50:00 | 1016.60 | 2025-12-23 10:45:00 | 1012.76 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-24 09:50:00 | 1025.85 | 2025-12-24 11:05:00 | 1030.05 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-12-24 09:50:00 | 1025.85 | 2025-12-24 15:20:00 | 1030.00 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-30 10:55:00 | 1020.45 | 2025-12-30 11:00:00 | 1022.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1055.65 | 2025-12-31 11:15:00 | 1059.45 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1055.65 | 2025-12-31 13:20:00 | 1056.20 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2026-01-01 11:05:00 | 1052.80 | 2026-01-01 11:15:00 | 1054.54 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-02 10:45:00 | 1047.90 | 2026-01-02 11:30:00 | 1052.34 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-02 10:45:00 | 1047.90 | 2026-01-02 15:20:00 | 1058.60 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2026-01-07 11:05:00 | 1039.00 | 2026-01-07 11:15:00 | 1036.52 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-01-07 11:05:00 | 1039.00 | 2026-01-07 13:10:00 | 1028.70 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-01-28 10:45:00 | 958.70 | 2026-01-28 10:50:00 | 961.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-29 10:15:00 | 938.10 | 2026-01-29 10:45:00 | 934.10 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-29 10:15:00 | 938.10 | 2026-01-29 12:35:00 | 930.00 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-02-11 10:50:00 | 1025.55 | 2026-02-11 11:05:00 | 1022.42 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1006.35 | 2026-02-18 11:00:00 | 1002.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1006.35 | 2026-02-18 14:20:00 | 1004.70 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1002.05 | 2026-02-24 10:20:00 | 996.96 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1002.05 | 2026-02-24 14:50:00 | 991.55 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2026-03-11 09:45:00 | 955.90 | 2026-03-11 10:20:00 | 953.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 10:10:00 | 1010.00 | 2026-04-15 10:25:00 | 1006.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-17 10:20:00 | 1036.70 | 2026-04-17 10:30:00 | 1043.88 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-17 10:20:00 | 1036.70 | 2026-04-17 11:40:00 | 1036.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1151.90 | 2026-05-04 11:00:00 | 1160.86 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-05-04 10:55:00 | 1151.90 | 2026-05-04 12:30:00 | 1151.90 | STOP_HIT | 0.50 | 0.00% |
