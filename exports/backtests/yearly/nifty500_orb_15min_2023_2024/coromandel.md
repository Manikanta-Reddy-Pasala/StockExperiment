# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 1928.90
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
| ENTRY1 | 108 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 23 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 85
- **Target hits / Stop hits / Partials:** 23 / 85 / 42
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 17.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 23 | 33.3% | 6 | 46 | 17 | 0.03% | 1.8% |
| BUY @ 2nd Alert (retest1) | 69 | 23 | 33.3% | 6 | 46 | 17 | 0.03% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 81 | 42 | 51.9% | 17 | 39 | 25 | 0.20% | 15.9% |
| SELL @ 2nd Alert (retest1) | 81 | 42 | 51.9% | 17 | 39 | 25 | 0.20% | 15.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 150 | 65 | 43.3% | 23 | 85 | 42 | 0.12% | 17.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 09:40:00 | 926.50 | 928.28 | 0.00 | ORB-short ORB[929.25,939.50] vol=1.9x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 10:10:00 | 921.38 | 926.06 | 0.00 | T1 1.5R @ 921.38 |
| Target hit | 2023-05-23 12:10:00 | 921.60 | 921.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2023-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 11:05:00 | 947.15 | 938.12 | 0.00 | ORB-long ORB[932.70,943.35] vol=4.3x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 11:10:00 | 951.33 | 940.09 | 0.00 | T1 1.5R @ 951.33 |
| Stop hit — per-position SL triggered | 2023-05-29 11:20:00 | 947.15 | 943.95 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-06-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 10:35:00 | 948.00 | 952.95 | 0.00 | ORB-short ORB[951.05,961.10] vol=4.5x ATR=2.82 |
| Stop hit — per-position SL triggered | 2023-06-05 10:40:00 | 950.82 | 952.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 11:00:00 | 948.05 | 958.23 | 0.00 | ORB-short ORB[956.00,964.70] vol=1.7x ATR=2.96 |
| Stop hit — per-position SL triggered | 2023-06-06 11:15:00 | 951.01 | 957.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:40:00 | 966.00 | 961.35 | 0.00 | ORB-long ORB[955.70,961.60] vol=1.8x ATR=3.12 |
| Stop hit — per-position SL triggered | 2023-06-07 09:50:00 | 962.88 | 962.34 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 10:30:00 | 962.75 | 962.98 | 0.00 | ORB-short ORB[963.75,972.00] vol=3.3x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:50:00 | 958.79 | 962.76 | 0.00 | T1 1.5R @ 958.79 |
| Target hit | 2023-06-08 13:25:00 | 960.30 | 959.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2023-06-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:45:00 | 949.80 | 939.40 | 0.00 | ORB-long ORB[931.35,936.55] vol=2.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2023-06-13 09:50:00 | 946.76 | 940.68 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 09:40:00 | 948.70 | 942.69 | 0.00 | ORB-long ORB[934.65,941.20] vol=3.0x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-06-14 09:45:00 | 945.78 | 943.18 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 10:45:00 | 942.30 | 946.48 | 0.00 | ORB-short ORB[944.00,950.00] vol=1.6x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 11:30:00 | 939.89 | 944.79 | 0.00 | T1 1.5R @ 939.89 |
| Target hit | 2023-06-15 13:30:00 | 941.95 | 941.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2023-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:20:00 | 944.90 | 940.33 | 0.00 | ORB-long ORB[932.00,943.00] vol=3.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-06-20 10:30:00 | 941.79 | 940.53 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:40:00 | 937.00 | 940.60 | 0.00 | ORB-short ORB[940.10,948.00] vol=1.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-06-21 12:20:00 | 938.56 | 939.26 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 960.45 | 955.36 | 0.00 | ORB-long ORB[946.00,959.00] vol=2.2x ATR=3.61 |
| Stop hit — per-position SL triggered | 2023-06-22 10:00:00 | 956.84 | 957.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:35:00 | 945.75 | 940.84 | 0.00 | ORB-long ORB[935.00,943.35] vol=1.7x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:40:00 | 949.72 | 941.87 | 0.00 | T1 1.5R @ 949.72 |
| Stop hit — per-position SL triggered | 2023-06-27 09:50:00 | 945.75 | 942.57 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 11:00:00 | 960.85 | 958.08 | 0.00 | ORB-long ORB[948.80,957.05] vol=3.8x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 12:20:00 | 963.57 | 959.38 | 0.00 | T1 1.5R @ 963.57 |
| Stop hit — per-position SL triggered | 2023-07-05 12:25:00 | 960.85 | 959.80 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:55:00 | 964.40 | 973.52 | 0.00 | ORB-short ORB[969.80,979.90] vol=1.6x ATR=3.03 |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 967.43 | 972.61 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 10:55:00 | 955.25 | 958.56 | 0.00 | ORB-short ORB[956.40,968.00] vol=3.2x ATR=2.23 |
| Stop hit — per-position SL triggered | 2023-07-11 11:05:00 | 957.48 | 958.35 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:20:00 | 954.65 | 957.64 | 0.00 | ORB-short ORB[955.25,963.00] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-07-13 12:20:00 | 956.45 | 955.68 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:40:00 | 934.15 | 928.17 | 0.00 | ORB-long ORB[921.25,930.00] vol=2.4x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 09:45:00 | 938.99 | 928.91 | 0.00 | T1 1.5R @ 938.99 |
| Stop hit — per-position SL triggered | 2023-07-14 09:50:00 | 934.15 | 928.98 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:00:00 | 952.05 | 954.73 | 0.00 | ORB-short ORB[952.30,960.00] vol=5.3x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 11:55:00 | 949.05 | 953.72 | 0.00 | T1 1.5R @ 949.05 |
| Stop hit — per-position SL triggered | 2023-07-19 12:50:00 | 952.05 | 953.33 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 11:05:00 | 964.05 | 959.76 | 0.00 | ORB-long ORB[955.90,963.80] vol=1.7x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 11:35:00 | 966.97 | 961.93 | 0.00 | T1 1.5R @ 966.97 |
| Target hit | 2023-07-21 15:20:00 | 981.00 | 975.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:15:00 | 981.40 | 977.39 | 0.00 | ORB-long ORB[967.00,978.60] vol=2.4x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-07-26 10:20:00 | 979.51 | 977.97 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:35:00 | 1029.35 | 1032.82 | 0.00 | ORB-short ORB[1031.10,1040.00] vol=2.0x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-08-02 10:40:00 | 1032.21 | 1032.69 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 09:30:00 | 1029.40 | 1024.49 | 0.00 | ORB-long ORB[1017.05,1028.15] vol=1.6x ATR=3.55 |
| Stop hit — per-position SL triggered | 2023-08-03 09:40:00 | 1025.85 | 1025.09 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 10:25:00 | 1040.35 | 1036.28 | 0.00 | ORB-long ORB[1030.80,1039.00] vol=2.1x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 10:30:00 | 1044.66 | 1037.40 | 0.00 | T1 1.5R @ 1044.66 |
| Stop hit — per-position SL triggered | 2023-08-04 10:45:00 | 1040.35 | 1037.86 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:20:00 | 1060.60 | 1055.70 | 0.00 | ORB-long ORB[1048.00,1057.90] vol=1.8x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-08-08 11:05:00 | 1057.63 | 1057.93 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:15:00 | 1065.00 | 1067.89 | 0.00 | ORB-short ORB[1067.00,1073.20] vol=3.0x ATR=1.90 |
| Stop hit — per-position SL triggered | 2023-08-22 12:50:00 | 1066.90 | 1067.04 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:35:00 | 1077.05 | 1074.12 | 0.00 | ORB-long ORB[1065.75,1076.25] vol=1.7x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 10:20:00 | 1081.72 | 1075.89 | 0.00 | T1 1.5R @ 1081.72 |
| Target hit | 2023-08-24 13:35:00 | 1080.00 | 1080.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2023-08-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 11:05:00 | 1078.35 | 1072.79 | 0.00 | ORB-long ORB[1060.30,1071.85] vol=1.9x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 11:45:00 | 1081.89 | 1076.59 | 0.00 | T1 1.5R @ 1081.89 |
| Target hit | 2023-08-28 15:20:00 | 1088.05 | 1083.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2023-08-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 10:35:00 | 1083.75 | 1088.15 | 0.00 | ORB-short ORB[1091.30,1104.00] vol=2.5x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 10:45:00 | 1078.81 | 1087.32 | 0.00 | T1 1.5R @ 1078.81 |
| Target hit | 2023-08-31 13:20:00 | 1082.55 | 1082.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2023-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:35:00 | 1105.35 | 1100.46 | 0.00 | ORB-long ORB[1087.00,1096.00] vol=1.5x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-09-01 10:55:00 | 1102.27 | 1102.39 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 1113.05 | 1117.54 | 0.00 | ORB-short ORB[1115.15,1124.50] vol=2.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2023-09-05 09:45:00 | 1116.44 | 1117.46 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:45:00 | 1137.80 | 1142.94 | 0.00 | ORB-short ORB[1138.30,1149.00] vol=2.3x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 11:20:00 | 1132.69 | 1140.55 | 0.00 | T1 1.5R @ 1132.69 |
| Target hit | 2023-09-08 15:20:00 | 1124.30 | 1133.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 1113.65 | 1118.26 | 0.00 | ORB-short ORB[1116.15,1127.80] vol=3.3x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:40:00 | 1110.15 | 1116.64 | 0.00 | T1 1.5R @ 1110.15 |
| Stop hit — per-position SL triggered | 2023-09-12 10:00:00 | 1113.65 | 1112.45 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:15:00 | 1120.40 | 1125.62 | 0.00 | ORB-short ORB[1121.00,1136.85] vol=2.0x ATR=3.72 |
| Stop hit — per-position SL triggered | 2023-09-14 10:25:00 | 1124.12 | 1125.45 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:50:00 | 1073.20 | 1078.48 | 0.00 | ORB-short ORB[1077.00,1086.50] vol=5.3x ATR=3.09 |
| Stop hit — per-position SL triggered | 2023-09-22 09:55:00 | 1076.29 | 1078.12 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:40:00 | 1119.50 | 1115.25 | 0.00 | ORB-long ORB[1107.05,1117.55] vol=3.6x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 10:45:00 | 1124.01 | 1117.01 | 0.00 | T1 1.5R @ 1124.01 |
| Target hit | 2023-09-27 15:20:00 | 1128.50 | 1123.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2023-09-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 10:50:00 | 1135.55 | 1130.74 | 0.00 | ORB-long ORB[1124.95,1131.90] vol=3.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-09-28 11:00:00 | 1133.23 | 1131.61 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 09:30:00 | 1166.90 | 1160.95 | 0.00 | ORB-long ORB[1147.10,1161.85] vol=5.8x ATR=4.54 |
| Stop hit — per-position SL triggered | 2023-10-05 09:45:00 | 1162.36 | 1164.56 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 10:00:00 | 1169.70 | 1173.10 | 0.00 | ORB-short ORB[1170.55,1177.80] vol=1.8x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 11:40:00 | 1165.20 | 1171.46 | 0.00 | T1 1.5R @ 1165.20 |
| Target hit | 2023-10-11 15:20:00 | 1148.10 | 1159.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-10-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:35:00 | 1174.35 | 1168.85 | 0.00 | ORB-long ORB[1161.70,1170.00] vol=2.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-10-18 09:45:00 | 1171.24 | 1170.44 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:35:00 | 1156.55 | 1157.00 | 0.00 | ORB-short ORB[1157.00,1162.30] vol=3.1x ATR=2.11 |
| Stop hit — per-position SL triggered | 2023-10-20 10:45:00 | 1158.66 | 1157.06 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 09:55:00 | 1145.40 | 1139.95 | 0.00 | ORB-long ORB[1130.95,1142.90] vol=2.9x ATR=3.62 |
| Stop hit — per-position SL triggered | 2023-10-23 10:05:00 | 1141.78 | 1140.25 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 10:25:00 | 1122.65 | 1116.50 | 0.00 | ORB-long ORB[1109.20,1118.30] vol=4.4x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-10-25 10:40:00 | 1118.85 | 1118.17 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 11:15:00 | 1090.00 | 1078.60 | 0.00 | ORB-long ORB[1065.00,1078.85] vol=1.6x ATR=4.36 |
| Stop hit — per-position SL triggered | 2023-10-27 11:35:00 | 1085.64 | 1079.54 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 10:55:00 | 1053.10 | 1047.88 | 0.00 | ORB-long ORB[1040.00,1051.00] vol=1.8x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:20:00 | 1058.96 | 1048.81 | 0.00 | T1 1.5R @ 1058.96 |
| Stop hit — per-position SL triggered | 2023-10-31 12:10:00 | 1053.10 | 1049.55 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:45:00 | 1062.25 | 1056.43 | 0.00 | ORB-long ORB[1043.15,1057.60] vol=2.0x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-11-02 10:25:00 | 1058.92 | 1058.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 09:30:00 | 1058.10 | 1061.16 | 0.00 | ORB-short ORB[1058.20,1066.80] vol=1.7x ATR=3.88 |
| Stop hit — per-position SL triggered | 2023-11-03 10:10:00 | 1061.98 | 1060.16 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 10:55:00 | 1098.95 | 1107.07 | 0.00 | ORB-short ORB[1103.60,1114.75] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2023-11-08 11:00:00 | 1101.83 | 1106.19 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:55:00 | 1114.00 | 1106.57 | 0.00 | ORB-long ORB[1100.20,1109.15] vol=1.6x ATR=2.62 |
| Stop hit — per-position SL triggered | 2023-11-10 11:20:00 | 1111.38 | 1107.36 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:45:00 | 1121.65 | 1116.58 | 0.00 | ORB-long ORB[1109.35,1120.05] vol=2.2x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-11-16 10:55:00 | 1119.37 | 1117.26 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:55:00 | 1127.65 | 1122.09 | 0.00 | ORB-long ORB[1116.50,1125.30] vol=2.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-11-17 10:15:00 | 1124.48 | 1123.77 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:35:00 | 1122.05 | 1116.84 | 0.00 | ORB-long ORB[1111.30,1118.10] vol=3.5x ATR=2.80 |
| Stop hit — per-position SL triggered | 2023-11-21 10:00:00 | 1119.25 | 1117.64 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 11:00:00 | 1125.65 | 1122.02 | 0.00 | ORB-long ORB[1113.30,1124.65] vol=2.4x ATR=2.19 |
| Stop hit — per-position SL triggered | 2023-11-22 11:05:00 | 1123.46 | 1122.22 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:30:00 | 1122.25 | 1127.61 | 0.00 | ORB-short ORB[1124.00,1129.85] vol=1.7x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 10:55:00 | 1117.77 | 1126.60 | 0.00 | T1 1.5R @ 1117.77 |
| Stop hit — per-position SL triggered | 2023-11-23 11:30:00 | 1122.25 | 1123.29 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 09:55:00 | 1123.60 | 1129.62 | 0.00 | ORB-short ORB[1127.40,1140.70] vol=2.3x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:45:00 | 1117.70 | 1126.26 | 0.00 | T1 1.5R @ 1117.70 |
| Stop hit — per-position SL triggered | 2023-11-28 14:20:00 | 1123.60 | 1121.60 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:10:00 | 1131.85 | 1130.07 | 0.00 | ORB-long ORB[1116.55,1130.00] vol=10.9x ATR=2.53 |
| Stop hit — per-position SL triggered | 2023-11-29 10:30:00 | 1129.32 | 1129.99 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 11:15:00 | 1161.95 | 1165.08 | 0.00 | ORB-short ORB[1165.00,1172.70] vol=1.5x ATR=3.47 |
| Stop hit — per-position SL triggered | 2023-12-01 11:35:00 | 1165.42 | 1164.97 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 1233.00 | 1224.47 | 0.00 | ORB-long ORB[1213.00,1228.25] vol=5.3x ATR=4.28 |
| Stop hit — per-position SL triggered | 2023-12-06 09:35:00 | 1228.72 | 1225.33 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:55:00 | 1241.15 | 1231.44 | 0.00 | ORB-long ORB[1224.40,1240.50] vol=2.5x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 13:10:00 | 1246.08 | 1237.48 | 0.00 | T1 1.5R @ 1246.08 |
| Stop hit — per-position SL triggered | 2023-12-07 13:35:00 | 1241.15 | 1239.17 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1233.75 | 1242.54 | 0.00 | ORB-short ORB[1239.10,1252.95] vol=1.5x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-12-08 11:10:00 | 1236.77 | 1242.21 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:40:00 | 1250.15 | 1248.40 | 0.00 | ORB-long ORB[1242.60,1250.00] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2023-12-13 09:50:00 | 1247.10 | 1248.38 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 10:30:00 | 1239.55 | 1246.72 | 0.00 | ORB-short ORB[1247.90,1259.00] vol=1.8x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-12-14 11:05:00 | 1242.77 | 1244.13 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:40:00 | 1262.80 | 1260.13 | 0.00 | ORB-long ORB[1248.75,1262.40] vol=2.1x ATR=3.52 |
| Stop hit — per-position SL triggered | 2023-12-15 09:55:00 | 1259.28 | 1260.61 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:10:00 | 1229.60 | 1234.95 | 0.00 | ORB-short ORB[1237.10,1244.00] vol=1.9x ATR=3.05 |
| Stop hit — per-position SL triggered | 2023-12-19 10:15:00 | 1232.65 | 1234.86 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 11:15:00 | 1243.50 | 1241.51 | 0.00 | ORB-long ORB[1228.20,1242.10] vol=1.8x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-12-20 11:30:00 | 1240.93 | 1241.55 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 11:00:00 | 1220.65 | 1219.19 | 0.00 | ORB-long ORB[1211.00,1218.85] vol=1.5x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 11:40:00 | 1226.91 | 1220.13 | 0.00 | T1 1.5R @ 1226.91 |
| Stop hit — per-position SL triggered | 2023-12-22 12:35:00 | 1220.65 | 1221.32 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:30:00 | 1235.10 | 1249.00 | 0.00 | ORB-short ORB[1258.55,1271.55] vol=10.4x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:35:00 | 1226.32 | 1239.55 | 0.00 | T1 1.5R @ 1226.32 |
| Stop hit — per-position SL triggered | 2023-12-27 10:40:00 | 1235.10 | 1237.12 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:20:00 | 1238.70 | 1241.72 | 0.00 | ORB-short ORB[1241.25,1251.90] vol=1.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2023-12-29 10:35:00 | 1242.98 | 1241.27 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 11:05:00 | 1239.45 | 1230.73 | 0.00 | ORB-long ORB[1224.20,1234.90] vol=2.2x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 11:45:00 | 1244.24 | 1233.53 | 0.00 | T1 1.5R @ 1244.24 |
| Stop hit — per-position SL triggered | 2024-01-03 12:20:00 | 1239.45 | 1234.84 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-01-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:35:00 | 1260.60 | 1256.92 | 0.00 | ORB-long ORB[1251.00,1259.95] vol=2.2x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-01-04 09:45:00 | 1257.27 | 1256.94 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:45:00 | 1250.70 | 1256.02 | 0.00 | ORB-short ORB[1255.00,1268.50] vol=1.7x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 11:15:00 | 1246.68 | 1255.01 | 0.00 | T1 1.5R @ 1246.68 |
| Target hit | 2024-01-05 15:20:00 | 1246.25 | 1250.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2024-01-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:10:00 | 1228.70 | 1232.39 | 0.00 | ORB-short ORB[1236.50,1246.00] vol=2.2x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:40:00 | 1222.61 | 1230.33 | 0.00 | T1 1.5R @ 1222.61 |
| Target hit | 2024-01-08 15:20:00 | 1200.70 | 1211.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2024-01-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:45:00 | 1179.95 | 1174.55 | 0.00 | ORB-long ORB[1166.70,1176.25] vol=1.7x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-01-11 10:15:00 | 1176.54 | 1175.51 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:50:00 | 1170.50 | 1175.48 | 0.00 | ORB-short ORB[1171.05,1182.00] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 12:05:00 | 1166.31 | 1173.28 | 0.00 | T1 1.5R @ 1166.31 |
| Target hit | 2024-01-17 15:20:00 | 1161.65 | 1166.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2024-01-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:10:00 | 1174.00 | 1164.22 | 0.00 | ORB-long ORB[1152.70,1163.50] vol=1.6x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-01-19 10:15:00 | 1170.77 | 1164.58 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 11:10:00 | 1138.80 | 1141.84 | 0.00 | ORB-short ORB[1141.05,1155.00] vol=2.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 11:50:00 | 1134.23 | 1140.63 | 0.00 | T1 1.5R @ 1134.23 |
| Target hit | 2024-01-25 15:20:00 | 1129.95 | 1135.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-02-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:50:00 | 1107.10 | 1103.44 | 0.00 | ORB-long ORB[1091.00,1102.95] vol=2.7x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-02-06 12:55:00 | 1103.52 | 1104.91 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 10:30:00 | 1089.00 | 1095.39 | 0.00 | ORB-short ORB[1097.15,1111.95] vol=2.7x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 14:55:00 | 1083.41 | 1091.19 | 0.00 | T1 1.5R @ 1083.41 |
| Target hit | 2024-02-07 15:20:00 | 1082.60 | 1090.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — SELL (started 2024-02-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:30:00 | 1072.65 | 1077.91 | 0.00 | ORB-short ORB[1080.40,1088.90] vol=1.6x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:00:00 | 1068.41 | 1076.63 | 0.00 | T1 1.5R @ 1068.41 |
| Target hit | 2024-02-08 12:25:00 | 1070.35 | 1070.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — BUY (started 2024-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:45:00 | 1087.25 | 1079.61 | 0.00 | ORB-long ORB[1072.80,1083.10] vol=1.6x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 11:10:00 | 1091.94 | 1082.51 | 0.00 | T1 1.5R @ 1091.94 |
| Stop hit — per-position SL triggered | 2024-02-13 12:50:00 | 1087.25 | 1085.67 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 11:05:00 | 1109.00 | 1104.55 | 0.00 | ORB-long ORB[1093.00,1106.50] vol=2.8x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-02-15 11:30:00 | 1106.02 | 1104.98 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 11:05:00 | 1097.35 | 1099.23 | 0.00 | ORB-short ORB[1100.05,1110.95] vol=1.5x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 11:15:00 | 1093.75 | 1098.31 | 0.00 | T1 1.5R @ 1093.75 |
| Target hit | 2024-02-16 14:25:00 | 1092.65 | 1092.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 83 — SELL (started 2024-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:40:00 | 1110.00 | 1117.65 | 0.00 | ORB-short ORB[1110.85,1124.00] vol=1.8x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:10:00 | 1103.56 | 1114.70 | 0.00 | T1 1.5R @ 1103.56 |
| Target hit | 2024-02-20 15:20:00 | 1095.95 | 1098.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2024-02-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:30:00 | 1091.95 | 1095.20 | 0.00 | ORB-short ORB[1092.75,1101.05] vol=2.8x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 10:35:00 | 1087.53 | 1094.62 | 0.00 | T1 1.5R @ 1087.53 |
| Stop hit — per-position SL triggered | 2024-02-21 12:10:00 | 1091.95 | 1090.64 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:20:00 | 1083.05 | 1086.30 | 0.00 | ORB-short ORB[1085.45,1093.90] vol=2.3x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-02-22 10:30:00 | 1086.19 | 1086.17 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 11:15:00 | 1092.00 | 1097.41 | 0.00 | ORB-short ORB[1094.05,1100.95] vol=2.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 11:20:00 | 1088.75 | 1094.59 | 0.00 | T1 1.5R @ 1088.75 |
| Target hit | 2024-03-05 12:05:00 | 1084.90 | 1084.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 87 — SELL (started 2024-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:55:00 | 1085.60 | 1087.64 | 0.00 | ORB-short ORB[1089.00,1099.75] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-03-06 12:30:00 | 1088.82 | 1087.11 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:55:00 | 1092.85 | 1104.40 | 0.00 | ORB-short ORB[1110.35,1123.65] vol=1.8x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:30:00 | 1086.96 | 1101.32 | 0.00 | T1 1.5R @ 1086.96 |
| Target hit | 2024-03-13 15:20:00 | 1075.35 | 1084.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:15:00 | 1070.00 | 1070.89 | 0.00 | ORB-short ORB[1075.95,1090.25] vol=12.1x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:55:00 | 1065.83 | 1070.04 | 0.00 | T1 1.5R @ 1065.83 |
| Stop hit — per-position SL triggered | 2024-03-15 13:10:00 | 1070.00 | 1069.93 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:10:00 | 1057.40 | 1065.84 | 0.00 | ORB-short ORB[1073.05,1079.00] vol=2.6x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-03-18 11:30:00 | 1061.10 | 1065.25 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:25:00 | 1056.10 | 1060.78 | 0.00 | ORB-short ORB[1060.05,1069.60] vol=2.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-03-19 10:30:00 | 1059.29 | 1060.63 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:15:00 | 1038.70 | 1047.17 | 0.00 | ORB-short ORB[1049.40,1058.30] vol=2.1x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-03-20 10:40:00 | 1042.12 | 1045.21 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-03-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:55:00 | 1066.55 | 1064.15 | 0.00 | ORB-long ORB[1059.00,1065.50] vol=1.7x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-03-21 11:45:00 | 1063.99 | 1065.65 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 10:30:00 | 1060.70 | 1062.80 | 0.00 | ORB-short ORB[1063.50,1070.70] vol=3.7x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-03-22 11:05:00 | 1063.44 | 1062.59 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-03-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:00:00 | 1050.20 | 1055.64 | 0.00 | ORB-short ORB[1060.00,1067.50] vol=2.0x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 10:05:00 | 1043.91 | 1054.29 | 0.00 | T1 1.5R @ 1043.91 |
| Stop hit — per-position SL triggered | 2024-03-26 10:20:00 | 1050.20 | 1049.83 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-03-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 11:05:00 | 1062.30 | 1065.96 | 0.00 | ORB-short ORB[1064.90,1076.10] vol=1.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-03-28 11:15:00 | 1064.20 | 1065.86 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-04-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 11:05:00 | 1124.50 | 1118.86 | 0.00 | ORB-long ORB[1110.65,1123.05] vol=2.4x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 11:20:00 | 1128.90 | 1120.39 | 0.00 | T1 1.5R @ 1128.90 |
| Target hit | 2024-04-02 15:20:00 | 1133.40 | 1127.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 98 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 1139.10 | 1145.49 | 0.00 | ORB-short ORB[1143.15,1154.00] vol=1.7x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-04-04 09:35:00 | 1142.33 | 1145.12 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:15:00 | 1153.55 | 1148.17 | 0.00 | ORB-long ORB[1141.80,1149.05] vol=1.5x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 10:20:00 | 1157.89 | 1150.11 | 0.00 | T1 1.5R @ 1157.89 |
| Target hit | 2024-04-05 13:10:00 | 1156.60 | 1157.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 100 — BUY (started 2024-04-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:50:00 | 1165.10 | 1159.88 | 0.00 | ORB-long ORB[1150.50,1158.00] vol=1.5x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 11:05:00 | 1168.58 | 1160.86 | 0.00 | T1 1.5R @ 1168.58 |
| Stop hit — per-position SL triggered | 2024-04-09 11:10:00 | 1165.10 | 1161.04 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:50:00 | 1159.65 | 1153.76 | 0.00 | ORB-long ORB[1147.95,1156.65] vol=2.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-04-10 11:00:00 | 1157.02 | 1153.86 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-04-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:30:00 | 1170.60 | 1165.04 | 0.00 | ORB-long ORB[1155.00,1166.85] vol=3.9x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-04-12 09:45:00 | 1166.43 | 1167.76 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 09:40:00 | 1081.95 | 1085.63 | 0.00 | ORB-short ORB[1082.55,1097.95] vol=1.6x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-04-23 09:50:00 | 1085.88 | 1085.19 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 11:00:00 | 1193.00 | 1194.65 | 0.00 | ORB-short ORB[1198.30,1209.25] vol=8.0x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-04-30 11:05:00 | 1196.51 | 1195.08 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-05-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 11:10:00 | 1215.90 | 1211.91 | 0.00 | ORB-long ORB[1202.85,1214.55] vol=2.1x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-05-02 11:15:00 | 1213.03 | 1211.93 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2024-05-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:05:00 | 1201.05 | 1203.24 | 0.00 | ORB-short ORB[1202.60,1215.00] vol=2.2x ATR=3.78 |
| Stop hit — per-position SL triggered | 2024-05-03 10:10:00 | 1204.83 | 1203.28 | 0.00 | SL hit |

### Cycle 107 — SELL (started 2024-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:35:00 | 1208.50 | 1216.25 | 0.00 | ORB-short ORB[1215.25,1228.60] vol=1.9x ATR=5.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:10:00 | 1200.74 | 1213.42 | 0.00 | T1 1.5R @ 1200.74 |
| Target hit | 2024-05-07 14:45:00 | 1200.70 | 1198.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 108 — SELL (started 2024-05-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:55:00 | 1193.05 | 1203.71 | 0.00 | ORB-short ORB[1206.00,1216.05] vol=3.5x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-05-09 11:05:00 | 1196.75 | 1202.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-23 09:40:00 | 926.50 | 2023-05-23 10:10:00 | 921.38 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-05-23 09:40:00 | 926.50 | 2023-05-23 12:10:00 | 921.60 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-05-29 11:05:00 | 947.15 | 2023-05-29 11:10:00 | 951.33 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-05-29 11:05:00 | 947.15 | 2023-05-29 11:20:00 | 947.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 10:35:00 | 948.00 | 2023-06-05 10:40:00 | 950.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-06 11:00:00 | 948.05 | 2023-06-06 11:15:00 | 951.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-07 09:40:00 | 966.00 | 2023-06-07 09:50:00 | 962.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-06-08 10:30:00 | 962.75 | 2023-06-08 10:50:00 | 958.79 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-06-08 10:30:00 | 962.75 | 2023-06-08 13:25:00 | 960.30 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2023-06-13 09:45:00 | 949.80 | 2023-06-13 09:50:00 | 946.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-06-14 09:40:00 | 948.70 | 2023-06-14 09:45:00 | 945.78 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-06-15 10:45:00 | 942.30 | 2023-06-15 11:30:00 | 939.89 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-15 10:45:00 | 942.30 | 2023-06-15 13:30:00 | 941.95 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2023-06-20 10:20:00 | 944.90 | 2023-06-20 10:30:00 | 941.79 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-21 10:40:00 | 937.00 | 2023-06-21 12:20:00 | 938.56 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-22 09:40:00 | 960.45 | 2023-06-22 10:00:00 | 956.84 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-06-27 09:35:00 | 945.75 | 2023-06-27 09:40:00 | 949.72 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-06-27 09:35:00 | 945.75 | 2023-06-27 09:50:00 | 945.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 11:00:00 | 960.85 | 2023-07-05 12:20:00 | 963.57 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-07-05 11:00:00 | 960.85 | 2023-07-05 12:25:00 | 960.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-07 10:55:00 | 964.40 | 2023-07-07 11:15:00 | 967.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-07-11 10:55:00 | 955.25 | 2023-07-11 11:05:00 | 957.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-13 10:20:00 | 954.65 | 2023-07-13 12:20:00 | 956.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-14 09:40:00 | 934.15 | 2023-07-14 09:45:00 | 938.99 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-14 09:40:00 | 934.15 | 2023-07-14 09:50:00 | 934.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-19 11:00:00 | 952.05 | 2023-07-19 11:55:00 | 949.05 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-07-19 11:00:00 | 952.05 | 2023-07-19 12:50:00 | 952.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-21 11:05:00 | 964.05 | 2023-07-21 11:35:00 | 966.97 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-07-21 11:05:00 | 964.05 | 2023-07-21 15:20:00 | 981.00 | TARGET_HIT | 0.50 | 1.76% |
| BUY | retest1 | 2023-07-26 10:15:00 | 981.40 | 2023-07-26 10:20:00 | 979.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-08-02 10:35:00 | 1029.35 | 2023-08-02 10:40:00 | 1032.21 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-03 09:30:00 | 1029.40 | 2023-08-03 09:40:00 | 1025.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-04 10:25:00 | 1040.35 | 2023-08-04 10:30:00 | 1044.66 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-08-04 10:25:00 | 1040.35 | 2023-08-04 10:45:00 | 1040.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-08 10:20:00 | 1060.60 | 2023-08-08 11:05:00 | 1057.63 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-22 11:15:00 | 1065.00 | 2023-08-22 12:50:00 | 1066.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-08-24 09:35:00 | 1077.05 | 2023-08-24 10:20:00 | 1081.72 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-08-24 09:35:00 | 1077.05 | 2023-08-24 13:35:00 | 1080.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2023-08-28 11:05:00 | 1078.35 | 2023-08-28 11:45:00 | 1081.89 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-08-28 11:05:00 | 1078.35 | 2023-08-28 15:20:00 | 1088.05 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2023-08-31 10:35:00 | 1083.75 | 2023-08-31 10:45:00 | 1078.81 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-08-31 10:35:00 | 1083.75 | 2023-08-31 13:20:00 | 1082.55 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2023-09-01 10:35:00 | 1105.35 | 2023-09-01 10:55:00 | 1102.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-09-05 09:40:00 | 1113.05 | 2023-09-05 09:45:00 | 1116.44 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-08 09:45:00 | 1137.80 | 2023-09-08 11:20:00 | 1132.69 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-09-08 09:45:00 | 1137.80 | 2023-09-08 15:20:00 | 1124.30 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2023-09-12 09:35:00 | 1113.65 | 2023-09-12 09:40:00 | 1110.15 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-09-12 09:35:00 | 1113.65 | 2023-09-12 10:00:00 | 1113.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-14 10:15:00 | 1120.40 | 2023-09-14 10:25:00 | 1124.12 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-22 09:50:00 | 1073.20 | 2023-09-22 09:55:00 | 1076.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-27 09:40:00 | 1119.50 | 2023-09-27 10:45:00 | 1124.01 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-09-27 09:40:00 | 1119.50 | 2023-09-27 15:20:00 | 1128.50 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2023-09-28 10:50:00 | 1135.55 | 2023-09-28 11:00:00 | 1133.23 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-05 09:30:00 | 1166.90 | 2023-10-05 09:45:00 | 1162.36 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-10-11 10:00:00 | 1169.70 | 2023-10-11 11:40:00 | 1165.20 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-10-11 10:00:00 | 1169.70 | 2023-10-11 15:20:00 | 1148.10 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2023-10-18 09:35:00 | 1174.35 | 2023-10-18 09:45:00 | 1171.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-20 10:35:00 | 1156.55 | 2023-10-20 10:45:00 | 1158.66 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-23 09:55:00 | 1145.40 | 2023-10-23 10:05:00 | 1141.78 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-10-25 10:25:00 | 1122.65 | 2023-10-25 10:40:00 | 1118.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-10-27 11:15:00 | 1090.00 | 2023-10-27 11:35:00 | 1085.64 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-10-31 10:55:00 | 1053.10 | 2023-10-31 11:20:00 | 1058.96 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-10-31 10:55:00 | 1053.10 | 2023-10-31 12:10:00 | 1053.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 09:45:00 | 1062.25 | 2023-11-02 10:25:00 | 1058.92 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-11-03 09:30:00 | 1058.10 | 2023-11-03 10:10:00 | 1061.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-11-08 10:55:00 | 1098.95 | 2023-11-08 11:00:00 | 1101.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-10 10:55:00 | 1114.00 | 2023-11-10 11:20:00 | 1111.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-16 10:45:00 | 1121.65 | 2023-11-16 10:55:00 | 1119.37 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-17 09:55:00 | 1127.65 | 2023-11-17 10:15:00 | 1124.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-21 09:35:00 | 1122.05 | 2023-11-21 10:00:00 | 1119.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-22 11:00:00 | 1125.65 | 2023-11-22 11:05:00 | 1123.46 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-23 10:30:00 | 1122.25 | 2023-11-23 10:55:00 | 1117.77 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-23 10:30:00 | 1122.25 | 2023-11-23 11:30:00 | 1122.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-28 09:55:00 | 1123.60 | 2023-11-28 11:45:00 | 1117.70 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-11-28 09:55:00 | 1123.60 | 2023-11-28 14:20:00 | 1123.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 10:10:00 | 1131.85 | 2023-11-29 10:30:00 | 1129.32 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-12-01 11:15:00 | 1161.95 | 2023-12-01 11:35:00 | 1165.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-06 09:30:00 | 1233.00 | 2023-12-06 09:35:00 | 1228.72 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-07 10:55:00 | 1241.15 | 2023-12-07 13:10:00 | 1246.08 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-07 10:55:00 | 1241.15 | 2023-12-07 13:35:00 | 1241.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1233.75 | 2023-12-08 11:10:00 | 1236.77 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-12-13 09:40:00 | 1250.15 | 2023-12-13 09:50:00 | 1247.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-12-14 10:30:00 | 1239.55 | 2023-12-14 11:05:00 | 1242.77 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-15 09:40:00 | 1262.80 | 2023-12-15 09:55:00 | 1259.28 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-19 10:10:00 | 1229.60 | 2023-12-19 10:15:00 | 1232.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-12-20 11:15:00 | 1243.50 | 2023-12-20 11:30:00 | 1240.93 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-22 11:00:00 | 1220.65 | 2023-12-22 11:40:00 | 1226.91 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-12-22 11:00:00 | 1220.65 | 2023-12-22 12:35:00 | 1220.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 10:30:00 | 1235.10 | 2023-12-27 10:35:00 | 1226.32 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2023-12-27 10:30:00 | 1235.10 | 2023-12-27 10:40:00 | 1235.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-29 10:20:00 | 1238.70 | 2023-12-29 10:35:00 | 1242.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-03 11:05:00 | 1239.45 | 2024-01-03 11:45:00 | 1244.24 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-01-03 11:05:00 | 1239.45 | 2024-01-03 12:20:00 | 1239.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 09:35:00 | 1260.60 | 2024-01-04 09:45:00 | 1257.27 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-05 10:45:00 | 1250.70 | 2024-01-05 11:15:00 | 1246.68 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-01-05 10:45:00 | 1250.70 | 2024-01-05 15:20:00 | 1246.25 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-01-08 10:10:00 | 1228.70 | 2024-01-08 10:40:00 | 1222.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-08 10:10:00 | 1228.70 | 2024-01-08 15:20:00 | 1200.70 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2024-01-11 09:45:00 | 1179.95 | 2024-01-11 10:15:00 | 1176.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-17 10:50:00 | 1170.50 | 2024-01-17 12:05:00 | 1166.31 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-01-17 10:50:00 | 1170.50 | 2024-01-17 15:20:00 | 1161.65 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-01-19 10:10:00 | 1174.00 | 2024-01-19 10:15:00 | 1170.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-25 11:10:00 | 1138.80 | 2024-01-25 11:50:00 | 1134.23 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-01-25 11:10:00 | 1138.80 | 2024-01-25 15:20:00 | 1129.95 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2024-02-06 10:50:00 | 1107.10 | 2024-02-06 12:55:00 | 1103.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-07 10:30:00 | 1089.00 | 2024-02-07 14:55:00 | 1083.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-02-07 10:30:00 | 1089.00 | 2024-02-07 15:20:00 | 1082.60 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2024-02-08 10:30:00 | 1072.65 | 2024-02-08 11:00:00 | 1068.41 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-08 10:30:00 | 1072.65 | 2024-02-08 12:25:00 | 1070.35 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-02-13 10:45:00 | 1087.25 | 2024-02-13 11:10:00 | 1091.94 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-02-13 10:45:00 | 1087.25 | 2024-02-13 12:50:00 | 1087.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-15 11:05:00 | 1109.00 | 2024-02-15 11:30:00 | 1106.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-16 11:05:00 | 1097.35 | 2024-02-16 11:15:00 | 1093.75 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-02-16 11:05:00 | 1097.35 | 2024-02-16 14:25:00 | 1092.65 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-02-20 09:40:00 | 1110.00 | 2024-02-20 10:10:00 | 1103.56 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-02-20 09:40:00 | 1110.00 | 2024-02-20 15:20:00 | 1095.95 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2024-02-21 10:30:00 | 1091.95 | 2024-02-21 10:35:00 | 1087.53 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-02-21 10:30:00 | 1091.95 | 2024-02-21 12:10:00 | 1091.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-22 10:20:00 | 1083.05 | 2024-02-22 10:30:00 | 1086.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-05 11:15:00 | 1092.00 | 2024-03-05 11:20:00 | 1088.75 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-03-05 11:15:00 | 1092.00 | 2024-03-05 12:05:00 | 1084.90 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-03-06 10:55:00 | 1085.60 | 2024-03-06 12:30:00 | 1088.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-13 10:55:00 | 1092.85 | 2024-03-13 11:30:00 | 1086.96 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-03-13 10:55:00 | 1092.85 | 2024-03-13 15:20:00 | 1075.35 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2024-03-15 11:15:00 | 1070.00 | 2024-03-15 11:55:00 | 1065.83 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-15 11:15:00 | 1070.00 | 2024-03-15 13:10:00 | 1070.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-18 11:10:00 | 1057.40 | 2024-03-18 11:30:00 | 1061.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-19 10:25:00 | 1056.10 | 2024-03-19 10:30:00 | 1059.29 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-20 10:15:00 | 1038.70 | 2024-03-20 10:40:00 | 1042.12 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-21 09:55:00 | 1066.55 | 2024-03-21 11:45:00 | 1063.99 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-03-22 10:30:00 | 1060.70 | 2024-03-22 11:05:00 | 1063.44 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-03-26 10:00:00 | 1050.20 | 2024-03-26 10:05:00 | 1043.91 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-03-26 10:00:00 | 1050.20 | 2024-03-26 10:20:00 | 1050.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-28 11:05:00 | 1062.30 | 2024-03-28 11:15:00 | 1064.20 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-02 11:05:00 | 1124.50 | 2024-04-02 11:20:00 | 1128.90 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-04-02 11:05:00 | 1124.50 | 2024-04-02 15:20:00 | 1133.40 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2024-04-04 09:30:00 | 1139.10 | 2024-04-04 09:35:00 | 1142.33 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-05 10:15:00 | 1153.55 | 2024-04-05 10:20:00 | 1157.89 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-04-05 10:15:00 | 1153.55 | 2024-04-05 13:10:00 | 1156.60 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-04-09 10:50:00 | 1165.10 | 2024-04-09 11:05:00 | 1168.58 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-09 10:50:00 | 1165.10 | 2024-04-09 11:10:00 | 1165.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-10 10:50:00 | 1159.65 | 2024-04-10 11:00:00 | 1157.02 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-12 09:30:00 | 1170.60 | 2024-04-12 09:45:00 | 1166.43 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-23 09:40:00 | 1081.95 | 2024-04-23 09:50:00 | 1085.88 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-30 11:00:00 | 1193.00 | 2024-04-30 11:05:00 | 1196.51 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-02 11:10:00 | 1215.90 | 2024-05-02 11:15:00 | 1213.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-03 10:05:00 | 1201.05 | 2024-05-03 10:10:00 | 1204.83 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-07 09:35:00 | 1208.50 | 2024-05-07 10:10:00 | 1200.74 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-07 09:35:00 | 1208.50 | 2024-05-07 14:45:00 | 1200.70 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-05-09 10:55:00 | 1193.05 | 2024-05-09 11:05:00 | 1196.75 | STOP_HIT | 1.00 | -0.31% |
