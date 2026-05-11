# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1272.00
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
| ENTRY1 | 107 |
| ENTRY2 | 0 |
| PARTIAL | 44 |
| TARGET_HIT | 20 |
| STOP_HIT | 87 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 64 / 87
- **Target hits / Stop hits / Partials:** 20 / 87 / 44
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 20.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 29 | 41.4% | 8 | 41 | 21 | 0.10% | 7.3% |
| BUY @ 2nd Alert (retest1) | 70 | 29 | 41.4% | 8 | 41 | 21 | 0.10% | 7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 81 | 35 | 43.2% | 12 | 46 | 23 | 0.16% | 13.4% |
| SELL @ 2nd Alert (retest1) | 81 | 35 | 43.2% | 12 | 46 | 23 | 0.16% | 13.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 151 | 64 | 42.4% | 20 | 87 | 44 | 0.14% | 20.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:45:00 | 1007.00 | 1002.53 | 0.00 | ORB-long ORB[992.00,1003.50] vol=1.9x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 10:20:00 | 1011.45 | 1004.86 | 0.00 | T1 1.5R @ 1011.45 |
| Stop hit — per-position SL triggered | 2025-05-14 10:30:00 | 1007.00 | 1005.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:05:00 | 1024.60 | 1030.63 | 0.00 | ORB-short ORB[1031.00,1044.60] vol=8.2x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-05-16 11:35:00 | 1027.06 | 1029.99 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:15:00 | 1017.90 | 1014.76 | 0.00 | ORB-long ORB[1008.20,1017.00] vol=3.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-05-21 11:20:00 | 1015.30 | 1014.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:30:00 | 988.80 | 993.11 | 0.00 | ORB-short ORB[992.00,1003.00] vol=2.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-05-22 09:35:00 | 991.23 | 992.78 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:55:00 | 1019.70 | 1014.72 | 0.00 | ORB-long ORB[1008.70,1017.70] vol=3.7x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-05-23 11:05:00 | 1016.91 | 1015.73 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 09:50:00 | 1006.50 | 1013.79 | 0.00 | ORB-short ORB[1009.20,1022.50] vol=2.1x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-05-26 09:55:00 | 1009.99 | 1012.94 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:05:00 | 1022.50 | 1023.53 | 0.00 | ORB-short ORB[1024.10,1033.10] vol=1.8x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-05-27 11:30:00 | 1024.86 | 1023.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-05-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:55:00 | 1008.70 | 1016.37 | 0.00 | ORB-short ORB[1013.70,1023.70] vol=1.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-05-29 11:25:00 | 1011.58 | 1014.52 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:50:00 | 977.60 | 978.46 | 0.00 | ORB-short ORB[978.00,987.35] vol=2.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-06-03 11:05:00 | 980.32 | 978.61 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:10:00 | 968.60 | 972.17 | 0.00 | ORB-short ORB[969.30,974.55] vol=3.8x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:30:00 | 964.27 | 970.61 | 0.00 | T1 1.5R @ 964.27 |
| Stop hit — per-position SL triggered | 2025-06-04 13:00:00 | 968.60 | 969.47 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:35:00 | 984.70 | 976.83 | 0.00 | ORB-long ORB[967.65,976.90] vol=2.6x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:45:00 | 989.90 | 979.74 | 0.00 | T1 1.5R @ 989.90 |
| Target hit | 2025-06-06 15:20:00 | 1004.60 | 998.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 11:10:00 | 1010.20 | 1007.36 | 0.00 | ORB-long ORB[1002.00,1009.00] vol=6.2x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 11:20:00 | 1013.52 | 1008.91 | 0.00 | T1 1.5R @ 1013.52 |
| Stop hit — per-position SL triggered | 2025-06-09 11:30:00 | 1010.20 | 1009.17 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:45:00 | 1016.00 | 1012.69 | 0.00 | ORB-long ORB[1006.80,1015.00] vol=2.0x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-06-10 09:55:00 | 1013.68 | 1013.03 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 11:10:00 | 995.25 | 991.56 | 0.00 | ORB-long ORB[985.00,992.85] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-06-19 11:20:00 | 993.35 | 991.67 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1002.00 | 998.13 | 0.00 | ORB-long ORB[993.95,1000.90] vol=1.6x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 10:25:00 | 1006.42 | 999.26 | 0.00 | T1 1.5R @ 1006.42 |
| Target hit | 2025-06-20 12:20:00 | 1011.25 | 1011.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-06-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 10:50:00 | 990.10 | 990.25 | 0.00 | ORB-short ORB[993.45,1005.20] vol=2.0x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-06-23 11:25:00 | 992.78 | 990.00 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-06-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 10:05:00 | 1004.30 | 1007.08 | 0.00 | ORB-short ORB[1006.00,1013.00] vol=1.8x ATR=3.06 |
| Stop hit — per-position SL triggered | 2025-06-24 10:20:00 | 1007.36 | 1006.82 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:55:00 | 1016.30 | 1012.56 | 0.00 | ORB-long ORB[1008.00,1015.70] vol=1.7x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1013.72 | 1013.08 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1040.60 | 1036.44 | 0.00 | ORB-long ORB[1031.00,1040.50] vol=3.2x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:20:00 | 1044.48 | 1037.18 | 0.00 | T1 1.5R @ 1044.48 |
| Target hit | 2025-07-02 15:20:00 | 1060.00 | 1053.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:15:00 | 1039.80 | 1043.49 | 0.00 | ORB-short ORB[1040.70,1051.70] vol=1.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:40:00 | 1036.96 | 1042.72 | 0.00 | T1 1.5R @ 1036.96 |
| Stop hit — per-position SL triggered | 2025-07-04 14:35:00 | 1039.80 | 1038.97 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 1038.30 | 1040.02 | 0.00 | ORB-short ORB[1039.10,1047.70] vol=2.2x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-10 11:30:00 | 1040.09 | 1040.23 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 1049.50 | 1046.13 | 0.00 | ORB-long ORB[1039.50,1047.70] vol=1.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1047.06 | 1047.17 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:00:00 | 1025.00 | 1029.21 | 0.00 | ORB-short ORB[1028.00,1032.30] vol=5.2x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 1026.63 | 1028.55 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1034.10 | 1040.15 | 0.00 | ORB-short ORB[1038.70,1046.70] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:40:00 | 1029.90 | 1036.02 | 0.00 | T1 1.5R @ 1029.90 |
| Stop hit — per-position SL triggered | 2025-07-18 13:55:00 | 1034.10 | 1033.34 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:10:00 | 1040.50 | 1042.50 | 0.00 | ORB-short ORB[1040.60,1052.30] vol=3.5x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:00:00 | 1036.41 | 1041.34 | 0.00 | T1 1.5R @ 1036.41 |
| Target hit | 2025-07-30 15:20:00 | 1037.00 | 1040.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-08-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 10:55:00 | 1046.90 | 1040.89 | 0.00 | ORB-long ORB[1026.40,1034.60] vol=2.6x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-08-04 11:05:00 | 1044.25 | 1041.90 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:55:00 | 1054.00 | 1056.83 | 0.00 | ORB-short ORB[1055.00,1063.40] vol=2.4x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:50:00 | 1050.39 | 1055.68 | 0.00 | T1 1.5R @ 1050.39 |
| Stop hit — per-position SL triggered | 2025-08-05 13:30:00 | 1054.00 | 1053.48 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 1047.20 | 1050.71 | 0.00 | ORB-short ORB[1051.30,1056.20] vol=2.2x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 1049.17 | 1050.56 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:45:00 | 1051.00 | 1054.97 | 0.00 | ORB-short ORB[1053.30,1059.90] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1053.08 | 1054.52 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:40:00 | 1056.80 | 1056.00 | 0.00 | ORB-long ORB[1050.30,1056.00] vol=3.0x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-08-13 10:05:00 | 1054.74 | 1056.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:10:00 | 1072.00 | 1078.07 | 0.00 | ORB-short ORB[1075.90,1088.00] vol=2.2x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-08-19 11:20:00 | 1074.31 | 1077.61 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 1076.70 | 1072.06 | 0.00 | ORB-long ORB[1067.30,1073.80] vol=1.9x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:40:00 | 1079.76 | 1073.83 | 0.00 | T1 1.5R @ 1079.76 |
| Stop hit — per-position SL triggered | 2025-08-20 12:00:00 | 1076.70 | 1077.10 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:50:00 | 1083.60 | 1081.41 | 0.00 | ORB-long ORB[1077.70,1083.50] vol=3.0x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-08-21 10:30:00 | 1081.27 | 1082.33 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:05:00 | 1040.50 | 1039.20 | 0.00 | ORB-long ORB[1031.20,1038.70] vol=3.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:35:00 | 1042.89 | 1039.76 | 0.00 | T1 1.5R @ 1042.89 |
| Target hit | 2025-09-02 13:20:00 | 1043.20 | 1043.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 11:15:00 | 1066.90 | 1069.31 | 0.00 | ORB-short ORB[1069.00,1076.60] vol=2.2x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:45:00 | 1064.21 | 1068.94 | 0.00 | T1 1.5R @ 1064.21 |
| Stop hit — per-position SL triggered | 2025-09-05 12:55:00 | 1066.90 | 1068.05 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 1094.80 | 1099.13 | 0.00 | ORB-short ORB[1096.10,1107.90] vol=2.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-09-12 11:20:00 | 1096.36 | 1098.02 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:40:00 | 1108.50 | 1105.58 | 0.00 | ORB-long ORB[1100.00,1105.00] vol=1.8x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:55:00 | 1111.40 | 1107.60 | 0.00 | T1 1.5R @ 1111.40 |
| Stop hit — per-position SL triggered | 2025-09-16 10:45:00 | 1108.50 | 1108.49 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:00:00 | 1114.90 | 1117.48 | 0.00 | ORB-short ORB[1115.80,1122.80] vol=2.9x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 1116.50 | 1117.30 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:45:00 | 1126.30 | 1122.84 | 0.00 | ORB-long ORB[1117.10,1126.00] vol=2.5x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-09-23 11:05:00 | 1124.08 | 1123.67 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 1144.50 | 1140.59 | 0.00 | ORB-long ORB[1133.00,1141.40] vol=1.7x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-09-24 09:45:00 | 1141.37 | 1141.19 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:40:00 | 1150.00 | 1154.84 | 0.00 | ORB-short ORB[1151.50,1163.20] vol=2.0x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-09-26 10:45:00 | 1152.80 | 1156.11 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:50:00 | 1141.10 | 1138.73 | 0.00 | ORB-long ORB[1127.10,1136.90] vol=2.4x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:35:00 | 1145.88 | 1139.51 | 0.00 | T1 1.5R @ 1145.88 |
| Target hit | 2025-09-30 14:15:00 | 1142.90 | 1143.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 1137.00 | 1141.51 | 0.00 | ORB-short ORB[1139.90,1147.90] vol=1.6x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-10-01 12:00:00 | 1139.17 | 1140.18 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:40:00 | 1164.20 | 1159.17 | 0.00 | ORB-long ORB[1145.90,1163.30] vol=2.8x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 1160.81 | 1159.80 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:15:00 | 1152.00 | 1154.71 | 0.00 | ORB-short ORB[1152.50,1164.90] vol=4.3x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 1154.27 | 1154.07 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:50:00 | 1167.50 | 1166.18 | 0.00 | ORB-long ORB[1159.00,1167.10] vol=2.1x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:40:00 | 1171.10 | 1167.45 | 0.00 | T1 1.5R @ 1171.10 |
| Stop hit — per-position SL triggered | 2025-10-07 10:55:00 | 1167.50 | 1168.13 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 1149.20 | 1155.08 | 0.00 | ORB-short ORB[1153.00,1162.70] vol=2.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-10-08 11:50:00 | 1151.58 | 1152.86 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:30:00 | 1160.30 | 1157.55 | 0.00 | ORB-long ORB[1146.00,1159.10] vol=2.5x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-10-09 09:40:00 | 1157.34 | 1158.04 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:40:00 | 1159.50 | 1163.50 | 0.00 | ORB-short ORB[1162.10,1169.60] vol=5.0x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:00:00 | 1156.16 | 1162.23 | 0.00 | T1 1.5R @ 1156.16 |
| Target hit | 2025-10-14 15:20:00 | 1147.40 | 1149.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:00:00 | 1159.30 | 1155.18 | 0.00 | ORB-long ORB[1145.00,1153.60] vol=3.3x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:55:00 | 1162.33 | 1156.35 | 0.00 | T1 1.5R @ 1162.33 |
| Stop hit — per-position SL triggered | 2025-10-15 12:30:00 | 1159.30 | 1157.45 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:45:00 | 1155.90 | 1151.32 | 0.00 | ORB-long ORB[1144.30,1151.90] vol=1.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-10-27 10:05:00 | 1153.35 | 1152.69 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:50:00 | 1158.70 | 1152.59 | 0.00 | ORB-long ORB[1145.00,1152.90] vol=1.5x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-10-28 10:40:00 | 1155.56 | 1154.56 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 10:00:00 | 1205.40 | 1207.28 | 0.00 | ORB-short ORB[1206.10,1217.50] vol=1.8x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:50:00 | 1200.20 | 1204.82 | 0.00 | T1 1.5R @ 1200.20 |
| Stop hit — per-position SL triggered | 2025-11-03 11:10:00 | 1205.40 | 1204.20 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 1190.00 | 1193.62 | 0.00 | ORB-short ORB[1191.00,1202.60] vol=2.2x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:50:00 | 1186.13 | 1192.52 | 0.00 | T1 1.5R @ 1186.13 |
| Target hit | 2025-11-04 15:20:00 | 1180.50 | 1183.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:15:00 | 1171.80 | 1175.74 | 0.00 | ORB-short ORB[1175.30,1183.50] vol=2.4x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:45:00 | 1166.99 | 1174.39 | 0.00 | T1 1.5R @ 1166.99 |
| Target hit | 2025-11-06 14:40:00 | 1170.30 | 1170.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — SELL (started 2025-11-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:30:00 | 1171.20 | 1175.91 | 0.00 | ORB-short ORB[1177.10,1187.00] vol=2.5x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:35:00 | 1166.91 | 1173.39 | 0.00 | T1 1.5R @ 1166.91 |
| Stop hit — per-position SL triggered | 2025-11-11 11:40:00 | 1171.20 | 1173.35 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 1181.10 | 1184.96 | 0.00 | ORB-short ORB[1184.30,1194.90] vol=1.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-11-12 11:05:00 | 1183.60 | 1184.87 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:45:00 | 1195.40 | 1189.93 | 0.00 | ORB-long ORB[1178.10,1191.70] vol=2.4x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-11-13 11:00:00 | 1192.65 | 1190.03 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:15:00 | 1168.50 | 1173.25 | 0.00 | ORB-short ORB[1170.20,1178.00] vol=1.9x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-11-17 12:45:00 | 1170.62 | 1171.70 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:05:00 | 1162.60 | 1165.47 | 0.00 | ORB-short ORB[1165.80,1179.90] vol=3.0x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:25:00 | 1158.40 | 1162.76 | 0.00 | T1 1.5R @ 1158.40 |
| Target hit | 2025-11-18 11:05:00 | 1160.40 | 1159.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 1156.60 | 1160.09 | 0.00 | ORB-short ORB[1157.40,1166.90] vol=2.6x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:05:00 | 1152.42 | 1157.85 | 0.00 | T1 1.5R @ 1152.42 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 1156.60 | 1157.64 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:55:00 | 1122.00 | 1134.25 | 0.00 | ORB-short ORB[1135.80,1145.00] vol=3.0x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:10:00 | 1117.65 | 1130.69 | 0.00 | T1 1.5R @ 1117.65 |
| Target hit | 2025-11-24 15:20:00 | 1106.40 | 1114.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2025-12-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:35:00 | 1148.50 | 1152.91 | 0.00 | ORB-short ORB[1159.20,1168.00] vol=1.9x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:35:00 | 1143.64 | 1150.82 | 0.00 | T1 1.5R @ 1143.64 |
| Target hit | 2025-12-08 15:20:00 | 1115.90 | 1131.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-12-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:50:00 | 1109.00 | 1113.06 | 0.00 | ORB-short ORB[1110.10,1120.40] vol=2.0x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-12-09 10:10:00 | 1112.92 | 1111.67 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 1108.50 | 1103.02 | 0.00 | ORB-long ORB[1095.50,1104.90] vol=1.9x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-12-11 11:35:00 | 1105.97 | 1105.10 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 1122.30 | 1115.15 | 0.00 | ORB-long ORB[1108.80,1115.80] vol=1.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-12-12 09:40:00 | 1119.13 | 1116.11 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 1111.70 | 1114.21 | 0.00 | ORB-short ORB[1112.50,1122.60] vol=2.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-12-15 11:25:00 | 1113.51 | 1114.16 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:05:00 | 1084.60 | 1079.96 | 0.00 | ORB-long ORB[1076.40,1084.00] vol=1.8x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:25:00 | 1088.60 | 1081.33 | 0.00 | T1 1.5R @ 1088.60 |
| Stop hit — per-position SL triggered | 2025-12-18 11:45:00 | 1084.60 | 1081.76 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:55:00 | 1080.50 | 1084.60 | 0.00 | ORB-short ORB[1082.00,1091.00] vol=1.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-12-19 11:35:00 | 1082.96 | 1084.33 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 1092.60 | 1090.83 | 0.00 | ORB-long ORB[1077.70,1091.90] vol=3.3x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:55:00 | 1095.73 | 1091.62 | 0.00 | T1 1.5R @ 1095.73 |
| Target hit | 2025-12-22 15:10:00 | 1094.40 | 1094.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — SELL (started 2025-12-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:50:00 | 1091.20 | 1096.02 | 0.00 | ORB-short ORB[1095.90,1103.30] vol=2.1x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-12-24 10:55:00 | 1093.04 | 1095.52 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:35:00 | 1102.90 | 1099.06 | 0.00 | ORB-long ORB[1092.20,1101.90] vol=1.8x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 09:45:00 | 1106.94 | 1101.53 | 0.00 | T1 1.5R @ 1106.94 |
| Target hit | 2025-12-29 10:10:00 | 1103.60 | 1103.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2025-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:05:00 | 1096.30 | 1094.79 | 0.00 | ORB-long ORB[1089.50,1095.10] vol=1.5x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:15:00 | 1099.56 | 1095.36 | 0.00 | T1 1.5R @ 1099.56 |
| Stop hit — per-position SL triggered | 2025-12-30 11:50:00 | 1096.30 | 1096.75 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:25:00 | 1182.00 | 1177.46 | 0.00 | ORB-long ORB[1169.10,1177.70] vol=1.8x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-01-02 10:55:00 | 1179.55 | 1178.58 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1167.30 | 1174.97 | 0.00 | ORB-short ORB[1178.00,1190.00] vol=1.5x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:55:00 | 1161.99 | 1170.41 | 0.00 | T1 1.5R @ 1161.99 |
| Target hit | 2026-01-08 15:20:00 | 1155.50 | 1162.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:45:00 | 1162.30 | 1159.74 | 0.00 | ORB-long ORB[1154.40,1160.90] vol=2.4x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-01-09 10:55:00 | 1160.01 | 1159.81 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:20:00 | 1190.20 | 1185.08 | 0.00 | ORB-long ORB[1179.30,1189.20] vol=1.6x ATR=3.15 |
| Stop hit — per-position SL triggered | 2026-01-13 10:45:00 | 1187.05 | 1186.88 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 1193.20 | 1185.03 | 0.00 | ORB-long ORB[1170.30,1184.30] vol=1.9x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:15:00 | 1198.17 | 1188.66 | 0.00 | T1 1.5R @ 1198.17 |
| Stop hit — per-position SL triggered | 2026-01-14 12:05:00 | 1193.20 | 1191.05 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1194.50 | 1187.46 | 0.00 | ORB-long ORB[1179.50,1187.90] vol=1.7x ATR=3.05 |
| Stop hit — per-position SL triggered | 2026-01-16 10:35:00 | 1191.45 | 1188.43 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 09:40:00 | 1193.70 | 1190.74 | 0.00 | ORB-long ORB[1184.60,1191.90] vol=1.9x ATR=3.04 |
| Stop hit — per-position SL triggered | 2026-01-19 09:45:00 | 1190.66 | 1190.58 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:05:00 | 1175.60 | 1183.56 | 0.00 | ORB-short ORB[1185.90,1195.00] vol=1.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:00:00 | 1171.15 | 1181.36 | 0.00 | T1 1.5R @ 1171.15 |
| Target hit | 2026-01-20 15:20:00 | 1157.70 | 1173.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 1180.40 | 1186.10 | 0.00 | ORB-short ORB[1181.10,1193.30] vol=2.4x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:20:00 | 1176.16 | 1185.08 | 0.00 | T1 1.5R @ 1176.16 |
| Target hit | 2026-01-23 15:20:00 | 1169.40 | 1174.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2026-02-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:55:00 | 1178.30 | 1190.04 | 0.00 | ORB-short ORB[1185.00,1200.00] vol=1.5x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-02-02 11:30:00 | 1182.44 | 1187.21 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:20:00 | 1226.10 | 1221.52 | 0.00 | ORB-long ORB[1214.20,1226.00] vol=1.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-02-05 10:25:00 | 1223.10 | 1221.60 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:55:00 | 1225.30 | 1230.04 | 0.00 | ORB-short ORB[1231.20,1238.60] vol=5.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-02-06 11:00:00 | 1227.54 | 1229.94 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-02-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:05:00 | 1237.80 | 1240.55 | 0.00 | ORB-short ORB[1238.50,1248.50] vol=2.1x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 1240.53 | 1239.87 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-02-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:00:00 | 1252.40 | 1245.71 | 0.00 | ORB-long ORB[1236.70,1247.70] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 1256.67 | 1247.67 | 0.00 | T1 1.5R @ 1256.67 |
| Stop hit — per-position SL triggered | 2026-02-10 11:35:00 | 1252.40 | 1251.84 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1243.00 | 1244.99 | 0.00 | ORB-short ORB[1243.40,1249.90] vol=2.8x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 1245.74 | 1244.95 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 1228.80 | 1232.64 | 0.00 | ORB-short ORB[1229.90,1247.80] vol=1.6x ATR=3.21 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 1232.01 | 1231.10 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1260.80 | 1255.37 | 0.00 | ORB-long ORB[1244.20,1257.00] vol=2.4x ATR=3.05 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 1257.75 | 1257.43 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1238.50 | 1241.72 | 0.00 | ORB-short ORB[1247.80,1255.50] vol=2.1x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:00:00 | 1234.48 | 1240.84 | 0.00 | T1 1.5R @ 1234.48 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 1238.50 | 1240.64 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 1240.40 | 1232.57 | 0.00 | ORB-long ORB[1223.20,1234.50] vol=2.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 1244.85 | 1234.30 | 0.00 | T1 1.5R @ 1244.85 |
| Stop hit — per-position SL triggered | 2026-02-20 15:05:00 | 1240.40 | 1241.60 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 1264.90 | 1269.36 | 0.00 | ORB-short ORB[1265.10,1277.80] vol=1.8x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 1267.93 | 1269.07 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 1203.80 | 1215.48 | 0.00 | ORB-short ORB[1212.00,1222.90] vol=1.9x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:00:00 | 1197.64 | 1211.88 | 0.00 | T1 1.5R @ 1197.64 |
| Target hit | 2026-03-11 15:20:00 | 1175.60 | 1196.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 1147.30 | 1155.82 | 0.00 | ORB-short ORB[1154.40,1169.20] vol=1.9x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 1140.16 | 1149.18 | 0.00 | T1 1.5R @ 1140.16 |
| Target hit | 2026-03-13 11:25:00 | 1146.80 | 1142.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 96 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1169.50 | 1166.13 | 0.00 | ORB-long ORB[1160.10,1167.20] vol=1.9x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 1165.93 | 1166.93 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 1149.50 | 1154.17 | 0.00 | ORB-short ORB[1155.20,1165.90] vol=2.8x ATR=4.26 |
| Stop hit — per-position SL triggered | 2026-03-19 10:25:00 | 1153.76 | 1154.05 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2026-04-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:50:00 | 1153.80 | 1149.98 | 0.00 | ORB-long ORB[1138.20,1150.30] vol=2.9x ATR=3.89 |
| Stop hit — per-position SL triggered | 2026-04-01 11:00:00 | 1149.91 | 1150.10 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2026-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:55:00 | 1218.00 | 1223.58 | 0.00 | ORB-short ORB[1218.70,1232.00] vol=3.3x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:20:00 | 1213.02 | 1221.62 | 0.00 | T1 1.5R @ 1213.02 |
| Stop hit — per-position SL triggered | 2026-04-15 12:55:00 | 1218.00 | 1218.48 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 1237.60 | 1233.56 | 0.00 | ORB-long ORB[1221.20,1236.00] vol=2.2x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 1242.41 | 1235.28 | 0.00 | T1 1.5R @ 1242.41 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 1237.60 | 1235.83 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1254.90 | 1249.08 | 0.00 | ORB-long ORB[1233.00,1246.90] vol=2.2x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:30:00 | 1259.30 | 1251.12 | 0.00 | T1 1.5R @ 1259.30 |
| Target hit | 2026-04-20 15:20:00 | 1271.50 | 1267.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 102 — SELL (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 1270.00 | 1274.40 | 0.00 | ORB-short ORB[1270.10,1286.60] vol=1.7x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:55:00 | 1264.93 | 1272.42 | 0.00 | T1 1.5R @ 1264.93 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 1270.00 | 1270.07 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 1253.00 | 1263.59 | 0.00 | ORB-short ORB[1253.30,1268.50] vol=1.7x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-04-23 11:45:00 | 1256.36 | 1262.38 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 1250.10 | 1252.73 | 0.00 | ORB-short ORB[1254.20,1268.00] vol=4.1x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 1252.56 | 1250.96 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 1301.40 | 1294.46 | 0.00 | ORB-long ORB[1281.00,1295.00] vol=1.8x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 1306.70 | 1297.55 | 0.00 | T1 1.5R @ 1306.70 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 1301.40 | 1298.06 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 1254.00 | 1256.05 | 0.00 | ORB-short ORB[1257.20,1273.40] vol=1.8x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 1257.44 | 1255.95 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1280.10 | 1277.26 | 0.00 | ORB-long ORB[1268.60,1279.40] vol=1.6x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 1275.96 | 1277.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:45:00 | 1007.00 | 2025-05-14 10:20:00 | 1011.45 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-05-14 09:45:00 | 1007.00 | 2025-05-14 10:30:00 | 1007.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-16 11:05:00 | 1024.60 | 2025-05-16 11:35:00 | 1027.06 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-21 11:15:00 | 1017.90 | 2025-05-21 11:20:00 | 1015.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-22 09:30:00 | 988.80 | 2025-05-22 09:35:00 | 991.23 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-23 10:55:00 | 1019.70 | 2025-05-23 11:05:00 | 1016.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-26 09:50:00 | 1006.50 | 2025-05-26 09:55:00 | 1009.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-05-27 11:05:00 | 1022.50 | 2025-05-27 11:30:00 | 1024.86 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-29 10:55:00 | 1008.70 | 2025-05-29 11:25:00 | 1011.58 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-03 10:50:00 | 977.60 | 2025-06-03 11:05:00 | 980.32 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-04 11:10:00 | 968.60 | 2025-06-04 11:30:00 | 964.27 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-04 11:10:00 | 968.60 | 2025-06-04 13:00:00 | 968.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 10:35:00 | 984.70 | 2025-06-06 10:45:00 | 989.90 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-06 10:35:00 | 984.70 | 2025-06-06 15:20:00 | 1004.60 | TARGET_HIT | 0.50 | 2.02% |
| BUY | retest1 | 2025-06-09 11:10:00 | 1010.20 | 2025-06-09 11:20:00 | 1013.52 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-06-09 11:10:00 | 1010.20 | 2025-06-09 11:30:00 | 1010.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 09:45:00 | 1016.00 | 2025-06-10 09:55:00 | 1013.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-19 11:10:00 | 995.25 | 2025-06-19 11:20:00 | 993.35 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-20 10:15:00 | 1002.00 | 2025-06-20 10:25:00 | 1006.42 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-20 10:15:00 | 1002.00 | 2025-06-20 12:20:00 | 1011.25 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2025-06-23 10:50:00 | 990.10 | 2025-06-23 11:25:00 | 992.78 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-24 10:05:00 | 1004.30 | 2025-06-24 10:20:00 | 1007.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-25 09:55:00 | 1016.30 | 2025-06-25 10:15:00 | 1013.72 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-02 11:15:00 | 1040.60 | 2025-07-02 11:20:00 | 1044.48 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-02 11:15:00 | 1040.60 | 2025-07-02 15:20:00 | 1060.00 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2025-07-04 11:15:00 | 1039.80 | 2025-07-04 11:40:00 | 1036.96 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-04 11:15:00 | 1039.80 | 2025-07-04 14:35:00 | 1039.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 11:10:00 | 1038.30 | 2025-07-10 11:30:00 | 1040.09 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1049.50 | 2025-07-11 10:15:00 | 1047.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-17 11:00:00 | 1025.00 | 2025-07-17 11:10:00 | 1026.63 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1034.10 | 2025-07-18 11:40:00 | 1029.90 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1034.10 | 2025-07-18 13:55:00 | 1034.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-30 11:10:00 | 1040.50 | 2025-07-30 12:00:00 | 1036.41 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-30 11:10:00 | 1040.50 | 2025-07-30 15:20:00 | 1037.00 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-04 10:55:00 | 1046.90 | 2025-08-04 11:05:00 | 1044.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-05 10:55:00 | 1054.00 | 2025-08-05 11:50:00 | 1050.39 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-05 10:55:00 | 1054.00 | 2025-08-05 13:30:00 | 1054.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:05:00 | 1047.20 | 2025-08-07 11:15:00 | 1049.17 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-12 10:45:00 | 1051.00 | 2025-08-12 11:15:00 | 1053.08 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-13 09:40:00 | 1056.80 | 2025-08-13 10:05:00 | 1054.74 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-19 11:10:00 | 1072.00 | 2025-08-19 11:20:00 | 1074.31 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-08-20 10:10:00 | 1076.70 | 2025-08-20 10:40:00 | 1079.76 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-08-20 10:10:00 | 1076.70 | 2025-08-20 12:00:00 | 1076.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 09:50:00 | 1083.60 | 2025-08-21 10:30:00 | 1081.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-02 11:05:00 | 1040.50 | 2025-09-02 11:35:00 | 1042.89 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-09-02 11:05:00 | 1040.50 | 2025-09-02 13:20:00 | 1043.20 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-09-05 11:15:00 | 1066.90 | 2025-09-05 11:45:00 | 1064.21 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-09-05 11:15:00 | 1066.90 | 2025-09-05 12:55:00 | 1066.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 11:00:00 | 1094.80 | 2025-09-12 11:20:00 | 1096.36 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-16 09:40:00 | 1108.50 | 2025-09-16 09:55:00 | 1111.40 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-09-16 09:40:00 | 1108.50 | 2025-09-16 10:45:00 | 1108.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-19 11:00:00 | 1114.90 | 2025-09-19 11:25:00 | 1116.50 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-23 10:45:00 | 1126.30 | 2025-09-23 11:05:00 | 1124.08 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-24 09:35:00 | 1144.50 | 2025-09-24 09:45:00 | 1141.37 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-26 10:40:00 | 1150.00 | 2025-09-26 10:45:00 | 1152.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-30 10:50:00 | 1141.10 | 2025-09-30 11:35:00 | 1145.88 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-09-30 10:50:00 | 1141.10 | 2025-09-30 14:15:00 | 1142.90 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-10-01 11:15:00 | 1137.00 | 2025-10-01 12:00:00 | 1139.17 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-03 10:40:00 | 1164.20 | 2025-10-03 11:15:00 | 1160.81 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-06 11:15:00 | 1152.00 | 2025-10-06 12:15:00 | 1154.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-07 09:50:00 | 1167.50 | 2025-10-07 10:40:00 | 1171.10 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-10-07 09:50:00 | 1167.50 | 2025-10-07 10:55:00 | 1167.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 11:05:00 | 1149.20 | 2025-10-08 11:50:00 | 1151.58 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-09 09:30:00 | 1160.30 | 2025-10-09 09:40:00 | 1157.34 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-14 10:40:00 | 1159.50 | 2025-10-14 11:00:00 | 1156.16 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-14 10:40:00 | 1159.50 | 2025-10-14 15:20:00 | 1147.40 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2025-10-15 11:00:00 | 1159.30 | 2025-10-15 11:55:00 | 1162.33 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-15 11:00:00 | 1159.30 | 2025-10-15 12:30:00 | 1159.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:45:00 | 1155.90 | 2025-10-27 10:05:00 | 1153.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-28 09:50:00 | 1158.70 | 2025-10-28 10:40:00 | 1155.56 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-03 10:00:00 | 1205.40 | 2025-11-03 10:50:00 | 1200.20 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-03 10:00:00 | 1205.40 | 2025-11-03 11:10:00 | 1205.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:35:00 | 1190.00 | 2025-11-04 10:50:00 | 1186.13 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-04 10:35:00 | 1190.00 | 2025-11-04 15:20:00 | 1180.50 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2025-11-06 10:15:00 | 1171.80 | 2025-11-06 10:45:00 | 1166.99 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-06 10:15:00 | 1171.80 | 2025-11-06 14:40:00 | 1170.30 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-11-11 10:30:00 | 1171.20 | 2025-11-11 11:35:00 | 1166.91 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-11 10:30:00 | 1171.20 | 2025-11-11 11:40:00 | 1171.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 10:55:00 | 1181.10 | 2025-11-12 11:05:00 | 1183.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-13 10:45:00 | 1195.40 | 2025-11-13 11:00:00 | 1192.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-17 11:15:00 | 1168.50 | 2025-11-17 12:45:00 | 1170.62 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-18 10:05:00 | 1162.60 | 2025-11-18 10:25:00 | 1158.40 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-18 10:05:00 | 1162.60 | 2025-11-18 11:05:00 | 1160.40 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-11-21 09:45:00 | 1156.60 | 2025-11-21 10:05:00 | 1152.42 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-21 09:45:00 | 1156.60 | 2025-11-21 10:10:00 | 1156.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-24 10:55:00 | 1122.00 | 2025-11-24 12:10:00 | 1117.65 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-24 10:55:00 | 1122.00 | 2025-11-24 15:20:00 | 1106.40 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2025-12-08 10:35:00 | 1148.50 | 2025-12-08 11:35:00 | 1143.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-12-08 10:35:00 | 1148.50 | 2025-12-08 15:20:00 | 1115.90 | TARGET_HIT | 0.50 | 2.84% |
| SELL | retest1 | 2025-12-09 09:50:00 | 1109.00 | 2025-12-09 10:10:00 | 1112.92 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-11 11:10:00 | 1108.50 | 2025-12-11 11:35:00 | 1105.97 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-12 09:35:00 | 1122.30 | 2025-12-12 09:40:00 | 1119.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-15 11:15:00 | 1111.70 | 2025-12-15 11:25:00 | 1113.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-18 11:05:00 | 1084.60 | 2025-12-18 11:25:00 | 1088.60 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-18 11:05:00 | 1084.60 | 2025-12-18 11:45:00 | 1084.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 10:55:00 | 1080.50 | 2025-12-19 11:35:00 | 1082.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-22 11:00:00 | 1092.60 | 2025-12-22 11:55:00 | 1095.73 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-22 11:00:00 | 1092.60 | 2025-12-22 15:10:00 | 1094.40 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-12-24 10:50:00 | 1091.20 | 2025-12-24 10:55:00 | 1093.04 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-29 09:35:00 | 1102.90 | 2025-12-29 09:45:00 | 1106.94 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-29 09:35:00 | 1102.90 | 2025-12-29 10:10:00 | 1103.60 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-12-30 10:05:00 | 1096.30 | 2025-12-30 10:15:00 | 1099.56 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-30 10:05:00 | 1096.30 | 2025-12-30 11:50:00 | 1096.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:25:00 | 1182.00 | 2026-01-02 10:55:00 | 1179.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-08 10:00:00 | 1167.30 | 2026-01-08 10:55:00 | 1161.99 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-08 10:00:00 | 1167.30 | 2026-01-08 15:20:00 | 1155.50 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2026-01-09 10:45:00 | 1162.30 | 2026-01-09 10:55:00 | 1160.01 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-13 10:20:00 | 1190.20 | 2026-01-13 10:45:00 | 1187.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-14 10:55:00 | 1193.20 | 2026-01-14 11:15:00 | 1198.17 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-14 10:55:00 | 1193.20 | 2026-01-14 12:05:00 | 1193.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:15:00 | 1194.50 | 2026-01-16 10:35:00 | 1191.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-19 09:40:00 | 1193.70 | 2026-01-19 09:45:00 | 1190.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-20 11:05:00 | 1175.60 | 2026-01-20 12:00:00 | 1171.15 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-20 11:05:00 | 1175.60 | 2026-01-20 15:20:00 | 1157.70 | TARGET_HIT | 0.50 | 1.52% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1180.40 | 2026-01-23 11:20:00 | 1176.16 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1180.40 | 2026-01-23 15:20:00 | 1169.40 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-02-02 10:55:00 | 1178.30 | 2026-02-02 11:30:00 | 1182.44 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-05 10:20:00 | 1226.10 | 2026-02-05 10:25:00 | 1223.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-06 10:55:00 | 1225.30 | 2026-02-06 11:00:00 | 1227.54 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-09 10:05:00 | 1237.80 | 2026-02-09 10:15:00 | 1240.53 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-10 10:00:00 | 1252.40 | 2026-02-10 10:15:00 | 1256.67 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-10 10:00:00 | 1252.40 | 2026-02-10 11:35:00 | 1252.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 11:05:00 | 1243.00 | 2026-02-11 11:20:00 | 1245.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 09:40:00 | 1228.80 | 2026-02-13 10:25:00 | 1232.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-18 09:30:00 | 1260.80 | 2026-02-18 09:50:00 | 1257.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1238.50 | 2026-02-19 12:00:00 | 1234.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1238.50 | 2026-02-19 12:15:00 | 1238.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 1240.40 | 2026-02-20 11:15:00 | 1244.85 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-20 10:45:00 | 1240.40 | 2026-02-20 15:05:00 | 1240.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:10:00 | 1264.90 | 2026-02-27 10:35:00 | 1267.93 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-11 09:45:00 | 1203.80 | 2026-03-11 10:00:00 | 1197.64 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-11 09:45:00 | 1203.80 | 2026-03-11 15:20:00 | 1175.60 | TARGET_HIT | 0.50 | 2.34% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1147.30 | 2026-03-13 10:00:00 | 1140.16 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1147.30 | 2026-03-13 11:25:00 | 1146.80 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1169.50 | 2026-03-18 09:55:00 | 1165.93 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-19 10:20:00 | 1149.50 | 2026-03-19 10:25:00 | 1153.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-01 10:50:00 | 1153.80 | 2026-04-01 11:00:00 | 1149.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-15 10:55:00 | 1218.00 | 2026-04-15 11:20:00 | 1213.02 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-15 10:55:00 | 1218.00 | 2026-04-15 12:55:00 | 1218.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:30:00 | 1237.60 | 2026-04-16 09:40:00 | 1242.41 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-16 09:30:00 | 1237.60 | 2026-04-16 09:50:00 | 1237.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-20 11:15:00 | 1254.90 | 2026-04-20 11:30:00 | 1259.30 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-20 11:15:00 | 1254.90 | 2026-04-20 15:20:00 | 1271.50 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2026-04-22 09:40:00 | 1270.00 | 2026-04-22 09:55:00 | 1264.93 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-22 09:40:00 | 1270.00 | 2026-04-22 10:55:00 | 1270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1253.00 | 2026-04-23 11:45:00 | 1256.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1250.10 | 2026-04-24 12:10:00 | 1252.56 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-28 09:50:00 | 1301.40 | 2026-04-28 10:15:00 | 1306.70 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-28 09:50:00 | 1301.40 | 2026-04-28 10:25:00 | 1301.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:30:00 | 1254.00 | 2026-04-30 11:05:00 | 1257.44 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 09:45:00 | 1280.10 | 2026-05-07 10:05:00 | 1275.96 | STOP_HIT | 1.00 | -0.32% |
