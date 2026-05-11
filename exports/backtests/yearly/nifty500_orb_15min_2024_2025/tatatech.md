# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-07-04 15:25:00 (19758 bars)
- **Last close:** 707.80
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
| ENTRY1 | 103 |
| ENTRY2 | 0 |
| PARTIAL | 43 |
| TARGET_HIT | 23 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 146 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 80
- **Target hits / Stop hits / Partials:** 23 / 80 / 43
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 21.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 36 | 46.2% | 13 | 42 | 23 | 0.11% | 8.6% |
| BUY @ 2nd Alert (retest1) | 78 | 36 | 46.2% | 13 | 42 | 23 | 0.11% | 8.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 30 | 44.1% | 10 | 38 | 20 | 0.19% | 12.8% |
| SELL @ 2nd Alert (retest1) | 68 | 30 | 44.1% | 10 | 38 | 20 | 0.19% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 146 | 66 | 45.2% | 23 | 80 | 43 | 0.15% | 21.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 1045.10 | 1047.32 | 0.00 | ORB-short ORB[1045.90,1053.50] vol=1.9x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-05-17 09:55:00 | 1047.56 | 1046.84 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 1058.15 | 1053.99 | 0.00 | ORB-long ORB[1048.70,1058.00] vol=3.6x ATR=3.08 |
| Stop hit — per-position SL triggered | 2024-05-21 10:40:00 | 1055.07 | 1055.03 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:05:00 | 1050.70 | 1053.63 | 0.00 | ORB-short ORB[1052.00,1059.80] vol=1.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-05-23 10:20:00 | 1052.64 | 1053.24 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1111.05 | 1101.62 | 0.00 | ORB-long ORB[1087.80,1102.85] vol=3.8x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 1104.80 | 1106.05 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 1045.00 | 1050.86 | 0.00 | ORB-short ORB[1049.00,1056.00] vol=1.9x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-05-31 11:05:00 | 1048.16 | 1047.95 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:35:00 | 1046.50 | 1041.35 | 0.00 | ORB-long ORB[1036.30,1043.00] vol=4.5x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 1052.06 | 1044.54 | 0.00 | T1 1.5R @ 1052.06 |
| Target hit | 2024-06-06 11:30:00 | 1047.90 | 1049.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:35:00 | 1059.80 | 1063.67 | 0.00 | ORB-short ORB[1060.00,1069.45] vol=1.7x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-06-10 10:10:00 | 1062.99 | 1062.28 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 1032.50 | 1038.16 | 0.00 | ORB-short ORB[1034.00,1043.90] vol=2.1x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:15:00 | 1028.33 | 1035.37 | 0.00 | T1 1.5R @ 1028.33 |
| Target hit | 2024-06-19 15:20:00 | 1028.45 | 1029.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 1003.55 | 1009.59 | 0.00 | ORB-short ORB[1006.65,1013.90] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:50:00 | 1000.54 | 1008.65 | 0.00 | T1 1.5R @ 1000.54 |
| Stop hit — per-position SL triggered | 2024-06-21 11:10:00 | 1003.55 | 1008.09 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:55:00 | 1008.55 | 1004.60 | 0.00 | ORB-long ORB[1001.50,1005.95] vol=6.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-06-26 10:00:00 | 1006.48 | 1004.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:55:00 | 1012.00 | 1008.66 | 0.00 | ORB-long ORB[1003.00,1011.00] vol=4.2x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:05:00 | 1015.25 | 1011.49 | 0.00 | T1 1.5R @ 1015.25 |
| Target hit | 2024-06-27 12:55:00 | 1016.55 | 1016.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-06-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:55:00 | 1026.65 | 1021.53 | 0.00 | ORB-long ORB[1018.00,1022.00] vol=2.1x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-06-28 10:30:00 | 1023.85 | 1023.64 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1007.70 | 1009.74 | 0.00 | ORB-short ORB[1009.00,1013.35] vol=2.1x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 1008.82 | 1009.71 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 1017.15 | 1013.73 | 0.00 | ORB-long ORB[1009.40,1015.90] vol=2.2x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-07-05 09:35:00 | 1015.49 | 1014.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 1005.10 | 1007.21 | 0.00 | ORB-short ORB[1006.50,1015.75] vol=2.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-07-08 13:40:00 | 1006.61 | 1006.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:55:00 | 1012.95 | 1008.39 | 0.00 | ORB-long ORB[1004.50,1010.00] vol=3.8x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:00:00 | 1016.03 | 1010.27 | 0.00 | T1 1.5R @ 1016.03 |
| Target hit | 2024-07-09 12:25:00 | 1014.65 | 1017.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:00:00 | 1026.40 | 1018.89 | 0.00 | ORB-long ORB[1011.95,1018.00] vol=7.5x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-07-10 10:05:00 | 1023.26 | 1019.97 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1022.95 | 1019.90 | 0.00 | ORB-long ORB[1016.00,1021.00] vol=4.0x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 1020.54 | 1020.02 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1008.60 | 1014.11 | 0.00 | ORB-short ORB[1012.10,1020.00] vol=2.4x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 1011.42 | 1012.32 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:40:00 | 1002.35 | 1000.59 | 0.00 | ORB-long ORB[995.00,1000.00] vol=3.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-07-23 09:45:00 | 999.92 | 1000.70 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:35:00 | 999.35 | 997.58 | 0.00 | ORB-long ORB[994.00,998.00] vol=2.3x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-07-25 10:05:00 | 997.23 | 997.90 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 1003.15 | 1000.22 | 0.00 | ORB-long ORB[997.10,1003.00] vol=4.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-07-26 11:10:00 | 1001.66 | 1000.38 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:45:00 | 999.00 | 1000.43 | 0.00 | ORB-short ORB[999.70,1004.00] vol=2.1x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:45:00 | 997.11 | 999.63 | 0.00 | T1 1.5R @ 997.11 |
| Stop hit — per-position SL triggered | 2024-08-01 11:25:00 | 999.00 | 999.38 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:10:00 | 994.00 | 997.59 | 0.00 | ORB-short ORB[995.00,1004.30] vol=1.5x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 996.02 | 996.96 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:05:00 | 990.50 | 985.38 | 0.00 | ORB-long ORB[980.35,988.45] vol=2.3x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:20:00 | 993.65 | 985.90 | 0.00 | T1 1.5R @ 993.65 |
| Stop hit — per-position SL triggered | 2024-08-12 11:50:00 | 990.50 | 986.74 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:35:00 | 984.35 | 987.26 | 0.00 | ORB-short ORB[984.65,996.00] vol=2.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 986.76 | 986.95 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 1002.20 | 999.83 | 0.00 | ORB-long ORB[993.15,1001.75] vol=4.1x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:45:00 | 1005.71 | 1003.46 | 0.00 | T1 1.5R @ 1005.71 |
| Target hit | 2024-08-16 10:10:00 | 1003.30 | 1004.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 999.15 | 1001.30 | 0.00 | ORB-short ORB[1000.20,1005.00] vol=2.0x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-08-20 09:50:00 | 1001.05 | 1000.81 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:40:00 | 1011.50 | 1008.04 | 0.00 | ORB-long ORB[1000.45,1011.00] vol=1.6x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:30:00 | 1017.07 | 1010.83 | 0.00 | T1 1.5R @ 1017.07 |
| Target hit | 2024-08-21 12:20:00 | 1027.00 | 1027.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 1066.00 | 1077.04 | 0.00 | ORB-short ORB[1075.70,1089.90] vol=1.7x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-08-29 12:35:00 | 1070.90 | 1074.25 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:50:00 | 1074.00 | 1068.43 | 0.00 | ORB-long ORB[1060.55,1072.30] vol=3.8x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-08-30 10:20:00 | 1069.35 | 1070.29 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:25:00 | 1056.70 | 1068.60 | 0.00 | ORB-short ORB[1066.25,1076.00] vol=1.6x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-09-02 10:45:00 | 1061.43 | 1066.94 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 1062.15 | 1055.30 | 0.00 | ORB-long ORB[1045.55,1060.90] vol=2.3x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:50:00 | 1068.18 | 1059.36 | 0.00 | T1 1.5R @ 1068.18 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 1062.15 | 1059.76 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:50:00 | 1068.20 | 1064.93 | 0.00 | ORB-long ORB[1058.25,1067.00] vol=2.2x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:55:00 | 1072.50 | 1067.68 | 0.00 | T1 1.5R @ 1072.50 |
| Target hit | 2024-09-05 10:25:00 | 1072.00 | 1075.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2024-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:55:00 | 1098.30 | 1091.93 | 0.00 | ORB-long ORB[1085.05,1098.00] vol=2.2x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:05:00 | 1104.26 | 1093.55 | 0.00 | T1 1.5R @ 1104.26 |
| Stop hit — per-position SL triggered | 2024-09-10 12:45:00 | 1098.30 | 1097.95 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:15:00 | 1101.70 | 1092.85 | 0.00 | ORB-long ORB[1087.70,1098.00] vol=3.7x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:30:00 | 1107.57 | 1094.46 | 0.00 | T1 1.5R @ 1107.57 |
| Stop hit — per-position SL triggered | 2024-09-11 11:40:00 | 1101.70 | 1094.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:10:00 | 1093.10 | 1087.39 | 0.00 | ORB-long ORB[1081.70,1092.70] vol=2.3x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-09-12 11:25:00 | 1089.25 | 1087.72 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:10:00 | 1075.35 | 1079.86 | 0.00 | ORB-short ORB[1080.10,1086.85] vol=2.4x ATR=2.95 |
| Stop hit — per-position SL triggered | 2024-09-18 10:20:00 | 1078.30 | 1079.65 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 1091.95 | 1085.47 | 0.00 | ORB-long ORB[1077.05,1090.70] vol=3.1x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-09-19 09:45:00 | 1087.43 | 1087.98 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 1074.35 | 1070.59 | 0.00 | ORB-long ORB[1067.00,1074.00] vol=2.4x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:40:00 | 1079.84 | 1075.24 | 0.00 | T1 1.5R @ 1079.84 |
| Target hit | 2024-09-20 10:15:00 | 1077.55 | 1079.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2024-09-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:45:00 | 1098.05 | 1105.15 | 0.00 | ORB-short ORB[1103.40,1113.30] vol=2.2x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-09-24 10:55:00 | 1101.46 | 1104.78 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-09-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:05:00 | 1096.55 | 1090.67 | 0.00 | ORB-long ORB[1085.40,1095.60] vol=2.6x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-09-26 10:10:00 | 1093.57 | 1090.86 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:30:00 | 1117.00 | 1110.34 | 0.00 | ORB-long ORB[1098.70,1110.80] vol=4.5x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:35:00 | 1123.28 | 1118.81 | 0.00 | T1 1.5R @ 1123.28 |
| Target hit | 2024-09-27 09:50:00 | 1121.05 | 1122.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 1095.40 | 1103.08 | 0.00 | ORB-short ORB[1103.60,1110.80] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-10-01 11:45:00 | 1098.57 | 1102.28 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 1075.20 | 1081.32 | 0.00 | ORB-short ORB[1076.25,1087.70] vol=2.5x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:25:00 | 1071.15 | 1080.72 | 0.00 | T1 1.5R @ 1071.15 |
| Stop hit — per-position SL triggered | 2024-10-03 11:40:00 | 1075.20 | 1080.12 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:50:00 | 1055.00 | 1062.12 | 0.00 | ORB-short ORB[1060.10,1070.65] vol=1.6x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:55:00 | 1049.15 | 1060.66 | 0.00 | T1 1.5R @ 1049.15 |
| Target hit | 2024-10-07 15:20:00 | 1024.85 | 1035.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:55:00 | 1037.80 | 1031.04 | 0.00 | ORB-long ORB[1021.55,1034.00] vol=1.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-10-08 12:30:00 | 1033.71 | 1033.24 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:55:00 | 1051.55 | 1054.32 | 0.00 | ORB-short ORB[1052.30,1062.45] vol=1.7x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:50:00 | 1048.27 | 1052.83 | 0.00 | T1 1.5R @ 1048.27 |
| Stop hit — per-position SL triggered | 2024-10-14 12:35:00 | 1051.55 | 1052.48 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 11:10:00 | 1057.30 | 1052.08 | 0.00 | ORB-long ORB[1042.00,1056.95] vol=3.5x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-10-16 11:25:00 | 1054.77 | 1052.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 10:50:00 | 1084.70 | 1073.23 | 0.00 | ORB-long ORB[1066.50,1077.00] vol=3.3x ATR=4.86 |
| Stop hit — per-position SL triggered | 2024-10-17 11:00:00 | 1079.84 | 1074.06 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-10-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 10:10:00 | 1071.60 | 1062.17 | 0.00 | ORB-long ORB[1057.30,1065.65] vol=2.8x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 1067.21 | 1063.16 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:45:00 | 1038.20 | 1047.29 | 0.00 | ORB-short ORB[1049.10,1063.80] vol=1.6x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:50:00 | 1032.45 | 1046.27 | 0.00 | T1 1.5R @ 1032.45 |
| Stop hit — per-position SL triggered | 2024-10-22 10:55:00 | 1038.20 | 1046.03 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:40:00 | 1017.70 | 1024.18 | 0.00 | ORB-short ORB[1021.40,1034.45] vol=3.3x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-10-23 09:50:00 | 1021.82 | 1023.85 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 10:00:00 | 1036.90 | 1042.41 | 0.00 | ORB-short ORB[1040.10,1051.30] vol=2.1x ATR=4.16 |
| Stop hit — per-position SL triggered | 2024-10-24 10:10:00 | 1041.06 | 1041.17 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 1022.05 | 1025.18 | 0.00 | ORB-short ORB[1022.30,1032.90] vol=1.6x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 1017.02 | 1023.20 | 0.00 | T1 1.5R @ 1017.02 |
| Target hit | 2024-10-25 11:15:00 | 1013.95 | 1013.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — BUY (started 2024-10-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:55:00 | 1025.85 | 1017.99 | 0.00 | ORB-long ORB[1013.10,1023.00] vol=2.0x ATR=3.77 |
| Stop hit — per-position SL triggered | 2024-10-28 11:00:00 | 1022.08 | 1018.17 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 1011.10 | 1006.13 | 0.00 | ORB-long ORB[1000.20,1009.70] vol=1.7x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-10-31 10:05:00 | 1008.29 | 1006.91 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:40:00 | 1003.70 | 1013.16 | 0.00 | ORB-short ORB[1015.20,1029.00] vol=3.4x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-11-04 10:50:00 | 1007.21 | 1012.76 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 1020.70 | 1018.07 | 0.00 | ORB-long ORB[1014.25,1020.00] vol=2.0x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:35:00 | 1024.78 | 1019.27 | 0.00 | T1 1.5R @ 1024.78 |
| Target hit | 2024-11-06 10:25:00 | 1023.50 | 1024.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 985.00 | 990.82 | 0.00 | ORB-short ORB[990.00,997.10] vol=3.5x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:35:00 | 980.69 | 988.61 | 0.00 | T1 1.5R @ 980.69 |
| Target hit | 2024-11-13 15:20:00 | 962.00 | 970.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:35:00 | 946.40 | 942.82 | 0.00 | ORB-long ORB[938.45,944.50] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-11-28 09:40:00 | 944.26 | 943.05 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:45:00 | 952.45 | 946.57 | 0.00 | ORB-long ORB[936.40,945.30] vol=5.4x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-12-02 10:00:00 | 949.34 | 947.45 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:35:00 | 961.75 | 956.54 | 0.00 | ORB-long ORB[951.00,956.65] vol=2.1x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:40:00 | 964.89 | 958.44 | 0.00 | T1 1.5R @ 964.89 |
| Stop hit — per-position SL triggered | 2024-12-03 09:45:00 | 961.75 | 958.97 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 960.45 | 964.05 | 0.00 | ORB-short ORB[961.00,966.00] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-12-04 11:05:00 | 962.59 | 963.87 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 952.40 | 957.07 | 0.00 | ORB-short ORB[953.00,961.75] vol=2.0x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 954.49 | 956.29 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:50:00 | 952.45 | 955.54 | 0.00 | ORB-short ORB[953.35,959.85] vol=1.8x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-12-06 11:00:00 | 955.00 | 955.42 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-12-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:10:00 | 950.30 | 954.04 | 0.00 | ORB-short ORB[951.55,960.00] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 952.44 | 953.98 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 950.00 | 947.70 | 0.00 | ORB-long ORB[943.20,949.90] vol=2.7x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:40:00 | 952.69 | 949.65 | 0.00 | T1 1.5R @ 952.69 |
| Stop hit — per-position SL triggered | 2024-12-12 09:45:00 | 950.00 | 949.85 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:55:00 | 935.50 | 938.64 | 0.00 | ORB-short ORB[936.95,942.00] vol=1.8x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:25:00 | 933.13 | 937.06 | 0.00 | T1 1.5R @ 933.13 |
| Target hit | 2024-12-16 15:20:00 | 931.50 | 935.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2024-12-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:05:00 | 931.65 | 932.99 | 0.00 | ORB-short ORB[932.10,936.60] vol=2.4x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:25:00 | 929.61 | 931.97 | 0.00 | T1 1.5R @ 929.61 |
| Stop hit — per-position SL triggered | 2024-12-17 10:55:00 | 931.65 | 931.62 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-12-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:40:00 | 902.05 | 895.32 | 0.00 | ORB-long ORB[888.00,897.20] vol=3.3x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 09:45:00 | 906.66 | 899.95 | 0.00 | T1 1.5R @ 906.66 |
| Target hit | 2024-12-24 12:10:00 | 909.45 | 910.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:15:00 | 904.90 | 910.49 | 0.00 | ORB-short ORB[908.00,914.00] vol=2.2x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:40:00 | 902.23 | 909.73 | 0.00 | T1 1.5R @ 902.23 |
| Target hit | 2024-12-26 15:20:00 | 901.55 | 904.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 906.95 | 904.80 | 0.00 | ORB-long ORB[901.60,905.75] vol=1.5x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 905.21 | 905.25 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:45:00 | 888.30 | 891.01 | 0.00 | ORB-short ORB[889.95,896.95] vol=4.1x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-01-01 09:50:00 | 890.69 | 890.99 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 899.40 | 896.01 | 0.00 | ORB-long ORB[890.10,898.75] vol=3.1x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:35:00 | 902.34 | 897.10 | 0.00 | T1 1.5R @ 902.34 |
| Target hit | 2025-01-02 10:15:00 | 901.75 | 902.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 910.20 | 908.71 | 0.00 | ORB-long ORB[904.00,909.70] vol=2.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 908.30 | 909.09 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-01-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:35:00 | 889.70 | 896.30 | 0.00 | ORB-short ORB[896.00,900.90] vol=1.5x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:50:00 | 886.25 | 894.98 | 0.00 | T1 1.5R @ 886.25 |
| Target hit | 2025-01-06 12:55:00 | 888.55 | 887.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 78 — SELL (started 2025-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:35:00 | 822.50 | 828.57 | 0.00 | ORB-short ORB[826.50,834.40] vol=1.8x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:05:00 | 817.56 | 824.78 | 0.00 | T1 1.5R @ 817.56 |
| Target hit | 2025-01-13 15:20:00 | 796.00 | 805.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:00:00 | 815.00 | 810.72 | 0.00 | ORB-long ORB[806.15,812.00] vol=1.7x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:45:00 | 818.96 | 812.49 | 0.00 | T1 1.5R @ 818.96 |
| Stop hit — per-position SL triggered | 2025-01-16 11:05:00 | 815.00 | 812.88 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 816.60 | 808.79 | 0.00 | ORB-long ORB[800.00,806.85] vol=3.2x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-01-23 10:05:00 | 813.75 | 809.66 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-01-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:20:00 | 762.55 | 773.80 | 0.00 | ORB-short ORB[772.50,781.40] vol=2.2x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 11:00:00 | 757.53 | 769.06 | 0.00 | T1 1.5R @ 757.53 |
| Stop hit — per-position SL triggered | 2025-01-27 11:20:00 | 762.55 | 767.71 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 789.60 | 792.42 | 0.00 | ORB-short ORB[790.00,794.45] vol=1.6x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:10:00 | 787.19 | 792.15 | 0.00 | T1 1.5R @ 787.19 |
| Target hit | 2025-02-01 15:20:00 | 775.75 | 782.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2025-02-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 10:05:00 | 763.55 | 767.38 | 0.00 | ORB-short ORB[765.45,774.20] vol=2.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 10:15:00 | 759.26 | 766.26 | 0.00 | T1 1.5R @ 759.26 |
| Stop hit — per-position SL triggered | 2025-02-03 10:20:00 | 763.55 | 766.19 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 779.25 | 782.15 | 0.00 | ORB-short ORB[780.05,789.75] vol=2.0x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-02-07 11:20:00 | 781.56 | 782.03 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 11:15:00 | 773.50 | 778.08 | 0.00 | ORB-short ORB[778.40,786.40] vol=1.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:55:00 | 770.38 | 777.28 | 0.00 | T1 1.5R @ 770.38 |
| Target hit | 2025-02-10 15:20:00 | 769.50 | 774.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2025-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:30:00 | 760.00 | 763.79 | 0.00 | ORB-short ORB[761.00,769.30] vol=2.4x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-02-11 09:35:00 | 762.08 | 763.50 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:50:00 | 745.80 | 742.06 | 0.00 | ORB-long ORB[736.65,745.65] vol=1.6x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-02-13 10:10:00 | 743.00 | 742.38 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:50:00 | 726.50 | 721.43 | 0.00 | ORB-long ORB[712.55,722.00] vol=1.7x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:15:00 | 730.65 | 724.82 | 0.00 | T1 1.5R @ 730.65 |
| Stop hit — per-position SL triggered | 2025-02-19 10:50:00 | 726.50 | 726.12 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:40:00 | 724.65 | 726.31 | 0.00 | ORB-short ORB[725.00,729.90] vol=2.1x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-02-27 09:45:00 | 726.61 | 726.23 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 673.75 | 680.11 | 0.00 | ORB-short ORB[677.55,684.40] vol=2.1x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-03-06 09:35:00 | 677.55 | 679.67 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:45:00 | 682.55 | 675.18 | 0.00 | ORB-long ORB[667.55,676.10] vol=1.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-03-07 10:50:00 | 680.15 | 675.44 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 654.90 | 649.77 | 0.00 | ORB-long ORB[643.70,651.85] vol=1.6x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-03-11 11:10:00 | 652.29 | 649.91 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-03-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:55:00 | 646.55 | 649.74 | 0.00 | ORB-short ORB[648.90,653.60] vol=1.7x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:10:00 | 643.76 | 648.16 | 0.00 | T1 1.5R @ 643.76 |
| Stop hit — per-position SL triggered | 2025-03-12 10:35:00 | 646.55 | 646.59 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2025-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:50:00 | 639.90 | 642.88 | 0.00 | ORB-short ORB[640.75,647.05] vol=2.2x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 11:25:00 | 637.29 | 642.18 | 0.00 | T1 1.5R @ 637.29 |
| Stop hit — per-position SL triggered | 2025-03-13 15:05:00 | 639.90 | 638.15 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:15:00 | 656.75 | 653.39 | 0.00 | ORB-long ORB[646.10,653.70] vol=2.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:00:00 | 659.34 | 653.94 | 0.00 | T1 1.5R @ 659.34 |
| Target hit | 2025-03-19 15:20:00 | 670.60 | 661.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 96 — SELL (started 2025-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:50:00 | 669.60 | 676.09 | 0.00 | ORB-short ORB[673.30,682.00] vol=1.6x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-03-20 10:25:00 | 672.51 | 673.95 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 675.80 | 671.62 | 0.00 | ORB-long ORB[665.90,674.35] vol=1.6x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:40:00 | 678.59 | 672.95 | 0.00 | T1 1.5R @ 678.59 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 675.80 | 673.25 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 717.50 | 712.00 | 0.00 | ORB-long ORB[707.05,713.70] vol=3.3x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:40:00 | 722.76 | 714.72 | 0.00 | T1 1.5R @ 722.76 |
| Stop hit — per-position SL triggered | 2025-03-25 10:00:00 | 717.50 | 718.28 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2025-04-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 10:40:00 | 684.70 | 681.62 | 0.00 | ORB-long ORB[674.70,681.20] vol=1.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-04-01 10:50:00 | 681.74 | 681.72 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2025-04-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:45:00 | 689.40 | 683.57 | 0.00 | ORB-long ORB[677.95,687.50] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-04-03 11:00:00 | 686.52 | 683.82 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2025-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:05:00 | 730.55 | 723.80 | 0.00 | ORB-long ORB[719.75,726.10] vol=2.9x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-04-24 10:15:00 | 727.46 | 724.73 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2025-05-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 10:45:00 | 664.30 | 660.34 | 0.00 | ORB-long ORB[657.10,662.30] vol=1.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-05-02 10:50:00 | 662.16 | 660.46 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 655.80 | 651.93 | 0.00 | ORB-long ORB[645.80,654.70] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:45:00 | 659.33 | 653.00 | 0.00 | T1 1.5R @ 659.33 |
| Target hit | 2025-05-05 15:20:00 | 665.60 | 661.98 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 09:35:00 | 1045.10 | 2024-05-17 09:55:00 | 1047.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-21 10:00:00 | 1058.15 | 2024-05-21 10:40:00 | 1055.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-23 10:05:00 | 1050.70 | 2024-05-23 10:20:00 | 1052.64 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1111.05 | 2024-05-27 09:40:00 | 1104.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-05-31 10:00:00 | 1045.00 | 2024-05-31 11:05:00 | 1048.16 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-06 09:35:00 | 1046.50 | 2024-06-06 09:45:00 | 1052.06 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-06 09:35:00 | 1046.50 | 2024-06-06 11:30:00 | 1047.90 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-06-10 09:35:00 | 1059.80 | 2024-06-10 10:10:00 | 1062.99 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-19 09:30:00 | 1032.50 | 2024-06-19 10:15:00 | 1028.33 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-19 09:30:00 | 1032.50 | 2024-06-19 15:20:00 | 1028.45 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-21 10:45:00 | 1003.55 | 2024-06-21 10:50:00 | 1000.54 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-06-21 10:45:00 | 1003.55 | 2024-06-21 11:10:00 | 1003.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:55:00 | 1008.55 | 2024-06-26 10:00:00 | 1006.48 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-27 10:55:00 | 1012.00 | 2024-06-27 11:05:00 | 1015.25 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-06-27 10:55:00 | 1012.00 | 2024-06-27 12:55:00 | 1016.55 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-28 09:55:00 | 1026.65 | 2024-06-28 10:30:00 | 1023.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-04 10:30:00 | 1007.70 | 2024-07-04 10:35:00 | 1008.82 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2024-07-05 09:30:00 | 1017.15 | 2024-07-05 09:35:00 | 1015.49 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-07-08 11:10:00 | 1005.10 | 2024-07-08 13:40:00 | 1006.61 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-07-09 09:55:00 | 1012.95 | 2024-07-09 10:00:00 | 1016.03 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-09 09:55:00 | 1012.95 | 2024-07-09 12:25:00 | 1014.65 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-07-10 10:00:00 | 1026.40 | 2024-07-10 10:05:00 | 1023.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-12 09:30:00 | 1022.95 | 2024-07-12 09:35:00 | 1020.54 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1008.60 | 2024-07-18 10:15:00 | 1011.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-23 09:40:00 | 1002.35 | 2024-07-23 09:45:00 | 999.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-25 09:35:00 | 999.35 | 2024-07-25 10:05:00 | 997.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-26 11:05:00 | 1003.15 | 2024-07-26 11:10:00 | 1001.66 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-08-01 09:45:00 | 999.00 | 2024-08-01 10:45:00 | 997.11 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2024-08-01 09:45:00 | 999.00 | 2024-08-01 11:25:00 | 999.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 11:10:00 | 994.00 | 2024-08-08 11:20:00 | 996.02 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-12 11:05:00 | 990.50 | 2024-08-12 11:20:00 | 993.65 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-12 11:05:00 | 990.50 | 2024-08-12 11:50:00 | 990.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 09:35:00 | 984.35 | 2024-08-14 09:45:00 | 986.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-16 09:40:00 | 1002.20 | 2024-08-16 09:45:00 | 1005.71 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-16 09:40:00 | 1002.20 | 2024-08-16 10:10:00 | 1003.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-08-20 09:30:00 | 999.15 | 2024-08-20 09:50:00 | 1001.05 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-21 09:40:00 | 1011.50 | 2024-08-21 10:30:00 | 1017.07 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-21 09:40:00 | 1011.50 | 2024-08-21 12:20:00 | 1027.00 | TARGET_HIT | 0.50 | 1.53% |
| SELL | retest1 | 2024-08-29 10:55:00 | 1066.00 | 2024-08-29 12:35:00 | 1070.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-08-30 09:50:00 | 1074.00 | 2024-08-30 10:20:00 | 1069.35 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-09-02 10:25:00 | 1056.70 | 2024-09-02 10:45:00 | 1061.43 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-03 09:30:00 | 1062.15 | 2024-09-03 09:50:00 | 1068.18 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-03 09:30:00 | 1062.15 | 2024-09-03 09:55:00 | 1062.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 09:50:00 | 1068.20 | 2024-09-05 09:55:00 | 1072.50 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-05 09:50:00 | 1068.20 | 2024-09-05 10:25:00 | 1072.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-09-10 10:55:00 | 1098.30 | 2024-09-10 11:05:00 | 1104.26 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-10 10:55:00 | 1098.30 | 2024-09-10 12:45:00 | 1098.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 11:15:00 | 1101.70 | 2024-09-11 11:30:00 | 1107.57 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-09-11 11:15:00 | 1101.70 | 2024-09-11 11:40:00 | 1101.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-12 11:10:00 | 1093.10 | 2024-09-12 11:25:00 | 1089.25 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-18 10:10:00 | 1075.35 | 2024-09-18 10:20:00 | 1078.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-19 09:35:00 | 1091.95 | 2024-09-19 09:45:00 | 1087.43 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-09-20 09:30:00 | 1074.35 | 2024-09-20 09:40:00 | 1079.84 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-20 09:30:00 | 1074.35 | 2024-09-20 10:15:00 | 1077.55 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-24 10:45:00 | 1098.05 | 2024-09-24 10:55:00 | 1101.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-26 10:05:00 | 1096.55 | 2024-09-26 10:10:00 | 1093.57 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-27 09:30:00 | 1117.00 | 2024-09-27 09:35:00 | 1123.28 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-27 09:30:00 | 1117.00 | 2024-09-27 09:50:00 | 1121.05 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-01 11:10:00 | 1095.40 | 2024-10-01 11:45:00 | 1098.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-03 11:15:00 | 1075.20 | 2024-10-03 11:25:00 | 1071.15 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-10-03 11:15:00 | 1075.20 | 2024-10-03 11:40:00 | 1075.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 09:50:00 | 1055.00 | 2024-10-07 09:55:00 | 1049.15 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-07 09:50:00 | 1055.00 | 2024-10-07 15:20:00 | 1024.85 | TARGET_HIT | 0.50 | 2.86% |
| BUY | retest1 | 2024-10-08 10:55:00 | 1037.80 | 2024-10-08 12:30:00 | 1033.71 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-14 09:55:00 | 1051.55 | 2024-10-14 11:50:00 | 1048.27 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-10-14 09:55:00 | 1051.55 | 2024-10-14 12:35:00 | 1051.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 11:10:00 | 1057.30 | 2024-10-16 11:25:00 | 1054.77 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-17 10:50:00 | 1084.70 | 2024-10-17 11:00:00 | 1079.84 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-21 10:10:00 | 1071.60 | 2024-10-21 10:15:00 | 1067.21 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-22 10:45:00 | 1038.20 | 2024-10-22 10:50:00 | 1032.45 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-22 10:45:00 | 1038.20 | 2024-10-22 10:55:00 | 1038.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-23 09:40:00 | 1017.70 | 2024-10-23 09:50:00 | 1021.82 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-24 10:00:00 | 1036.90 | 2024-10-24 10:10:00 | 1041.06 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-25 09:30:00 | 1022.05 | 2024-10-25 09:40:00 | 1017.02 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-25 09:30:00 | 1022.05 | 2024-10-25 11:15:00 | 1013.95 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2024-10-28 10:55:00 | 1025.85 | 2024-10-28 11:00:00 | 1022.08 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-31 09:45:00 | 1011.10 | 2024-10-31 10:05:00 | 1008.29 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-04 10:40:00 | 1003.70 | 2024-11-04 10:50:00 | 1007.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-06 09:30:00 | 1020.70 | 2024-11-06 09:35:00 | 1024.78 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-06 09:30:00 | 1020.70 | 2024-11-06 10:25:00 | 1023.50 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2024-11-13 09:30:00 | 985.00 | 2024-11-13 09:35:00 | 980.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-13 09:30:00 | 985.00 | 2024-11-13 15:20:00 | 962.00 | TARGET_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2024-11-28 09:35:00 | 946.40 | 2024-11-28 09:40:00 | 944.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-02 09:45:00 | 952.45 | 2024-12-02 10:00:00 | 949.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-03 09:35:00 | 961.75 | 2024-12-03 09:40:00 | 964.89 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-12-03 09:35:00 | 961.75 | 2024-12-03 09:45:00 | 961.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 10:45:00 | 960.45 | 2024-12-04 11:05:00 | 962.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 10:55:00 | 952.40 | 2024-12-05 12:05:00 | 954.49 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-06 10:50:00 | 952.45 | 2024-12-06 11:00:00 | 955.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-09 10:10:00 | 950.30 | 2024-12-09 10:15:00 | 952.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-12 09:30:00 | 950.00 | 2024-12-12 09:40:00 | 952.69 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-12-12 09:30:00 | 950.00 | 2024-12-12 09:45:00 | 950.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 10:55:00 | 935.50 | 2024-12-16 12:25:00 | 933.13 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-12-16 10:55:00 | 935.50 | 2024-12-16 15:20:00 | 931.50 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-17 10:05:00 | 931.65 | 2024-12-17 10:25:00 | 929.61 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-12-17 10:05:00 | 931.65 | 2024-12-17 10:55:00 | 931.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:40:00 | 902.05 | 2024-12-24 09:45:00 | 906.66 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-24 09:40:00 | 902.05 | 2024-12-24 12:10:00 | 909.45 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2024-12-26 11:15:00 | 904.90 | 2024-12-26 11:40:00 | 902.23 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-26 11:15:00 | 904.90 | 2024-12-26 15:20:00 | 901.55 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-27 09:50:00 | 906.95 | 2024-12-27 10:15:00 | 905.21 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-01-01 09:45:00 | 888.30 | 2025-01-01 09:50:00 | 890.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-02 09:30:00 | 899.40 | 2025-01-02 09:35:00 | 902.34 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-02 09:30:00 | 899.40 | 2025-01-02 10:15:00 | 901.75 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-01-03 09:30:00 | 910.20 | 2025-01-03 09:50:00 | 908.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-06 10:35:00 | 889.70 | 2025-01-06 10:50:00 | 886.25 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-06 10:35:00 | 889.70 | 2025-01-06 12:55:00 | 888.55 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-01-13 09:35:00 | 822.50 | 2025-01-13 10:05:00 | 817.56 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-01-13 09:35:00 | 822.50 | 2025-01-13 15:20:00 | 796.00 | TARGET_HIT | 0.50 | 3.22% |
| BUY | retest1 | 2025-01-16 10:00:00 | 815.00 | 2025-01-16 10:45:00 | 818.96 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-01-16 10:00:00 | 815.00 | 2025-01-16 11:05:00 | 815.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 09:55:00 | 816.60 | 2025-01-23 10:05:00 | 813.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-27 10:20:00 | 762.55 | 2025-01-27 11:00:00 | 757.53 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-01-27 10:20:00 | 762.55 | 2025-01-27 11:20:00 | 762.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 11:00:00 | 789.60 | 2025-02-01 11:10:00 | 787.19 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-02-01 11:00:00 | 789.60 | 2025-02-01 15:20:00 | 775.75 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2025-02-03 10:05:00 | 763.55 | 2025-02-03 10:15:00 | 759.26 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-03 10:05:00 | 763.55 | 2025-02-03 10:20:00 | 763.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-07 11:10:00 | 779.25 | 2025-02-07 11:20:00 | 781.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-10 11:15:00 | 773.50 | 2025-02-10 11:55:00 | 770.38 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-10 11:15:00 | 773.50 | 2025-02-10 15:20:00 | 769.50 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-11 09:30:00 | 760.00 | 2025-02-11 09:35:00 | 762.08 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-13 09:50:00 | 745.80 | 2025-02-13 10:10:00 | 743.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-19 09:50:00 | 726.50 | 2025-02-19 10:15:00 | 730.65 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-02-19 09:50:00 | 726.50 | 2025-02-19 10:50:00 | 726.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-27 09:40:00 | 724.65 | 2025-02-27 09:45:00 | 726.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-06 09:30:00 | 673.75 | 2025-03-06 09:35:00 | 677.55 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-03-07 10:45:00 | 682.55 | 2025-03-07 10:50:00 | 680.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-11 11:00:00 | 654.90 | 2025-03-11 11:10:00 | 652.29 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-12 09:55:00 | 646.55 | 2025-03-12 10:10:00 | 643.76 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-12 09:55:00 | 646.55 | 2025-03-12 10:35:00 | 646.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-13 10:50:00 | 639.90 | 2025-03-13 11:25:00 | 637.29 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-03-13 10:50:00 | 639.90 | 2025-03-13 15:05:00 | 639.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 11:15:00 | 656.75 | 2025-03-19 12:00:00 | 659.34 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-19 11:15:00 | 656.75 | 2025-03-19 15:20:00 | 670.60 | TARGET_HIT | 0.50 | 2.11% |
| SELL | retest1 | 2025-03-20 09:50:00 | 669.60 | 2025-03-20 10:25:00 | 672.51 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-21 09:30:00 | 675.80 | 2025-03-21 09:40:00 | 678.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-21 09:30:00 | 675.80 | 2025-03-21 09:45:00 | 675.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 09:30:00 | 717.50 | 2025-03-25 09:40:00 | 722.76 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-03-25 09:30:00 | 717.50 | 2025-03-25 10:00:00 | 717.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-01 10:40:00 | 684.70 | 2025-04-01 10:50:00 | 681.74 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-04-03 10:45:00 | 689.40 | 2025-04-03 11:00:00 | 686.52 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-24 10:05:00 | 730.55 | 2025-04-24 10:15:00 | 727.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-02 10:45:00 | 664.30 | 2025-05-02 10:50:00 | 662.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-05 09:40:00 | 655.80 | 2025-05-05 09:45:00 | 659.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-05 09:40:00 | 655.80 | 2025-05-05 15:20:00 | 665.60 | TARGET_HIT | 0.50 | 1.49% |
