# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1088.90
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
| ENTRY1 | 92 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 19 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 73
- **Target hits / Stop hits / Partials:** 19 / 73 / 35
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 19.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 26 | 41.3% | 10 | 37 | 16 | 0.12% | 7.7% |
| BUY @ 2nd Alert (retest1) | 63 | 26 | 41.3% | 10 | 37 | 16 | 0.12% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 28 | 43.8% | 9 | 36 | 19 | 0.18% | 11.6% |
| SELL @ 2nd Alert (retest1) | 64 | 28 | 43.8% | 9 | 36 | 19 | 0.18% | 11.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 54 | 42.5% | 19 | 73 | 35 | 0.15% | 19.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 1062.60 | 1070.01 | 0.00 | ORB-short ORB[1067.50,1078.50] vol=1.8x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:35:00 | 1057.66 | 1067.08 | 0.00 | T1 1.5R @ 1057.66 |
| Stop hit — per-position SL triggered | 2025-05-19 10:50:00 | 1062.60 | 1060.21 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:20:00 | 1060.40 | 1052.27 | 0.00 | ORB-long ORB[1042.10,1056.30] vol=2.5x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-05-21 10:35:00 | 1056.36 | 1053.02 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:10:00 | 1054.30 | 1051.20 | 0.00 | ORB-long ORB[1045.10,1054.00] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-05-26 10:25:00 | 1051.41 | 1052.41 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 1040.40 | 1042.78 | 0.00 | ORB-short ORB[1041.40,1051.40] vol=2.5x ATR=3.08 |
| Stop hit — per-position SL triggered | 2025-05-27 09:40:00 | 1043.48 | 1042.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:35:00 | 1060.30 | 1055.73 | 0.00 | ORB-long ORB[1042.60,1055.80] vol=2.2x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-05-28 09:55:00 | 1056.08 | 1057.52 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:00:00 | 1033.80 | 1039.76 | 0.00 | ORB-short ORB[1043.00,1049.40] vol=2.2x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-05-30 11:25:00 | 1036.16 | 1039.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:50:00 | 1058.70 | 1055.26 | 0.00 | ORB-long ORB[1050.70,1058.40] vol=2.3x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-06-04 11:30:00 | 1055.57 | 1056.58 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1131.90 | 1122.77 | 0.00 | ORB-long ORB[1110.50,1123.40] vol=2.3x ATR=6.11 |
| Stop hit — per-position SL triggered | 2025-06-09 09:50:00 | 1125.79 | 1126.51 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:10:00 | 1121.30 | 1125.15 | 0.00 | ORB-short ORB[1126.00,1136.20] vol=5.1x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:20:00 | 1116.74 | 1124.91 | 0.00 | T1 1.5R @ 1116.74 |
| Stop hit — per-position SL triggered | 2025-06-10 12:25:00 | 1121.30 | 1123.86 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:05:00 | 1119.70 | 1125.29 | 0.00 | ORB-short ORB[1122.40,1132.80] vol=1.6x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-06-11 10:20:00 | 1122.79 | 1123.81 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 1109.70 | 1111.98 | 0.00 | ORB-short ORB[1110.00,1121.40] vol=2.5x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:40:00 | 1103.97 | 1110.72 | 0.00 | T1 1.5R @ 1103.97 |
| Target hit | 2025-06-12 15:20:00 | 1086.30 | 1102.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1039.10 | 1051.43 | 0.00 | ORB-short ORB[1050.10,1061.30] vol=3.0x ATR=3.02 |
| Stop hit — per-position SL triggered | 2025-06-19 11:20:00 | 1042.12 | 1051.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:10:00 | 1053.20 | 1044.15 | 0.00 | ORB-long ORB[1032.00,1046.00] vol=1.9x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-06-23 11:20:00 | 1049.29 | 1044.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:50:00 | 1099.00 | 1093.20 | 0.00 | ORB-long ORB[1086.10,1097.00] vol=2.8x ATR=6.30 |
| Stop hit — per-position SL triggered | 2025-06-26 09:55:00 | 1092.70 | 1093.49 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 09:55:00 | 1076.40 | 1079.29 | 0.00 | ORB-short ORB[1077.00,1091.90] vol=1.5x ATR=3.95 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1080.35 | 1079.21 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 1076.60 | 1082.11 | 0.00 | ORB-short ORB[1077.50,1088.20] vol=2.0x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-07-08 11:40:00 | 1078.99 | 1081.24 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:15:00 | 1070.20 | 1074.88 | 0.00 | ORB-short ORB[1076.10,1083.90] vol=3.4x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-07-11 11:30:00 | 1073.19 | 1074.78 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:55:00 | 1080.50 | 1087.29 | 0.00 | ORB-short ORB[1087.70,1096.70] vol=3.0x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-07-15 10:00:00 | 1083.70 | 1087.06 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:00:00 | 1112.90 | 1106.82 | 0.00 | ORB-long ORB[1106.20,1110.40] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 1110.46 | 1107.50 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:55:00 | 1020.70 | 1029.95 | 0.00 | ORB-short ORB[1033.30,1039.60] vol=3.2x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1015.61 | 1026.80 | 0.00 | T1 1.5R @ 1015.61 |
| Target hit | 2025-07-25 15:20:00 | 1010.70 | 1012.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 774.20 | 770.91 | 0.00 | ORB-long ORB[766.65,772.95] vol=1.7x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:40:00 | 777.82 | 772.81 | 0.00 | T1 1.5R @ 777.82 |
| Stop hit — per-position SL triggered | 2025-08-06 10:00:00 | 774.20 | 773.96 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 11:05:00 | 777.40 | 773.62 | 0.00 | ORB-long ORB[770.45,777.00] vol=5.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 11:10:00 | 779.97 | 775.42 | 0.00 | T1 1.5R @ 779.97 |
| Target hit | 2025-08-18 12:20:00 | 779.85 | 779.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:35:00 | 795.80 | 790.35 | 0.00 | ORB-long ORB[783.65,791.40] vol=4.2x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-08-19 09:40:00 | 793.34 | 791.26 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 11:00:00 | 798.65 | 800.13 | 0.00 | ORB-short ORB[798.90,806.50] vol=4.9x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:25:00 | 796.53 | 799.32 | 0.00 | T1 1.5R @ 796.53 |
| Stop hit — per-position SL triggered | 2025-08-25 12:25:00 | 798.65 | 798.49 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 790.40 | 794.84 | 0.00 | ORB-short ORB[793.05,802.25] vol=3.0x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 793.01 | 794.09 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 774.35 | 766.20 | 0.00 | ORB-long ORB[760.10,769.70] vol=2.2x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:50:00 | 778.48 | 769.94 | 0.00 | T1 1.5R @ 778.48 |
| Target hit | 2025-09-02 15:20:00 | 786.00 | 783.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:45:00 | 810.45 | 807.10 | 0.00 | ORB-long ORB[802.75,809.75] vol=2.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:05:00 | 813.20 | 810.79 | 0.00 | T1 1.5R @ 813.20 |
| Target hit | 2025-09-10 11:45:00 | 812.90 | 812.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:00:00 | 817.00 | 814.78 | 0.00 | ORB-long ORB[811.00,815.25] vol=1.8x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 10:15:00 | 819.64 | 817.30 | 0.00 | T1 1.5R @ 819.64 |
| Target hit | 2025-09-12 12:35:00 | 824.80 | 825.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 829.50 | 833.46 | 0.00 | ORB-short ORB[832.55,836.95] vol=2.2x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-09-17 10:00:00 | 831.47 | 832.80 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:10:00 | 864.00 | 869.63 | 0.00 | ORB-short ORB[868.10,874.60] vol=3.0x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-09-22 11:40:00 | 866.08 | 868.64 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:25:00 | 868.75 | 865.62 | 0.00 | ORB-long ORB[859.05,868.00] vol=1.8x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:50:00 | 873.04 | 868.12 | 0.00 | T1 1.5R @ 873.04 |
| Target hit | 2025-09-23 15:20:00 | 890.90 | 882.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-09-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:40:00 | 897.75 | 890.52 | 0.00 | ORB-long ORB[880.00,887.50] vol=1.7x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-09-24 11:00:00 | 894.80 | 891.27 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:30:00 | 872.20 | 866.97 | 0.00 | ORB-long ORB[862.65,869.15] vol=2.0x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:10:00 | 877.13 | 870.00 | 0.00 | T1 1.5R @ 877.13 |
| Target hit | 2025-10-01 15:00:00 | 881.65 | 883.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 880.00 | 886.85 | 0.00 | ORB-short ORB[886.70,896.00] vol=3.5x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-10-03 11:00:00 | 882.86 | 886.70 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:45:00 | 899.20 | 893.98 | 0.00 | ORB-long ORB[888.00,896.50] vol=3.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-10-06 09:50:00 | 896.33 | 894.27 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:40:00 | 882.95 | 889.28 | 0.00 | ORB-short ORB[888.20,898.20] vol=2.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-10-07 12:00:00 | 885.45 | 885.57 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:40:00 | 875.35 | 875.64 | 0.00 | ORB-short ORB[876.00,881.95] vol=5.1x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-10-09 09:50:00 | 878.46 | 875.67 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:35:00 | 864.60 | 870.88 | 0.00 | ORB-short ORB[870.90,878.00] vol=3.8x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:55:00 | 860.67 | 866.03 | 0.00 | T1 1.5R @ 860.67 |
| Target hit | 2025-10-10 15:20:00 | 855.55 | 859.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 849.15 | 852.29 | 0.00 | ORB-short ORB[850.50,856.35] vol=1.8x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:55:00 | 845.81 | 849.42 | 0.00 | T1 1.5R @ 845.81 |
| Target hit | 2025-10-14 15:10:00 | 843.65 | 843.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2025-10-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:50:00 | 859.50 | 846.16 | 0.00 | ORB-long ORB[839.55,847.85] vol=4.9x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-10-20 11:05:00 | 856.16 | 850.60 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:45:00 | 882.95 | 876.39 | 0.00 | ORB-long ORB[871.00,880.00] vol=2.0x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:55:00 | 887.65 | 878.56 | 0.00 | T1 1.5R @ 887.65 |
| Target hit | 2025-10-23 13:30:00 | 884.20 | 884.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 896.65 | 892.62 | 0.00 | ORB-long ORB[885.15,891.15] vol=5.0x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:50:00 | 901.42 | 895.12 | 0.00 | T1 1.5R @ 901.42 |
| Target hit | 2025-10-24 10:20:00 | 899.50 | 901.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2025-10-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:00:00 | 928.75 | 921.59 | 0.00 | ORB-long ORB[911.30,923.90] vol=1.6x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-10-27 10:30:00 | 925.39 | 924.28 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:55:00 | 926.40 | 929.27 | 0.00 | ORB-short ORB[929.95,936.00] vol=1.5x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:05:00 | 923.07 | 928.81 | 0.00 | T1 1.5R @ 923.07 |
| Stop hit — per-position SL triggered | 2025-10-31 11:30:00 | 926.40 | 928.38 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:10:00 | 949.15 | 936.17 | 0.00 | ORB-long ORB[926.55,935.20] vol=1.9x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 946.11 | 938.00 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 918.70 | 924.07 | 0.00 | ORB-short ORB[920.20,931.30] vol=1.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-11-06 09:35:00 | 921.40 | 923.49 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:25:00 | 889.30 | 890.15 | 0.00 | ORB-short ORB[892.15,901.80] vol=3.7x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:05:00 | 884.67 | 889.48 | 0.00 | T1 1.5R @ 884.67 |
| Stop hit — per-position SL triggered | 2025-11-11 11:50:00 | 889.30 | 888.93 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:50:00 | 910.10 | 903.60 | 0.00 | ORB-long ORB[894.30,902.90] vol=1.7x ATR=2.77 |
| Stop hit — per-position SL triggered | 2025-11-13 10:35:00 | 907.33 | 906.04 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:45:00 | 922.25 | 917.48 | 0.00 | ORB-long ORB[907.85,918.75] vol=2.2x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-11-14 09:55:00 | 919.84 | 918.53 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:00:00 | 920.00 | 916.97 | 0.00 | ORB-long ORB[912.20,918.70] vol=1.7x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-11-17 10:40:00 | 917.65 | 917.77 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:40:00 | 902.55 | 905.07 | 0.00 | ORB-short ORB[902.70,907.30] vol=2.1x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:50:00 | 898.91 | 903.91 | 0.00 | T1 1.5R @ 898.91 |
| Target hit | 2025-11-21 14:00:00 | 893.10 | 892.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — SELL (started 2025-11-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:05:00 | 879.20 | 882.53 | 0.00 | ORB-short ORB[882.10,892.00] vol=7.4x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 881.85 | 881.75 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 11:00:00 | 915.75 | 909.22 | 0.00 | ORB-long ORB[902.00,910.45] vol=2.1x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-12-01 11:40:00 | 913.61 | 911.56 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:45:00 | 914.55 | 911.62 | 0.00 | ORB-long ORB[905.90,913.60] vol=1.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-12-02 11:45:00 | 912.11 | 913.71 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:40:00 | 888.10 | 893.39 | 0.00 | ORB-short ORB[892.45,901.00] vol=1.9x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:05:00 | 883.71 | 891.07 | 0.00 | T1 1.5R @ 883.71 |
| Target hit | 2025-12-03 15:15:00 | 881.00 | 879.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — BUY (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 893.60 | 888.40 | 0.00 | ORB-long ORB[878.00,888.90] vol=3.4x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:10:00 | 898.51 | 890.92 | 0.00 | T1 1.5R @ 898.51 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 893.60 | 893.12 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 889.40 | 897.05 | 0.00 | ORB-short ORB[899.55,906.00] vol=1.7x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:40:00 | 885.74 | 894.64 | 0.00 | T1 1.5R @ 885.74 |
| Target hit | 2025-12-08 15:20:00 | 873.45 | 881.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-12-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:00:00 | 857.90 | 860.17 | 0.00 | ORB-short ORB[861.80,872.55] vol=8.5x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-12-09 10:05:00 | 861.10 | 860.59 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:50:00 | 885.10 | 883.38 | 0.00 | ORB-long ORB[877.80,885.00] vol=2.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:25:00 | 888.37 | 884.52 | 0.00 | T1 1.5R @ 888.37 |
| Stop hit — per-position SL triggered | 2025-12-10 10:30:00 | 885.10 | 884.61 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:50:00 | 891.50 | 884.28 | 0.00 | ORB-long ORB[878.95,888.65] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 888.53 | 885.17 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:25:00 | 908.40 | 911.45 | 0.00 | ORB-short ORB[910.00,917.50] vol=1.7x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:55:00 | 904.45 | 910.52 | 0.00 | T1 1.5R @ 904.45 |
| Stop hit — per-position SL triggered | 2025-12-17 11:45:00 | 908.40 | 909.75 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:40:00 | 886.95 | 893.04 | 0.00 | ORB-short ORB[890.05,897.00] vol=1.6x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:55:00 | 882.27 | 890.60 | 0.00 | T1 1.5R @ 882.27 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 886.95 | 888.52 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 905.00 | 901.23 | 0.00 | ORB-long ORB[893.15,904.35] vol=1.6x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-12-19 10:05:00 | 901.48 | 902.92 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 943.00 | 937.14 | 0.00 | ORB-long ORB[928.50,940.10] vol=2.0x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-12-24 09:55:00 | 940.35 | 940.07 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 955.55 | 962.98 | 0.00 | ORB-short ORB[961.75,970.85] vol=1.5x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:20:00 | 952.34 | 962.13 | 0.00 | T1 1.5R @ 952.34 |
| Target hit | 2025-12-26 15:20:00 | 942.70 | 953.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 936.00 | 938.99 | 0.00 | ORB-short ORB[937.00,946.85] vol=1.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-12-29 11:45:00 | 938.09 | 938.53 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:00:00 | 960.70 | 959.28 | 0.00 | ORB-long ORB[951.55,957.00] vol=6.7x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:25:00 | 964.83 | 959.78 | 0.00 | T1 1.5R @ 964.83 |
| Target hit | 2026-01-01 15:20:00 | 986.20 | 983.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 995.05 | 989.42 | 0.00 | ORB-long ORB[980.55,987.60] vol=2.4x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:20:00 | 1000.79 | 993.03 | 0.00 | T1 1.5R @ 1000.79 |
| Stop hit — per-position SL triggered | 2026-01-02 10:50:00 | 995.05 | 993.82 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:55:00 | 991.00 | 1000.07 | 0.00 | ORB-short ORB[1000.30,1010.65] vol=1.5x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-01-07 10:10:00 | 994.91 | 996.21 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 998.55 | 1002.17 | 0.00 | ORB-short ORB[999.75,1007.10] vol=2.1x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:05:00 | 994.68 | 1001.12 | 0.00 | T1 1.5R @ 994.68 |
| Target hit | 2026-01-08 15:20:00 | 980.35 | 991.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2026-01-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:35:00 | 966.50 | 976.20 | 0.00 | ORB-short ORB[979.90,992.65] vol=3.8x ATR=4.40 |
| Stop hit — per-position SL triggered | 2026-01-13 11:55:00 | 970.90 | 973.67 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:15:00 | 973.20 | 980.99 | 0.00 | ORB-short ORB[975.00,986.20] vol=1.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2026-01-16 11:25:00 | 975.88 | 980.76 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 09:30:00 | 959.90 | 956.37 | 0.00 | ORB-long ORB[950.40,959.10] vol=1.7x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-01-19 09:40:00 | 956.39 | 956.73 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 839.30 | 832.65 | 0.00 | ORB-long ORB[826.45,833.60] vol=3.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-01-30 09:40:00 | 835.80 | 833.91 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:20:00 | 840.50 | 834.66 | 0.00 | ORB-long ORB[820.15,827.90] vol=2.0x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-02-04 10:25:00 | 837.34 | 835.16 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 842.05 | 848.55 | 0.00 | ORB-short ORB[846.65,855.00] vol=1.8x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-02-05 10:20:00 | 845.28 | 846.32 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:45:00 | 864.15 | 859.59 | 0.00 | ORB-long ORB[852.00,857.55] vol=4.1x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-02-09 09:50:00 | 860.80 | 859.68 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 853.20 | 860.55 | 0.00 | ORB-short ORB[861.65,869.85] vol=2.1x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-02-10 10:10:00 | 855.97 | 859.88 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 831.75 | 837.43 | 0.00 | ORB-short ORB[838.50,848.00] vol=4.4x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 834.59 | 837.20 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 846.35 | 841.20 | 0.00 | ORB-long ORB[830.15,840.40] vol=3.1x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 844.01 | 841.87 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 853.95 | 848.77 | 0.00 | ORB-long ORB[841.00,853.55] vol=3.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:10:00 | 856.66 | 851.95 | 0.00 | T1 1.5R @ 856.66 |
| Target hit | 2026-02-17 15:20:00 | 863.60 | 860.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 854.15 | 858.95 | 0.00 | ORB-short ORB[859.00,868.55] vol=2.5x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 851.24 | 856.93 | 0.00 | T1 1.5R @ 851.24 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 854.15 | 856.56 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 850.15 | 847.65 | 0.00 | ORB-long ORB[838.40,848.80] vol=4.4x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 855.09 | 849.51 | 0.00 | T1 1.5R @ 855.09 |
| Stop hit — per-position SL triggered | 2026-02-20 13:25:00 | 850.15 | 850.77 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 846.75 | 850.20 | 0.00 | ORB-short ORB[848.50,854.25] vol=3.5x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:05:00 | 843.69 | 849.93 | 0.00 | T1 1.5R @ 843.69 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 846.75 | 846.43 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 824.95 | 827.52 | 0.00 | ORB-short ORB[827.05,838.00] vol=2.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-24 10:40:00 | 827.38 | 827.30 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 759.00 | 763.82 | 0.00 | ORB-short ORB[762.65,774.00] vol=3.6x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-03-12 09:45:00 | 762.39 | 762.77 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 770.50 | 774.47 | 0.00 | ORB-short ORB[772.60,778.50] vol=1.9x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 766.16 | 772.55 | 0.00 | T1 1.5R @ 766.16 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 770.50 | 771.13 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 772.15 | 764.54 | 0.00 | ORB-long ORB[758.40,765.50] vol=2.3x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 768.48 | 769.58 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-03-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:10:00 | 782.20 | 772.73 | 0.00 | ORB-long ORB[767.15,773.15] vol=2.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 779.26 | 774.95 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1055.05 | 1043.82 | 0.00 | ORB-long ORB[1036.45,1050.10] vol=2.2x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:25:00 | 1060.63 | 1048.07 | 0.00 | T1 1.5R @ 1060.63 |
| Stop hit — per-position SL triggered | 2026-04-29 14:45:00 | 1055.05 | 1056.56 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1049.50 | 1044.65 | 0.00 | ORB-long ORB[1036.60,1045.00] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2026-05-05 10:30:00 | 1046.78 | 1048.58 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 1060.90 | 1054.76 | 0.00 | ORB-long ORB[1048.30,1058.80] vol=1.8x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 1056.85 | 1056.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 09:30:00 | 1062.60 | 2025-05-19 09:35:00 | 1057.66 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-05-19 09:30:00 | 1062.60 | 2025-05-19 10:50:00 | 1062.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-21 10:20:00 | 1060.40 | 2025-05-21 10:35:00 | 1056.36 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-26 10:10:00 | 1054.30 | 2025-05-26 10:25:00 | 1051.41 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-27 09:30:00 | 1040.40 | 2025-05-27 09:40:00 | 1043.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-28 09:35:00 | 1060.30 | 2025-05-28 09:55:00 | 1056.08 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-30 11:00:00 | 1033.80 | 2025-05-30 11:25:00 | 1036.16 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-04 10:50:00 | 1058.70 | 2025-06-04 11:30:00 | 1055.57 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1131.90 | 2025-06-09 09:50:00 | 1125.79 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-06-10 11:10:00 | 1121.30 | 2025-06-10 11:20:00 | 1116.74 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-10 11:10:00 | 1121.30 | 2025-06-10 12:25:00 | 1121.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-11 10:05:00 | 1119.70 | 2025-06-11 10:20:00 | 1122.79 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-12 11:10:00 | 1109.70 | 2025-06-12 11:40:00 | 1103.97 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-12 11:10:00 | 1109.70 | 2025-06-12 15:20:00 | 1086.30 | TARGET_HIT | 0.50 | 2.11% |
| SELL | retest1 | 2025-06-19 11:15:00 | 1039.10 | 2025-06-19 11:20:00 | 1042.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-23 11:10:00 | 1053.20 | 2025-06-23 11:20:00 | 1049.29 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-26 09:50:00 | 1099.00 | 2025-06-26 09:55:00 | 1092.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-07-03 09:55:00 | 1076.40 | 2025-07-03 10:15:00 | 1080.35 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-08 11:10:00 | 1076.60 | 2025-07-08 11:40:00 | 1078.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-11 11:15:00 | 1070.20 | 2025-07-11 11:30:00 | 1073.19 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-15 09:55:00 | 1080.50 | 2025-07-15 10:00:00 | 1083.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-17 11:00:00 | 1112.90 | 2025-07-17 11:10:00 | 1110.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-25 10:55:00 | 1020.70 | 2025-07-25 11:15:00 | 1015.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-25 10:55:00 | 1020.70 | 2025-07-25 15:20:00 | 1010.70 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-08-06 09:30:00 | 774.20 | 2025-08-06 09:40:00 | 777.82 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-06 09:30:00 | 774.20 | 2025-08-06 10:00:00 | 774.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 11:05:00 | 777.40 | 2025-08-18 11:10:00 | 779.97 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-08-18 11:05:00 | 777.40 | 2025-08-18 12:20:00 | 779.85 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-19 09:35:00 | 795.80 | 2025-08-19 09:40:00 | 793.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-25 11:00:00 | 798.65 | 2025-08-25 11:25:00 | 796.53 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-08-25 11:00:00 | 798.65 | 2025-08-25 12:25:00 | 798.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:35:00 | 790.40 | 2025-08-26 09:45:00 | 793.01 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-02 09:40:00 | 774.35 | 2025-09-02 09:50:00 | 778.48 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-02 09:40:00 | 774.35 | 2025-09-02 15:20:00 | 786.00 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2025-09-10 09:45:00 | 810.45 | 2025-09-10 10:05:00 | 813.20 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-10 09:45:00 | 810.45 | 2025-09-10 11:45:00 | 812.90 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-12 10:00:00 | 817.00 | 2025-09-12 10:15:00 | 819.64 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-12 10:00:00 | 817.00 | 2025-09-12 12:35:00 | 824.80 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-09-17 09:45:00 | 829.50 | 2025-09-17 10:00:00 | 831.47 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-22 11:10:00 | 864.00 | 2025-09-22 11:40:00 | 866.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-23 10:25:00 | 868.75 | 2025-09-23 11:50:00 | 873.04 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-23 10:25:00 | 868.75 | 2025-09-23 15:20:00 | 890.90 | TARGET_HIT | 0.50 | 2.55% |
| BUY | retest1 | 2025-09-24 10:40:00 | 897.75 | 2025-09-24 11:00:00 | 894.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-01 10:30:00 | 872.20 | 2025-10-01 11:10:00 | 877.13 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-01 10:30:00 | 872.20 | 2025-10-01 15:00:00 | 881.65 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-10-03 10:55:00 | 880.00 | 2025-10-03 11:00:00 | 882.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-06 09:45:00 | 899.20 | 2025-10-06 09:50:00 | 896.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-07 10:40:00 | 882.95 | 2025-10-07 12:00:00 | 885.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-09 09:40:00 | 875.35 | 2025-10-09 09:50:00 | 878.46 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-10 09:35:00 | 864.60 | 2025-10-10 10:55:00 | 860.67 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-10 09:35:00 | 864.60 | 2025-10-10 15:20:00 | 855.55 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-10-14 09:35:00 | 849.15 | 2025-10-14 10:55:00 | 845.81 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-14 09:35:00 | 849.15 | 2025-10-14 15:10:00 | 843.65 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2025-10-20 10:50:00 | 859.50 | 2025-10-20 11:05:00 | 856.16 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-23 09:45:00 | 882.95 | 2025-10-23 09:55:00 | 887.65 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-23 09:45:00 | 882.95 | 2025-10-23 13:30:00 | 884.20 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-10-24 09:40:00 | 896.65 | 2025-10-24 09:50:00 | 901.42 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-24 09:40:00 | 896.65 | 2025-10-24 10:20:00 | 899.50 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-27 10:00:00 | 928.75 | 2025-10-27 10:30:00 | 925.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-31 10:55:00 | 926.40 | 2025-10-31 11:05:00 | 923.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-31 10:55:00 | 926.40 | 2025-10-31 11:30:00 | 926.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:10:00 | 949.15 | 2025-11-03 10:15:00 | 946.11 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-06 09:30:00 | 918.70 | 2025-11-06 09:35:00 | 921.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-11 10:25:00 | 889.30 | 2025-11-11 11:05:00 | 884.67 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-11-11 10:25:00 | 889.30 | 2025-11-11 11:50:00 | 889.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:50:00 | 910.10 | 2025-11-13 10:35:00 | 907.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-14 09:45:00 | 922.25 | 2025-11-14 09:55:00 | 919.84 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-17 10:00:00 | 920.00 | 2025-11-17 10:40:00 | 917.65 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 09:40:00 | 902.55 | 2025-11-21 09:50:00 | 898.91 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-21 09:40:00 | 902.55 | 2025-11-21 14:00:00 | 893.10 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-11-24 10:05:00 | 879.20 | 2025-11-24 10:15:00 | 881.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-01 11:00:00 | 915.75 | 2025-12-01 11:40:00 | 913.61 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-02 09:45:00 | 914.55 | 2025-12-02 11:45:00 | 912.11 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-03 09:40:00 | 888.10 | 2025-12-03 10:05:00 | 883.71 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-03 09:40:00 | 888.10 | 2025-12-03 15:15:00 | 881.00 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2025-12-05 10:05:00 | 893.60 | 2025-12-05 10:10:00 | 898.51 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-12-05 10:05:00 | 893.60 | 2025-12-05 10:25:00 | 893.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 11:00:00 | 889.40 | 2025-12-08 11:40:00 | 885.74 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-08 11:00:00 | 889.40 | 2025-12-08 15:20:00 | 873.45 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2025-12-09 10:00:00 | 857.90 | 2025-12-09 10:05:00 | 861.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-10 09:50:00 | 885.10 | 2025-12-10 10:25:00 | 888.37 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-10 09:50:00 | 885.10 | 2025-12-10 10:30:00 | 885.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:50:00 | 891.50 | 2025-12-11 11:15:00 | 888.53 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-17 10:25:00 | 908.40 | 2025-12-17 10:55:00 | 904.45 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-17 10:25:00 | 908.40 | 2025-12-17 11:45:00 | 908.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:40:00 | 886.95 | 2025-12-18 09:55:00 | 882.27 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-12-18 09:40:00 | 886.95 | 2025-12-18 10:15:00 | 886.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:30:00 | 905.00 | 2025-12-19 10:05:00 | 901.48 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-24 09:30:00 | 943.00 | 2025-12-24 09:55:00 | 940.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-26 11:05:00 | 955.55 | 2025-12-26 11:20:00 | 952.34 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-26 11:05:00 | 955.55 | 2025-12-26 15:20:00 | 942.70 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2025-12-29 11:10:00 | 936.00 | 2025-12-29 11:45:00 | 938.09 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-01 10:00:00 | 960.70 | 2026-01-01 10:25:00 | 964.83 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-01-01 10:00:00 | 960.70 | 2026-01-01 15:20:00 | 986.20 | TARGET_HIT | 0.50 | 2.65% |
| BUY | retest1 | 2026-01-02 10:00:00 | 995.05 | 2026-01-02 10:20:00 | 1000.79 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-01-02 10:00:00 | 995.05 | 2026-01-02 10:50:00 | 995.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:55:00 | 991.00 | 2026-01-07 10:10:00 | 994.91 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-08 10:50:00 | 998.55 | 2026-01-08 11:05:00 | 994.68 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-08 10:50:00 | 998.55 | 2026-01-08 15:20:00 | 980.35 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2026-01-13 10:35:00 | 966.50 | 2026-01-13 11:55:00 | 970.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-01-16 11:15:00 | 973.20 | 2026-01-16 11:25:00 | 975.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-19 09:30:00 | 959.90 | 2026-01-19 09:40:00 | 956.39 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-30 09:30:00 | 839.30 | 2026-01-30 09:40:00 | 835.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-04 10:20:00 | 840.50 | 2026-02-04 10:25:00 | 837.34 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-05 09:40:00 | 842.05 | 2026-02-05 10:20:00 | 845.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-09 09:45:00 | 864.15 | 2026-02-09 09:50:00 | 860.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-10 10:05:00 | 853.20 | 2026-02-10 10:10:00 | 855.97 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-13 10:20:00 | 831.75 | 2026-02-13 10:30:00 | 834.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-16 10:50:00 | 846.35 | 2026-02-16 11:30:00 | 844.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-17 11:05:00 | 853.95 | 2026-02-17 11:10:00 | 856.66 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 11:05:00 | 853.95 | 2026-02-17 15:20:00 | 863.60 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-02-19 10:50:00 | 854.15 | 2026-02-19 11:15:00 | 851.24 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 10:50:00 | 854.15 | 2026-02-19 11:25:00 | 854.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:35:00 | 850.15 | 2026-02-20 12:15:00 | 855.09 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-20 09:35:00 | 850.15 | 2026-02-20 13:25:00 | 850.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 846.75 | 2026-02-23 11:05:00 | 843.69 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-23 11:00:00 | 846.75 | 2026-02-23 11:15:00 | 846.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 10:25:00 | 824.95 | 2026-02-24 10:40:00 | 827.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-12 09:30:00 | 759.00 | 2026-03-12 09:45:00 | 762.39 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-13 09:45:00 | 770.50 | 2026-03-13 09:55:00 | 766.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-13 09:45:00 | 770.50 | 2026-03-13 10:30:00 | 770.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-16 09:30:00 | 772.15 | 2026-03-16 10:10:00 | 768.48 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-17 10:10:00 | 782.20 | 2026-03-17 10:40:00 | 779.26 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1055.05 | 2026-04-29 11:25:00 | 1060.63 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1055.05 | 2026-04-29 14:45:00 | 1055.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:15:00 | 1049.50 | 2026-05-05 10:30:00 | 1046.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-06 09:35:00 | 1060.90 | 2026-05-06 09:55:00 | 1056.85 | STOP_HIT | 1.00 | -0.38% |
