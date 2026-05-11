# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1176.60
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
| PARTIAL | 34 |
| TARGET_HIT | 18 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 67
- **Target hits / Stop hits / Partials:** 18 / 67 / 34
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 12.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 28 | 48.3% | 10 | 30 | 18 | 0.14% | 8.0% |
| BUY @ 2nd Alert (retest1) | 58 | 28 | 48.3% | 10 | 30 | 18 | 0.14% | 8.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 24 | 39.3% | 8 | 37 | 16 | 0.08% | 4.8% |
| SELL @ 2nd Alert (retest1) | 61 | 24 | 39.3% | 8 | 37 | 16 | 0.08% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 119 | 52 | 43.7% | 18 | 67 | 34 | 0.11% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 10:35:00 | 1130.80 | 1135.45 | 0.00 | ORB-short ORB[1134.00,1150.30] vol=1.8x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:05:00 | 1125.86 | 1133.85 | 0.00 | T1 1.5R @ 1125.86 |
| Stop hit — per-position SL triggered | 2025-05-13 11:15:00 | 1130.80 | 1133.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1112.00 | 1116.33 | 0.00 | ORB-short ORB[1114.10,1125.30] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-05-15 09:50:00 | 1114.97 | 1116.03 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 11:05:00 | 1139.60 | 1141.71 | 0.00 | ORB-short ORB[1144.00,1155.20] vol=4.6x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 13:35:00 | 1135.93 | 1140.22 | 0.00 | T1 1.5R @ 1135.93 |
| Target hit | 2025-05-20 15:20:00 | 1128.10 | 1134.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 1147.50 | 1142.26 | 0.00 | ORB-long ORB[1123.80,1136.40] vol=1.7x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-05-21 11:30:00 | 1144.47 | 1142.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:55:00 | 1148.60 | 1142.94 | 0.00 | ORB-long ORB[1124.00,1140.90] vol=2.3x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-05-23 11:50:00 | 1145.80 | 1144.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:35:00 | 1130.00 | 1133.46 | 0.00 | ORB-short ORB[1134.00,1149.80] vol=1.7x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-05-27 10:55:00 | 1132.72 | 1132.71 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:20:00 | 1126.50 | 1127.31 | 0.00 | ORB-short ORB[1129.00,1138.80] vol=1.5x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 1129.67 | 1126.88 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 1101.30 | 1108.28 | 0.00 | ORB-short ORB[1103.00,1117.00] vol=1.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-05-29 11:30:00 | 1104.11 | 1107.86 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:05:00 | 1115.60 | 1110.00 | 0.00 | ORB-long ORB[1099.20,1114.20] vol=1.7x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 1113.16 | 1110.26 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 1122.20 | 1115.47 | 0.00 | ORB-long ORB[1109.40,1115.90] vol=3.3x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-06-06 11:35:00 | 1119.91 | 1116.56 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:35:00 | 1073.80 | 1068.93 | 0.00 | ORB-long ORB[1061.60,1071.50] vol=1.6x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:55:00 | 1077.86 | 1071.73 | 0.00 | T1 1.5R @ 1077.86 |
| Target hit | 2025-06-19 11:40:00 | 1079.80 | 1080.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2025-06-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 11:00:00 | 1135.70 | 1138.75 | 0.00 | ORB-short ORB[1137.60,1147.20] vol=2.2x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 13:20:00 | 1132.36 | 1137.36 | 0.00 | T1 1.5R @ 1132.36 |
| Target hit | 2025-06-27 15:20:00 | 1120.20 | 1130.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1111.80 | 1115.95 | 0.00 | ORB-short ORB[1115.00,1126.70] vol=1.7x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 09:45:00 | 1107.13 | 1113.62 | 0.00 | T1 1.5R @ 1107.13 |
| Target hit | 2025-06-30 15:15:00 | 1099.80 | 1099.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2025-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:00:00 | 1089.50 | 1093.08 | 0.00 | ORB-short ORB[1090.50,1100.70] vol=6.4x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:10:00 | 1086.41 | 1092.68 | 0.00 | T1 1.5R @ 1086.41 |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 1089.50 | 1092.51 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 09:55:00 | 1089.70 | 1091.38 | 0.00 | ORB-short ORB[1090.30,1095.00] vol=1.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-07-04 11:30:00 | 1091.76 | 1090.29 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1104.00 | 1100.64 | 0.00 | ORB-long ORB[1096.30,1102.90] vol=2.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-07-09 11:40:00 | 1102.00 | 1101.41 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:55:00 | 1080.20 | 1074.93 | 0.00 | ORB-long ORB[1070.20,1079.40] vol=1.9x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:25:00 | 1083.42 | 1077.01 | 0.00 | T1 1.5R @ 1083.42 |
| Stop hit — per-position SL triggered | 2025-07-14 11:30:00 | 1080.20 | 1077.52 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 10:40:00 | 1069.40 | 1072.23 | 0.00 | ORB-short ORB[1070.00,1075.70] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-07-15 10:50:00 | 1071.33 | 1071.94 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:55:00 | 1071.80 | 1076.21 | 0.00 | ORB-short ORB[1074.10,1089.90] vol=2.1x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-07-16 11:05:00 | 1074.21 | 1073.02 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:50:00 | 1091.90 | 1086.68 | 0.00 | ORB-long ORB[1080.00,1088.70] vol=3.3x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-07-17 10:55:00 | 1089.79 | 1087.88 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:30:00 | 1100.70 | 1102.26 | 0.00 | ORB-short ORB[1102.50,1108.30] vol=5.2x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 1103.06 | 1101.28 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:40:00 | 1074.40 | 1070.36 | 0.00 | ORB-long ORB[1056.00,1070.00] vol=1.8x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:45:00 | 1079.21 | 1073.89 | 0.00 | T1 1.5R @ 1079.21 |
| Target hit | 2025-07-30 13:20:00 | 1076.30 | 1076.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1073.40 | 1068.92 | 0.00 | ORB-long ORB[1054.50,1069.90] vol=4.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-07-31 11:50:00 | 1070.24 | 1069.45 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:50:00 | 1062.10 | 1066.64 | 0.00 | ORB-short ORB[1066.90,1072.90] vol=2.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-08-04 11:45:00 | 1064.44 | 1065.69 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 1045.00 | 1049.46 | 0.00 | ORB-short ORB[1048.40,1056.50] vol=2.0x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:25:00 | 1042.17 | 1048.34 | 0.00 | T1 1.5R @ 1042.17 |
| Stop hit — per-position SL triggered | 2025-08-07 12:00:00 | 1045.00 | 1047.24 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:50:00 | 1053.40 | 1048.40 | 0.00 | ORB-long ORB[1042.10,1052.40] vol=2.0x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:15:00 | 1057.51 | 1049.30 | 0.00 | T1 1.5R @ 1057.51 |
| Stop hit — per-position SL triggered | 2025-08-11 11:50:00 | 1053.40 | 1051.10 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 1045.20 | 1046.77 | 0.00 | ORB-short ORB[1047.80,1052.90] vol=1.9x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-08-13 10:55:00 | 1047.00 | 1046.74 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:05:00 | 1071.30 | 1072.53 | 0.00 | ORB-short ORB[1071.40,1076.70] vol=2.4x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-08-19 11:25:00 | 1073.06 | 1072.43 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:00:00 | 1088.00 | 1092.39 | 0.00 | ORB-short ORB[1092.80,1104.00] vol=1.9x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-08-21 11:05:00 | 1090.37 | 1092.35 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 1072.30 | 1077.63 | 0.00 | ORB-short ORB[1074.10,1084.50] vol=2.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 1074.81 | 1076.20 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 11:05:00 | 1071.50 | 1074.43 | 0.00 | ORB-short ORB[1074.20,1085.20] vol=1.8x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 12:15:00 | 1068.50 | 1073.52 | 0.00 | T1 1.5R @ 1068.50 |
| Target hit | 2025-08-28 15:20:00 | 1063.30 | 1066.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-08-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1059.50 | 1069.56 | 0.00 | ORB-short ORB[1063.00,1077.50] vol=1.6x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-08-29 10:55:00 | 1062.85 | 1065.13 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:30:00 | 1087.40 | 1081.61 | 0.00 | ORB-long ORB[1074.10,1082.00] vol=2.1x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:35:00 | 1091.06 | 1084.86 | 0.00 | T1 1.5R @ 1091.06 |
| Target hit | 2025-09-02 12:00:00 | 1089.70 | 1090.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2025-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1089.20 | 1085.83 | 0.00 | ORB-long ORB[1082.10,1088.90] vol=1.6x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:20:00 | 1092.83 | 1086.80 | 0.00 | T1 1.5R @ 1092.83 |
| Target hit | 2025-09-10 15:20:00 | 1101.70 | 1092.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1107.00 | 1101.95 | 0.00 | ORB-long ORB[1097.20,1104.80] vol=2.1x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-09-11 10:35:00 | 1104.44 | 1102.37 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 1094.40 | 1097.47 | 0.00 | ORB-short ORB[1096.80,1104.40] vol=3.0x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 11:10:00 | 1092.33 | 1097.20 | 0.00 | T1 1.5R @ 1092.33 |
| Stop hit — per-position SL triggered | 2025-09-15 11:30:00 | 1094.40 | 1095.72 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 10:35:00 | 1097.00 | 1099.20 | 0.00 | ORB-short ORB[1097.60,1105.40] vol=2.6x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 13:30:00 | 1093.84 | 1097.43 | 0.00 | T1 1.5R @ 1093.84 |
| Target hit | 2025-09-16 15:20:00 | 1092.30 | 1094.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:05:00 | 1134.10 | 1129.28 | 0.00 | ORB-long ORB[1119.90,1127.90] vol=3.0x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1131.72 | 1129.56 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 1123.40 | 1124.43 | 0.00 | ORB-short ORB[1123.80,1129.60] vol=3.1x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:40:00 | 1120.18 | 1124.05 | 0.00 | T1 1.5R @ 1120.18 |
| Stop hit — per-position SL triggered | 2025-09-24 09:55:00 | 1123.40 | 1123.89 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:40:00 | 1133.60 | 1141.13 | 0.00 | ORB-short ORB[1138.00,1146.50] vol=1.8x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-09-25 11:00:00 | 1136.41 | 1140.41 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:15:00 | 1130.20 | 1136.02 | 0.00 | ORB-short ORB[1135.70,1144.90] vol=2.0x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 1126.88 | 1135.34 | 0.00 | T1 1.5R @ 1126.88 |
| Stop hit — per-position SL triggered | 2025-10-07 11:55:00 | 1130.20 | 1133.43 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1122.10 | 1119.24 | 0.00 | ORB-long ORB[1108.40,1119.00] vol=6.2x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-10-10 11:05:00 | 1119.72 | 1119.48 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1114.10 | 1116.73 | 0.00 | ORB-short ORB[1115.00,1122.70] vol=3.0x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-10-14 13:25:00 | 1116.62 | 1115.46 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 1157.40 | 1152.31 | 0.00 | ORB-long ORB[1142.50,1152.40] vol=1.6x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:10:00 | 1162.55 | 1156.58 | 0.00 | T1 1.5R @ 1162.55 |
| Target hit | 2025-10-17 15:20:00 | 1166.50 | 1163.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1155.00 | 1158.97 | 0.00 | ORB-short ORB[1156.30,1164.10] vol=2.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-11-12 10:25:00 | 1157.27 | 1158.22 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 1160.50 | 1157.91 | 0.00 | ORB-long ORB[1146.80,1158.80] vol=4.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-11-14 11:20:00 | 1157.65 | 1158.42 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:45:00 | 1159.50 | 1166.32 | 0.00 | ORB-short ORB[1162.00,1172.00] vol=1.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-11-17 11:10:00 | 1162.05 | 1165.70 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:35:00 | 1159.90 | 1166.79 | 0.00 | ORB-short ORB[1173.20,1178.70] vol=4.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-11-18 10:45:00 | 1162.46 | 1166.03 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:25:00 | 1176.30 | 1173.32 | 0.00 | ORB-long ORB[1163.00,1169.70] vol=1.9x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1173.87 | 1174.30 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 1159.90 | 1162.38 | 0.00 | ORB-short ORB[1160.20,1171.00] vol=2.6x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:25:00 | 1156.30 | 1161.77 | 0.00 | T1 1.5R @ 1156.30 |
| Target hit | 2025-12-08 15:20:00 | 1144.10 | 1152.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 1147.00 | 1142.97 | 0.00 | ORB-long ORB[1136.10,1146.00] vol=2.0x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:15:00 | 1150.48 | 1144.14 | 0.00 | T1 1.5R @ 1150.48 |
| Target hit | 2025-12-11 13:55:00 | 1148.30 | 1148.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — SELL (started 2025-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:35:00 | 1141.80 | 1143.85 | 0.00 | ORB-short ORB[1142.10,1147.30] vol=4.2x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 1144.78 | 1143.19 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:40:00 | 1160.60 | 1149.72 | 0.00 | ORB-long ORB[1140.10,1149.00] vol=2.2x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:45:00 | 1164.31 | 1150.87 | 0.00 | T1 1.5R @ 1164.31 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 1160.60 | 1155.22 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 10:45:00 | 1180.70 | 1178.17 | 0.00 | ORB-long ORB[1170.00,1177.20] vol=7.4x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:10:00 | 1184.26 | 1179.39 | 0.00 | T1 1.5R @ 1184.26 |
| Stop hit — per-position SL triggered | 2025-12-17 12:00:00 | 1180.70 | 1180.37 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 1183.60 | 1177.91 | 0.00 | ORB-long ORB[1174.50,1178.80] vol=1.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-12-23 11:40:00 | 1181.46 | 1179.89 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:40:00 | 1180.60 | 1182.76 | 0.00 | ORB-short ORB[1182.00,1189.50] vol=4.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:55:00 | 1177.63 | 1182.28 | 0.00 | T1 1.5R @ 1177.63 |
| Target hit | 2025-12-24 14:55:00 | 1179.50 | 1178.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 1178.60 | 1176.72 | 0.00 | ORB-long ORB[1170.50,1177.20] vol=2.4x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1176.21 | 1176.80 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:20:00 | 1179.50 | 1183.07 | 0.00 | ORB-short ORB[1187.50,1195.50] vol=5.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-01-01 10:25:00 | 1181.78 | 1183.01 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:25:00 | 1180.90 | 1175.30 | 0.00 | ORB-long ORB[1165.00,1177.10] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:35:00 | 1185.10 | 1177.64 | 0.00 | T1 1.5R @ 1185.10 |
| Stop hit — per-position SL triggered | 2026-01-05 14:35:00 | 1180.90 | 1181.30 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 1198.90 | 1201.95 | 0.00 | ORB-short ORB[1201.00,1212.00] vol=1.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1201.43 | 1201.79 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:40:00 | 1182.50 | 1182.15 | 0.00 | ORB-long ORB[1173.50,1182.10] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:45:00 | 1186.94 | 1182.65 | 0.00 | T1 1.5R @ 1186.94 |
| Stop hit — per-position SL triggered | 2026-01-12 11:05:00 | 1182.50 | 1182.94 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:05:00 | 1187.50 | 1181.62 | 0.00 | ORB-long ORB[1166.30,1181.40] vol=1.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:10:00 | 1192.07 | 1182.31 | 0.00 | T1 1.5R @ 1192.07 |
| Target hit | 2026-01-16 11:55:00 | 1190.10 | 1190.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1180.40 | 1178.09 | 0.00 | ORB-long ORB[1163.70,1172.50] vol=1.7x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-01-22 11:50:00 | 1177.26 | 1178.37 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:00:00 | 1173.90 | 1182.20 | 0.00 | ORB-short ORB[1174.10,1183.00] vol=3.1x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:10:00 | 1168.98 | 1179.43 | 0.00 | T1 1.5R @ 1168.98 |
| Stop hit — per-position SL triggered | 2026-01-23 12:25:00 | 1173.90 | 1178.92 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:05:00 | 1101.50 | 1114.03 | 0.00 | ORB-short ORB[1115.70,1131.40] vol=2.7x ATR=3.01 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 1104.51 | 1112.61 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 11:10:00 | 1133.00 | 1122.41 | 0.00 | ORB-long ORB[1100.10,1114.30] vol=2.1x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:15:00 | 1137.70 | 1123.43 | 0.00 | T1 1.5R @ 1137.70 |
| Stop hit — per-position SL triggered | 2026-01-30 11:20:00 | 1133.00 | 1123.76 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:45:00 | 1147.60 | 1136.07 | 0.00 | ORB-long ORB[1125.90,1139.00] vol=2.1x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-02-01 11:00:00 | 1143.60 | 1137.22 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 1104.80 | 1094.08 | 0.00 | ORB-long ORB[1084.00,1094.00] vol=2.2x ATR=8.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:35:00 | 1117.75 | 1104.74 | 0.00 | T1 1.5R @ 1117.75 |
| Target hit | 2026-02-02 15:20:00 | 1127.10 | 1114.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2026-02-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 11:00:00 | 1169.40 | 1162.18 | 0.00 | ORB-long ORB[1145.90,1158.00] vol=2.0x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-02-04 11:25:00 | 1167.04 | 1162.64 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:10:00 | 1148.50 | 1154.03 | 0.00 | ORB-short ORB[1149.00,1163.60] vol=3.4x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-05 13:50:00 | 1151.28 | 1151.31 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 1159.60 | 1162.06 | 0.00 | ORB-short ORB[1161.10,1167.80] vol=3.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 1156.63 | 1160.43 | 0.00 | T1 1.5R @ 1156.63 |
| Stop hit — per-position SL triggered | 2026-02-10 12:00:00 | 1159.60 | 1159.60 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1139.80 | 1144.02 | 0.00 | ORB-short ORB[1144.30,1154.00] vol=1.5x ATR=2.66 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 1142.46 | 1143.25 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1158.50 | 1152.53 | 0.00 | ORB-long ORB[1145.80,1154.50] vol=2.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 1156.24 | 1153.43 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1173.00 | 1176.29 | 0.00 | ORB-short ORB[1175.70,1185.00] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 1175.03 | 1176.25 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1135.30 | 1145.62 | 0.00 | ORB-short ORB[1146.50,1159.50] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 1138.10 | 1144.22 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 1075.00 | 1065.52 | 0.00 | ORB-long ORB[1048.00,1060.80] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:30:00 | 1080.49 | 1069.79 | 0.00 | T1 1.5R @ 1080.49 |
| Target hit | 2026-03-13 13:15:00 | 1078.20 | 1078.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1072.30 | 1080.61 | 0.00 | ORB-short ORB[1078.10,1093.00] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 1075.90 | 1080.34 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 1100.00 | 1098.08 | 0.00 | ORB-long ORB[1088.00,1096.20] vol=1.8x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 1096.82 | 1097.60 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:15:00 | 1029.90 | 1030.97 | 0.00 | ORB-short ORB[1030.90,1044.60] vol=1.6x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 1033.15 | 1030.91 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:05:00 | 1024.40 | 1019.24 | 0.00 | ORB-long ORB[1007.20,1022.00] vol=2.0x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-02 11:30:00 | 1021.23 | 1019.73 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1090.10 | 1087.07 | 0.00 | ORB-long ORB[1080.60,1088.50] vol=3.0x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-10 12:55:00 | 1087.17 | 1088.95 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1121.40 | 1116.05 | 0.00 | ORB-long ORB[1106.70,1119.90] vol=1.5x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:40:00 | 1126.54 | 1120.05 | 0.00 | T1 1.5R @ 1126.54 |
| Target hit | 2026-04-20 13:55:00 | 1122.10 | 1122.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — BUY (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 1178.10 | 1161.43 | 0.00 | ORB-long ORB[1139.80,1156.00] vol=2.0x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:10:00 | 1184.62 | 1172.03 | 0.00 | T1 1.5R @ 1184.62 |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 1178.10 | 1177.09 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 1192.00 | 1190.21 | 0.00 | ORB-long ORB[1174.00,1190.00] vol=2.4x ATR=4.95 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 1187.05 | 1190.00 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 1157.60 | 1159.96 | 0.00 | ORB-short ORB[1159.00,1166.90] vol=2.2x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:35:00 | 1152.91 | 1159.17 | 0.00 | T1 1.5R @ 1152.91 |
| Target hit | 2026-05-06 14:05:00 | 1152.20 | 1149.81 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-13 10:35:00 | 1130.80 | 2025-05-13 11:05:00 | 1125.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-05-13 10:35:00 | 1130.80 | 2025-05-13 11:15:00 | 1130.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-15 09:45:00 | 1112.00 | 2025-05-15 09:50:00 | 1114.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-20 11:05:00 | 1139.60 | 2025-05-20 13:35:00 | 1135.93 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-05-20 11:05:00 | 1139.60 | 2025-05-20 15:20:00 | 1128.10 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-05-21 11:00:00 | 1147.50 | 2025-05-21 11:30:00 | 1144.47 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-23 10:55:00 | 1148.60 | 2025-05-23 11:50:00 | 1145.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-27 10:35:00 | 1130.00 | 2025-05-27 10:55:00 | 1132.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-28 10:20:00 | 1126.50 | 2025-05-28 12:15:00 | 1129.67 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-29 11:10:00 | 1101.30 | 2025-05-29 11:30:00 | 1104.11 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-02 11:05:00 | 1115.60 | 2025-06-02 11:15:00 | 1113.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-06 11:15:00 | 1122.20 | 2025-06-06 11:35:00 | 1119.91 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-19 09:35:00 | 1073.80 | 2025-06-19 09:55:00 | 1077.86 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-19 09:35:00 | 1073.80 | 2025-06-19 11:40:00 | 1079.80 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-27 11:00:00 | 1135.70 | 2025-06-27 13:20:00 | 1132.36 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-27 11:00:00 | 1135.70 | 2025-06-27 15:20:00 | 1120.20 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-06-30 09:30:00 | 1111.80 | 2025-06-30 09:45:00 | 1107.13 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-30 09:30:00 | 1111.80 | 2025-06-30 15:15:00 | 1099.80 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-07-01 11:00:00 | 1089.50 | 2025-07-01 11:10:00 | 1086.41 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-01 11:00:00 | 1089.50 | 2025-07-01 11:15:00 | 1089.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 09:55:00 | 1089.70 | 2025-07-04 11:30:00 | 1091.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-09 11:15:00 | 1104.00 | 2025-07-09 11:40:00 | 1102.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-14 10:55:00 | 1080.20 | 2025-07-14 11:25:00 | 1083.42 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-14 10:55:00 | 1080.20 | 2025-07-14 11:30:00 | 1080.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-15 10:40:00 | 1069.40 | 2025-07-15 10:50:00 | 1071.33 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-16 09:55:00 | 1071.80 | 2025-07-16 11:05:00 | 1074.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-17 10:50:00 | 1091.90 | 2025-07-17 10:55:00 | 1089.79 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-18 10:30:00 | 1100.70 | 2025-07-18 11:05:00 | 1103.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-30 09:40:00 | 1074.40 | 2025-07-30 10:45:00 | 1079.21 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-30 09:40:00 | 1074.40 | 2025-07-30 13:20:00 | 1076.30 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-07-31 11:15:00 | 1073.40 | 2025-07-31 11:50:00 | 1070.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-04 10:50:00 | 1062.10 | 2025-08-04 11:45:00 | 1064.44 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-07 11:00:00 | 1045.00 | 2025-08-07 11:25:00 | 1042.17 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-08-07 11:00:00 | 1045.00 | 2025-08-07 12:00:00 | 1045.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 10:50:00 | 1053.40 | 2025-08-11 11:15:00 | 1057.51 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-11 10:50:00 | 1053.40 | 2025-08-11 11:50:00 | 1053.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 10:45:00 | 1045.20 | 2025-08-13 10:55:00 | 1047.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-19 11:05:00 | 1071.30 | 2025-08-19 11:25:00 | 1073.06 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-08-21 11:00:00 | 1088.00 | 2025-08-21 11:05:00 | 1090.37 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-26 09:35:00 | 1072.30 | 2025-08-26 09:55:00 | 1074.81 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-28 11:05:00 | 1071.50 | 2025-08-28 12:15:00 | 1068.50 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-28 11:05:00 | 1071.50 | 2025-08-28 15:20:00 | 1063.30 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-08-29 10:00:00 | 1059.50 | 2025-08-29 10:55:00 | 1062.85 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-02 09:30:00 | 1087.40 | 2025-09-02 09:35:00 | 1091.06 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-02 09:30:00 | 1087.40 | 2025-09-02 12:00:00 | 1089.70 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-09-10 10:30:00 | 1089.20 | 2025-09-10 11:20:00 | 1092.83 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-10 10:30:00 | 1089.20 | 2025-09-10 15:20:00 | 1101.70 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-09-11 10:15:00 | 1107.00 | 2025-09-11 10:35:00 | 1104.44 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-15 11:00:00 | 1094.40 | 2025-09-15 11:10:00 | 1092.33 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-09-15 11:00:00 | 1094.40 | 2025-09-15 11:30:00 | 1094.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 10:35:00 | 1097.00 | 2025-09-16 13:30:00 | 1093.84 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-16 10:35:00 | 1097.00 | 2025-09-16 15:20:00 | 1092.30 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-22 11:05:00 | 1134.10 | 2025-09-22 11:15:00 | 1131.72 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-24 09:30:00 | 1123.40 | 2025-09-24 09:40:00 | 1120.18 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-24 09:30:00 | 1123.40 | 2025-09-24 09:55:00 | 1123.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 10:40:00 | 1133.60 | 2025-09-25 11:00:00 | 1136.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-07 11:15:00 | 1130.20 | 2025-10-07 11:20:00 | 1126.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-07 11:15:00 | 1130.20 | 2025-10-07 11:55:00 | 1130.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 11:00:00 | 1122.10 | 2025-10-10 11:05:00 | 1119.72 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1114.10 | 2025-10-14 13:25:00 | 1116.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-17 10:15:00 | 1157.40 | 2025-10-17 11:10:00 | 1162.55 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-17 10:15:00 | 1157.40 | 2025-10-17 15:20:00 | 1166.50 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-11-12 10:00:00 | 1155.00 | 2025-11-12 10:25:00 | 1157.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1160.50 | 2025-11-14 11:20:00 | 1157.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-17 10:45:00 | 1159.50 | 2025-11-17 11:10:00 | 1162.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-18 10:35:00 | 1159.90 | 2025-11-18 10:45:00 | 1162.46 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-20 10:25:00 | 1176.30 | 2025-11-20 11:15:00 | 1173.87 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-08 11:05:00 | 1159.90 | 2025-12-08 11:25:00 | 1156.30 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-08 11:05:00 | 1159.90 | 2025-12-08 15:20:00 | 1144.10 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1147.00 | 2025-12-11 11:15:00 | 1150.48 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1147.00 | 2025-12-11 13:55:00 | 1148.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-12-12 10:35:00 | 1141.80 | 2025-12-12 12:15:00 | 1144.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-15 10:40:00 | 1160.60 | 2025-12-15 10:45:00 | 1164.31 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-15 10:40:00 | 1160.60 | 2025-12-15 11:05:00 | 1160.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 10:45:00 | 1180.70 | 2025-12-17 11:10:00 | 1184.26 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-17 10:45:00 | 1180.70 | 2025-12-17 12:00:00 | 1180.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 10:55:00 | 1183.60 | 2025-12-23 11:40:00 | 1181.46 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-24 10:40:00 | 1180.60 | 2025-12-24 10:55:00 | 1177.63 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-24 10:40:00 | 1180.60 | 2025-12-24 14:55:00 | 1179.50 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-12-29 11:10:00 | 1178.60 | 2025-12-29 11:15:00 | 1176.21 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-01 10:20:00 | 1179.50 | 2026-01-01 10:25:00 | 1181.78 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1180.90 | 2026-01-05 11:35:00 | 1185.10 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1180.90 | 2026-01-05 14:35:00 | 1180.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:10:00 | 1198.90 | 2026-01-08 11:15:00 | 1201.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-12 10:40:00 | 1182.50 | 2026-01-12 10:45:00 | 1186.94 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-12 10:40:00 | 1182.50 | 2026-01-12 11:05:00 | 1182.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:05:00 | 1187.50 | 2026-01-16 10:10:00 | 1192.07 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-16 10:05:00 | 1187.50 | 2026-01-16 11:55:00 | 1190.10 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2026-01-22 11:15:00 | 1180.40 | 2026-01-22 11:50:00 | 1177.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-23 11:00:00 | 1173.90 | 2026-01-23 12:10:00 | 1168.98 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-23 11:00:00 | 1173.90 | 2026-01-23 12:25:00 | 1173.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:05:00 | 1101.50 | 2026-01-29 11:15:00 | 1104.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-30 11:10:00 | 1133.00 | 2026-01-30 11:15:00 | 1137.70 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-30 11:10:00 | 1133.00 | 2026-01-30 11:20:00 | 1133.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 10:45:00 | 1147.60 | 2026-02-01 11:00:00 | 1143.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-02 09:30:00 | 1104.80 | 2026-02-02 11:35:00 | 1117.75 | PARTIAL | 0.50 | 1.17% |
| BUY | retest1 | 2026-02-02 09:30:00 | 1104.80 | 2026-02-02 15:20:00 | 1127.10 | TARGET_HIT | 0.50 | 2.02% |
| BUY | retest1 | 2026-02-04 11:00:00 | 1169.40 | 2026-02-04 11:25:00 | 1167.04 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-05 11:10:00 | 1148.50 | 2026-02-05 13:50:00 | 1151.28 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 11:10:00 | 1156.63 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 12:00:00 | 1159.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:30:00 | 1139.80 | 2026-02-13 10:55:00 | 1142.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-18 11:00:00 | 1158.50 | 2026-02-18 11:25:00 | 1156.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1173.00 | 2026-02-25 11:05:00 | 1175.03 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1135.30 | 2026-02-27 10:30:00 | 1138.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 11:30:00 | 1080.49 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 13:15:00 | 1078.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-16 10:50:00 | 1072.30 | 2026-03-16 11:00:00 | 1075.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 10:25:00 | 1100.00 | 2026-03-17 10:30:00 | 1096.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-24 11:15:00 | 1029.90 | 2026-03-24 11:40:00 | 1033.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-02 11:05:00 | 1024.40 | 2026-04-02 11:30:00 | 1021.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1090.10 | 2026-04-10 12:55:00 | 1087.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 11:40:00 | 1126.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 13:55:00 | 1122.10 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 12:10:00 | 1184.62 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 14:15:00 | 1178.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:40:00 | 1192.00 | 2026-04-24 09:50:00 | 1187.05 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 10:35:00 | 1152.91 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 14:05:00 | 1152.20 | TARGET_HIT | 0.50 | 0.47% |
