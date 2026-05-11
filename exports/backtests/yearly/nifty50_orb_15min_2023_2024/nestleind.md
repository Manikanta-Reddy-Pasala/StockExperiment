# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-06-04 15:25:00 (19704 bars)
- **Last close:** 1216.08
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 13 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 74
- **Target hits / Stop hits / Partials:** 13 / 74 / 30
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 1.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 17 | 31.5% | 5 | 37 | 12 | -0.00% | -0.2% |
| BUY @ 2nd Alert (retest1) | 54 | 17 | 31.5% | 5 | 37 | 12 | -0.00% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 26 | 41.3% | 8 | 37 | 18 | 0.03% | 1.8% |
| SELL @ 2nd Alert (retest1) | 63 | 26 | 41.3% | 8 | 37 | 18 | 0.03% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 117 | 43 | 36.8% | 13 | 74 | 30 | 0.01% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 11:05:00 | 1092.99 | 1091.67 | 0.00 | ORB-long ORB[1086.99,1092.87] vol=11.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2023-05-17 11:10:00 | 1091.27 | 1091.49 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:10:00 | 1080.00 | 1085.72 | 0.00 | ORB-short ORB[1085.00,1090.00] vol=1.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:30:00 | 1076.93 | 1084.78 | 0.00 | T1 1.5R @ 1076.93 |
| Stop hit — per-position SL triggered | 2023-05-19 10:45:00 | 1080.00 | 1083.17 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:55:00 | 1088.15 | 1086.75 | 0.00 | ORB-long ORB[1076.72,1087.40] vol=1.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-05-31 11:00:00 | 1086.46 | 1086.67 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 11:00:00 | 1095.16 | 1099.03 | 0.00 | ORB-short ORB[1096.66,1102.79] vol=8.1x ATR=2.23 |
| Stop hit — per-position SL triggered | 2023-06-05 12:00:00 | 1097.39 | 1097.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:05:00 | 1109.24 | 1100.37 | 0.00 | ORB-long ORB[1087.49,1101.47] vol=1.5x ATR=2.90 |
| Stop hit — per-position SL triggered | 2023-06-07 10:35:00 | 1106.34 | 1102.46 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:40:00 | 1129.77 | 1124.93 | 0.00 | ORB-long ORB[1114.62,1126.13] vol=1.7x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:00:00 | 1133.32 | 1127.59 | 0.00 | T1 1.5R @ 1133.32 |
| Stop hit — per-position SL triggered | 2023-06-13 10:20:00 | 1129.77 | 1128.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 1132.39 | 1129.55 | 0.00 | ORB-long ORB[1125.98,1131.22] vol=2.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-06-14 11:05:00 | 1130.96 | 1129.58 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 11:05:00 | 1137.50 | 1140.37 | 0.00 | ORB-short ORB[1138.06,1146.64] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-06-20 11:10:00 | 1138.94 | 1140.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:10:00 | 1142.60 | 1145.33 | 0.00 | ORB-short ORB[1146.50,1153.48] vol=7.2x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-06-21 11:25:00 | 1144.22 | 1144.65 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 11:10:00 | 1133.78 | 1137.45 | 0.00 | ORB-short ORB[1136.46,1144.54] vol=1.8x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:35:00 | 1131.57 | 1136.45 | 0.00 | T1 1.5R @ 1131.57 |
| Stop hit — per-position SL triggered | 2023-06-22 11:55:00 | 1133.78 | 1136.13 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 10:30:00 | 1130.86 | 1127.54 | 0.00 | ORB-long ORB[1122.68,1127.49] vol=1.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-06-23 10:35:00 | 1128.85 | 1127.64 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 11:05:00 | 1130.00 | 1135.89 | 0.00 | ORB-short ORB[1131.57,1143.50] vol=5.9x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-06-27 11:10:00 | 1131.93 | 1135.73 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:45:00 | 1138.88 | 1142.31 | 0.00 | ORB-short ORB[1143.08,1147.50] vol=2.2x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 11:10:00 | 1136.49 | 1140.77 | 0.00 | T1 1.5R @ 1136.49 |
| Target hit | 2023-07-03 13:10:00 | 1134.84 | 1134.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2023-07-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:55:00 | 1142.50 | 1137.94 | 0.00 | ORB-long ORB[1129.00,1137.16] vol=2.3x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 11:05:00 | 1145.37 | 1138.86 | 0.00 | T1 1.5R @ 1145.37 |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 1142.50 | 1139.54 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:30:00 | 1159.75 | 1156.04 | 0.00 | ORB-long ORB[1145.66,1157.00] vol=4.5x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:35:00 | 1164.55 | 1159.01 | 0.00 | T1 1.5R @ 1164.55 |
| Stop hit — per-position SL triggered | 2023-07-06 10:15:00 | 1159.75 | 1161.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:45:00 | 1141.71 | 1136.05 | 0.00 | ORB-long ORB[1127.52,1137.50] vol=1.6x ATR=2.71 |
| Stop hit — per-position SL triggered | 2023-07-11 10:00:00 | 1139.00 | 1137.02 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:05:00 | 1149.33 | 1155.19 | 0.00 | ORB-short ORB[1154.01,1159.50] vol=1.6x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:35:00 | 1146.29 | 1154.20 | 0.00 | T1 1.5R @ 1146.29 |
| Stop hit — per-position SL triggered | 2023-07-18 12:00:00 | 1149.33 | 1153.68 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:50:00 | 1141.09 | 1146.01 | 0.00 | ORB-short ORB[1142.50,1149.80] vol=3.1x ATR=2.35 |
| Stop hit — per-position SL triggered | 2023-07-19 11:10:00 | 1143.44 | 1145.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 10:40:00 | 1142.50 | 1144.53 | 0.00 | ORB-short ORB[1142.99,1152.43] vol=1.9x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:45:00 | 1139.25 | 1143.71 | 0.00 | T1 1.5R @ 1139.25 |
| Stop hit — per-position SL triggered | 2023-07-24 12:20:00 | 1142.50 | 1142.94 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 11:10:00 | 1137.00 | 1144.29 | 0.00 | ORB-short ORB[1144.47,1157.00] vol=1.6x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-07-25 11:25:00 | 1138.75 | 1143.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 11:00:00 | 1113.50 | 1132.44 | 0.00 | ORB-short ORB[1132.37,1145.00] vol=6.7x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:05:00 | 1106.86 | 1127.65 | 0.00 | T1 1.5R @ 1106.86 |
| Stop hit — per-position SL triggered | 2023-07-27 11:10:00 | 1113.50 | 1126.30 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:25:00 | 1113.99 | 1117.21 | 0.00 | ORB-short ORB[1118.45,1124.99] vol=1.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 11:00:00 | 1111.15 | 1116.47 | 0.00 | T1 1.5R @ 1111.15 |
| Stop hit — per-position SL triggered | 2023-08-07 12:55:00 | 1113.99 | 1113.72 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:10:00 | 1111.51 | 1115.38 | 0.00 | ORB-short ORB[1117.61,1124.95] vol=3.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 11:30:00 | 1109.07 | 1112.66 | 0.00 | T1 1.5R @ 1109.07 |
| Target hit | 2023-08-08 14:30:00 | 1110.15 | 1109.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2023-08-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:45:00 | 1110.45 | 1111.69 | 0.00 | ORB-short ORB[1111.44,1117.81] vol=3.3x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-08-09 10:55:00 | 1111.98 | 1111.67 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 1100.00 | 1103.02 | 0.00 | ORB-short ORB[1104.00,1112.23] vol=1.9x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:30:00 | 1096.99 | 1102.50 | 0.00 | T1 1.5R @ 1096.99 |
| Target hit | 2023-08-10 13:15:00 | 1099.21 | 1099.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 1083.50 | 1088.57 | 0.00 | ORB-short ORB[1089.53,1101.42] vol=3.6x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:55:00 | 1081.23 | 1087.31 | 0.00 | T1 1.5R @ 1081.23 |
| Stop hit — per-position SL triggered | 2023-08-17 12:10:00 | 1083.50 | 1086.95 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 11:15:00 | 1099.55 | 1096.24 | 0.00 | ORB-long ORB[1086.77,1094.20] vol=2.5x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 13:30:00 | 1101.88 | 1099.30 | 0.00 | T1 1.5R @ 1101.88 |
| Stop hit — per-position SL triggered | 2023-08-21 14:30:00 | 1099.55 | 1099.69 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:40:00 | 1104.24 | 1100.62 | 0.00 | ORB-long ORB[1095.30,1101.72] vol=3.1x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-08-22 09:45:00 | 1101.99 | 1100.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 09:50:00 | 1098.20 | 1103.97 | 0.00 | ORB-short ORB[1101.76,1111.25] vol=1.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2023-08-23 10:00:00 | 1100.38 | 1103.30 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:05:00 | 1096.04 | 1106.38 | 0.00 | ORB-short ORB[1108.62,1114.45] vol=1.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:20:00 | 1093.28 | 1105.51 | 0.00 | T1 1.5R @ 1093.28 |
| Target hit | 2023-08-31 14:45:00 | 1093.90 | 1093.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2023-09-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:10:00 | 1087.55 | 1090.91 | 0.00 | ORB-short ORB[1089.50,1098.99] vol=1.5x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 10:45:00 | 1084.78 | 1089.55 | 0.00 | T1 1.5R @ 1084.78 |
| Target hit | 2023-09-04 13:00:00 | 1086.25 | 1086.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:15:00 | 1096.36 | 1092.66 | 0.00 | ORB-long ORB[1087.73,1093.59] vol=1.6x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 12:10:00 | 1099.10 | 1093.98 | 0.00 | T1 1.5R @ 1099.10 |
| Stop hit — per-position SL triggered | 2023-09-05 12:35:00 | 1096.36 | 1094.49 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 11:05:00 | 1102.45 | 1097.04 | 0.00 | ORB-long ORB[1088.73,1098.50] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2023-09-06 12:50:00 | 1100.48 | 1098.95 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:30:00 | 1101.80 | 1099.48 | 0.00 | ORB-long ORB[1093.67,1101.10] vol=1.5x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 09:55:00 | 1104.35 | 1101.29 | 0.00 | T1 1.5R @ 1104.35 |
| Target hit | 2023-09-11 10:20:00 | 1102.14 | 1102.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2023-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 11:05:00 | 1108.06 | 1103.78 | 0.00 | ORB-long ORB[1100.86,1106.71] vol=2.2x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 11:10:00 | 1110.86 | 1104.08 | 0.00 | T1 1.5R @ 1110.86 |
| Target hit | 2023-09-12 15:20:00 | 1115.02 | 1112.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2023-09-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 09:45:00 | 1124.98 | 1119.11 | 0.00 | ORB-long ORB[1110.39,1117.25] vol=2.0x ATR=2.74 |
| Stop hit — per-position SL triggered | 2023-09-13 10:25:00 | 1122.24 | 1122.77 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:35:00 | 1119.20 | 1115.98 | 0.00 | ORB-long ORB[1110.83,1114.75] vol=2.0x ATR=2.05 |
| Stop hit — per-position SL triggered | 2023-09-14 09:55:00 | 1117.15 | 1117.12 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:55:00 | 1139.84 | 1141.16 | 0.00 | ORB-short ORB[1143.01,1150.28] vol=2.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-09-27 11:10:00 | 1141.83 | 1141.16 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-10-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 09:55:00 | 1164.22 | 1158.56 | 0.00 | ORB-long ORB[1147.77,1159.95] vol=1.5x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-10-13 10:10:00 | 1161.36 | 1159.72 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:50:00 | 1169.75 | 1167.41 | 0.00 | ORB-long ORB[1160.00,1167.00] vol=2.2x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 1167.80 | 1167.46 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:50:00 | 1166.33 | 1161.27 | 0.00 | ORB-long ORB[1155.71,1164.74] vol=8.9x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-10-19 10:55:00 | 1164.07 | 1161.93 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 11:05:00 | 1222.10 | 1216.96 | 0.00 | ORB-long ORB[1207.66,1221.77] vol=1.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-10-25 11:15:00 | 1220.09 | 1217.12 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 11:00:00 | 1182.13 | 1195.92 | 0.00 | ORB-short ORB[1196.66,1213.03] vol=1.6x ATR=2.69 |
| Stop hit — per-position SL triggered | 2023-10-26 11:05:00 | 1184.82 | 1195.52 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:30:00 | 1207.35 | 1211.84 | 0.00 | ORB-short ORB[1210.31,1216.50] vol=1.6x ATR=2.76 |
| Stop hit — per-position SL triggered | 2023-10-31 11:20:00 | 1210.11 | 1210.69 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-11-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 11:05:00 | 1195.77 | 1202.67 | 0.00 | ORB-short ORB[1204.78,1210.01] vol=1.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 13:10:00 | 1193.02 | 1199.32 | 0.00 | T1 1.5R @ 1193.02 |
| Stop hit — per-position SL triggered | 2023-11-01 14:25:00 | 1195.77 | 1197.11 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-11-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 10:25:00 | 1190.90 | 1193.81 | 0.00 | ORB-short ORB[1197.50,1204.70] vol=9.1x ATR=2.66 |
| Stop hit — per-position SL triggered | 2023-11-02 11:05:00 | 1193.56 | 1193.55 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:05:00 | 1211.20 | 1207.30 | 0.00 | ORB-long ORB[1203.49,1209.76] vol=1.7x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-11-16 11:30:00 | 1209.36 | 1208.16 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:55:00 | 1218.63 | 1216.24 | 0.00 | ORB-long ORB[1203.23,1212.40] vol=1.7x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-11-17 11:05:00 | 1216.29 | 1216.54 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:30:00 | 1228.50 | 1225.78 | 0.00 | ORB-long ORB[1218.35,1227.50] vol=1.5x ATR=2.05 |
| Stop hit — per-position SL triggered | 2023-11-23 09:35:00 | 1226.45 | 1226.14 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-11-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:35:00 | 1216.11 | 1216.80 | 0.00 | ORB-short ORB[1216.17,1223.64] vol=1.5x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 11:05:00 | 1213.36 | 1216.26 | 0.00 | T1 1.5R @ 1213.36 |
| Target hit | 2023-11-24 15:20:00 | 1207.20 | 1209.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2023-11-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:55:00 | 1200.19 | 1203.56 | 0.00 | ORB-short ORB[1202.26,1209.30] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 1202.12 | 1202.37 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 1246.24 | 1241.27 | 0.00 | ORB-long ORB[1235.10,1244.50] vol=1.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2023-12-06 10:00:00 | 1243.60 | 1243.61 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 11:00:00 | 1252.22 | 1245.53 | 0.00 | ORB-long ORB[1231.95,1246.10] vol=1.5x ATR=2.48 |
| Stop hit — per-position SL triggered | 2023-12-11 11:55:00 | 1249.74 | 1246.70 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-12-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 11:05:00 | 1244.99 | 1245.26 | 0.00 | ORB-short ORB[1245.02,1250.09] vol=1.6x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 11:30:00 | 1241.84 | 1245.10 | 0.00 | T1 1.5R @ 1241.84 |
| Stop hit — per-position SL triggered | 2023-12-13 13:40:00 | 1244.99 | 1243.95 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-18 09:55:00 | 1219.00 | 1221.98 | 0.00 | ORB-short ORB[1220.76,1229.03] vol=2.1x ATR=2.43 |
| Target hit | 2023-12-18 15:20:00 | 1217.50 | 1219.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 1242.00 | 1235.63 | 0.00 | ORB-long ORB[1225.55,1239.75] vol=2.3x ATR=3.37 |
| Stop hit — per-position SL triggered | 2023-12-19 09:45:00 | 1238.63 | 1236.76 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 1297.62 | 1293.48 | 0.00 | ORB-long ORB[1288.76,1294.74] vol=1.9x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:55:00 | 1301.18 | 1296.31 | 0.00 | T1 1.5R @ 1301.18 |
| Stop hit — per-position SL triggered | 2023-12-28 10:05:00 | 1297.62 | 1296.76 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 11:10:00 | 1346.79 | 1356.68 | 0.00 | ORB-short ORB[1363.08,1370.87] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-01-03 11:20:00 | 1349.26 | 1356.52 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 10:45:00 | 1298.03 | 1308.32 | 0.00 | ORB-short ORB[1312.53,1320.15] vol=1.7x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-01-09 10:50:00 | 1300.78 | 1307.90 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 1252.08 | 1262.38 | 0.00 | ORB-short ORB[1260.90,1275.00] vol=1.9x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-01-18 09:50:00 | 1255.64 | 1261.19 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 10:45:00 | 1249.88 | 1252.56 | 0.00 | ORB-short ORB[1250.50,1256.00] vol=3.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-01-30 10:50:00 | 1252.31 | 1252.53 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 11:15:00 | 1224.55 | 1227.90 | 0.00 | ORB-short ORB[1231.18,1238.97] vol=5.4x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-02-06 11:30:00 | 1226.80 | 1227.65 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:35:00 | 1236.33 | 1233.91 | 0.00 | ORB-long ORB[1228.78,1235.00] vol=1.6x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-02-07 09:40:00 | 1233.66 | 1233.63 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 09:45:00 | 1231.00 | 1243.82 | 0.00 | ORB-short ORB[1239.20,1256.93] vol=1.5x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-02-08 10:35:00 | 1236.18 | 1235.11 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:40:00 | 1221.38 | 1224.05 | 0.00 | ORB-short ORB[1222.50,1228.50] vol=1.7x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-02-14 10:00:00 | 1224.34 | 1223.79 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 11:00:00 | 1282.83 | 1280.01 | 0.00 | ORB-long ORB[1275.97,1282.50] vol=2.3x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 13:10:00 | 1286.76 | 1282.78 | 0.00 | T1 1.5R @ 1286.76 |
| Target hit | 2024-02-23 15:20:00 | 1289.20 | 1285.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2024-02-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 09:50:00 | 1290.50 | 1283.28 | 0.00 | ORB-long ORB[1273.53,1287.35] vol=2.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-02-29 10:10:00 | 1286.58 | 1286.53 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:45:00 | 1281.47 | 1289.86 | 0.00 | ORB-short ORB[1288.75,1297.60] vol=2.7x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 11:20:00 | 1277.57 | 1286.77 | 0.00 | T1 1.5R @ 1277.57 |
| Stop hit — per-position SL triggered | 2024-03-05 11:40:00 | 1281.47 | 1286.06 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 09:40:00 | 1282.53 | 1274.80 | 0.00 | ORB-long ORB[1264.70,1276.18] vol=1.7x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-03-06 09:50:00 | 1279.34 | 1276.15 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 10:05:00 | 1287.50 | 1282.37 | 0.00 | ORB-long ORB[1274.28,1285.28] vol=3.4x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:15:00 | 1292.89 | 1284.50 | 0.00 | T1 1.5R @ 1292.89 |
| Target hit | 2024-03-11 11:45:00 | 1299.97 | 1300.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — SELL (started 2024-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:05:00 | 1286.45 | 1291.98 | 0.00 | ORB-short ORB[1291.33,1307.53] vol=3.7x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-03-12 10:20:00 | 1291.13 | 1291.17 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-13 09:40:00 | 1316.00 | 1307.80 | 0.00 | ORB-long ORB[1294.00,1311.75] vol=3.1x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-03-13 09:55:00 | 1311.71 | 1311.32 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:15:00 | 1300.08 | 1305.78 | 0.00 | ORB-short ORB[1307.00,1314.43] vol=2.0x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-03-15 11:30:00 | 1303.10 | 1305.40 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 11:05:00 | 1268.30 | 1261.23 | 0.00 | ORB-long ORB[1249.03,1264.50] vol=1.6x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 13:05:00 | 1274.38 | 1264.84 | 0.00 | T1 1.5R @ 1274.38 |
| Target hit | 2024-03-20 15:20:00 | 1276.10 | 1269.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 11:00:00 | 1285.68 | 1289.52 | 0.00 | ORB-short ORB[1286.40,1295.28] vol=1.5x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-03-27 11:15:00 | 1288.65 | 1289.11 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 1300.00 | 1293.81 | 0.00 | ORB-long ORB[1283.53,1295.45] vol=1.8x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-03-28 09:50:00 | 1296.86 | 1294.89 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:40:00 | 1306.63 | 1300.35 | 0.00 | ORB-long ORB[1291.50,1304.10] vol=2.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-04-02 10:50:00 | 1303.70 | 1300.65 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-04-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:35:00 | 1264.75 | 1271.16 | 0.00 | ORB-short ORB[1274.00,1285.00] vol=1.8x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 10:50:00 | 1260.26 | 1269.90 | 0.00 | T1 1.5R @ 1260.26 |
| Stop hit — per-position SL triggered | 2024-04-04 11:10:00 | 1264.75 | 1268.88 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-04-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:45:00 | 1261.18 | 1267.57 | 0.00 | ORB-short ORB[1262.60,1273.63] vol=1.6x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 11:05:00 | 1257.46 | 1266.02 | 0.00 | T1 1.5R @ 1257.46 |
| Target hit | 2024-04-08 15:20:00 | 1249.83 | 1254.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 1275.90 | 1267.12 | 0.00 | ORB-long ORB[1258.10,1267.95] vol=2.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-04-12 09:40:00 | 1272.11 | 1267.58 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:40:00 | 1290.18 | 1283.39 | 0.00 | ORB-long ORB[1271.30,1288.50] vol=2.2x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-04-16 09:45:00 | 1285.84 | 1283.60 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:45:00 | 1215.93 | 1221.41 | 0.00 | ORB-short ORB[1218.25,1230.53] vol=2.5x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-04-22 11:20:00 | 1219.56 | 1220.61 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 1264.55 | 1262.30 | 0.00 | ORB-long ORB[1253.75,1263.93] vol=3.2x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 1260.04 | 1262.52 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-04-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:25:00 | 1263.33 | 1257.00 | 0.00 | ORB-long ORB[1248.90,1259.97] vol=2.1x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-04-25 10:30:00 | 1259.48 | 1257.32 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:45:00 | 1256.50 | 1252.54 | 0.00 | ORB-long ORB[1245.00,1253.70] vol=1.8x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 11:05:00 | 1261.05 | 1254.14 | 0.00 | T1 1.5R @ 1261.05 |
| Stop hit — per-position SL triggered | 2024-04-29 11:25:00 | 1256.50 | 1254.49 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-05-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:20:00 | 1243.43 | 1251.03 | 0.00 | ORB-short ORB[1253.50,1261.53] vol=1.6x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-03 10:30:00 | 1246.88 | 1250.40 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-05-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:25:00 | 1256.05 | 1258.54 | 0.00 | ORB-short ORB[1260.50,1274.95] vol=4.0x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-05-09 10:35:00 | 1259.70 | 1258.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-17 11:05:00 | 1092.99 | 2023-05-17 11:10:00 | 1091.27 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-05-19 10:10:00 | 1080.00 | 2023-05-19 10:30:00 | 1076.93 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-19 10:10:00 | 1080.00 | 2023-05-19 10:45:00 | 1080.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-31 10:55:00 | 1088.15 | 2023-05-31 11:00:00 | 1086.46 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-05 11:00:00 | 1095.16 | 2023-06-05 12:00:00 | 1097.39 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-07 10:05:00 | 1109.24 | 2023-06-07 10:35:00 | 1106.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-13 09:40:00 | 1129.77 | 2023-06-13 10:00:00 | 1133.32 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-13 09:40:00 | 1129.77 | 2023-06-13 10:20:00 | 1129.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 11:00:00 | 1132.39 | 2023-06-14 11:05:00 | 1130.96 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-06-20 11:05:00 | 1137.50 | 2023-06-20 11:10:00 | 1138.94 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-06-21 11:10:00 | 1142.60 | 2023-06-21 11:25:00 | 1144.22 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-22 11:10:00 | 1133.78 | 2023-06-22 11:35:00 | 1131.57 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-06-22 11:10:00 | 1133.78 | 2023-06-22 11:55:00 | 1133.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-23 10:30:00 | 1130.86 | 2023-06-23 10:35:00 | 1128.85 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-27 11:05:00 | 1130.00 | 2023-06-27 11:10:00 | 1131.93 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-03 10:45:00 | 1138.88 | 2023-07-03 11:10:00 | 1136.49 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-07-03 10:45:00 | 1138.88 | 2023-07-03 13:10:00 | 1134.84 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-05 10:55:00 | 1142.50 | 2023-07-05 11:05:00 | 1145.37 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-07-05 10:55:00 | 1142.50 | 2023-07-05 11:15:00 | 1142.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-06 09:30:00 | 1159.75 | 2023-07-06 09:35:00 | 1164.55 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-06 09:30:00 | 1159.75 | 2023-07-06 10:15:00 | 1159.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 09:45:00 | 1141.71 | 2023-07-11 10:00:00 | 1139.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-18 11:05:00 | 1149.33 | 2023-07-18 11:35:00 | 1146.29 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-07-18 11:05:00 | 1149.33 | 2023-07-18 12:00:00 | 1149.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-19 10:50:00 | 1141.09 | 2023-07-19 11:10:00 | 1143.44 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-07-24 10:40:00 | 1142.50 | 2023-07-24 11:45:00 | 1139.25 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-07-24 10:40:00 | 1142.50 | 2023-07-24 12:20:00 | 1142.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-25 11:10:00 | 1137.00 | 2023-07-25 11:25:00 | 1138.75 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-07-27 11:00:00 | 1113.50 | 2023-07-27 11:05:00 | 1106.86 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-07-27 11:00:00 | 1113.50 | 2023-07-27 11:10:00 | 1113.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-07 10:25:00 | 1113.99 | 2023-08-07 11:00:00 | 1111.15 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-08-07 10:25:00 | 1113.99 | 2023-08-07 12:55:00 | 1113.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 11:10:00 | 1111.51 | 2023-08-08 11:30:00 | 1109.07 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-08 11:10:00 | 1111.51 | 2023-08-08 14:30:00 | 1110.15 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2023-08-09 10:45:00 | 1110.45 | 2023-08-09 10:55:00 | 1111.98 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1100.00 | 2023-08-10 10:30:00 | 1096.99 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1100.00 | 2023-08-10 13:15:00 | 1099.21 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2023-08-17 11:15:00 | 1083.50 | 2023-08-17 11:55:00 | 1081.23 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-08-17 11:15:00 | 1083.50 | 2023-08-17 12:10:00 | 1083.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-21 11:15:00 | 1099.55 | 2023-08-21 13:30:00 | 1101.88 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-08-21 11:15:00 | 1099.55 | 2023-08-21 14:30:00 | 1099.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 09:40:00 | 1104.24 | 2023-08-22 09:45:00 | 1101.99 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-23 09:50:00 | 1098.20 | 2023-08-23 10:00:00 | 1100.38 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-31 11:05:00 | 1096.04 | 2023-08-31 11:20:00 | 1093.28 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-08-31 11:05:00 | 1096.04 | 2023-08-31 14:45:00 | 1093.90 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-09-04 10:10:00 | 1087.55 | 2023-09-04 10:45:00 | 1084.78 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-09-04 10:10:00 | 1087.55 | 2023-09-04 13:00:00 | 1086.25 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2023-09-05 11:15:00 | 1096.36 | 2023-09-05 12:10:00 | 1099.10 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-09-05 11:15:00 | 1096.36 | 2023-09-05 12:35:00 | 1096.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-06 11:05:00 | 1102.45 | 2023-09-06 12:50:00 | 1100.48 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-09-11 09:30:00 | 1101.80 | 2023-09-11 09:55:00 | 1104.35 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-09-11 09:30:00 | 1101.80 | 2023-09-11 10:20:00 | 1102.14 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2023-09-12 11:05:00 | 1108.06 | 2023-09-12 11:10:00 | 1110.86 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-09-12 11:05:00 | 1108.06 | 2023-09-12 15:20:00 | 1115.02 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2023-09-13 09:45:00 | 1124.98 | 2023-09-13 10:25:00 | 1122.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-14 09:35:00 | 1119.20 | 2023-09-14 09:55:00 | 1117.15 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-09-27 10:55:00 | 1139.84 | 2023-09-27 11:10:00 | 1141.83 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-13 09:55:00 | 1164.22 | 2023-10-13 10:10:00 | 1161.36 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-18 10:50:00 | 1169.75 | 2023-10-18 10:55:00 | 1167.80 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-19 10:50:00 | 1166.33 | 2023-10-19 10:55:00 | 1164.07 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-25 11:05:00 | 1222.10 | 2023-10-25 11:15:00 | 1220.09 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-10-26 11:00:00 | 1182.13 | 2023-10-26 11:05:00 | 1184.82 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-31 09:30:00 | 1207.35 | 2023-10-31 11:20:00 | 1210.11 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-11-01 11:05:00 | 1195.77 | 2023-11-01 13:10:00 | 1193.02 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-01 11:05:00 | 1195.77 | 2023-11-01 14:25:00 | 1195.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-02 10:25:00 | 1190.90 | 2023-11-02 11:05:00 | 1193.56 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-16 11:05:00 | 1211.20 | 2023-11-16 11:30:00 | 1209.36 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-11-17 10:55:00 | 1218.63 | 2023-11-17 11:05:00 | 1216.29 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-23 09:30:00 | 1228.50 | 2023-11-23 09:35:00 | 1226.45 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-11-24 10:35:00 | 1216.11 | 2023-11-24 11:05:00 | 1213.36 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-24 10:35:00 | 1216.11 | 2023-11-24 15:20:00 | 1207.20 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2023-11-30 09:55:00 | 1200.19 | 2023-11-30 10:15:00 | 1202.12 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-12-06 09:30:00 | 1246.24 | 2023-12-06 10:00:00 | 1243.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-11 11:00:00 | 1252.22 | 2023-12-11 11:55:00 | 1249.74 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-13 11:05:00 | 1244.99 | 2023-12-13 11:30:00 | 1241.84 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-12-13 11:05:00 | 1244.99 | 2023-12-13 13:40:00 | 1244.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-18 09:55:00 | 1219.00 | 2023-12-18 15:20:00 | 1217.50 | TARGET_HIT | 1.00 | 0.12% |
| BUY | retest1 | 2023-12-19 09:35:00 | 1242.00 | 2023-12-19 09:45:00 | 1238.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-28 09:30:00 | 1297.62 | 2023-12-28 09:55:00 | 1301.18 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-12-28 09:30:00 | 1297.62 | 2023-12-28 10:05:00 | 1297.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-03 11:10:00 | 1346.79 | 2024-01-03 11:20:00 | 1349.26 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-09 10:45:00 | 1298.03 | 2024-01-09 10:50:00 | 1300.78 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-01-18 09:45:00 | 1252.08 | 2024-01-18 09:50:00 | 1255.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-30 10:45:00 | 1249.88 | 2024-01-30 10:50:00 | 1252.31 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-06 11:15:00 | 1224.55 | 2024-02-06 11:30:00 | 1226.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-02-07 09:35:00 | 1236.33 | 2024-02-07 09:40:00 | 1233.66 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-08 09:45:00 | 1231.00 | 2024-02-08 10:35:00 | 1236.18 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-02-14 09:40:00 | 1221.38 | 2024-02-14 10:00:00 | 1224.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-23 11:00:00 | 1282.83 | 2024-02-23 13:10:00 | 1286.76 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-02-23 11:00:00 | 1282.83 | 2024-02-23 15:20:00 | 1289.20 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-02-29 09:50:00 | 1290.50 | 2024-02-29 10:10:00 | 1286.58 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-05 10:45:00 | 1281.47 | 2024-03-05 11:20:00 | 1277.57 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-03-05 10:45:00 | 1281.47 | 2024-03-05 11:40:00 | 1281.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-06 09:40:00 | 1282.53 | 2024-03-06 09:50:00 | 1279.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-11 10:05:00 | 1287.50 | 2024-03-11 10:15:00 | 1292.89 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-03-11 10:05:00 | 1287.50 | 2024-03-11 11:45:00 | 1299.97 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-03-12 10:05:00 | 1286.45 | 2024-03-12 10:20:00 | 1291.13 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-13 09:40:00 | 1316.00 | 2024-03-13 09:55:00 | 1311.71 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-15 11:15:00 | 1300.08 | 2024-03-15 11:30:00 | 1303.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-20 11:05:00 | 1268.30 | 2024-03-20 13:05:00 | 1274.38 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-03-20 11:05:00 | 1268.30 | 2024-03-20 15:20:00 | 1276.10 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2024-03-27 11:00:00 | 1285.68 | 2024-03-27 11:15:00 | 1288.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-28 09:45:00 | 1300.00 | 2024-03-28 09:50:00 | 1296.86 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-02 10:40:00 | 1306.63 | 2024-04-02 10:50:00 | 1303.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-04-04 10:35:00 | 1264.75 | 2024-04-04 10:50:00 | 1260.26 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-04-04 10:35:00 | 1264.75 | 2024-04-04 11:10:00 | 1264.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-08 10:45:00 | 1261.18 | 2024-04-08 11:05:00 | 1257.46 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-04-08 10:45:00 | 1261.18 | 2024-04-08 15:20:00 | 1249.83 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-04-12 09:35:00 | 1275.90 | 2024-04-12 09:40:00 | 1272.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-16 09:40:00 | 1290.18 | 2024-04-16 09:45:00 | 1285.84 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-22 10:45:00 | 1215.93 | 2024-04-22 11:20:00 | 1219.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-24 10:00:00 | 1264.55 | 2024-04-24 10:15:00 | 1260.04 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-04-25 10:25:00 | 1263.33 | 2024-04-25 10:30:00 | 1259.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-29 10:45:00 | 1256.50 | 2024-04-29 11:05:00 | 1261.05 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-04-29 10:45:00 | 1256.50 | 2024-04-29 11:25:00 | 1256.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 10:20:00 | 1243.43 | 2024-05-03 10:30:00 | 1246.88 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-09 10:25:00 | 1256.05 | 2024-05-09 10:35:00 | 1259.70 | STOP_HIT | 1.00 | -0.29% |
