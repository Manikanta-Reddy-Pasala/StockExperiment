# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-11-06 15:25:00 (7663 bars)
- **Last close:** 1041.30
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
| ENTRY1 | 28 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 25
- **Target hits / Stop hits / Partials:** 3 / 25 / 10
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 10 | 34.5% | 2 | 19 | 8 | 0.02% | 0.7% |
| BUY @ 2nd Alert (retest1) | 29 | 10 | 34.5% | 2 | 19 | 8 | 0.02% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.14% | 1.2% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.14% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 13 | 34.2% | 3 | 25 | 10 | 0.05% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:25:00 | 874.85 | 867.48 | 0.00 | ORB-long ORB[855.00,862.00] vol=1.6x ATR=5.89 |
| Stop hit — per-position SL triggered | 2025-05-12 13:55:00 | 868.96 | 870.13 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:10:00 | 910.00 | 903.27 | 0.00 | ORB-long ORB[895.25,908.00] vol=1.6x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-05-14 11:50:00 | 906.66 | 906.17 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:10:00 | 939.00 | 929.61 | 0.00 | ORB-long ORB[921.40,929.00] vol=2.5x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 11:15:00 | 945.06 | 934.40 | 0.00 | T1 1.5R @ 945.06 |
| Target hit | 2025-05-15 15:20:00 | 946.85 | 941.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 11:15:00 | 960.30 | 953.14 | 0.00 | ORB-long ORB[944.00,957.95] vol=4.1x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-05-16 11:20:00 | 957.48 | 953.21 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:25:00 | 1000.40 | 1007.51 | 0.00 | ORB-short ORB[1004.35,1019.00] vol=1.8x ATR=4.38 |
| Stop hit — per-position SL triggered | 2025-05-26 10:35:00 | 1004.78 | 1006.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:05:00 | 969.45 | 964.30 | 0.00 | ORB-long ORB[952.80,967.30] vol=1.8x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-05-30 11:25:00 | 964.96 | 967.08 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:50:00 | 994.05 | 987.00 | 0.00 | ORB-long ORB[973.30,986.00] vol=4.5x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:55:00 | 998.76 | 992.65 | 0.00 | T1 1.5R @ 998.76 |
| Target hit | 2025-06-11 10:25:00 | 996.70 | 996.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:45:00 | 976.90 | 968.98 | 0.00 | ORB-long ORB[955.00,965.40] vol=3.4x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 972.33 | 969.78 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:55:00 | 973.85 | 964.02 | 0.00 | ORB-long ORB[944.75,957.65] vol=1.7x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 12:10:00 | 978.86 | 967.79 | 0.00 | T1 1.5R @ 978.86 |
| Stop hit — per-position SL triggered | 2025-06-23 12:30:00 | 973.85 | 969.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 1030.55 | 1022.15 | 0.00 | ORB-long ORB[1014.55,1026.95] vol=1.5x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:30:00 | 1036.49 | 1024.15 | 0.00 | T1 1.5R @ 1036.49 |
| Stop hit — per-position SL triggered | 2025-06-27 12:45:00 | 1030.55 | 1026.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1068.00 | 1079.43 | 0.00 | ORB-short ORB[1084.50,1098.00] vol=2.0x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-07-08 10:55:00 | 1072.81 | 1077.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 1126.00 | 1117.07 | 0.00 | ORB-long ORB[1105.00,1121.00] vol=3.3x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:45:00 | 1134.19 | 1124.37 | 0.00 | T1 1.5R @ 1134.19 |
| Stop hit — per-position SL triggered | 2025-07-11 10:10:00 | 1126.00 | 1128.10 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:50:00 | 1158.70 | 1149.83 | 0.00 | ORB-long ORB[1139.40,1151.90] vol=2.2x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:55:00 | 1165.74 | 1155.29 | 0.00 | T1 1.5R @ 1165.74 |
| Stop hit — per-position SL triggered | 2025-07-21 10:00:00 | 1158.70 | 1155.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:15:00 | 1175.90 | 1167.75 | 0.00 | ORB-long ORB[1162.10,1171.50] vol=1.6x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:30:00 | 1181.48 | 1172.49 | 0.00 | T1 1.5R @ 1181.48 |
| Stop hit — per-position SL triggered | 2025-07-22 10:35:00 | 1175.90 | 1172.77 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 1172.60 | 1178.29 | 0.00 | ORB-short ORB[1174.10,1185.00] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:50:00 | 1167.37 | 1176.88 | 0.00 | T1 1.5R @ 1167.37 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1172.60 | 1175.35 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 11:00:00 | 1183.90 | 1182.02 | 0.00 | ORB-long ORB[1170.10,1182.40] vol=2.6x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-07-25 11:10:00 | 1179.78 | 1181.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:35:00 | 1218.20 | 1225.17 | 0.00 | ORB-short ORB[1222.80,1236.80] vol=2.5x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-08-01 09:50:00 | 1222.91 | 1224.61 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:25:00 | 1163.80 | 1159.58 | 0.00 | ORB-long ORB[1142.00,1158.90] vol=3.7x ATR=4.75 |
| Stop hit — per-position SL triggered | 2025-08-14 11:40:00 | 1159.05 | 1159.73 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:00:00 | 1169.00 | 1172.59 | 0.00 | ORB-short ORB[1169.20,1182.90] vol=1.8x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 13:30:00 | 1162.21 | 1171.09 | 0.00 | T1 1.5R @ 1162.21 |
| Target hit | 2025-08-19 15:20:00 | 1144.90 | 1162.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:00:00 | 1173.70 | 1163.66 | 0.00 | ORB-long ORB[1152.90,1167.00] vol=2.1x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-08-21 10:35:00 | 1169.93 | 1166.49 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1182.00 | 1167.27 | 0.00 | ORB-long ORB[1154.40,1170.50] vol=2.5x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-08-22 10:50:00 | 1178.37 | 1168.51 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 1126.00 | 1121.58 | 0.00 | ORB-long ORB[1111.20,1125.80] vol=3.4x ATR=4.30 |
| Stop hit — per-position SL triggered | 2025-09-05 11:05:00 | 1121.70 | 1123.01 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 11:15:00 | 1032.70 | 1040.29 | 0.00 | ORB-short ORB[1036.60,1043.60] vol=3.3x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-10-10 11:30:00 | 1035.45 | 1039.84 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:45:00 | 1035.00 | 1027.09 | 0.00 | ORB-long ORB[1022.00,1034.80] vol=1.7x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 13:55:00 | 1041.43 | 1033.50 | 0.00 | T1 1.5R @ 1041.43 |
| Stop hit — per-position SL triggered | 2025-10-13 15:05:00 | 1035.00 | 1037.67 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:55:00 | 1043.00 | 1036.72 | 0.00 | ORB-long ORB[1030.00,1038.70] vol=2.3x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-10-16 11:10:00 | 1040.02 | 1037.24 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:45:00 | 1043.60 | 1036.73 | 0.00 | ORB-long ORB[1031.70,1040.00] vol=2.2x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-10-23 09:50:00 | 1039.87 | 1037.43 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 1040.90 | 1043.96 | 0.00 | ORB-short ORB[1044.00,1051.50] vol=1.9x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-10-24 09:50:00 | 1044.29 | 1043.90 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:50:00 | 1092.90 | 1081.57 | 0.00 | ORB-long ORB[1070.00,1079.80] vol=3.3x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-10-29 10:55:00 | 1088.37 | 1081.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:25:00 | 874.85 | 2025-05-12 13:55:00 | 868.96 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2025-05-14 10:10:00 | 910.00 | 2025-05-14 11:50:00 | 906.66 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-05-15 10:10:00 | 939.00 | 2025-05-15 11:15:00 | 945.06 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-15 10:10:00 | 939.00 | 2025-05-15 15:20:00 | 946.85 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2025-05-16 11:15:00 | 960.30 | 2025-05-16 11:20:00 | 957.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-26 10:25:00 | 1000.40 | 2025-05-26 10:35:00 | 1004.78 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-30 10:05:00 | 969.45 | 2025-05-30 11:25:00 | 964.96 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-11 09:50:00 | 994.05 | 2025-06-11 09:55:00 | 998.76 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-11 09:50:00 | 994.05 | 2025-06-11 10:25:00 | 996.70 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-19 09:45:00 | 976.90 | 2025-06-19 09:50:00 | 972.33 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-06-23 10:55:00 | 973.85 | 2025-06-23 12:10:00 | 978.86 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-23 10:55:00 | 973.85 | 2025-06-23 12:30:00 | 973.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 11:15:00 | 1030.55 | 2025-06-27 11:30:00 | 1036.49 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-27 11:15:00 | 1030.55 | 2025-06-27 12:45:00 | 1030.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 10:15:00 | 1068.00 | 2025-07-08 10:55:00 | 1072.81 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1126.00 | 2025-07-11 09:45:00 | 1134.19 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1126.00 | 2025-07-11 10:10:00 | 1126.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:50:00 | 1158.70 | 2025-07-21 09:55:00 | 1165.74 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-07-21 09:50:00 | 1158.70 | 2025-07-21 10:00:00 | 1158.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 10:15:00 | 1175.90 | 2025-07-22 10:30:00 | 1181.48 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-22 10:15:00 | 1175.90 | 2025-07-22 10:35:00 | 1175.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 09:35:00 | 1172.60 | 2025-07-24 09:50:00 | 1167.37 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-24 09:35:00 | 1172.60 | 2025-07-24 10:15:00 | 1172.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-25 11:00:00 | 1183.90 | 2025-07-25 11:10:00 | 1179.78 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-01 09:35:00 | 1218.20 | 2025-08-01 09:50:00 | 1222.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-14 10:25:00 | 1163.80 | 2025-08-14 11:40:00 | 1159.05 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-19 11:00:00 | 1169.00 | 2025-08-19 13:30:00 | 1162.21 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-08-19 11:00:00 | 1169.00 | 2025-08-19 15:20:00 | 1144.90 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2025-08-21 10:00:00 | 1173.70 | 2025-08-21 10:35:00 | 1169.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-22 10:45:00 | 1182.00 | 2025-08-22 10:50:00 | 1178.37 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-05 10:15:00 | 1126.00 | 2025-09-05 11:05:00 | 1121.70 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-10 11:15:00 | 1032.70 | 2025-10-10 11:30:00 | 1035.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-13 09:45:00 | 1035.00 | 2025-10-13 13:55:00 | 1041.43 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-10-13 09:45:00 | 1035.00 | 2025-10-13 15:05:00 | 1035.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 10:55:00 | 1043.00 | 2025-10-16 11:10:00 | 1040.02 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-23 09:45:00 | 1043.60 | 2025-10-23 09:50:00 | 1039.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-24 09:40:00 | 1040.90 | 2025-10-24 09:50:00 | 1044.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-29 10:50:00 | 1092.90 | 2025-10-29 10:55:00 | 1088.37 | STOP_HIT | 1.00 | -0.41% |
