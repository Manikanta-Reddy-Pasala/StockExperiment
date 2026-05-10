# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1179.90
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 11
- **Target hits / Stop hits / Partials:** 6 / 11 / 6
- **Avg / median % per leg:** 0.18% / 0.28%
- **Sum % (uncompounded):** 4.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.33% | 2.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.33% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 8 | 47.1% | 4 | 9 | 4 | 0.13% | 2.1% |
| SELL @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 4 | 9 | 4 | 0.13% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 12 | 52.2% | 6 | 11 | 6 | 0.18% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 1220.10 | 1217.19 | 0.00 | ORB-long ORB[1200.20,1212.00] vol=2.1x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:15:00 | 1226.25 | 1220.11 | 0.00 | T1 1.5R @ 1226.25 |
| Target hit | 2026-02-10 15:20:00 | 1231.00 | 1223.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 1222.10 | 1230.65 | 0.00 | ORB-short ORB[1228.20,1241.00] vol=1.9x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-02-11 09:40:00 | 1225.12 | 1230.26 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 11:05:00 | 1246.10 | 1237.11 | 0.00 | ORB-long ORB[1231.50,1245.80] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-02-12 11:25:00 | 1242.24 | 1239.58 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:05:00 | 1224.90 | 1228.74 | 0.00 | ORB-short ORB[1228.50,1246.80] vol=2.1x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-02-13 10:10:00 | 1229.13 | 1228.55 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1232.20 | 1239.60 | 0.00 | ORB-short ORB[1233.40,1243.90] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:30:00 | 1227.46 | 1238.58 | 0.00 | T1 1.5R @ 1227.46 |
| Target hit | 2026-02-16 15:20:00 | 1214.10 | 1222.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1203.90 | 1206.09 | 0.00 | ORB-short ORB[1204.00,1215.70] vol=2.6x ATR=3.80 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 1207.70 | 1205.79 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 1198.90 | 1189.87 | 0.00 | ORB-long ORB[1181.90,1197.60] vol=5.0x ATR=5.27 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 1193.63 | 1190.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 1184.70 | 1196.71 | 0.00 | ORB-short ORB[1192.00,1207.60] vol=1.7x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1188.06 | 1192.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1176.60 | 1179.17 | 0.00 | ORB-short ORB[1179.50,1194.40] vol=2.1x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 1179.53 | 1178.92 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 1120.10 | 1129.91 | 0.00 | ORB-short ORB[1130.10,1144.00] vol=2.2x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:50:00 | 1112.82 | 1128.27 | 0.00 | T1 1.5R @ 1112.82 |
| Target hit | 2026-03-05 14:45:00 | 1115.50 | 1114.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 1032.40 | 1042.91 | 0.00 | ORB-short ORB[1045.20,1057.60] vol=1.9x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-03-13 11:00:00 | 1037.15 | 1041.10 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:25:00 | 1048.70 | 1052.86 | 0.00 | ORB-short ORB[1050.30,1061.00] vol=2.6x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-01 10:30:00 | 1053.26 | 1052.97 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 1100.20 | 1106.16 | 0.00 | ORB-short ORB[1103.40,1117.80] vol=1.6x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-04-16 11:30:00 | 1103.27 | 1105.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1116.70 | 1125.29 | 0.00 | ORB-short ORB[1120.70,1137.20] vol=1.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 1110.33 | 1120.15 | 0.00 | T1 1.5R @ 1110.33 |
| Target hit | 2026-04-24 14:20:00 | 1107.70 | 1107.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 1084.40 | 1097.53 | 0.00 | ORB-short ORB[1097.40,1111.50] vol=1.7x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:20:00 | 1078.58 | 1089.34 | 0.00 | T1 1.5R @ 1078.58 |
| Target hit | 2026-05-05 13:25:00 | 1081.40 | 1079.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 1089.20 | 1093.02 | 0.00 | ORB-short ORB[1090.80,1099.70] vol=2.0x ATR=3.85 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 1093.05 | 1093.17 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 1169.20 | 1156.12 | 0.00 | ORB-long ORB[1141.40,1154.90] vol=1.8x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:50:00 | 1176.32 | 1158.76 | 0.00 | T1 1.5R @ 1176.32 |
| Target hit | 2026-05-08 15:20:00 | 1177.70 | 1172.44 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:50:00 | 1220.10 | 2026-02-10 12:15:00 | 1226.25 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-10 10:50:00 | 1220.10 | 2026-02-10 15:20:00 | 1231.00 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-02-11 09:35:00 | 1222.10 | 2026-02-11 09:40:00 | 1225.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-12 11:05:00 | 1246.10 | 2026-02-12 11:25:00 | 1242.24 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-13 10:05:00 | 1224.90 | 2026-02-13 10:10:00 | 1229.13 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-16 11:15:00 | 1232.20 | 2026-02-16 11:30:00 | 1227.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-16 11:15:00 | 1232.20 | 2026-02-16 15:20:00 | 1214.10 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1203.90 | 2026-02-19 10:50:00 | 1207.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-20 09:45:00 | 1198.90 | 2026-02-20 10:10:00 | 1193.63 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-23 10:50:00 | 1184.70 | 2026-02-23 11:15:00 | 1188.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-24 11:10:00 | 1176.60 | 2026-02-24 11:40:00 | 1179.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1120.10 | 2026-03-05 10:50:00 | 1112.82 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1120.10 | 2026-03-05 14:45:00 | 1115.50 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-13 10:40:00 | 1032.40 | 2026-03-13 11:00:00 | 1037.15 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-01 10:25:00 | 1048.70 | 2026-04-01 10:30:00 | 1053.26 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-16 11:10:00 | 1100.20 | 2026-04-16 11:30:00 | 1103.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1116.70 | 2026-04-24 10:30:00 | 1110.33 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1116.70 | 2026-04-24 14:20:00 | 1107.70 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1084.40 | 2026-05-05 10:20:00 | 1078.58 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1084.40 | 2026-05-05 13:25:00 | 1081.40 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2026-05-06 09:50:00 | 1089.20 | 2026-05-06 10:05:00 | 1093.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-08 10:45:00 | 1169.20 | 2026-05-08 10:50:00 | 1176.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-08 10:45:00 | 1169.20 | 2026-05-08 15:20:00 | 1177.70 | TARGET_HIT | 0.50 | 0.73% |
