# Affle 3i Ltd. (AFFLE)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1510.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 98 |
| ALERT2 | 96 |
| ALERT2_SKIP | 55 |
| ALERT3 | 247 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 127 |
| PARTIAL | 14 |
| TARGET_HIT | 10 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 88
- **Target hits / Stop hits / Partials:** 10 / 121 / 14
- **Avg / median % per leg:** 0.93% / -0.45%
- **Sum % (uncompounded):** 134.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 20 | 27.8% | 9 | 63 | 0 | 0.60% | 43.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.86% | -1.9% |
| BUY @ 3rd Alert (retest2) | 71 | 20 | 28.2% | 9 | 62 | 0 | 0.64% | 45.4% |
| SELL (all) | 73 | 37 | 50.7% | 1 | 58 | 14 | 1.25% | 91.2% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 2.28% | 6.8% |
| SELL @ 3rd Alert (retest2) | 70 | 34 | 48.6% | 1 | 55 | 14 | 1.20% | 84.3% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.24% | 5.0% |
| retest2 (combined) | 141 | 54 | 38.3% | 10 | 117 | 14 | 0.92% | 129.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1086.90 | 1079.11 | 1078.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 1092.00 | 1083.14 | 1080.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 1086.20 | 1088.31 | 1084.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:30:00 | 1086.60 | 1088.31 | 1084.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1092.45 | 1089.80 | 1086.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 1099.25 | 1090.91 | 1088.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-18 11:15:00 | 1209.18 | 1175.61 | 1141.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1196.15 | 1228.74 | 1229.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 1194.75 | 1221.94 | 1226.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 10:15:00 | 1148.60 | 1146.61 | 1160.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 11:00:00 | 1148.60 | 1146.61 | 1160.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 1151.00 | 1148.50 | 1158.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:00:00 | 1151.00 | 1148.50 | 1158.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 1154.50 | 1151.25 | 1157.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1087.25 | 1151.25 | 1157.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 10:00:00 | 1138.10 | 1111.75 | 1118.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 1162.80 | 1121.96 | 1122.88 | SL hit (close>static) qty=1.00 sl=1157.75 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1163.00 | 1130.17 | 1126.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 1200.30 | 1166.69 | 1151.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 11:15:00 | 1254.10 | 1259.03 | 1243.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 15:15:00 | 1251.45 | 1254.62 | 1246.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1251.45 | 1254.62 | 1246.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 1248.70 | 1254.62 | 1246.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1261.05 | 1255.91 | 1247.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1265.60 | 1255.06 | 1250.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 1239.10 | 1250.78 | 1249.72 | SL hit (close<static) qty=1.00 sl=1242.20 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 1241.30 | 1247.61 | 1248.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 14:15:00 | 1235.80 | 1245.25 | 1247.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 1252.80 | 1240.56 | 1243.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 11:15:00 | 1252.80 | 1240.56 | 1243.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1252.80 | 1240.56 | 1243.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 1248.05 | 1240.56 | 1243.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 1245.00 | 1241.45 | 1243.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 1241.90 | 1241.91 | 1243.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:00:00 | 1239.25 | 1241.91 | 1243.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 1242.00 | 1242.85 | 1243.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:00:00 | 1242.00 | 1243.24 | 1243.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1240.10 | 1242.61 | 1243.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 1240.10 | 1242.61 | 1243.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1286.30 | 1250.93 | 1246.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1286.30 | 1250.93 | 1246.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 1317.00 | 1273.25 | 1262.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 1290.50 | 1299.94 | 1283.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1290.50 | 1299.94 | 1283.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1290.50 | 1299.94 | 1283.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 1290.50 | 1299.94 | 1283.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1281.25 | 1296.21 | 1283.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 1281.25 | 1296.21 | 1283.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1295.50 | 1296.06 | 1284.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 1279.65 | 1296.06 | 1284.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1296.00 | 1295.28 | 1287.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:30:00 | 1296.30 | 1295.38 | 1288.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 1297.05 | 1295.38 | 1288.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 14:30:00 | 1339.90 | 1312.03 | 1298.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 1327.90 | 1353.80 | 1355.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 1327.90 | 1353.80 | 1355.18 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 1406.25 | 1356.08 | 1353.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 1432.55 | 1409.18 | 1399.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1416.35 | 1420.91 | 1412.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 1416.35 | 1420.91 | 1412.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1416.35 | 1420.91 | 1412.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 1412.95 | 1420.91 | 1412.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1420.45 | 1420.82 | 1412.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 1411.80 | 1420.82 | 1412.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1406.40 | 1420.19 | 1418.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1406.40 | 1420.19 | 1418.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1411.30 | 1418.42 | 1417.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 1411.30 | 1418.42 | 1417.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1390.70 | 1412.65 | 1414.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1359.35 | 1385.57 | 1397.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 1382.45 | 1379.70 | 1391.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 14:00:00 | 1382.45 | 1379.70 | 1391.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1367.00 | 1377.16 | 1389.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:30:00 | 1390.00 | 1377.16 | 1389.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1368.85 | 1369.77 | 1379.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 1376.65 | 1369.77 | 1379.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1375.00 | 1370.82 | 1379.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 1375.00 | 1370.82 | 1379.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1351.55 | 1366.83 | 1375.96 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1411.55 | 1381.96 | 1379.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 1438.90 | 1406.39 | 1394.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 1485.60 | 1489.00 | 1473.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:45:00 | 1486.30 | 1489.00 | 1473.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1479.00 | 1485.64 | 1474.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1497.15 | 1485.64 | 1474.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 12:15:00 | 1484.95 | 1500.18 | 1493.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 1472.80 | 1491.17 | 1490.48 | SL hit (close<static) qty=1.00 sl=1474.25 alert=retest2 |

### Cycle 10 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 1464.80 | 1485.90 | 1488.14 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 09:15:00 | 1532.15 | 1490.28 | 1487.06 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1451.75 | 1482.58 | 1483.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1438.50 | 1473.76 | 1479.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1475.05 | 1459.49 | 1468.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1475.05 | 1459.49 | 1468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1475.05 | 1459.49 | 1468.61 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 1492.55 | 1474.19 | 1473.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1550.25 | 1500.34 | 1491.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 10:15:00 | 1566.25 | 1567.55 | 1540.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 11:00:00 | 1566.25 | 1567.55 | 1540.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1563.00 | 1565.24 | 1551.17 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 1523.80 | 1542.78 | 1545.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1490.50 | 1532.32 | 1540.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 1510.75 | 1506.39 | 1522.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 1510.75 | 1506.39 | 1522.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 1510.75 | 1506.39 | 1522.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 1510.75 | 1506.39 | 1522.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1528.70 | 1512.07 | 1522.00 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 1552.80 | 1530.79 | 1528.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 1555.00 | 1539.03 | 1532.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 12:15:00 | 1556.00 | 1558.39 | 1545.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 12:45:00 | 1556.90 | 1558.39 | 1545.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1633.00 | 1645.51 | 1630.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:30:00 | 1646.95 | 1643.79 | 1632.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:00:00 | 1645.30 | 1643.79 | 1632.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:00:00 | 1646.00 | 1644.23 | 1633.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 1647.00 | 1637.56 | 1633.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1659.00 | 1672.21 | 1661.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:45:00 | 1658.00 | 1672.21 | 1661.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1652.95 | 1668.36 | 1660.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 1653.70 | 1668.36 | 1660.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1655.00 | 1665.69 | 1660.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1665.00 | 1665.69 | 1660.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 12:15:00 | 1646.55 | 1657.19 | 1657.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 1646.55 | 1657.19 | 1657.62 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1660.80 | 1658.44 | 1658.15 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1645.25 | 1656.92 | 1657.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 15:15:00 | 1641.85 | 1650.45 | 1653.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1632.05 | 1623.29 | 1631.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1632.05 | 1623.29 | 1631.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1632.05 | 1623.29 | 1631.41 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1637.40 | 1623.83 | 1623.31 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1584.65 | 1617.66 | 1621.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 1572.05 | 1601.96 | 1613.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 1590.00 | 1586.87 | 1601.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 1600.65 | 1586.87 | 1601.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1586.95 | 1586.88 | 1599.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:30:00 | 1575.00 | 1588.64 | 1596.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 1578.45 | 1568.73 | 1569.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 15:15:00 | 1575.00 | 1569.98 | 1569.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 15:15:00 | 1575.00 | 1569.98 | 1569.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 1582.50 | 1573.93 | 1571.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1571.15 | 1576.17 | 1573.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 1571.15 | 1576.17 | 1573.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1571.15 | 1576.17 | 1573.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 1563.90 | 1576.17 | 1573.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1571.00 | 1575.14 | 1573.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 1571.00 | 1575.14 | 1573.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 1562.70 | 1571.71 | 1572.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 1553.00 | 1564.82 | 1568.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1530.90 | 1527.46 | 1542.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 1530.90 | 1527.46 | 1542.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1545.60 | 1531.49 | 1541.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 1545.60 | 1531.49 | 1541.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1542.25 | 1533.64 | 1541.71 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 1570.00 | 1548.84 | 1546.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 1582.95 | 1555.66 | 1550.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 1558.25 | 1571.20 | 1563.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 1558.25 | 1571.20 | 1563.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1558.25 | 1571.20 | 1563.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:15:00 | 1561.85 | 1571.20 | 1563.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1556.55 | 1568.27 | 1562.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:15:00 | 1552.05 | 1568.27 | 1562.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1557.90 | 1566.19 | 1562.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 1554.00 | 1566.19 | 1562.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1561.05 | 1565.11 | 1562.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1561.05 | 1565.11 | 1562.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1554.85 | 1563.06 | 1561.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 1565.75 | 1562.21 | 1561.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 1568.00 | 1561.81 | 1561.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 1557.20 | 1561.64 | 1561.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1557.20 | 1561.64 | 1561.65 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 1570.95 | 1561.62 | 1561.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 1590.35 | 1567.36 | 1564.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 1576.50 | 1586.97 | 1579.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 1576.50 | 1586.97 | 1579.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1576.50 | 1586.97 | 1579.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1576.50 | 1586.97 | 1579.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 1587.25 | 1587.02 | 1580.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:30:00 | 1574.05 | 1587.02 | 1580.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 1579.90 | 1585.59 | 1581.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 1579.90 | 1585.59 | 1581.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 1587.20 | 1585.92 | 1581.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 1592.05 | 1593.20 | 1585.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 1572.50 | 1599.82 | 1595.60 | SL hit (close<static) qty=1.00 sl=1577.55 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1557.30 | 1586.08 | 1589.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 1540.30 | 1566.49 | 1577.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1575.05 | 1567.41 | 1576.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 1575.05 | 1567.41 | 1576.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1575.05 | 1567.41 | 1576.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 1575.05 | 1567.41 | 1576.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1555.95 | 1565.12 | 1574.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 1547.10 | 1561.79 | 1571.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 1542.25 | 1561.79 | 1571.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:00:00 | 1546.60 | 1526.36 | 1528.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 1551.40 | 1531.37 | 1530.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1551.40 | 1531.37 | 1530.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 1581.65 | 1541.43 | 1535.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 1591.70 | 1598.36 | 1580.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 1591.70 | 1598.36 | 1580.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 1604.00 | 1612.26 | 1603.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1636.00 | 1612.26 | 1603.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 1613.75 | 1611.25 | 1604.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 1600.05 | 1609.01 | 1603.72 | SL hit (close<static) qty=1.00 sl=1601.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 1583.05 | 1600.68 | 1601.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 1578.75 | 1588.18 | 1594.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 1586.00 | 1580.25 | 1587.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 13:00:00 | 1586.00 | 1580.25 | 1587.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1582.85 | 1580.77 | 1587.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 15:15:00 | 1572.25 | 1581.56 | 1586.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 14:15:00 | 1493.64 | 1525.98 | 1548.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 1502.25 | 1490.26 | 1509.95 | SL hit (close>ema200) qty=0.50 sl=1490.26 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1502.80 | 1470.62 | 1467.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 13:15:00 | 1505.65 | 1488.29 | 1477.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 1515.00 | 1525.02 | 1508.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 1515.00 | 1525.02 | 1508.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1515.00 | 1525.02 | 1508.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1515.00 | 1525.02 | 1508.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1517.50 | 1523.52 | 1509.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 1545.95 | 1520.76 | 1510.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 1586.95 | 1601.59 | 1602.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 1586.95 | 1601.59 | 1602.70 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1605.00 | 1601.25 | 1601.06 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 1599.15 | 1600.89 | 1600.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 1584.85 | 1597.68 | 1599.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1557.10 | 1539.62 | 1558.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1557.10 | 1539.62 | 1558.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1557.10 | 1539.62 | 1558.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1557.10 | 1539.62 | 1558.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1550.95 | 1541.89 | 1557.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1558.35 | 1541.89 | 1557.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1592.30 | 1551.97 | 1560.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 1592.30 | 1551.97 | 1560.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1588.85 | 1559.34 | 1563.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:30:00 | 1589.35 | 1559.34 | 1563.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 14:15:00 | 1579.95 | 1566.67 | 1566.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 15:15:00 | 1581.95 | 1569.73 | 1567.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 09:15:00 | 1559.35 | 1567.65 | 1566.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 1559.35 | 1567.65 | 1566.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1559.35 | 1567.65 | 1566.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:30:00 | 1550.10 | 1567.65 | 1566.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 1558.35 | 1565.79 | 1566.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 1539.10 | 1557.43 | 1561.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1562.00 | 1557.80 | 1561.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1562.00 | 1557.80 | 1561.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1562.00 | 1557.80 | 1561.26 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 1590.00 | 1566.65 | 1564.40 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 1548.00 | 1562.00 | 1562.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 1530.50 | 1550.84 | 1556.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1563.50 | 1553.37 | 1557.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1563.50 | 1553.37 | 1557.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1563.50 | 1553.37 | 1557.12 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 1578.55 | 1562.96 | 1560.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1589.15 | 1576.09 | 1568.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 1577.75 | 1582.67 | 1576.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 10:15:00 | 1577.75 | 1582.67 | 1576.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1577.75 | 1582.67 | 1576.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 1607.00 | 1587.45 | 1581.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 1601.45 | 1595.32 | 1586.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 1601.85 | 1596.09 | 1587.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:45:00 | 1611.65 | 1597.87 | 1589.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1594.75 | 1608.48 | 1601.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1594.75 | 1608.48 | 1601.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1595.60 | 1605.90 | 1600.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1611.40 | 1605.90 | 1600.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 09:15:00 | 1767.70 | 1699.70 | 1659.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1770.00 | 1782.45 | 1783.34 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 1786.80 | 1781.73 | 1781.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 1819.50 | 1791.41 | 1786.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 1787.90 | 1793.09 | 1788.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 1787.90 | 1793.09 | 1788.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1787.90 | 1793.09 | 1788.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1787.90 | 1793.09 | 1788.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1818.00 | 1798.07 | 1791.21 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 1779.00 | 1788.69 | 1789.99 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1825.25 | 1790.31 | 1789.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 1855.00 | 1812.34 | 1800.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 1831.90 | 1834.98 | 1820.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 14:00:00 | 1831.90 | 1834.98 | 1820.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1825.00 | 1832.11 | 1822.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 1836.00 | 1832.11 | 1822.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1830.50 | 1831.79 | 1822.82 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 1763.35 | 1812.33 | 1817.74 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 13:15:00 | 1803.00 | 1772.80 | 1772.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 09:15:00 | 1813.60 | 1786.26 | 1778.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 12:15:00 | 1791.70 | 1792.72 | 1784.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-24 13:00:00 | 1791.70 | 1792.72 | 1784.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1768.65 | 1787.91 | 1782.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:00:00 | 1768.65 | 1787.91 | 1782.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1780.20 | 1786.37 | 1782.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:30:00 | 1770.45 | 1786.37 | 1782.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1774.90 | 1784.07 | 1781.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 1781.00 | 1784.07 | 1781.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 1743.15 | 1775.89 | 1778.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 1736.95 | 1763.15 | 1771.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 1793.50 | 1764.80 | 1768.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 1793.50 | 1764.80 | 1768.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1793.50 | 1764.80 | 1768.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 1798.05 | 1764.80 | 1768.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 1801.00 | 1772.04 | 1771.50 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 1777.80 | 1781.15 | 1781.18 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 1787.00 | 1782.19 | 1781.62 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 1777.00 | 1781.15 | 1781.20 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 1797.15 | 1784.35 | 1782.65 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 1771.60 | 1781.23 | 1782.04 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 1790.40 | 1780.67 | 1780.46 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 1769.75 | 1778.53 | 1779.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1746.60 | 1771.72 | 1776.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1735.95 | 1732.97 | 1749.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 1720.05 | 1732.97 | 1749.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1734.00 | 1732.59 | 1741.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1689.40 | 1732.59 | 1741.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1604.93 | 1659.28 | 1683.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 1520.46 | 1562.76 | 1607.71 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 1600.00 | 1591.79 | 1591.08 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1559.65 | 1590.32 | 1594.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1541.35 | 1567.45 | 1576.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1532.30 | 1522.63 | 1539.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1532.30 | 1522.63 | 1539.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1532.30 | 1522.63 | 1539.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1532.30 | 1522.63 | 1539.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1562.55 | 1530.67 | 1540.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1555.75 | 1530.67 | 1540.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1552.50 | 1535.04 | 1541.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 1547.80 | 1539.68 | 1543.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 12:15:00 | 1566.10 | 1544.96 | 1545.15 | SL hit (close>static) qty=1.00 sl=1564.40 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1561.50 | 1548.27 | 1546.63 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 1537.75 | 1546.43 | 1547.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1450.00 | 1527.14 | 1538.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1507.00 | 1473.87 | 1487.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1507.00 | 1473.87 | 1487.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1507.00 | 1473.87 | 1487.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1507.00 | 1473.87 | 1487.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1499.90 | 1479.07 | 1488.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:15:00 | 1500.00 | 1479.07 | 1488.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1500.35 | 1494.28 | 1493.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1507.50 | 1496.93 | 1494.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 1489.40 | 1495.42 | 1494.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 10:15:00 | 1489.40 | 1495.42 | 1494.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1489.40 | 1495.42 | 1494.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 1489.40 | 1495.42 | 1494.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1501.25 | 1496.59 | 1494.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 1484.80 | 1496.59 | 1494.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1491.05 | 1496.03 | 1495.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 1494.15 | 1496.03 | 1495.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1500.00 | 1496.83 | 1495.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:45:00 | 1513.90 | 1501.89 | 1498.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1512.20 | 1502.57 | 1499.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 1480.75 | 1504.40 | 1502.65 | SL hit (close<static) qty=1.00 sl=1488.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1487.30 | 1502.94 | 1503.01 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 1517.00 | 1503.24 | 1501.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1588.80 | 1520.67 | 1509.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1619.25 | 1650.20 | 1631.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1619.25 | 1650.20 | 1631.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1619.25 | 1650.20 | 1631.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 1605.00 | 1650.20 | 1631.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1637.00 | 1647.56 | 1631.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 1643.30 | 1641.70 | 1632.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:45:00 | 1647.35 | 1641.96 | 1633.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 1614.80 | 1636.68 | 1632.48 | SL hit (close<static) qty=1.00 sl=1617.85 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 1568.95 | 1617.98 | 1624.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 13:15:00 | 1561.30 | 1606.64 | 1618.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1555.00 | 1540.87 | 1566.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 1555.00 | 1540.87 | 1566.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1571.20 | 1549.58 | 1565.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 1571.20 | 1549.58 | 1565.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1574.80 | 1554.62 | 1566.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 1582.40 | 1554.62 | 1566.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 1555.70 | 1556.82 | 1565.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 1549.95 | 1556.82 | 1565.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 1611.85 | 1566.89 | 1567.91 | SL hit (close>static) qty=1.00 sl=1567.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 1525.10 | 1490.75 | 1489.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 1546.80 | 1501.96 | 1494.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 1495.45 | 1517.02 | 1505.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 1495.45 | 1517.02 | 1505.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1495.45 | 1517.02 | 1505.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 1495.45 | 1517.02 | 1505.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1491.45 | 1511.91 | 1504.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 1491.45 | 1511.91 | 1504.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1505.55 | 1509.56 | 1504.59 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 1489.75 | 1502.04 | 1502.09 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1503.75 | 1502.38 | 1502.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 11:15:00 | 1518.55 | 1506.99 | 1504.45 | Break + close above crossover candle high |

### Cycle 64 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1474.80 | 1503.97 | 1504.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 1467.05 | 1496.58 | 1501.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 1395.85 | 1382.18 | 1412.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 1395.85 | 1382.18 | 1412.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1417.55 | 1394.70 | 1410.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1417.55 | 1394.70 | 1410.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1433.00 | 1402.36 | 1412.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 1433.00 | 1402.36 | 1412.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 1427.75 | 1407.44 | 1414.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 1431.75 | 1407.44 | 1414.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1437.95 | 1419.82 | 1418.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1454.95 | 1427.32 | 1422.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 1428.75 | 1434.79 | 1427.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 12:15:00 | 1428.75 | 1434.79 | 1427.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1428.75 | 1434.79 | 1427.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:00:00 | 1428.75 | 1434.79 | 1427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1436.00 | 1435.03 | 1428.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:15:00 | 1445.55 | 1434.49 | 1429.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:30:00 | 1441.80 | 1436.24 | 1433.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1437.85 | 1435.99 | 1433.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:45:00 | 1437.00 | 1453.61 | 1452.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1426.00 | 1448.08 | 1449.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1426.00 | 1448.08 | 1449.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1409.75 | 1440.42 | 1446.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 1437.80 | 1436.92 | 1443.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 12:00:00 | 1437.80 | 1436.92 | 1443.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1440.00 | 1437.53 | 1442.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:30:00 | 1447.60 | 1437.53 | 1442.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1446.55 | 1439.34 | 1443.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:45:00 | 1450.85 | 1439.34 | 1443.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1451.85 | 1441.84 | 1444.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 1451.85 | 1441.84 | 1444.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1453.95 | 1444.26 | 1444.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 1464.40 | 1444.26 | 1444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1443.50 | 1441.28 | 1442.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 1440.90 | 1441.28 | 1442.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1448.70 | 1442.76 | 1443.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 1448.15 | 1442.76 | 1443.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 1449.95 | 1444.20 | 1443.93 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 1432.85 | 1441.93 | 1442.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 1412.80 | 1436.10 | 1440.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 14:15:00 | 1430.00 | 1428.57 | 1434.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 1430.00 | 1428.57 | 1434.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1430.00 | 1428.57 | 1434.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:45:00 | 1435.25 | 1428.57 | 1434.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1430.50 | 1428.95 | 1434.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1425.15 | 1428.95 | 1434.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1416.15 | 1426.39 | 1432.50 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1440.60 | 1429.63 | 1429.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 1445.00 | 1432.71 | 1430.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 13:15:00 | 1601.35 | 1605.77 | 1590.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 1601.35 | 1605.77 | 1590.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1610.00 | 1612.72 | 1604.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 1632.30 | 1612.72 | 1604.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1630.40 | 1616.26 | 1607.13 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 1590.10 | 1602.24 | 1602.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 12:15:00 | 1586.20 | 1596.70 | 1599.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 11:15:00 | 1593.20 | 1590.25 | 1594.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 1593.20 | 1590.25 | 1594.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1593.20 | 1590.25 | 1594.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:30:00 | 1593.30 | 1590.25 | 1594.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1598.20 | 1591.84 | 1594.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 1598.20 | 1591.84 | 1594.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 1598.00 | 1593.07 | 1595.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 1597.45 | 1593.07 | 1595.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 1601.45 | 1594.75 | 1595.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 1601.45 | 1594.75 | 1595.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 1604.00 | 1596.60 | 1596.42 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1581.65 | 1593.61 | 1595.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1571.80 | 1586.53 | 1591.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1472.70 | 1461.71 | 1498.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1472.70 | 1461.71 | 1498.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1458.75 | 1425.68 | 1447.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:15:00 | 1463.00 | 1425.68 | 1447.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 1480.15 | 1436.58 | 1450.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:30:00 | 1479.25 | 1436.58 | 1450.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1509.05 | 1461.20 | 1459.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1526.00 | 1490.41 | 1475.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 1535.80 | 1539.14 | 1520.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 1535.80 | 1539.14 | 1520.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1521.40 | 1536.05 | 1522.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 1521.40 | 1536.05 | 1522.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1530.40 | 1534.92 | 1523.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1538.80 | 1534.92 | 1523.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1566.90 | 1585.61 | 1586.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1566.90 | 1585.61 | 1586.59 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 1605.10 | 1590.28 | 1588.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1610.40 | 1597.74 | 1592.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 11:15:00 | 1597.70 | 1598.46 | 1593.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 1597.70 | 1598.46 | 1593.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 1597.70 | 1598.46 | 1593.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 1596.00 | 1598.46 | 1593.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 1616.80 | 1602.13 | 1595.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:00:00 | 1622.00 | 1606.10 | 1598.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 1637.10 | 1610.85 | 1601.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:45:00 | 1626.10 | 1613.48 | 1603.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 10:30:00 | 1628.40 | 1616.98 | 1606.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1609.90 | 1619.86 | 1613.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 1594.20 | 1611.48 | 1610.41 | SL hit (close<static) qty=1.00 sl=1594.70 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 1589.60 | 1607.11 | 1608.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1580.00 | 1596.13 | 1602.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 11:15:00 | 1597.30 | 1590.73 | 1598.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 11:15:00 | 1597.30 | 1590.73 | 1598.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1597.30 | 1590.73 | 1598.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 1597.30 | 1590.73 | 1598.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1598.10 | 1592.20 | 1598.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:30:00 | 1598.00 | 1592.20 | 1598.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1593.60 | 1592.48 | 1597.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 1599.50 | 1592.48 | 1597.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1595.00 | 1592.98 | 1597.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1572.00 | 1590.82 | 1594.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1493.40 | 1525.87 | 1537.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1520.00 | 1510.49 | 1522.68 | SL hit (close>ema200) qty=0.50 sl=1510.49 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1593.50 | 1541.20 | 1534.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 1610.60 | 1595.01 | 1579.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 1696.00 | 1700.61 | 1674.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 1688.30 | 1700.61 | 1674.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1699.90 | 1694.60 | 1683.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 1710.00 | 1695.71 | 1689.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1721.10 | 1705.28 | 1699.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1698.00 | 1716.92 | 1717.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 1698.00 | 1716.92 | 1717.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 1686.00 | 1703.47 | 1710.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:45:00 | 1702.00 | 1687.83 | 1696.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1710.50 | 1692.36 | 1697.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 1709.00 | 1692.36 | 1697.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 1739.00 | 1708.56 | 1704.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1750.40 | 1725.09 | 1713.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 1725.00 | 1725.90 | 1716.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 1725.00 | 1725.90 | 1716.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1710.90 | 1727.00 | 1720.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1710.90 | 1727.00 | 1720.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1712.20 | 1724.04 | 1719.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 1712.20 | 1724.04 | 1719.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1697.80 | 1716.48 | 1716.92 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 1760.00 | 1722.63 | 1719.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 1780.00 | 1758.97 | 1741.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1778.00 | 1787.86 | 1765.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 1778.00 | 1787.86 | 1765.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1793.40 | 1797.99 | 1790.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1793.30 | 1797.99 | 1790.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1787.50 | 1795.89 | 1790.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 1787.50 | 1795.89 | 1790.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1793.80 | 1795.47 | 1790.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1808.30 | 1795.47 | 1790.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 1799.50 | 1795.40 | 1791.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 1795.80 | 1793.96 | 1791.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 1800.00 | 1793.96 | 1791.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 09:15:00 | 1979.45 | 1850.02 | 1824.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 1885.90 | 1892.67 | 1893.50 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 14:15:00 | 1935.00 | 1900.72 | 1897.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 15:15:00 | 1946.00 | 1909.78 | 1901.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 1889.20 | 1908.90 | 1901.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1900.00 | 1907.12 | 1901.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1900.00 | 1907.12 | 1901.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1902.10 | 1906.12 | 1901.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 1934.50 | 1910.88 | 1904.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:00:00 | 1910.20 | 1929.70 | 1924.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1901.10 | 1919.69 | 1920.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1901.10 | 1919.69 | 1920.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1897.50 | 1912.98 | 1917.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:00:00 | 1909.10 | 1905.64 | 1910.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1901.10 | 1904.73 | 1909.63 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1927.50 | 1911.43 | 1909.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 1973.80 | 1945.93 | 1932.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1991.60 | 1996.39 | 1983.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2011.60 | 1996.39 | 1983.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1987.90 | 1995.49 | 1985.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1987.90 | 1995.49 | 1985.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1987.90 | 1993.98 | 1985.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1984.80 | 1993.98 | 1985.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1974.10 | 1990.00 | 1984.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 1974.10 | 1990.00 | 1984.53 | SL hit (close<ema400) qty=1.00 sl=1984.53 alert=retest1 |

### Cycle 86 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 1965.00 | 1979.59 | 1980.71 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2008.00 | 1982.96 | 1981.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 2022.00 | 1990.77 | 1984.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 2024.70 | 2026.30 | 2010.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 2024.70 | 2026.30 | 2010.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2012.90 | 2022.14 | 2013.33 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1992.50 | 2006.39 | 2008.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1982.10 | 1996.67 | 2002.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1976.10 | 1961.13 | 1980.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1976.10 | 1961.13 | 1980.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1979.90 | 1964.88 | 1980.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 1985.00 | 1964.88 | 1980.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1972.40 | 1966.39 | 1979.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1994.70 | 1966.39 | 1979.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1983.50 | 1969.81 | 1979.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1997.00 | 1969.81 | 1979.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1987.80 | 1973.41 | 1980.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1987.80 | 1973.41 | 1980.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1979.40 | 1976.30 | 1980.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 1977.00 | 1977.80 | 1980.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 1974.50 | 1980.08 | 1981.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 1975.60 | 1980.65 | 1981.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1973.40 | 1970.85 | 1975.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1960.00 | 1968.68 | 1973.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 1970.00 | 1968.68 | 1973.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1990.00 | 1967.52 | 1971.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 1990.00 | 1967.52 | 1971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1975.00 | 1969.02 | 1971.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1976.80 | 1969.02 | 1971.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1987.90 | 1972.79 | 1972.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 2000.00 | 1972.79 | 1972.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1982.70 | 1974.77 | 1973.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1982.70 | 1974.77 | 1973.77 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1968.10 | 1972.52 | 1972.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 15:15:00 | 1957.10 | 1968.95 | 1971.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1974.80 | 1972.63 | 1972.45 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1964.70 | 1971.04 | 1971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1953.00 | 1965.22 | 1968.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 1917.00 | 1910.28 | 1924.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 1917.00 | 1910.28 | 1924.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1906.00 | 1907.95 | 1918.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 1899.70 | 1906.30 | 1916.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1804.71 | 1828.20 | 1852.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 1820.00 | 1806.91 | 1829.40 | SL hit (close>ema200) qty=0.50 sl=1806.91 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 1894.00 | 1838.15 | 1838.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 1922.60 | 1868.82 | 1853.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1958.10 | 1960.85 | 1925.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 15:15:00 | 1940.00 | 1957.90 | 1939.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1940.00 | 1957.90 | 1939.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:30:00 | 1972.40 | 1960.93 | 1944.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 1970.70 | 1962.62 | 1946.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 1916.50 | 1944.13 | 1944.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 1916.50 | 1944.13 | 1944.39 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 1950.00 | 1939.38 | 1938.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1960.00 | 1947.90 | 1943.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1946.20 | 1947.56 | 1943.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1954.60 | 1948.97 | 1944.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 1956.10 | 1948.97 | 1944.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1964.90 | 1952.15 | 1946.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1969.70 | 1948.50 | 1947.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 1974.60 | 1962.74 | 1955.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:15:00 | 1968.80 | 1965.74 | 1959.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 1970.80 | 1975.64 | 1969.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1972.30 | 1974.97 | 1969.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 1972.30 | 1974.97 | 1969.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1980.90 | 1976.16 | 1970.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 1987.50 | 1977.63 | 1971.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1985.00 | 1979.04 | 1972.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 1984.40 | 1979.25 | 1973.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1986.00 | 1978.40 | 1973.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1988.40 | 1980.40 | 1974.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 1970.00 | 1982.57 | 1980.61 | SL hit (close<static) qty=1.00 sl=1970.20 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1969.00 | 1977.83 | 1978.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1959.00 | 1972.16 | 1975.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 1945.00 | 1947.32 | 1958.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1963.80 | 1951.57 | 1956.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1963.80 | 1951.57 | 1956.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1965.00 | 1954.25 | 1957.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1942.60 | 1954.25 | 1957.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:00:00 | 1954.90 | 1952.01 | 1954.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 1912.70 | 1905.44 | 1905.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1912.70 | 1905.44 | 1905.03 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1897.00 | 1903.80 | 1904.64 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1931.20 | 1909.28 | 1907.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 1935.30 | 1914.49 | 1909.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1912.20 | 1916.61 | 1911.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1909.80 | 1915.25 | 1911.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 1909.00 | 1915.25 | 1911.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1902.70 | 1912.74 | 1910.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 1913.00 | 1912.74 | 1910.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1906.00 | 1911.39 | 1910.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1911.60 | 1911.39 | 1910.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1912.70 | 1911.81 | 1910.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1900.00 | 1909.82 | 1909.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1900.00 | 1909.82 | 1909.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1893.80 | 1906.61 | 1908.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1917.70 | 1905.66 | 1907.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1911.30 | 1906.79 | 1907.75 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1925.00 | 1910.43 | 1909.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 1932.10 | 1916.33 | 1912.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 2023.70 | 2042.73 | 2010.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 2023.70 | 2042.73 | 2010.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2074.30 | 2083.38 | 2064.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 2074.30 | 2083.38 | 2064.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2072.10 | 2079.78 | 2066.25 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 2055.60 | 2062.71 | 2063.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 2048.40 | 2059.84 | 2061.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:15:00 | 2064.40 | 2060.88 | 2062.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2069.00 | 2062.50 | 2062.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 2069.00 | 2062.50 | 2062.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 2064.10 | 2062.82 | 2062.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 2135.00 | 2077.26 | 2069.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 2127.60 | 2128.88 | 2107.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:45:00 | 2136.30 | 2128.88 | 2107.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 2098.70 | 2121.97 | 2107.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 2098.70 | 2121.97 | 2107.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2105.10 | 2118.59 | 2107.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 2130.00 | 2113.36 | 2106.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 2114.20 | 2131.35 | 2132.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 2114.20 | 2131.35 | 2132.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 2101.20 | 2125.32 | 2129.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 2068.20 | 2064.07 | 2086.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2023.70 | 2053.66 | 2070.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:00:00 | 2020.40 | 2047.01 | 2065.92 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:45:00 | 2019.20 | 2041.61 | 2061.75 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1969.90 | 1956.84 | 1973.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 1969.90 | 1956.84 | 1973.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1975.00 | 1960.47 | 1973.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1975.00 | 1960.47 | 1973.38 | SL hit (close>ema400) qty=1.00 sl=1973.38 alert=retest1 |

### Cycle 105 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1942.00 | 1929.90 | 1929.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1947.80 | 1933.64 | 1931.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1934.10 | 1951.91 | 1952.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1916.30 | 1944.79 | 1948.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1941.20 | 1933.69 | 1940.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1931.80 | 1933.31 | 1939.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 1929.00 | 1931.53 | 1938.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1964.70 | 1930.50 | 1934.61 | SL hit (close>static) qty=1.00 sl=1948.70 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1948.30 | 1937.50 | 1937.29 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1927.60 | 1938.00 | 1938.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1914.20 | 1931.16 | 1934.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1890.00 | 1889.04 | 1901.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:45:00 | 1889.10 | 1889.04 | 1901.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1900.00 | 1892.63 | 1901.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:15:00 | 1907.00 | 1892.63 | 1901.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1906.80 | 1895.46 | 1901.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1908.00 | 1895.46 | 1901.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1903.00 | 1896.97 | 1901.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 1908.40 | 1896.97 | 1901.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1896.20 | 1896.82 | 1901.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 1933.00 | 1896.82 | 1901.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1886.30 | 1894.42 | 1899.31 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1919.20 | 1900.23 | 1897.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1925.30 | 1905.24 | 1900.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 1920.80 | 1925.23 | 1914.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 1920.80 | 1925.23 | 1914.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1918.50 | 1923.88 | 1915.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1893.40 | 1923.88 | 1915.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1890.00 | 1917.11 | 1912.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:45:00 | 1888.70 | 1917.11 | 1912.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1894.00 | 1912.49 | 1911.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 1897.50 | 1912.49 | 1911.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1895.70 | 1909.13 | 1909.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1895.70 | 1909.13 | 1909.70 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1932.00 | 1912.09 | 1910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 1939.60 | 1917.59 | 1912.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1928.00 | 1931.42 | 1922.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 1928.00 | 1931.42 | 1922.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1925.00 | 1932.37 | 1926.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1918.80 | 1932.37 | 1926.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1898.00 | 1925.49 | 1924.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 1897.00 | 1925.49 | 1924.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1911.00 | 1922.60 | 1923.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 1826.90 | 1892.20 | 1907.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 1747.20 | 1738.43 | 1763.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 1747.20 | 1738.43 | 1763.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1741.80 | 1739.51 | 1759.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 1728.30 | 1736.63 | 1748.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 1764.00 | 1744.88 | 1749.52 | SL hit (close>static) qty=1.00 sl=1760.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1777.00 | 1754.76 | 1753.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 1779.00 | 1759.61 | 1755.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1755.10 | 1758.71 | 1755.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1749.00 | 1756.77 | 1755.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1749.00 | 1756.77 | 1755.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1746.00 | 1754.61 | 1754.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 1746.40 | 1754.61 | 1754.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1744.90 | 1752.92 | 1753.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1737.50 | 1749.84 | 1752.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 1729.40 | 1728.17 | 1734.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:30:00 | 1729.30 | 1728.17 | 1734.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1687.30 | 1720.25 | 1729.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1676.60 | 1700.45 | 1712.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1676.30 | 1688.23 | 1699.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 13:15:00 | 1717.60 | 1706.11 | 1705.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1717.60 | 1706.11 | 1705.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1720.20 | 1710.99 | 1707.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1708.60 | 1711.32 | 1708.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1717.10 | 1712.47 | 1709.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:45:00 | 1707.20 | 1712.47 | 1709.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1707.70 | 1713.32 | 1711.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1701.40 | 1713.32 | 1711.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 1693.50 | 1709.35 | 1709.58 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 1718.90 | 1710.13 | 1709.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 1720.00 | 1712.11 | 1710.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1701.80 | 1710.05 | 1709.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1700.00 | 1708.04 | 1708.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 1683.50 | 1697.92 | 1703.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1686.80 | 1678.94 | 1686.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1678.90 | 1678.93 | 1686.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1676.80 | 1678.78 | 1685.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 1674.80 | 1677.48 | 1683.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1672.10 | 1678.75 | 1682.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1701.60 | 1676.83 | 1677.90 | SL hit (close>static) qty=1.00 sl=1688.70 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 1709.50 | 1683.36 | 1680.77 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1656.80 | 1679.34 | 1680.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1653.20 | 1674.11 | 1678.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1645.10 | 1639.10 | 1648.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1643.00 | 1639.88 | 1647.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1647.30 | 1639.88 | 1647.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1634.10 | 1638.72 | 1646.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 1631.00 | 1638.72 | 1646.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 1628.00 | 1637.74 | 1645.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 1623.40 | 1630.36 | 1637.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1643.40 | 1626.67 | 1625.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1643.40 | 1626.67 | 1625.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1655.20 | 1634.51 | 1629.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1634.60 | 1638.46 | 1634.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1626.60 | 1636.09 | 1633.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1625.00 | 1636.09 | 1633.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1631.90 | 1635.25 | 1633.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1622.30 | 1635.25 | 1633.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1690.80 | 1704.65 | 1695.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 1690.80 | 1704.65 | 1695.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1682.90 | 1700.30 | 1693.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 1686.00 | 1700.30 | 1693.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1660.00 | 1687.52 | 1689.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1639.70 | 1673.86 | 1682.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1670.10 | 1667.74 | 1675.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1677.00 | 1670.63 | 1675.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1682.30 | 1670.63 | 1675.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1674.10 | 1671.32 | 1675.38 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1721.40 | 1684.69 | 1680.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 1745.00 | 1696.75 | 1686.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 1698.90 | 1700.54 | 1691.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 11:45:00 | 1702.20 | 1700.54 | 1691.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1769.20 | 1771.48 | 1753.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 1783.30 | 1771.55 | 1759.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 1776.70 | 1765.20 | 1762.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 1776.70 | 1775.27 | 1769.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1777.40 | 1775.64 | 1770.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1791.10 | 1782.50 | 1775.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 1781.90 | 1782.50 | 1775.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1779.70 | 1786.61 | 1781.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 1776.90 | 1786.61 | 1781.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1786.50 | 1786.59 | 1781.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1780.00 | 1786.59 | 1781.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1788.30 | 1786.93 | 1782.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 1785.50 | 1786.93 | 1782.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1783.10 | 1786.17 | 1782.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1783.10 | 1786.17 | 1782.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1779.00 | 1784.73 | 1782.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1762.10 | 1784.73 | 1782.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1771.30 | 1782.05 | 1781.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1768.70 | 1779.38 | 1780.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 1768.70 | 1779.38 | 1780.01 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1786.40 | 1781.38 | 1780.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1803.70 | 1785.53 | 1782.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1813.50 | 1814.03 | 1804.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1813.50 | 1814.03 | 1804.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1807.20 | 1813.02 | 1806.58 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1800.50 | 1804.07 | 1804.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1791.00 | 1801.46 | 1803.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1733.60 | 1738.23 | 1753.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 1735.20 | 1736.64 | 1751.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 1734.50 | 1734.11 | 1747.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1729.00 | 1731.45 | 1738.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1726.90 | 1730.15 | 1736.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 1720.00 | 1727.66 | 1734.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1708.10 | 1723.20 | 1729.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1646.92 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1648.44 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1647.77 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1642.55 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1634.00 | 1651.71 | 1676.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1622.69 | 1651.71 | 1676.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 1623.00 | 1622.96 | 1648.45 | SL hit (close>ema200) qty=0.50 sl=1622.96 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1567.00 | 1555.28 | 1554.41 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 1539.40 | 1552.11 | 1553.05 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 1560.20 | 1554.83 | 1554.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 1639.00 | 1571.66 | 1561.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 1585.10 | 1591.63 | 1577.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 1572.40 | 1591.63 | 1577.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1590.30 | 1591.37 | 1578.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1616.20 | 1587.43 | 1580.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 1646.10 | 1650.06 | 1650.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1646.10 | 1650.06 | 1650.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1621.40 | 1644.33 | 1647.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 1589.50 | 1581.34 | 1595.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 1589.50 | 1581.34 | 1595.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1586.30 | 1583.03 | 1592.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1585.10 | 1583.03 | 1592.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1588.40 | 1584.11 | 1591.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 1588.40 | 1584.11 | 1591.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1566.70 | 1580.62 | 1589.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 1564.50 | 1577.54 | 1587.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 1565.20 | 1577.54 | 1587.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1561.90 | 1574.03 | 1584.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1486.27 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1486.94 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1483.81 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1509.10 | 1495.34 | 1509.99 | SL hit (close>ema200) qty=0.50 sl=1495.34 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1403.60 | 1375.73 | 1371.98 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 1353.10 | 1369.19 | 1370.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 1347.00 | 1357.67 | 1362.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:30:00 | 1372.50 | 1354.79 | 1359.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 1386.90 | 1361.22 | 1362.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 1386.90 | 1361.22 | 1362.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 13:15:00 | 1386.00 | 1366.17 | 1364.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 14:15:00 | 1392.10 | 1371.36 | 1366.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 1376.90 | 1377.57 | 1371.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:45:00 | 1377.20 | 1377.57 | 1371.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1396.70 | 1381.40 | 1374.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 1400.00 | 1386.09 | 1377.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 1400.00 | 1397.72 | 1394.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:00:00 | 1401.00 | 1398.38 | 1394.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 1371.80 | 1393.06 | 1392.56 | SL hit (close<static) qty=1.00 sl=1372.70 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 1380.00 | 1390.45 | 1391.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1335.90 | 1379.54 | 1386.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 1345.70 | 1324.30 | 1321.58 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1300.70 | 1320.52 | 1320.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1300.20 | 1309.70 | 1314.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1311.50 | 1306.07 | 1311.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1310.10 | 1306.88 | 1311.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1312.90 | 1306.88 | 1311.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1293.90 | 1304.28 | 1309.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 1286.80 | 1302.03 | 1308.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 1307.80 | 1290.20 | 1289.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1307.80 | 1290.20 | 1289.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1375.20 | 1309.25 | 1298.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 11:15:00 | 1432.30 | 1432.67 | 1400.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:00:00 | 1432.30 | 1432.67 | 1400.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1435.90 | 1447.36 | 1431.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 1435.90 | 1447.36 | 1431.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 1438.00 | 1445.49 | 1431.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 1417.00 | 1445.49 | 1431.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1411.80 | 1438.75 | 1429.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 1411.80 | 1438.75 | 1429.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1414.40 | 1433.88 | 1428.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 1418.60 | 1433.88 | 1428.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1419.00 | 1425.87 | 1425.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 1419.00 | 1425.87 | 1425.96 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1430.00 | 1426.52 | 1426.18 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 13:15:00 | 1418.50 | 1424.98 | 1425.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1412.80 | 1420.70 | 1423.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:45:00 | 1422.10 | 1420.47 | 1422.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1412.10 | 1418.80 | 1421.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 1408.10 | 1417.46 | 1420.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 14:45:00 | 1404.40 | 1416.75 | 1420.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1406.90 | 1416.75 | 1420.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1433.60 | 1418.54 | 1420.35 | SL hit (close>static) qty=1.00 sl=1422.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 13:15:00 | 1415.70 | 1407.11 | 1406.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1422.20 | 1413.39 | 1409.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1418.20 | 1419.84 | 1416.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 11:45:00 | 1420.10 | 1419.84 | 1416.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1434.40 | 1422.75 | 1417.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:30:00 | 1452.30 | 1428.50 | 1420.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 1445.00 | 1453.04 | 1453.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 1445.00 | 1453.04 | 1453.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1440.70 | 1446.71 | 1450.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1439.90 | 1417.00 | 1425.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1440.20 | 1421.64 | 1426.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1440.00 | 1421.64 | 1426.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1456.00 | 1431.91 | 1430.75 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 1431.50 | 1437.39 | 1437.72 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1442.20 | 1438.35 | 1438.12 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 1435.00 | 1437.98 | 1438.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1430.60 | 1436.51 | 1437.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1426.00 | 1424.54 | 1428.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1424.90 | 1424.61 | 1428.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1418.00 | 1420.55 | 1426.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 1417.30 | 1414.99 | 1421.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 1415.00 | 1417.23 | 1420.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1417.70 | 1416.87 | 1418.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1423.00 | 1418.09 | 1418.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 1424.90 | 1418.09 | 1418.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1419.50 | 1418.38 | 1418.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 1422.70 | 1419.24 | 1419.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 1422.70 | 1419.24 | 1419.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1443.80 | 1424.15 | 1421.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 15:15:00 | 1099.25 | 2024-05-18 11:15:00 | 1209.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1087.25 | 2024-06-06 10:15:00 | 1162.80 | STOP_HIT | 1.00 | -6.95% |
| SELL | retest2 | 2024-06-06 10:00:00 | 1138.10 | 2024-06-06 10:15:00 | 1162.80 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-06-18 09:15:00 | 1265.60 | 2024-06-18 11:15:00 | 1239.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-06-19 14:30:00 | 1241.90 | 2024-06-21 09:15:00 | 1286.30 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-06-19 15:00:00 | 1239.25 | 2024-06-21 09:15:00 | 1286.30 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-06-20 10:30:00 | 1242.00 | 2024-06-21 09:15:00 | 1286.30 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-06-20 14:00:00 | 1242.00 | 2024-06-21 09:15:00 | 1286.30 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2024-06-26 09:30:00 | 1296.30 | 2024-07-05 09:15:00 | 1327.90 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2024-06-26 10:15:00 | 1297.05 | 2024-07-05 09:15:00 | 1327.90 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2024-06-26 14:30:00 | 1339.90 | 2024-07-05 09:15:00 | 1327.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1497.15 | 2024-08-01 13:15:00 | 1472.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-08-01 12:15:00 | 1484.95 | 2024-08-01 13:15:00 | 1472.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-08-23 10:30:00 | 1646.95 | 2024-08-28 12:15:00 | 1646.55 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-08-23 11:00:00 | 1645.30 | 2024-08-28 12:15:00 | 1646.55 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-08-23 12:00:00 | 1646.00 | 2024-08-28 12:15:00 | 1646.55 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-26 09:30:00 | 1647.00 | 2024-08-28 12:15:00 | 1646.55 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1665.00 | 2024-08-28 12:15:00 | 1646.55 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-10 14:30:00 | 1575.00 | 2024-09-13 15:15:00 | 1575.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-09-13 15:00:00 | 1578.45 | 2024-09-13 15:15:00 | 1575.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-09-25 11:15:00 | 1565.75 | 2024-09-26 10:15:00 | 1557.20 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-09-26 09:15:00 | 1568.00 | 2024-09-26 10:15:00 | 1557.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-10-01 09:30:00 | 1592.05 | 2024-10-03 09:15:00 | 1572.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-04 13:30:00 | 1547.10 | 2024-10-09 11:15:00 | 1551.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-10-04 14:15:00 | 1542.25 | 2024-10-09 11:15:00 | 1551.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-10-09 11:00:00 | 1546.60 | 2024-10-09 11:15:00 | 1551.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1636.00 | 2024-10-15 11:15:00 | 1600.05 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-10-15 10:30:00 | 1613.75 | 2024-10-15 11:15:00 | 1600.05 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-10-15 14:00:00 | 1610.55 | 2024-10-15 15:15:00 | 1599.85 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-10-17 15:15:00 | 1572.25 | 2024-10-21 14:15:00 | 1493.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 15:15:00 | 1572.25 | 2024-10-23 10:15:00 | 1502.25 | STOP_HIT | 0.50 | 4.45% |
| BUY | retest2 | 2024-11-04 12:15:00 | 1545.95 | 2024-11-11 10:15:00 | 1586.95 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest2 | 2024-11-27 09:15:00 | 1607.00 | 2024-12-03 09:15:00 | 1767.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 12:00:00 | 1601.45 | 2024-12-03 09:15:00 | 1761.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 12:45:00 | 1601.85 | 2024-12-03 09:15:00 | 1762.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 13:45:00 | 1611.65 | 2024-12-03 09:15:00 | 1772.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 09:15:00 | 1611.40 | 2024-12-03 09:15:00 | 1772.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1689.40 | 2025-01-10 09:15:00 | 1604.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1689.40 | 2025-01-13 13:15:00 | 1520.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1547.80 | 2025-01-23 12:15:00 | 1566.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-01-31 12:45:00 | 1513.90 | 2025-02-01 12:15:00 | 1480.75 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-01-31 15:15:00 | 1512.20 | 2025-02-01 12:15:00 | 1480.75 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-02-01 15:00:00 | 1522.70 | 2025-02-03 10:15:00 | 1487.30 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-02-10 14:00:00 | 1643.30 | 2025-02-11 09:15:00 | 1614.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-10 14:45:00 | 1647.35 | 2025-02-11 09:15:00 | 1614.80 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1549.95 | 2025-02-14 10:15:00 | 1611.85 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-02-14 12:15:00 | 1553.40 | 2025-02-17 09:15:00 | 1475.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 12:15:00 | 1553.40 | 2025-02-17 13:15:00 | 1538.75 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2025-02-21 09:30:00 | 1526.00 | 2025-02-21 11:15:00 | 1525.10 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-03-06 09:15:00 | 1445.55 | 2025-03-10 15:15:00 | 1426.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-06 14:30:00 | 1441.80 | 2025-03-10 15:15:00 | 1426.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-03-07 09:15:00 | 1437.85 | 2025-03-10 15:15:00 | 1426.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-03-10 14:45:00 | 1437.00 | 2025-03-10 15:15:00 | 1426.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1538.80 | 2025-04-25 10:15:00 | 1566.90 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2025-04-28 14:00:00 | 1622.00 | 2025-04-30 11:15:00 | 1594.20 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-04-29 09:15:00 | 1637.10 | 2025-04-30 11:15:00 | 1594.20 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-04-29 09:45:00 | 1626.10 | 2025-04-30 11:15:00 | 1594.20 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-04-29 10:30:00 | 1628.40 | 2025-04-30 11:15:00 | 1594.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1572.00 | 2025-05-09 09:15:00 | 1493.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1572.00 | 2025-05-09 15:15:00 | 1520.00 | STOP_HIT | 0.50 | 3.31% |
| BUY | retest2 | 2025-05-21 09:45:00 | 1710.00 | 2025-05-27 09:15:00 | 1698.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1721.10 | 2025-05-27 09:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1808.30 | 2025-06-10 09:15:00 | 1979.45 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2025-06-06 11:15:00 | 1799.50 | 2025-06-10 09:15:00 | 1975.38 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2025-06-06 14:30:00 | 1795.80 | 2025-06-10 09:15:00 | 1980.00 | TARGET_HIT | 1.00 | 10.26% |
| BUY | retest2 | 2025-06-06 15:00:00 | 1800.00 | 2025-06-13 12:15:00 | 1885.90 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-06-13 11:00:00 | 1892.00 | 2025-06-13 12:15:00 | 1885.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-06-16 14:00:00 | 1934.50 | 2025-06-18 11:15:00 | 1901.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-18 10:00:00 | 1910.20 | 2025-06-18 11:15:00 | 1901.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-07-01 09:15:00 | 2011.60 | 2025-07-01 12:15:00 | 1974.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-09 13:30:00 | 1977.00 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-07-09 14:30:00 | 1974.50 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-10 10:30:00 | 1975.60 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1973.40 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-22 11:00:00 | 1899.70 | 2025-07-25 14:15:00 | 1804.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 1899.70 | 2025-07-28 13:15:00 | 1820.00 | STOP_HIT | 0.50 | 4.20% |
| BUY | retest2 | 2025-08-01 10:30:00 | 1972.40 | 2025-08-04 10:15:00 | 1916.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-08-01 12:15:00 | 1970.70 | 2025-08-04 10:15:00 | 1916.50 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-08 09:15:00 | 1969.70 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-08-08 13:00:00 | 1974.60 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-08-11 10:15:00 | 1968.80 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-08-12 10:00:00 | 1970.80 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-12 12:30:00 | 1987.50 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1985.00 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-12 14:30:00 | 1984.40 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-08-13 09:15:00 | 1986.00 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1942.60 | 2025-09-01 15:15:00 | 1912.70 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-08-21 10:00:00 | 1954.90 | 2025-09-01 15:15:00 | 1912.70 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1911.60 | 2025-09-04 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1912.70 | 2025-09-04 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-19 09:15:00 | 2130.00 | 2025-09-23 12:15:00 | 2114.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest1 | 2025-09-26 09:15:00 | 2023.70 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.41% |
| SELL | retest1 | 2025-09-26 10:00:00 | 2020.40 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest1 | 2025-09-26 10:45:00 | 2019.20 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-10-03 09:15:00 | 1951.00 | 2025-10-09 09:15:00 | 1942.00 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-10-06 13:15:00 | 1957.50 | 2025-10-09 09:15:00 | 1942.00 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1929.00 | 2025-10-16 09:15:00 | 1964.70 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-29 11:15:00 | 1897.50 | 2025-10-29 11:15:00 | 1895.70 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-12 09:30:00 | 1728.30 | 2025-11-12 12:15:00 | 1764.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1676.60 | 2025-11-20 13:15:00 | 1717.60 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-11-20 09:30:00 | 1676.30 | 2025-11-20 13:15:00 | 1717.60 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1676.80 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-27 14:30:00 | 1674.80 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-28 11:15:00 | 1672.10 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-05 10:15:00 | 1631.00 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-05 11:15:00 | 1628.00 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-08 10:00:00 | 1623.40 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 15:00:00 | 1783.30 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-30 10:00:00 | 1776.70 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-30 14:30:00 | 1776.70 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-31 09:45:00 | 1777.40 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1733.60 | 2026-01-20 13:15:00 | 1646.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1735.20 | 2026-01-20 13:15:00 | 1648.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 1734.50 | 2026-01-20 13:15:00 | 1647.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1729.00 | 2026-01-20 13:15:00 | 1642.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1720.00 | 2026-01-21 09:15:00 | 1634.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1708.10 | 2026-01-21 09:15:00 | 1622.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1733.60 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1735.20 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2026-01-13 15:00:00 | 1734.50 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1729.00 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1720.00 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1708.10 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest2 | 2026-02-03 09:15:00 | 1616.20 | 2026-02-11 15:15:00 | 1646.10 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2026-02-17 13:30:00 | 1564.50 | 2026-02-20 09:15:00 | 1486.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1565.20 | 2026-02-20 09:15:00 | 1486.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1561.90 | 2026-02-20 09:15:00 | 1483.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:30:00 | 1564.50 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1565.20 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1561.90 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest2 | 2026-03-10 15:00:00 | 1400.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-12 13:00:00 | 1400.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-12 14:00:00 | 1401.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-03-20 14:15:00 | 1286.80 | 2026-03-24 14:15:00 | 1307.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1418.60 | 2026-04-02 15:15:00 | 1419.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2026-04-07 14:15:00 | 1408.10 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-04-07 14:45:00 | 1404.40 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-07 15:15:00 | 1406.90 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1408.10 | 2026-04-13 13:15:00 | 1415.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1379.40 | 2026-04-13 13:15:00 | 1415.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-16 13:30:00 | 1452.30 | 2026-04-22 11:15:00 | 1445.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-05-04 10:30:00 | 1418.00 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-05-04 14:30:00 | 1417.30 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-05-05 10:30:00 | 1415.00 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-05-06 11:15:00 | 1417.70 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.35% |
