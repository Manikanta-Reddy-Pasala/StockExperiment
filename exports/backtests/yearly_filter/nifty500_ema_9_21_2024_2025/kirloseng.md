# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1736.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 126 |
| ALERT1 | 88 |
| ALERT2 | 87 |
| ALERT2_SKIP | 40 |
| ALERT3 | 210 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 116 |
| PARTIAL | 24 |
| TARGET_HIT | 16 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 81
- **Target hits / Stop hits / Partials:** 16 / 102 / 24
- **Avg / median % per leg:** 1.28% / -0.75%
- **Sum % (uncompounded):** 181.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 12 | 24.5% | 9 | 40 | 0 | 0.28% | 13.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.98% | -1.0% |
| BUY @ 3rd Alert (retest2) | 48 | 12 | 25.0% | 9 | 39 | 0 | 0.31% | 14.7% |
| SELL (all) | 93 | 49 | 52.7% | 7 | 62 | 24 | 1.80% | 167.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.69% | -1.7% |
| SELL @ 3rd Alert (retest2) | 92 | 49 | 53.3% | 7 | 61 | 24 | 1.84% | 169.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.33% | -2.7% |
| retest2 (combined) | 140 | 61 | 43.6% | 16 | 100 | 24 | 1.31% | 184.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 1256.05 | 1286.24 | 1289.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 12:15:00 | 1251.00 | 1279.19 | 1285.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 1300.00 | 1278.58 | 1284.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 14:15:00 | 1300.00 | 1278.58 | 1284.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1300.00 | 1278.58 | 1284.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 1300.00 | 1278.58 | 1284.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 1300.00 | 1282.86 | 1285.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 1291.60 | 1282.86 | 1285.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:15:00 | 1227.02 | 1251.73 | 1263.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 1266.95 | 1241.47 | 1252.43 | SL hit (close>ema200) qty=0.50 sl=1241.47 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 1240.00 | 1224.21 | 1223.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1300.20 | 1239.41 | 1230.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1258.60 | 1288.03 | 1266.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1258.60 | 1288.03 | 1266.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1258.60 | 1288.03 | 1266.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1259.50 | 1288.03 | 1266.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1204.70 | 1271.36 | 1261.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 1203.75 | 1271.36 | 1261.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1191.40 | 1255.37 | 1254.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1191.40 | 1255.37 | 1254.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 1224.45 | 1249.19 | 1252.11 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1230.00 | 1215.84 | 1215.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 15:15:00 | 1288.85 | 1263.92 | 1251.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 1286.55 | 1288.67 | 1274.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 1286.55 | 1288.67 | 1274.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1286.55 | 1288.67 | 1274.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 1284.60 | 1288.67 | 1274.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1279.65 | 1285.67 | 1275.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 1284.75 | 1285.67 | 1275.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 1309.00 | 1286.15 | 1279.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 1413.23 | 1329.16 | 1308.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 1294.70 | 1313.32 | 1314.51 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 15:15:00 | 1324.65 | 1316.42 | 1315.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 1335.85 | 1320.12 | 1317.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 15:15:00 | 1330.00 | 1334.71 | 1327.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1395.30 | 1334.71 | 1327.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1390.10 | 1398.15 | 1384.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1381.65 | 1394.85 | 1384.63 | SL hit (close<ema400) qty=1.00 sl=1384.63 alert=retest1 |

### Cycle 7 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 1405.00 | 1410.89 | 1411.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 1388.80 | 1404.73 | 1408.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 15:15:00 | 1402.00 | 1401.80 | 1405.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 09:15:00 | 1399.00 | 1401.80 | 1405.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1399.00 | 1401.24 | 1404.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:00:00 | 1394.90 | 1399.29 | 1402.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:00:00 | 1393.90 | 1398.21 | 1402.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:00:00 | 1392.95 | 1393.94 | 1397.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1394.00 | 1393.54 | 1396.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1387.00 | 1392.23 | 1395.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 1373.05 | 1388.40 | 1393.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 1377.05 | 1379.33 | 1384.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 1374.50 | 1371.21 | 1376.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 12:15:00 | 1325.15 | 1347.76 | 1359.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 13:15:00 | 1324.20 | 1343.35 | 1356.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 13:15:00 | 1323.30 | 1343.35 | 1356.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 13:15:00 | 1324.30 | 1343.35 | 1356.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 1342.95 | 1338.43 | 1350.21 | SL hit (close>ema200) qty=0.50 sl=1338.43 alert=retest2 |

### Cycle 8 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 1239.50 | 1225.22 | 1225.04 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1222.00 | 1225.24 | 1225.55 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 10:15:00 | 1228.00 | 1225.88 | 1225.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 14:15:00 | 1233.45 | 1228.68 | 1227.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 1237.50 | 1238.20 | 1233.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 12:15:00 | 1223.80 | 1238.20 | 1233.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1209.55 | 1232.47 | 1230.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 1209.55 | 1232.47 | 1230.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 1208.00 | 1227.58 | 1228.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 1196.20 | 1221.30 | 1225.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1146.70 | 1138.80 | 1163.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1127.00 | 1131.95 | 1157.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1146.00 | 1118.65 | 1137.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 1146.00 | 1118.65 | 1137.49 | SL hit (close>ema400) qty=1.00 sl=1137.49 alert=retest1 |

### Cycle 12 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1168.40 | 1147.46 | 1145.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1189.85 | 1158.71 | 1151.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1160.60 | 1165.01 | 1156.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 1160.60 | 1165.01 | 1156.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1149.00 | 1161.72 | 1156.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1149.00 | 1161.72 | 1156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1159.95 | 1161.36 | 1156.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1179.95 | 1161.36 | 1156.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-19 14:15:00 | 1297.95 | 1268.76 | 1262.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 1291.80 | 1325.97 | 1329.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 1286.40 | 1297.60 | 1307.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 1304.75 | 1295.85 | 1303.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 1304.75 | 1295.85 | 1303.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1304.75 | 1295.85 | 1303.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 1304.75 | 1295.85 | 1303.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1313.00 | 1299.28 | 1304.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1297.05 | 1298.73 | 1303.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:00:00 | 1296.50 | 1298.73 | 1303.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 1296.50 | 1300.71 | 1303.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 1312.10 | 1305.11 | 1305.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1312.10 | 1305.11 | 1305.10 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 1290.60 | 1302.76 | 1304.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1284.40 | 1297.02 | 1301.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1304.25 | 1291.08 | 1295.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 1304.25 | 1291.08 | 1295.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1304.25 | 1291.08 | 1295.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 1302.60 | 1291.08 | 1295.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1301.50 | 1293.16 | 1295.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:15:00 | 1302.60 | 1293.16 | 1295.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1311.35 | 1298.18 | 1297.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1339.20 | 1306.39 | 1301.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 1310.50 | 1313.63 | 1307.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 1310.50 | 1313.63 | 1307.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1358.45 | 1351.56 | 1335.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:45:00 | 1353.70 | 1351.56 | 1335.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1399.90 | 1361.44 | 1342.51 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 1324.25 | 1346.27 | 1348.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 1318.00 | 1340.62 | 1345.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 1340.80 | 1337.71 | 1342.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 1340.80 | 1337.71 | 1342.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 1340.80 | 1337.71 | 1342.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:00:00 | 1340.80 | 1337.71 | 1342.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1330.85 | 1336.34 | 1341.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:45:00 | 1327.60 | 1332.70 | 1338.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 1261.22 | 1270.33 | 1278.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1265.00 | 1253.97 | 1263.98 | SL hit (close>ema200) qty=0.50 sl=1253.97 alert=retest2 |

### Cycle 18 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 1244.95 | 1221.37 | 1221.24 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 1219.05 | 1223.86 | 1224.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1211.75 | 1221.44 | 1223.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 1230.05 | 1223.16 | 1223.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 1230.05 | 1223.16 | 1223.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 1230.05 | 1223.16 | 1223.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 1230.05 | 1223.16 | 1223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 1225.00 | 1223.53 | 1223.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 1189.05 | 1223.53 | 1223.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1129.60 | 1157.64 | 1172.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 1133.80 | 1128.95 | 1144.68 | SL hit (close>ema200) qty=0.50 sl=1128.95 alert=retest2 |

### Cycle 20 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1172.75 | 1151.24 | 1150.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 1198.40 | 1179.89 | 1171.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 12:15:00 | 1210.00 | 1214.18 | 1201.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 13:00:00 | 1210.00 | 1214.18 | 1201.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1210.90 | 1213.52 | 1202.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:15:00 | 1202.20 | 1213.52 | 1202.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1203.00 | 1211.42 | 1202.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 1199.50 | 1211.42 | 1202.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1209.50 | 1211.03 | 1203.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 1208.40 | 1211.03 | 1203.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1195.35 | 1207.90 | 1202.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1195.35 | 1207.90 | 1202.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1202.00 | 1206.72 | 1202.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 1207.25 | 1206.72 | 1202.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1191.75 | 1203.72 | 1201.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 1191.85 | 1203.72 | 1201.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1209.55 | 1204.89 | 1202.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 1191.85 | 1204.89 | 1202.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1209.15 | 1206.45 | 1203.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1184.10 | 1206.45 | 1203.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1191.45 | 1203.45 | 1202.53 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1194.35 | 1201.63 | 1201.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 1180.75 | 1193.82 | 1197.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1122.85 | 1122.18 | 1140.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1122.85 | 1122.18 | 1140.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1131.15 | 1121.04 | 1133.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:00:00 | 1131.15 | 1121.04 | 1133.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 1129.05 | 1122.64 | 1133.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 1121.25 | 1122.64 | 1133.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1109.30 | 1119.97 | 1131.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 1101.10 | 1116.20 | 1128.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 1046.04 | 1079.63 | 1103.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 1044.20 | 1036.19 | 1064.50 | SL hit (close>ema200) qty=0.50 sl=1036.19 alert=retest2 |

### Cycle 22 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1090.50 | 1067.00 | 1065.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 1097.00 | 1073.00 | 1068.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1122.35 | 1140.73 | 1125.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1122.35 | 1140.73 | 1125.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1122.35 | 1140.73 | 1125.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 1115.05 | 1140.73 | 1125.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1126.65 | 1137.91 | 1125.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 1123.35 | 1137.91 | 1125.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1134.40 | 1137.21 | 1126.43 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 1112.50 | 1123.59 | 1124.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 14:15:00 | 1109.60 | 1119.25 | 1122.14 | Break + close below crossover candle low |

### Cycle 24 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1171.70 | 1127.94 | 1125.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 1182.90 | 1138.93 | 1130.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 1175.90 | 1179.78 | 1161.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 1175.90 | 1179.78 | 1161.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1190.85 | 1179.37 | 1168.08 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 1158.50 | 1170.96 | 1171.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 1146.90 | 1166.15 | 1169.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1136.65 | 1120.45 | 1136.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1136.65 | 1120.45 | 1136.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1136.65 | 1120.45 | 1136.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1136.65 | 1120.45 | 1136.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1126.50 | 1121.66 | 1135.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1142.25 | 1121.66 | 1135.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1161.25 | 1129.58 | 1137.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 1161.25 | 1129.58 | 1137.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1178.15 | 1139.29 | 1141.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1178.15 | 1139.29 | 1141.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 13:15:00 | 1161.70 | 1143.77 | 1143.42 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 1136.80 | 1142.38 | 1142.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 15:15:00 | 1130.00 | 1139.90 | 1141.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1140.70 | 1138.22 | 1140.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 11:15:00 | 1140.70 | 1138.22 | 1140.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 1140.70 | 1138.22 | 1140.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1128.10 | 1136.36 | 1139.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1158.55 | 1139.25 | 1139.55 | SL hit (close>static) qty=1.00 sl=1145.40 alert=retest2 |

### Cycle 28 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1159.75 | 1143.35 | 1141.38 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 13:15:00 | 1124.45 | 1137.43 | 1139.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1111.45 | 1130.26 | 1135.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 10:15:00 | 1078.50 | 1074.24 | 1095.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:00:00 | 1078.50 | 1074.24 | 1095.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1101.75 | 1077.00 | 1086.82 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1119.10 | 1094.52 | 1092.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 1142.00 | 1114.17 | 1105.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 1119.90 | 1123.50 | 1113.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 14:00:00 | 1119.90 | 1123.50 | 1113.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1122.00 | 1141.91 | 1132.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 1122.00 | 1141.91 | 1132.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1117.45 | 1137.02 | 1131.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:45:00 | 1125.90 | 1130.71 | 1129.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 1122.55 | 1127.71 | 1127.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 14:15:00 | 1122.55 | 1127.71 | 1127.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 1098.80 | 1121.09 | 1124.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1107.25 | 1101.83 | 1111.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1107.25 | 1101.83 | 1111.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1107.25 | 1101.83 | 1111.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1103.75 | 1101.83 | 1111.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1107.90 | 1103.05 | 1111.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 1104.50 | 1103.05 | 1111.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1098.85 | 1102.21 | 1109.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1093.90 | 1106.75 | 1109.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:30:00 | 1094.85 | 1101.99 | 1106.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:30:00 | 1096.90 | 1099.13 | 1102.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 1097.00 | 1099.97 | 1101.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1096.95 | 1099.37 | 1101.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 1106.95 | 1099.37 | 1101.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1115.80 | 1102.15 | 1102.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 1115.80 | 1102.15 | 1102.29 | SL hit (close>static) qty=1.00 sl=1110.95 alert=retest2 |

### Cycle 32 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1120.35 | 1105.79 | 1103.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1159.05 | 1121.78 | 1112.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 15:15:00 | 1170.00 | 1173.11 | 1157.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:15:00 | 1168.40 | 1173.11 | 1157.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1152.00 | 1168.89 | 1156.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 1152.00 | 1168.89 | 1156.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1160.90 | 1167.29 | 1157.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:15:00 | 1163.80 | 1167.29 | 1157.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 1164.40 | 1165.55 | 1158.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1163.15 | 1164.58 | 1159.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 10:30:00 | 1165.00 | 1163.43 | 1159.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1144.80 | 1159.70 | 1158.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 1144.80 | 1159.70 | 1158.37 | SL hit (close<static) qty=1.00 sl=1147.55 alert=retest2 |

### Cycle 33 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1145.75 | 1155.15 | 1156.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 1141.25 | 1151.01 | 1154.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 1112.40 | 1109.96 | 1122.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 14:45:00 | 1114.20 | 1109.96 | 1122.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1111.50 | 1110.56 | 1120.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 1119.60 | 1110.56 | 1120.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1013.60 | 1034.41 | 1053.99 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 1031.55 | 1025.05 | 1024.47 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 1014.30 | 1024.65 | 1025.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1011.25 | 1018.75 | 1021.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 1020.00 | 1019.00 | 1021.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 1020.00 | 1019.00 | 1021.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1018.70 | 1016.63 | 1019.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:30:00 | 1016.45 | 1016.63 | 1019.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1029.80 | 1019.26 | 1020.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 1029.80 | 1019.26 | 1020.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 1025.00 | 1020.41 | 1021.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 1017.45 | 1020.41 | 1021.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 1021.50 | 1015.96 | 1016.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 1030.40 | 1018.85 | 1018.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 1030.40 | 1018.85 | 1018.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 1037.40 | 1024.29 | 1020.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 1046.00 | 1046.36 | 1037.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 1024.60 | 1046.36 | 1037.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1021.75 | 1041.44 | 1035.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1021.75 | 1041.44 | 1035.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1005.65 | 1034.28 | 1033.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1005.65 | 1034.28 | 1033.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 993.90 | 1026.21 | 1029.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 985.10 | 1001.34 | 1010.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 919.00 | 918.12 | 936.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 919.00 | 918.12 | 936.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 931.70 | 919.59 | 932.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 931.70 | 919.59 | 932.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 939.30 | 923.53 | 933.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 940.80 | 923.53 | 933.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 941.45 | 927.11 | 933.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 941.00 | 927.11 | 933.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 951.30 | 934.41 | 936.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 949.00 | 934.41 | 936.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 948.05 | 939.52 | 938.36 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 15:15:00 | 942.00 | 942.56 | 942.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 14:15:00 | 931.80 | 937.61 | 939.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 13:15:00 | 923.85 | 921.17 | 926.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 13:15:00 | 923.85 | 921.17 | 926.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 923.85 | 921.17 | 926.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:00:00 | 923.85 | 921.17 | 926.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 931.70 | 923.27 | 926.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 931.70 | 923.27 | 926.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 925.00 | 923.62 | 926.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 910.05 | 923.62 | 926.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 923.45 | 925.94 | 927.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 922.70 | 925.94 | 927.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 922.80 | 926.35 | 927.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 930.00 | 927.08 | 927.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 930.00 | 927.08 | 927.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 922.70 | 926.20 | 926.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 917.25 | 922.44 | 925.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 864.55 | 893.10 | 907.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 877.28 | 893.10 | 907.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 876.57 | 893.10 | 907.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 876.66 | 893.10 | 907.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 871.39 | 893.10 | 907.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 831.11 | 871.22 | 889.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 40 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 890.20 | 876.37 | 876.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 899.95 | 887.62 | 882.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 866.20 | 898.22 | 893.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 866.20 | 898.22 | 893.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 866.20 | 898.22 | 893.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 866.20 | 898.22 | 893.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 877.10 | 894.00 | 891.59 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 873.90 | 889.98 | 889.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 855.00 | 880.59 | 885.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 10:15:00 | 854.40 | 853.20 | 865.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 854.50 | 853.54 | 860.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 854.50 | 853.54 | 860.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 854.50 | 853.54 | 860.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 855.00 | 853.83 | 859.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 858.75 | 853.83 | 859.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 849.55 | 852.78 | 856.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:45:00 | 847.35 | 850.78 | 855.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 804.98 | 812.73 | 823.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 762.62 | 770.80 | 795.36 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 42 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 601.40 | 599.34 | 599.34 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 593.80 | 598.95 | 599.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 585.85 | 594.28 | 596.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 13:15:00 | 566.35 | 565.54 | 578.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 14:00:00 | 566.35 | 565.54 | 578.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 576.30 | 567.69 | 577.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 576.30 | 567.69 | 577.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 577.05 | 569.56 | 577.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 584.40 | 569.56 | 577.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 572.55 | 570.16 | 577.36 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 591.15 | 579.10 | 578.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 603.45 | 591.31 | 585.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 650.40 | 651.01 | 637.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:45:00 | 650.30 | 651.01 | 637.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 644.90 | 649.45 | 641.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 653.95 | 650.33 | 642.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:30:00 | 650.75 | 650.48 | 643.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 634.30 | 641.34 | 641.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 634.30 | 641.34 | 641.56 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 648.00 | 641.03 | 640.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 658.10 | 646.03 | 643.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 652.75 | 652.88 | 648.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 15:00:00 | 652.75 | 652.88 | 648.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 671.95 | 656.23 | 650.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:30:00 | 674.20 | 659.08 | 652.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 698.45 | 665.25 | 658.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-18 11:15:00 | 741.62 | 696.73 | 675.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 730.90 | 737.37 | 737.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 720.00 | 732.77 | 735.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 733.05 | 731.71 | 734.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 733.05 | 731.71 | 734.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 733.05 | 731.71 | 734.66 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 760.50 | 732.34 | 728.99 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 726.25 | 734.57 | 734.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 717.85 | 731.22 | 733.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 713.45 | 700.33 | 712.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 713.45 | 700.33 | 712.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 713.45 | 700.33 | 712.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 713.45 | 700.33 | 712.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 712.00 | 702.66 | 712.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 721.65 | 702.66 | 712.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 713.75 | 704.88 | 712.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 707.45 | 705.62 | 712.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:00:00 | 710.45 | 707.48 | 711.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 709.05 | 709.65 | 712.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 721.85 | 709.58 | 708.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 721.85 | 709.58 | 708.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 728.00 | 713.27 | 710.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 11:15:00 | 748.55 | 750.34 | 741.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 12:00:00 | 748.55 | 750.34 | 741.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 753.75 | 749.35 | 743.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:30:00 | 760.80 | 753.34 | 748.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:15:00 | 758.00 | 756.61 | 751.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 757.60 | 753.44 | 751.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 757.95 | 755.21 | 753.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 759.95 | 756.48 | 754.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 729.30 | 749.30 | 751.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 729.30 | 749.30 | 751.74 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 750.25 | 744.77 | 744.03 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 739.20 | 744.88 | 745.12 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 746.80 | 744.19 | 743.96 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 734.30 | 743.15 | 743.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 726.50 | 736.91 | 740.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 725.75 | 721.77 | 727.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 725.75 | 721.77 | 727.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 725.75 | 721.77 | 727.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 711.80 | 718.73 | 724.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 676.21 | 702.64 | 715.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 722.10 | 687.82 | 698.35 | SL hit (close>ema200) qty=0.50 sl=687.82 alert=retest2 |

### Cycle 56 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 715.85 | 703.22 | 702.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 716.65 | 707.93 | 705.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 876.25 | 876.57 | 835.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 872.50 | 876.57 | 835.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 876.95 | 882.06 | 871.26 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 857.90 | 867.42 | 867.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 855.25 | 864.45 | 866.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 858.10 | 853.38 | 857.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 855.30 | 853.38 | 857.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 856.00 | 853.91 | 857.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 856.00 | 853.91 | 857.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 878.25 | 857.99 | 858.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 878.25 | 857.99 | 858.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 877.50 | 861.89 | 860.19 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 857.25 | 862.18 | 862.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 850.35 | 858.85 | 860.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 872.85 | 861.65 | 861.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 868.10 | 861.65 | 861.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 891.20 | 867.56 | 864.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 908.45 | 887.89 | 877.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 13:15:00 | 897.35 | 898.99 | 887.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:45:00 | 900.00 | 898.99 | 887.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 895.00 | 899.94 | 890.56 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 881.60 | 888.22 | 888.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 875.05 | 883.11 | 885.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 879.50 | 876.29 | 880.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 882.30 | 876.29 | 880.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 879.70 | 876.97 | 880.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 879.75 | 876.97 | 880.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 879.65 | 877.51 | 880.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 875.00 | 877.01 | 879.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 874.90 | 876.48 | 878.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:45:00 | 874.30 | 875.65 | 877.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 873.00 | 874.31 | 876.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 890.00 | 874.70 | 875.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 890.00 | 874.70 | 875.16 | SL hit (close>static) qty=1.00 sl=880.25 alert=retest2 |

### Cycle 62 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 888.25 | 877.41 | 876.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 912.45 | 886.27 | 880.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 899.05 | 900.37 | 892.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 898.80 | 900.37 | 892.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 903.75 | 900.84 | 895.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 932.20 | 903.61 | 899.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 917.40 | 903.61 | 899.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:30:00 | 922.15 | 910.94 | 903.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 890.55 | 898.68 | 899.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 890.55 | 898.68 | 899.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 11:15:00 | 884.50 | 895.84 | 898.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 871.00 | 868.92 | 879.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 871.00 | 868.92 | 879.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 884.75 | 871.78 | 878.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 884.75 | 871.78 | 878.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 875.40 | 872.51 | 878.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 873.10 | 872.51 | 878.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 870.15 | 871.62 | 877.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 829.44 | 844.31 | 852.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 826.64 | 844.31 | 852.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 842.75 | 841.62 | 848.13 | SL hit (close>ema200) qty=0.50 sl=841.62 alert=retest2 |

### Cycle 64 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 857.60 | 848.16 | 847.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 865.50 | 854.79 | 851.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 855.00 | 855.18 | 852.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 855.00 | 855.18 | 852.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 853.75 | 854.90 | 852.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 853.75 | 854.90 | 852.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 853.50 | 854.62 | 853.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 864.30 | 854.62 | 853.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:30:00 | 859.70 | 862.04 | 859.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:45:00 | 860.40 | 861.99 | 860.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 864.00 | 861.21 | 859.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 864.00 | 861.77 | 860.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 853.20 | 861.77 | 860.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 856.65 | 860.74 | 859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 850.40 | 860.74 | 859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 851.30 | 858.85 | 859.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 851.30 | 858.85 | 859.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 11:15:00 | 849.90 | 857.06 | 858.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 857.70 | 853.50 | 855.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 854.95 | 853.72 | 855.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 854.25 | 854.27 | 855.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 854.00 | 854.53 | 855.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 854.45 | 855.26 | 855.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 841.95 | 852.60 | 854.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 855.85 | 850.88 | 850.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 855.85 | 850.88 | 850.24 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 844.90 | 849.73 | 850.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 840.10 | 846.14 | 848.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 839.10 | 838.54 | 842.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 837.85 | 838.54 | 842.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 839.60 | 838.75 | 842.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 835.10 | 838.02 | 841.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 835.30 | 837.19 | 840.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 845.60 | 837.94 | 840.08 | SL hit (close>static) qty=1.00 sl=845.55 alert=retest2 |

### Cycle 68 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 854.10 | 843.50 | 842.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 12:15:00 | 865.40 | 847.88 | 844.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 912.35 | 913.70 | 889.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:30:00 | 915.20 | 913.70 | 889.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 930.50 | 940.06 | 933.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 928.80 | 937.81 | 932.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 920.00 | 934.25 | 931.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 920.00 | 934.25 | 931.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 917.20 | 928.14 | 929.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 912.25 | 916.93 | 920.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 919.10 | 913.48 | 917.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 917.80 | 913.48 | 917.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 915.85 | 913.95 | 917.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 912.75 | 913.71 | 916.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 913.75 | 913.75 | 915.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 922.95 | 915.59 | 916.49 | SL hit (close>static) qty=1.00 sl=919.85 alert=retest2 |

### Cycle 70 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 920.25 | 917.62 | 917.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 939.80 | 922.06 | 919.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 917.55 | 924.62 | 921.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 917.70 | 924.62 | 921.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 912.00 | 922.09 | 920.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 912.00 | 922.09 | 920.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 911.80 | 920.03 | 919.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 913.00 | 920.03 | 919.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 916.15 | 919.26 | 919.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 907.90 | 915.60 | 917.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 909.55 | 905.07 | 909.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 909.00 | 905.07 | 909.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 914.90 | 907.04 | 909.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 914.90 | 907.04 | 909.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 916.05 | 908.84 | 910.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 916.05 | 908.84 | 910.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 923.00 | 911.67 | 911.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 924.65 | 918.96 | 915.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 917.50 | 919.05 | 916.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 917.50 | 919.05 | 916.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 918.00 | 918.84 | 916.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 912.30 | 918.84 | 916.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 907.60 | 916.59 | 915.52 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 905.70 | 914.41 | 914.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 901.90 | 910.28 | 912.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 886.25 | 879.62 | 888.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 886.25 | 879.62 | 888.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 882.00 | 880.10 | 887.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 890.20 | 880.10 | 887.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 878.50 | 879.78 | 886.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 871.45 | 878.47 | 885.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 929.30 | 881.52 | 882.73 | SL hit (close>static) qty=1.00 sl=890.70 alert=retest2 |

### Cycle 74 — BUY (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 10:15:00 | 905.15 | 886.25 | 884.77 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 894.40 | 902.75 | 903.27 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 921.55 | 902.40 | 901.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 922.95 | 914.14 | 909.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 958.45 | 959.20 | 948.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 958.45 | 959.20 | 948.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 947.05 | 955.34 | 951.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 947.05 | 955.34 | 951.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 951.00 | 954.47 | 951.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 958.40 | 956.66 | 952.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 955.80 | 955.33 | 952.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 955.05 | 954.79 | 953.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 957.55 | 954.27 | 953.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 980.15 | 959.45 | 955.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 925.30 | 954.41 | 957.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 925.30 | 954.41 | 957.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 919.50 | 942.83 | 951.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 908.85 | 907.86 | 922.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 908.85 | 907.86 | 922.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 905.95 | 901.01 | 909.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 905.95 | 901.01 | 909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 916.00 | 904.01 | 909.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 915.70 | 904.01 | 909.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 915.50 | 906.31 | 910.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 921.20 | 906.31 | 910.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 928.05 | 914.44 | 913.51 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 890.05 | 911.88 | 913.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 865.05 | 879.39 | 889.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 14:15:00 | 865.95 | 864.43 | 872.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 865.95 | 864.43 | 872.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 869.70 | 865.49 | 871.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 872.60 | 865.49 | 871.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 872.00 | 866.79 | 871.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 876.30 | 866.79 | 871.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 867.55 | 866.94 | 871.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 864.85 | 866.54 | 870.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 876.55 | 867.66 | 869.73 | SL hit (close>static) qty=1.00 sl=872.30 alert=retest2 |

### Cycle 80 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 892.55 | 872.64 | 871.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 901.80 | 891.50 | 888.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 941.50 | 942.61 | 930.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:45:00 | 938.35 | 942.61 | 930.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 963.50 | 963.42 | 956.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 960.90 | 963.42 | 956.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 957.65 | 961.39 | 957.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 958.45 | 961.39 | 957.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 952.45 | 959.61 | 957.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 952.45 | 959.61 | 957.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 957.10 | 959.10 | 957.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 953.60 | 959.10 | 957.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 966.55 | 960.59 | 957.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 971.00 | 962.65 | 959.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 952.50 | 961.97 | 962.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 952.50 | 961.97 | 962.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 942.95 | 954.57 | 958.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 927.45 | 923.49 | 934.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 927.45 | 923.49 | 934.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 920.00 | 922.79 | 932.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 921.60 | 922.79 | 932.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 923.75 | 917.31 | 923.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 930.50 | 917.31 | 923.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 927.80 | 919.41 | 924.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 910.60 | 920.53 | 922.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 893.70 | 886.71 | 886.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 893.70 | 886.71 | 886.41 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 881.95 | 886.64 | 886.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 878.65 | 885.04 | 885.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 885.05 | 883.81 | 885.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 886.30 | 883.81 | 885.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 884.00 | 883.85 | 884.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 885.65 | 883.85 | 884.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 884.20 | 883.92 | 884.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 883.75 | 883.92 | 884.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 874.75 | 882.09 | 883.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:30:00 | 872.00 | 880.18 | 882.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 872.55 | 880.18 | 882.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 872.30 | 877.01 | 880.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:00:00 | 867.65 | 875.14 | 879.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 877.35 | 874.23 | 877.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 877.35 | 874.23 | 877.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 874.05 | 874.20 | 877.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 890.00 | 878.40 | 878.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 890.00 | 878.40 | 878.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 896.05 | 881.93 | 879.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 884.10 | 884.66 | 881.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:00:00 | 884.10 | 884.66 | 881.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 878.25 | 883.38 | 881.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 878.25 | 883.38 | 881.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 880.10 | 882.73 | 881.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 889.00 | 883.76 | 881.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 886.00 | 886.77 | 884.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:00:00 | 887.50 | 887.27 | 885.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 880.80 | 883.35 | 883.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 880.80 | 883.35 | 883.63 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 895.80 | 884.92 | 884.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 900.95 | 889.78 | 886.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 894.00 | 894.94 | 890.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 894.00 | 894.94 | 890.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 890.20 | 893.99 | 890.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 903.85 | 893.99 | 890.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 898.40 | 898.58 | 895.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:00:00 | 898.60 | 898.59 | 895.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 988.24 | 935.04 | 916.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 993.50 | 1009.05 | 1010.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 981.80 | 1003.60 | 1007.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 946.90 | 945.99 | 956.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:45:00 | 947.00 | 945.99 | 956.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 88 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1076.40 | 972.23 | 966.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1088.15 | 1012.73 | 987.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1162.50 | 1175.27 | 1160.30 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 1140.15 | 1157.38 | 1158.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 1134.00 | 1150.00 | 1155.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 1146.00 | 1141.13 | 1148.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:45:00 | 1144.25 | 1141.13 | 1148.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1151.90 | 1143.28 | 1148.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 1151.90 | 1143.28 | 1148.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1158.60 | 1146.35 | 1149.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 1155.65 | 1146.35 | 1149.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1156.00 | 1149.49 | 1150.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 1152.00 | 1149.49 | 1150.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1162.45 | 1153.14 | 1152.13 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1145.95 | 1151.13 | 1151.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 1139.30 | 1146.94 | 1149.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1147.70 | 1143.72 | 1147.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1147.70 | 1143.72 | 1147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1139.85 | 1142.95 | 1146.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 1133.25 | 1140.35 | 1144.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 1134.40 | 1132.71 | 1137.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1130.60 | 1132.19 | 1136.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1115.20 | 1119.05 | 1124.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1102.40 | 1115.72 | 1122.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:45:00 | 1095.00 | 1110.98 | 1119.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1076.59 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1077.68 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:15:00 | 1074.07 | 1103.64 | 1115.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1106.60 | 1097.38 | 1107.97 | SL hit (close>ema200) qty=0.50 sl=1097.38 alert=retest2 |

### Cycle 92 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1162.50 | 1123.29 | 1118.14 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1101.70 | 1127.99 | 1128.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1095.20 | 1121.44 | 1125.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1116.30 | 1111.44 | 1118.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 1116.30 | 1111.44 | 1118.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1119.50 | 1113.05 | 1118.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:30:00 | 1123.40 | 1113.05 | 1118.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1117.80 | 1114.00 | 1118.68 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1137.00 | 1122.28 | 1121.33 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 1114.40 | 1121.46 | 1121.55 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1131.30 | 1121.53 | 1120.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1138.50 | 1126.91 | 1123.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 1137.20 | 1139.89 | 1133.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 1137.20 | 1139.89 | 1133.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1136.10 | 1139.13 | 1133.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1152.70 | 1139.13 | 1133.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-17 09:15:00 | 1267.97 | 1211.08 | 1177.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1267.00 | 1280.74 | 1282.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 1260.00 | 1274.22 | 1279.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1264.00 | 1261.78 | 1269.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1264.00 | 1261.78 | 1269.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1266.70 | 1262.77 | 1269.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1266.70 | 1262.77 | 1269.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1270.00 | 1264.21 | 1269.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1260.60 | 1264.21 | 1269.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1233.10 | 1229.03 | 1228.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1233.10 | 1229.03 | 1228.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1244.00 | 1232.02 | 1229.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 1248.90 | 1252.21 | 1244.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:45:00 | 1246.50 | 1252.21 | 1244.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1249.40 | 1251.65 | 1244.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1243.20 | 1251.65 | 1244.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1251.30 | 1252.04 | 1246.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1264.70 | 1252.04 | 1246.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1261.00 | 1252.49 | 1250.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1222.10 | 1246.41 | 1247.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 1222.10 | 1246.41 | 1247.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1200.40 | 1220.54 | 1231.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 13:15:00 | 1135.00 | 1132.60 | 1148.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:45:00 | 1137.00 | 1132.60 | 1148.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1150.00 | 1138.54 | 1148.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1145.80 | 1138.83 | 1148.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 1165.30 | 1149.43 | 1151.04 | SL hit (close>static) qty=1.00 sl=1160.10 alert=retest2 |

### Cycle 100 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1166.00 | 1152.75 | 1152.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 1171.80 | 1158.76 | 1155.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1171.10 | 1173.66 | 1166.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1171.10 | 1173.66 | 1166.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1160.00 | 1170.93 | 1165.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1146.00 | 1170.93 | 1165.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1142.80 | 1165.30 | 1163.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1142.80 | 1165.30 | 1163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1130.00 | 1158.24 | 1160.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1104.70 | 1133.81 | 1145.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1095.10 | 1094.78 | 1112.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1095.10 | 1094.78 | 1112.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1106.00 | 1097.52 | 1109.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1125.50 | 1097.52 | 1109.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1123.50 | 1102.72 | 1110.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1123.70 | 1102.72 | 1110.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1101.00 | 1102.37 | 1110.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1097.20 | 1102.92 | 1109.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1097.20 | 1107.15 | 1109.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1094.10 | 1102.01 | 1105.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 1097.20 | 1100.88 | 1104.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1090.50 | 1093.64 | 1099.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1109.60 | 1093.64 | 1099.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1104.20 | 1095.75 | 1099.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1102.70 | 1095.75 | 1099.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1101.80 | 1096.96 | 1099.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1101.80 | 1096.96 | 1099.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1104.30 | 1098.43 | 1100.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1104.30 | 1098.43 | 1100.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1102.40 | 1099.22 | 1100.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 1113.90 | 1102.89 | 1101.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1113.90 | 1102.89 | 1101.97 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1088.50 | 1100.99 | 1101.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 1084.20 | 1097.63 | 1099.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1105.40 | 1099.19 | 1100.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:30:00 | 1105.60 | 1099.19 | 1100.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1110.00 | 1101.35 | 1101.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1121.50 | 1105.38 | 1103.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1149.00 | 1155.21 | 1137.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1145.70 | 1155.21 | 1137.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1140.20 | 1152.21 | 1137.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1136.60 | 1152.21 | 1137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1121.90 | 1146.15 | 1136.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1121.90 | 1146.15 | 1136.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1137.70 | 1144.46 | 1136.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1123.20 | 1144.46 | 1136.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1128.50 | 1140.57 | 1136.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1114.30 | 1140.57 | 1136.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1126.60 | 1137.77 | 1135.56 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1105.20 | 1131.26 | 1132.80 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1154.40 | 1135.58 | 1133.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1164.30 | 1141.33 | 1136.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 1198.00 | 1204.01 | 1188.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:00:00 | 1198.00 | 1204.01 | 1188.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1176.20 | 1196.59 | 1188.56 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1160.30 | 1182.71 | 1183.68 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1203.30 | 1183.60 | 1183.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 1224.60 | 1210.97 | 1199.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 14:15:00 | 1336.80 | 1350.07 | 1308.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 1336.80 | 1350.07 | 1308.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1415.10 | 1416.10 | 1397.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 1406.80 | 1416.10 | 1397.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1421.00 | 1417.08 | 1400.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1432.60 | 1418.73 | 1402.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1432.50 | 1420.02 | 1409.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 1428.20 | 1420.78 | 1410.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 1401.00 | 1412.18 | 1413.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 1401.00 | 1412.18 | 1413.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1397.00 | 1409.14 | 1411.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1391.50 | 1389.14 | 1396.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 1391.50 | 1389.14 | 1396.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1384.00 | 1388.11 | 1395.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1412.20 | 1388.11 | 1395.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1420.60 | 1394.61 | 1398.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1422.00 | 1394.61 | 1398.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1412.50 | 1398.19 | 1399.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1410.60 | 1398.19 | 1399.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1409.60 | 1401.60 | 1400.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1409.60 | 1401.60 | 1400.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1419.60 | 1405.42 | 1402.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1403.00 | 1406.73 | 1403.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 1403.00 | 1406.73 | 1403.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1392.80 | 1403.94 | 1402.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1394.40 | 1403.94 | 1402.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1391.00 | 1401.35 | 1401.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1371.00 | 1393.46 | 1397.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 1386.90 | 1385.22 | 1392.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 1387.20 | 1386.27 | 1391.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1387.20 | 1386.27 | 1391.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1387.20 | 1386.27 | 1391.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1395.00 | 1388.02 | 1391.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1358.90 | 1388.02 | 1391.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1290.95 | 1384.29 | 1389.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 1407.10 | 1373.31 | 1380.46 | SL hit (close>ema200) qty=0.50 sl=1373.31 alert=retest2 |

### Cycle 112 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1420.90 | 1382.92 | 1382.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 1439.30 | 1394.05 | 1387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1486.40 | 1496.98 | 1467.59 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 1428.70 | 1460.43 | 1460.97 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1467.80 | 1449.73 | 1448.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 15:15:00 | 1479.00 | 1460.35 | 1453.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 1447.40 | 1460.72 | 1456.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 1447.40 | 1460.72 | 1456.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 1433.70 | 1455.31 | 1454.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:45:00 | 1434.70 | 1455.31 | 1454.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 1442.40 | 1451.62 | 1452.50 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 09:15:00 | 1473.40 | 1455.97 | 1454.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 11:15:00 | 1501.80 | 1469.00 | 1460.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 14:15:00 | 1473.20 | 1474.04 | 1465.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 1473.20 | 1474.04 | 1465.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1461.00 | 1471.43 | 1465.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1456.00 | 1471.43 | 1465.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1450.90 | 1467.32 | 1463.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1450.20 | 1467.32 | 1463.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1451.90 | 1464.24 | 1462.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1451.90 | 1464.24 | 1462.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 1448.10 | 1461.01 | 1461.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 1435.20 | 1449.08 | 1453.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1318.50 | 1302.15 | 1336.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1318.50 | 1302.15 | 1336.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1390.00 | 1329.85 | 1339.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1390.00 | 1329.85 | 1339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1396.40 | 1343.16 | 1344.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 1395.00 | 1343.16 | 1344.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1393.70 | 1353.27 | 1349.23 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1330.50 | 1355.25 | 1357.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1319.40 | 1348.08 | 1353.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1398.90 | 1341.86 | 1344.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1406.40 | 1341.86 | 1344.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1382.70 | 1350.03 | 1347.62 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1334.00 | 1349.63 | 1350.18 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1355.20 | 1351.19 | 1350.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1367.70 | 1354.49 | 1352.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1393.00 | 1404.75 | 1388.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1424.50 | 1398.05 | 1391.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1566.95 | 1499.82 | 1479.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1611.90 | 1637.65 | 1638.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1592.50 | 1617.01 | 1626.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1675.70 | 1612.97 | 1617.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1675.70 | 1612.97 | 1617.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1685.00 | 1627.37 | 1623.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1720.00 | 1672.31 | 1650.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1700.00 | 1715.43 | 1696.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 1700.00 | 1715.43 | 1696.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1700.00 | 1709.70 | 1696.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1702.20 | 1709.70 | 1696.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1686.30 | 1705.02 | 1695.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1686.30 | 1705.02 | 1695.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1700.00 | 1704.02 | 1696.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 1709.10 | 1700.54 | 1696.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1708.50 | 1699.85 | 1696.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1684.30 | 1723.49 | 1720.33 | SL hit (close<static) qty=1.00 sl=1685.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 1692.40 | 1717.28 | 1717.79 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1752.50 | 1717.61 | 1716.28 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 09:15:00 | 1073.45 | 2024-05-15 15:15:00 | 1180.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-24 09:15:00 | 1291.60 | 2024-05-28 11:15:00 | 1227.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 09:15:00 | 1291.60 | 2024-05-29 09:15:00 | 1266.95 | STOP_HIT | 0.50 | 1.91% |
| BUY | retest2 | 2024-06-13 12:15:00 | 1284.75 | 2024-06-18 09:15:00 | 1413.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 09:15:00 | 1309.00 | 2024-06-19 12:15:00 | 1294.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest1 | 2024-06-21 09:15:00 | 1395.30 | 2024-06-25 13:15:00 | 1381.65 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-06-26 09:15:00 | 1404.60 | 2024-06-27 14:15:00 | 1368.05 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-06-28 09:15:00 | 1403.45 | 2024-07-02 15:15:00 | 1405.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-06-28 15:15:00 | 1401.00 | 2024-07-02 15:15:00 | 1405.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-07-02 15:15:00 | 1405.00 | 2024-07-02 15:15:00 | 1405.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-07-04 13:00:00 | 1394.90 | 2024-07-12 12:15:00 | 1325.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 14:00:00 | 1393.90 | 2024-07-12 13:15:00 | 1324.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-05 13:00:00 | 1392.95 | 2024-07-12 13:15:00 | 1323.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 09:15:00 | 1394.00 | 2024-07-12 13:15:00 | 1324.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 13:00:00 | 1394.90 | 2024-07-15 09:15:00 | 1342.95 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2024-07-04 14:00:00 | 1393.90 | 2024-07-15 09:15:00 | 1342.95 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2024-07-05 13:00:00 | 1392.95 | 2024-07-15 09:15:00 | 1342.95 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2024-07-08 09:15:00 | 1394.00 | 2024-07-15 09:15:00 | 1342.95 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2024-07-08 11:00:00 | 1373.05 | 2024-07-18 09:15:00 | 1304.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 09:30:00 | 1377.05 | 2024-07-18 09:15:00 | 1308.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:30:00 | 1374.50 | 2024-07-18 09:15:00 | 1305.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 11:00:00 | 1373.05 | 2024-07-19 12:15:00 | 1235.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-10 09:30:00 | 1377.05 | 2024-07-19 12:15:00 | 1239.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-11 10:30:00 | 1374.50 | 2024-07-19 12:15:00 | 1237.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-08-06 10:30:00 | 1127.00 | 2024-08-07 09:15:00 | 1146.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1179.95 | 2024-08-19 14:15:00 | 1297.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-28 09:30:00 | 1297.05 | 2024-08-28 14:15:00 | 1312.10 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-28 10:00:00 | 1296.50 | 2024-08-28 14:15:00 | 1312.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-28 12:30:00 | 1296.50 | 2024-08-28 14:15:00 | 1312.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-06 14:45:00 | 1327.60 | 2024-09-19 09:15:00 | 1261.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 14:45:00 | 1327.60 | 2024-09-20 09:15:00 | 1265.00 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1189.05 | 2024-10-07 10:15:00 | 1129.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1189.05 | 2024-10-08 12:15:00 | 1133.80 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2024-10-24 11:00:00 | 1101.10 | 2024-10-25 09:15:00 | 1046.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:00:00 | 1101.10 | 2024-10-28 09:15:00 | 1044.20 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1128.10 | 2024-11-19 09:15:00 | 1158.55 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-11-29 12:45:00 | 1125.90 | 2024-11-29 14:15:00 | 1122.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-04 09:15:00 | 1093.90 | 2024-12-06 11:15:00 | 1115.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-12-04 11:30:00 | 1094.85 | 2024-12-06 11:15:00 | 1115.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-12-05 10:30:00 | 1096.90 | 2024-12-06 11:15:00 | 1115.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-12-06 09:15:00 | 1097.00 | 2024-12-06 11:15:00 | 1115.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-12-11 11:15:00 | 1163.80 | 2024-12-12 11:15:00 | 1144.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-12-11 12:30:00 | 1164.40 | 2024-12-12 11:15:00 | 1144.80 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-12-12 09:15:00 | 1163.15 | 2024-12-12 11:15:00 | 1144.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-12-12 10:30:00 | 1165.00 | 2024-12-12 11:15:00 | 1144.80 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1017.45 | 2025-01-02 11:15:00 | 1030.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-01-02 10:30:00 | 1021.50 | 2025-01-02 11:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-23 09:15:00 | 910.05 | 2025-01-27 09:15:00 | 864.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 923.45 | 2025-01-27 09:15:00 | 877.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 922.70 | 2025-01-27 09:15:00 | 876.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 922.80 | 2025-01-27 09:15:00 | 876.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 917.25 | 2025-01-27 09:15:00 | 871.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 910.05 | 2025-01-28 09:15:00 | 831.11 | TARGET_HIT | 0.50 | 8.67% |
| SELL | retest2 | 2025-01-23 11:30:00 | 923.45 | 2025-01-28 09:15:00 | 830.43 | TARGET_HIT | 0.50 | 10.07% |
| SELL | retest2 | 2025-01-23 12:15:00 | 922.70 | 2025-01-28 09:15:00 | 830.52 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2025-01-23 13:15:00 | 922.80 | 2025-01-29 09:15:00 | 877.35 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2025-01-24 09:30:00 | 917.25 | 2025-01-29 09:15:00 | 877.35 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2025-02-06 10:45:00 | 847.35 | 2025-02-11 09:15:00 | 804.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 10:45:00 | 847.35 | 2025-02-12 09:15:00 | 762.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-11 10:30:00 | 653.95 | 2025-03-12 10:15:00 | 634.30 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-03-11 11:30:00 | 650.75 | 2025-03-12 10:15:00 | 634.30 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-03-17 10:30:00 | 674.20 | 2025-03-18 11:15:00 | 741.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 09:15:00 | 698.45 | 2025-03-26 12:15:00 | 730.90 | STOP_HIT | 1.00 | 4.65% |
| SELL | retest2 | 2025-04-08 10:30:00 | 707.45 | 2025-04-11 11:15:00 | 721.85 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-08 13:00:00 | 710.45 | 2025-04-11 11:15:00 | 721.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-04-08 15:15:00 | 709.05 | 2025-04-11 11:15:00 | 721.85 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-04-22 10:30:00 | 760.80 | 2025-04-25 09:15:00 | 729.30 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-04-22 14:15:00 | 758.00 | 2025-04-25 09:15:00 | 729.30 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-04-23 13:15:00 | 757.60 | 2025-04-25 09:15:00 | 729.30 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-04-23 14:45:00 | 757.95 | 2025-04-25 09:15:00 | 729.30 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-05-08 13:45:00 | 711.80 | 2025-05-09 09:15:00 | 676.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:45:00 | 711.80 | 2025-05-12 09:15:00 | 722.10 | STOP_HIT | 0.50 | -1.45% |
| SELL | retest2 | 2025-05-12 10:30:00 | 715.20 | 2025-05-13 09:15:00 | 715.85 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-05-12 11:00:00 | 714.95 | 2025-05-13 09:15:00 | 715.85 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-06-04 15:00:00 | 875.00 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-05 11:00:00 | 874.90 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-06 09:45:00 | 874.30 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-06-06 10:30:00 | 873.00 | 2025-06-09 09:15:00 | 890.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-12 09:30:00 | 932.20 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-06-12 10:00:00 | 917.40 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-06-12 12:30:00 | 922.15 | 2025-06-13 10:15:00 | 890.55 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-06-17 11:15:00 | 873.10 | 2025-06-20 09:15:00 | 829.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 870.15 | 2025-06-20 09:15:00 | 826.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 873.10 | 2025-06-20 14:15:00 | 842.75 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-06-17 11:45:00 | 870.15 | 2025-06-20 14:15:00 | 842.75 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2025-06-26 09:15:00 | 864.30 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-27 12:30:00 | 859.70 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-27 13:45:00 | 860.40 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-27 15:15:00 | 864.00 | 2025-06-30 10:15:00 | 851.30 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-07-01 10:45:00 | 854.95 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-01 13:15:00 | 854.25 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-01 13:45:00 | 854.00 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-07-02 09:15:00 | 854.45 | 2025-07-04 14:15:00 | 855.85 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-07-09 12:00:00 | 835.10 | 2025-07-10 09:15:00 | 845.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-09 13:30:00 | 835.30 | 2025-07-10 09:15:00 | 845.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-23 13:00:00 | 912.75 | 2025-07-24 09:15:00 | 922.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-24 09:15:00 | 913.75 | 2025-07-24 09:15:00 | 922.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-05 10:45:00 | 871.45 | 2025-08-06 09:15:00 | 929.30 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2025-08-22 09:45:00 | 958.40 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-22 11:30:00 | 955.80 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-08-22 13:30:00 | 955.05 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-08-25 09:30:00 | 957.55 | 2025-08-26 13:15:00 | 925.30 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-09-09 14:00:00 | 864.85 | 2025-09-10 09:15:00 | 876.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-24 10:45:00 | 971.00 | 2025-09-25 12:15:00 | 952.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-10-03 10:30:00 | 910.60 | 2025-10-10 13:15:00 | 893.70 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2025-10-14 11:30:00 | 872.00 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-10-14 12:00:00 | 872.55 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-14 13:45:00 | 872.30 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-10-14 15:00:00 | 867.65 | 2025-10-16 09:15:00 | 890.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-10-17 09:30:00 | 889.00 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-17 13:45:00 | 886.00 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-20 10:00:00 | 887.50 | 2025-10-20 13:15:00 | 880.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-24 09:15:00 | 903.85 | 2025-10-28 09:15:00 | 988.24 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2025-10-27 09:15:00 | 898.40 | 2025-10-28 09:15:00 | 988.46 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2025-10-27 10:00:00 | 898.60 | 2025-10-28 10:15:00 | 994.24 | TARGET_HIT | 1.00 | 10.64% |
| SELL | retest2 | 2025-11-28 10:45:00 | 1133.25 | 2025-12-03 12:15:00 | 1076.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 1134.40 | 2025-12-03 12:15:00 | 1077.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1130.60 | 2025-12-03 12:15:00 | 1074.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:45:00 | 1133.25 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2025-12-01 10:45:00 | 1134.40 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1130.60 | 2025-12-04 09:15:00 | 1106.60 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-12-03 09:30:00 | 1115.20 | 2025-12-04 11:15:00 | 1157.80 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-03 11:45:00 | 1095.00 | 2025-12-04 11:15:00 | 1157.80 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest2 | 2025-12-16 09:15:00 | 1152.70 | 2025-12-17 09:15:00 | 1267.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1260.60 | 2026-01-02 09:15:00 | 1233.10 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1264.70 | 2026-01-07 10:15:00 | 1222.10 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1261.00 | 2026-01-07 10:15:00 | 1222.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1145.80 | 2026-01-14 12:15:00 | 1165.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-23 13:15:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-27 10:15:00 | 1094.10 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-01-27 12:15:00 | 1097.20 | 2026-01-28 14:15:00 | 1113.90 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1432.60 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-02-19 09:30:00 | 1432.50 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-02-19 10:30:00 | 1428.20 | 2026-02-23 10:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1410.60 | 2026-02-25 14:15:00 | 1409.60 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1358.90 | 2026-03-02 09:15:00 | 1290.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1358.90 | 2026-03-02 14:15:00 | 1407.10 | STOP_HIT | 0.50 | -3.55% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1362.90 | 2026-03-04 14:15:00 | 1420.90 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1424.50 | 2026-04-15 09:15:00 | 1566.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:30:00 | 1709.10 | 2026-05-06 09:15:00 | 1684.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-04-30 15:15:00 | 1708.50 | 2026-05-06 09:15:00 | 1684.30 | STOP_HIT | 1.00 | -1.42% |
