# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1647.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 164 |
| ALERT1 | 113 |
| ALERT2 | 112 |
| ALERT2_SKIP | 59 |
| ALERT3 | 290 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 129 |
| PARTIAL | 14 |
| TARGET_HIT | 7 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 146 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 98
- **Target hits / Stop hits / Partials:** 7 / 125 / 14
- **Avg / median % per leg:** 0.10% / -1.27%
- **Sum % (uncompounded):** 13.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 18 | 24.7% | 5 | 68 | 0 | -0.60% | -44.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.73% | -5.2% |
| BUY @ 3rd Alert (retest2) | 70 | 18 | 25.7% | 5 | 65 | 0 | -0.55% | -38.8% |
| SELL (all) | 73 | 30 | 41.1% | 2 | 57 | 14 | 0.79% | 57.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 73 | 30 | 41.1% | 2 | 57 | 14 | 0.79% | 57.9% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.73% | -5.2% |
| retest2 (combined) | 143 | 48 | 33.6% | 7 | 122 | 14 | 0.13% | 19.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 1248.60 | 1223.43 | 1222.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1294.25 | 1241.05 | 1231.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 11:15:00 | 1242.55 | 1245.31 | 1235.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 12:00:00 | 1242.55 | 1245.31 | 1235.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1236.20 | 1243.47 | 1236.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 1236.20 | 1243.47 | 1236.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1228.65 | 1240.50 | 1235.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:45:00 | 1231.15 | 1240.50 | 1235.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 1236.80 | 1239.76 | 1235.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 1288.20 | 1239.76 | 1235.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 15:15:00 | 1290.80 | 1310.25 | 1311.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 1290.80 | 1310.25 | 1311.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 1277.20 | 1303.64 | 1308.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 1290.75 | 1290.10 | 1298.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:15:00 | 1290.90 | 1290.10 | 1298.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1290.90 | 1290.26 | 1297.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 1277.35 | 1290.26 | 1297.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:30:00 | 1280.00 | 1274.27 | 1282.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:00:00 | 1281.60 | 1274.27 | 1282.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 1213.48 | 1250.43 | 1262.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 1216.00 | 1250.43 | 1262.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 1217.52 | 1250.43 | 1262.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 1204.10 | 1191.11 | 1207.66 | SL hit (close>ema200) qty=0.50 sl=1191.11 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 1242.60 | 1212.49 | 1209.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 1250.50 | 1220.09 | 1213.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 14:15:00 | 1284.35 | 1284.79 | 1258.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 14:45:00 | 1284.60 | 1284.79 | 1258.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1289.35 | 1284.69 | 1263.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1252.65 | 1284.69 | 1263.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1255.15 | 1278.78 | 1262.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1255.15 | 1278.78 | 1262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1229.95 | 1269.02 | 1259.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:45:00 | 1229.25 | 1269.02 | 1259.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1237.70 | 1262.75 | 1257.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:45:00 | 1223.55 | 1262.75 | 1257.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 1244.50 | 1252.64 | 1253.69 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 1283.20 | 1258.75 | 1256.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 1299.45 | 1274.86 | 1265.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 10:15:00 | 1286.60 | 1286.72 | 1275.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 1288.40 | 1288.33 | 1281.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1288.40 | 1288.33 | 1281.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1331.50 | 1287.90 | 1284.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1316.40 | 1293.04 | 1289.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:00:00 | 1303.35 | 1296.70 | 1291.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1314.85 | 1300.02 | 1295.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1300.80 | 1300.17 | 1296.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:45:00 | 1308.30 | 1302.00 | 1297.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:00:00 | 1308.50 | 1303.30 | 1298.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 10:15:00 | 1330.40 | 1350.60 | 1351.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 1330.40 | 1350.60 | 1351.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 10:15:00 | 1309.65 | 1332.47 | 1340.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 1315.45 | 1309.91 | 1324.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 09:45:00 | 1308.70 | 1309.91 | 1324.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1317.50 | 1311.42 | 1323.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:45:00 | 1296.30 | 1308.67 | 1321.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 1310.00 | 1313.76 | 1320.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1338.90 | 1320.17 | 1322.26 | SL hit (close>static) qty=1.00 sl=1329.85 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 1343.55 | 1324.85 | 1324.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 1385.35 | 1344.90 | 1334.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 1400.00 | 1405.95 | 1385.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 14:30:00 | 1399.35 | 1405.95 | 1385.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1439.00 | 1472.16 | 1456.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 1437.85 | 1472.16 | 1456.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1440.65 | 1465.85 | 1454.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 1438.45 | 1465.85 | 1454.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1424.85 | 1453.10 | 1452.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 1422.85 | 1453.10 | 1452.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1464.10 | 1455.30 | 1453.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 1432.40 | 1455.30 | 1453.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 1433.00 | 1450.84 | 1451.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 1427.80 | 1443.35 | 1447.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 1405.00 | 1397.44 | 1414.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1405.00 | 1397.44 | 1414.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1405.00 | 1397.44 | 1414.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1402.05 | 1397.44 | 1414.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1416.30 | 1401.21 | 1414.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1416.30 | 1401.21 | 1414.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1405.70 | 1402.11 | 1413.90 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 1424.95 | 1415.96 | 1415.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 15:15:00 | 1429.00 | 1421.45 | 1418.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1412.70 | 1419.70 | 1418.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1412.70 | 1419.70 | 1418.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1412.70 | 1419.70 | 1418.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:15:00 | 1374.90 | 1419.70 | 1418.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1375.50 | 1410.86 | 1414.14 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 1458.80 | 1421.37 | 1417.12 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1407.30 | 1442.13 | 1442.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1388.85 | 1414.88 | 1425.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 1420.25 | 1414.07 | 1422.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 1420.25 | 1414.07 | 1422.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1420.25 | 1414.07 | 1422.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1420.25 | 1414.07 | 1422.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1446.30 | 1420.52 | 1424.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 1446.30 | 1420.52 | 1424.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1451.65 | 1426.74 | 1426.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1421.00 | 1426.74 | 1426.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 1461.55 | 1428.92 | 1424.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 1461.55 | 1428.92 | 1424.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 1496.15 | 1459.07 | 1447.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1472.35 | 1473.83 | 1459.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1472.35 | 1473.83 | 1459.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1472.35 | 1473.83 | 1459.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1493.45 | 1473.84 | 1465.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:15:00 | 1492.00 | 1484.36 | 1480.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:00:00 | 1492.40 | 1485.97 | 1481.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:00:00 | 1491.05 | 1486.99 | 1482.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1464.80 | 1483.29 | 1481.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 1465.05 | 1483.29 | 1481.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-30 10:15:00 | 1464.25 | 1479.48 | 1479.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 10:15:00 | 1464.25 | 1479.48 | 1479.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 1458.10 | 1472.61 | 1476.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 10:15:00 | 1467.85 | 1464.30 | 1470.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 10:15:00 | 1467.85 | 1464.30 | 1470.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1467.85 | 1464.30 | 1470.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 1467.95 | 1464.30 | 1470.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1466.00 | 1464.99 | 1469.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:45:00 | 1471.75 | 1464.99 | 1469.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1451.70 | 1461.50 | 1467.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 1463.90 | 1461.50 | 1467.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1437.05 | 1454.09 | 1462.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 1438.55 | 1454.09 | 1462.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1450.00 | 1445.14 | 1453.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1446.35 | 1445.14 | 1453.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1445.20 | 1445.15 | 1452.58 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 12:15:00 | 1474.95 | 1456.39 | 1456.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 13:15:00 | 1488.30 | 1462.77 | 1459.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1453.95 | 1476.28 | 1468.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 1453.95 | 1476.28 | 1468.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1453.95 | 1476.28 | 1468.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1453.95 | 1476.28 | 1468.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 1458.00 | 1472.62 | 1467.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:15:00 | 1473.00 | 1470.89 | 1467.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 14:15:00 | 1422.20 | 1464.45 | 1467.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1422.20 | 1464.45 | 1467.79 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1505.65 | 1472.49 | 1469.80 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 1474.00 | 1478.78 | 1478.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 1448.20 | 1464.59 | 1470.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 12:15:00 | 1458.40 | 1456.92 | 1463.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 13:00:00 | 1458.40 | 1456.92 | 1463.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1467.70 | 1449.87 | 1457.40 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 1476.90 | 1461.92 | 1461.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 14:15:00 | 1480.10 | 1467.87 | 1464.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 10:15:00 | 1632.45 | 1637.01 | 1593.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 11:00:00 | 1632.45 | 1637.01 | 1593.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1666.00 | 1677.67 | 1664.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 1671.95 | 1677.67 | 1664.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1678.70 | 1677.88 | 1666.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 14:00:00 | 1698.45 | 1682.33 | 1671.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 1705.85 | 1687.09 | 1676.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 1709.80 | 1691.93 | 1684.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 1725.35 | 1745.02 | 1745.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1725.35 | 1745.02 | 1745.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1720.05 | 1740.03 | 1743.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 1736.00 | 1733.19 | 1738.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 1710.05 | 1733.19 | 1738.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1715.95 | 1729.75 | 1736.35 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 1807.95 | 1739.14 | 1737.12 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 1720.25 | 1738.05 | 1738.24 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 1745.85 | 1730.99 | 1730.36 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 1720.05 | 1729.17 | 1729.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1716.10 | 1723.54 | 1726.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 1727.00 | 1722.22 | 1724.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 1727.00 | 1722.22 | 1724.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1727.00 | 1722.22 | 1724.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 1727.00 | 1722.22 | 1724.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1718.25 | 1721.43 | 1724.32 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 1747.00 | 1729.35 | 1726.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1766.95 | 1748.90 | 1740.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 1791.70 | 1799.36 | 1781.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1822.05 | 1803.38 | 1784.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1820.25 | 1806.25 | 1787.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 11:45:00 | 1820.05 | 1808.66 | 1790.40 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1789.25 | 1807.29 | 1797.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1789.25 | 1807.29 | 1797.06 | SL hit (close<ema400) qty=1.00 sl=1797.06 alert=retest1 |

### Cycle 26 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 1772.50 | 1788.23 | 1789.94 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1849.00 | 1801.17 | 1795.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 1865.95 | 1814.13 | 1801.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 1832.05 | 1848.73 | 1832.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 12:15:00 | 1832.05 | 1848.73 | 1832.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1832.05 | 1848.73 | 1832.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 1832.05 | 1848.73 | 1832.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1823.15 | 1843.61 | 1831.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 1825.15 | 1843.61 | 1831.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1804.85 | 1835.86 | 1829.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1804.85 | 1835.86 | 1829.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1832.85 | 1840.08 | 1833.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1832.85 | 1840.08 | 1833.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1832.40 | 1838.55 | 1833.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 1832.40 | 1838.55 | 1833.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1899.15 | 1850.67 | 1839.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:30:00 | 1901.05 | 1869.24 | 1851.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:15:00 | 1900.90 | 1875.00 | 1855.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 13:45:00 | 1900.25 | 1884.07 | 1863.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 1930.00 | 1893.26 | 1869.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1880.50 | 1906.00 | 1893.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 1880.50 | 1906.00 | 1893.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1873.55 | 1899.51 | 1891.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 1873.55 | 1899.51 | 1891.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-24 12:15:00 | 1808.50 | 1874.91 | 1881.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 1808.50 | 1874.91 | 1881.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 1721.70 | 1830.04 | 1856.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1696.55 | 1667.73 | 1726.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 1696.55 | 1667.73 | 1726.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1742.00 | 1682.59 | 1727.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 1688.75 | 1682.59 | 1727.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:15:00 | 1604.31 | 1665.33 | 1711.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 1648.70 | 1641.89 | 1676.96 | SL hit (close>ema200) qty=0.50 sl=1641.89 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 1725.20 | 1673.65 | 1670.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 1736.80 | 1686.28 | 1676.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 14:15:00 | 1710.00 | 1720.62 | 1703.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 1710.00 | 1720.62 | 1703.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1701.55 | 1716.81 | 1703.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1645.00 | 1716.81 | 1703.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1678.55 | 1709.16 | 1701.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1662.50 | 1709.16 | 1701.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1694.05 | 1706.13 | 1700.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 1677.65 | 1706.13 | 1700.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1692.60 | 1702.45 | 1699.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:45:00 | 1689.15 | 1702.45 | 1699.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1719.00 | 1705.76 | 1701.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1727.65 | 1706.46 | 1702.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 13:15:00 | 1688.05 | 1705.17 | 1704.00 | SL hit (close<static) qty=1.00 sl=1692.80 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1685.95 | 1701.32 | 1702.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 1662.40 | 1693.33 | 1698.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 13:15:00 | 1648.05 | 1646.10 | 1658.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 1648.05 | 1646.10 | 1658.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1619.80 | 1634.98 | 1649.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:30:00 | 1615.75 | 1629.76 | 1646.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 1664.90 | 1638.62 | 1642.79 | SL hit (close>static) qty=1.00 sl=1656.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 1661.10 | 1647.91 | 1646.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 1675.00 | 1659.60 | 1653.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 12:15:00 | 1692.95 | 1696.42 | 1686.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 12:15:00 | 1692.95 | 1696.42 | 1686.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1692.95 | 1696.42 | 1686.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:30:00 | 1684.15 | 1696.42 | 1686.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1686.65 | 1694.47 | 1686.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:15:00 | 1679.95 | 1694.47 | 1686.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1683.25 | 1692.22 | 1686.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1691.40 | 1691.12 | 1686.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:00:00 | 1690.90 | 1691.07 | 1686.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 1692.00 | 1690.27 | 1686.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:15:00 | 1689.80 | 1690.27 | 1686.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1693.55 | 1690.92 | 1687.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 13:15:00 | 1698.65 | 1690.92 | 1687.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 1681.65 | 1690.45 | 1688.04 | SL hit (close<static) qty=1.00 sl=1686.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 1666.55 | 1684.00 | 1685.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1647.60 | 1673.53 | 1680.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 13:15:00 | 1670.35 | 1668.34 | 1676.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 13:15:00 | 1670.35 | 1668.34 | 1676.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1670.35 | 1668.34 | 1676.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 1670.35 | 1668.34 | 1676.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1678.50 | 1670.37 | 1676.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1659.40 | 1669.51 | 1675.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:30:00 | 1651.35 | 1641.97 | 1648.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 1659.60 | 1643.46 | 1642.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 09:15:00 | 1659.60 | 1643.46 | 1642.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 1678.45 | 1656.21 | 1650.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 1659.10 | 1673.58 | 1662.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 1659.10 | 1673.58 | 1662.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1659.10 | 1673.58 | 1662.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 1659.10 | 1673.58 | 1662.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1653.05 | 1669.47 | 1661.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:45:00 | 1654.30 | 1669.47 | 1661.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1697.20 | 1692.50 | 1678.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 1714.10 | 1699.09 | 1692.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 1715.00 | 1702.54 | 1695.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 1668.00 | 1698.42 | 1694.63 | SL hit (close<static) qty=1.00 sl=1676.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 1638.00 | 1686.34 | 1689.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 1609.05 | 1670.88 | 1682.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 1642.35 | 1638.20 | 1655.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 11:15:00 | 1642.35 | 1638.20 | 1655.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1642.35 | 1638.20 | 1655.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 1642.35 | 1638.20 | 1655.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 1672.90 | 1645.14 | 1657.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 1672.90 | 1645.14 | 1657.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 1677.60 | 1651.63 | 1659.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:45:00 | 1656.30 | 1653.30 | 1659.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1734.60 | 1672.23 | 1666.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 1734.60 | 1672.23 | 1666.91 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 1670.80 | 1689.80 | 1691.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 1659.05 | 1678.88 | 1685.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1708.00 | 1680.15 | 1683.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1708.00 | 1680.15 | 1683.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1708.00 | 1680.15 | 1683.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 1675.95 | 1681.70 | 1683.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1675.70 | 1683.05 | 1683.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 1695.60 | 1676.74 | 1674.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 1695.60 | 1676.74 | 1674.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 14:15:00 | 1716.00 | 1688.07 | 1680.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 12:15:00 | 1703.65 | 1712.73 | 1697.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 1703.65 | 1712.73 | 1697.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1693.55 | 1708.89 | 1697.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:15:00 | 1686.05 | 1708.89 | 1697.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1714.00 | 1709.91 | 1699.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 1693.35 | 1709.91 | 1699.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1705.35 | 1726.40 | 1715.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 1705.35 | 1726.40 | 1715.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1696.65 | 1720.45 | 1713.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 1741.00 | 1720.45 | 1713.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 1724.40 | 1735.15 | 1725.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 1724.40 | 1735.15 | 1725.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 1725.05 | 1733.13 | 1725.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:15:00 | 1704.05 | 1733.13 | 1725.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1682.10 | 1722.92 | 1721.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 1682.10 | 1722.92 | 1721.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 10:15:00 | 1680.00 | 1714.34 | 1718.10 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 1740.15 | 1719.84 | 1718.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1798.50 | 1735.57 | 1726.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 1738.45 | 1759.41 | 1742.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 13:15:00 | 1738.45 | 1759.41 | 1742.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 1738.45 | 1759.41 | 1742.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 1738.45 | 1759.41 | 1742.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1808.85 | 1769.30 | 1748.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:45:00 | 1826.00 | 1804.16 | 1776.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:30:00 | 1823.90 | 1822.65 | 1793.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:30:00 | 1822.55 | 1824.31 | 1799.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 1821.00 | 1824.31 | 1799.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1793.00 | 1818.83 | 1807.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 1793.00 | 1818.83 | 1807.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1810.00 | 1817.06 | 1808.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 13:30:00 | 1825.40 | 1820.65 | 1811.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 15:15:00 | 2004.81 | 1899.71 | 1875.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 2118.00 | 2135.70 | 2136.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 2079.10 | 2105.95 | 2119.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 2110.30 | 2102.78 | 2115.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 11:15:00 | 2110.30 | 2102.78 | 2115.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 2110.30 | 2102.78 | 2115.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 2110.30 | 2102.78 | 2115.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2104.35 | 2103.09 | 2114.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:30:00 | 2097.05 | 2102.18 | 2113.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 15:15:00 | 2120.25 | 2106.02 | 2112.91 | SL hit (close>static) qty=1.00 sl=2117.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 2153.35 | 2119.97 | 2118.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 11:15:00 | 2164.15 | 2128.80 | 2122.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 2127.00 | 2132.02 | 2126.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 15:15:00 | 2127.00 | 2132.02 | 2126.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 2127.00 | 2132.02 | 2126.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 2156.60 | 2132.02 | 2126.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 12:15:00 | 2105.10 | 2130.24 | 2128.12 | SL hit (close<static) qty=1.00 sl=2124.30 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 2112.35 | 2126.67 | 2126.69 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 14:15:00 | 2147.35 | 2130.80 | 2128.57 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 2115.00 | 2126.90 | 2127.44 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 2137.60 | 2129.04 | 2128.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 12:15:00 | 2145.00 | 2132.23 | 2129.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 15:15:00 | 2135.00 | 2136.85 | 2132.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 15:15:00 | 2135.00 | 2136.85 | 2132.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2135.00 | 2136.85 | 2132.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2128.80 | 2136.85 | 2132.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2121.45 | 2133.77 | 2131.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 2121.45 | 2133.77 | 2131.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 10:15:00 | 2115.60 | 2130.14 | 2130.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 2097.10 | 2121.62 | 2126.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 2120.70 | 2106.62 | 2115.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 2120.70 | 2106.62 | 2115.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 2120.70 | 2106.62 | 2115.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 2074.30 | 2104.49 | 2112.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:00:00 | 2073.60 | 2104.49 | 2112.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 2146.95 | 2078.43 | 2069.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 2146.95 | 2078.43 | 2069.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 2179.50 | 2122.12 | 2102.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 2180.95 | 2204.31 | 2180.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 2180.95 | 2204.31 | 2180.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2180.95 | 2204.31 | 2180.43 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 2114.10 | 2164.68 | 2168.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 2095.60 | 2138.18 | 2153.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1769.55 | 1752.93 | 1820.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1769.55 | 1752.93 | 1820.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1822.70 | 1772.75 | 1781.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 1822.70 | 1772.75 | 1781.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1812.70 | 1787.46 | 1786.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1718.20 | 1777.54 | 1784.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 1708.50 | 1733.51 | 1755.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 1747.75 | 1735.79 | 1752.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 11:45:00 | 1749.05 | 1735.79 | 1752.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1765.00 | 1741.63 | 1754.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 1765.00 | 1741.63 | 1754.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1767.00 | 1746.71 | 1755.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 1768.75 | 1746.71 | 1755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1711.00 | 1740.09 | 1750.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1695.80 | 1740.09 | 1750.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1611.01 | 1661.88 | 1700.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 1612.10 | 1607.61 | 1653.90 | SL hit (close>ema200) qty=0.50 sl=1607.61 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 1688.25 | 1665.93 | 1665.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 10:15:00 | 1696.60 | 1678.54 | 1671.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 1688.00 | 1688.01 | 1678.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 1688.00 | 1688.01 | 1678.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1694.95 | 1689.40 | 1680.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 1689.05 | 1689.40 | 1680.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1693.45 | 1690.45 | 1682.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 13:30:00 | 1705.00 | 1689.55 | 1684.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 09:15:00 | 1657.35 | 1683.19 | 1682.69 | SL hit (close<static) qty=1.00 sl=1665.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 10:15:00 | 1645.00 | 1675.55 | 1679.26 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1689.30 | 1677.16 | 1675.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1721.25 | 1688.74 | 1681.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 1682.30 | 1693.31 | 1686.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 1682.30 | 1693.31 | 1686.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1682.30 | 1693.31 | 1686.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 1682.30 | 1693.31 | 1686.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1683.60 | 1691.37 | 1686.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 1680.35 | 1691.37 | 1686.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1649.35 | 1682.97 | 1683.09 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 1701.70 | 1683.57 | 1682.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 1710.00 | 1691.49 | 1686.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1633.80 | 1700.46 | 1696.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1633.80 | 1700.46 | 1696.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1633.80 | 1700.46 | 1696.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 1681.95 | 1700.46 | 1696.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1700.05 | 1700.38 | 1696.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1708.05 | 1700.38 | 1696.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1653.65 | 1695.72 | 1696.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1653.65 | 1695.72 | 1696.11 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 1705.00 | 1697.58 | 1696.92 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1680.90 | 1694.24 | 1695.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 1674.00 | 1689.82 | 1693.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1700.05 | 1687.52 | 1690.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1700.05 | 1687.52 | 1690.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1700.05 | 1687.52 | 1690.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:30:00 | 1704.15 | 1687.52 | 1690.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1695.05 | 1689.03 | 1691.29 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 1708.00 | 1695.83 | 1694.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1726.00 | 1705.95 | 1699.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1716.05 | 1735.72 | 1721.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1716.05 | 1735.72 | 1721.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1716.05 | 1735.72 | 1721.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1720.30 | 1735.72 | 1721.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1692.05 | 1726.98 | 1719.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1692.05 | 1726.98 | 1719.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1707.95 | 1723.18 | 1718.11 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 1701.55 | 1713.08 | 1714.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 15:15:00 | 1693.60 | 1709.18 | 1712.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 1710.70 | 1704.38 | 1709.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 1710.70 | 1704.38 | 1709.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1710.70 | 1704.38 | 1709.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1710.70 | 1704.38 | 1709.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1695.55 | 1702.61 | 1708.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 12:00:00 | 1688.00 | 1704.18 | 1707.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:15:00 | 1603.60 | 1650.07 | 1676.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 13:15:00 | 1519.20 | 1592.97 | 1641.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1642.85 | 1612.11 | 1611.99 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 1576.15 | 1613.75 | 1614.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 1562.95 | 1596.98 | 1606.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 1550.55 | 1518.41 | 1545.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 1550.55 | 1518.41 | 1545.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1550.55 | 1518.41 | 1545.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:45:00 | 1549.85 | 1518.41 | 1545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 1530.00 | 1520.73 | 1543.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:15:00 | 1526.00 | 1520.73 | 1543.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:30:00 | 1525.25 | 1522.80 | 1540.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 14:45:00 | 1525.00 | 1525.15 | 1538.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 1526.00 | 1525.15 | 1538.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1536.30 | 1527.52 | 1537.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1536.30 | 1527.52 | 1537.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1502.90 | 1522.59 | 1534.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:15:00 | 1492.55 | 1517.03 | 1529.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 10:15:00 | 1547.10 | 1522.70 | 1523.16 | SL hit (close>static) qty=1.00 sl=1542.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 1560.65 | 1530.29 | 1526.56 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 1514.95 | 1536.87 | 1537.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 1508.00 | 1528.74 | 1533.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 1501.70 | 1496.01 | 1508.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 15:15:00 | 1501.70 | 1496.01 | 1508.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 1501.70 | 1496.01 | 1508.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 1454.80 | 1496.01 | 1508.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 1382.06 | 1460.47 | 1480.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 1454.30 | 1449.49 | 1466.27 | SL hit (close>ema200) qty=0.50 sl=1449.49 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 12:15:00 | 1444.30 | 1419.21 | 1416.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 1473.75 | 1442.49 | 1430.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 1406.25 | 1442.20 | 1434.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 1406.25 | 1442.20 | 1434.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1406.25 | 1442.20 | 1434.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 1394.00 | 1442.20 | 1434.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1398.25 | 1433.41 | 1430.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 1400.00 | 1433.41 | 1430.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1366.70 | 1420.07 | 1425.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 09:15:00 | 1340.70 | 1395.13 | 1410.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 12:15:00 | 1340.45 | 1340.35 | 1362.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 13:00:00 | 1340.45 | 1340.35 | 1362.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 1360.40 | 1344.36 | 1362.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 1360.40 | 1344.36 | 1362.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1356.45 | 1346.78 | 1361.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 1355.00 | 1346.78 | 1361.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1387.85 | 1356.31 | 1363.60 | SL hit (close>static) qty=1.00 sl=1363.50 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 1432.70 | 1379.28 | 1373.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 1443.30 | 1400.68 | 1384.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1636.40 | 1652.09 | 1613.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1636.40 | 1652.09 | 1613.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1613.85 | 1640.44 | 1614.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1611.00 | 1640.44 | 1614.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1614.00 | 1635.15 | 1614.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 1609.95 | 1635.15 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1635.35 | 1635.19 | 1616.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1643.10 | 1625.52 | 1615.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 1601.95 | 1615.00 | 1614.34 | SL hit (close<static) qty=1.00 sl=1607.60 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 1585.00 | 1620.53 | 1623.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 1561.90 | 1596.70 | 1610.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1544.25 | 1540.70 | 1568.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1544.25 | 1540.70 | 1568.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1544.25 | 1540.70 | 1568.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1561.20 | 1540.70 | 1568.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1553.10 | 1545.31 | 1566.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 1537.55 | 1564.16 | 1566.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1383.80 | 1504.25 | 1528.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 1533.00 | 1504.36 | 1503.03 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 1485.00 | 1503.96 | 1504.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 12:15:00 | 1462.90 | 1492.72 | 1498.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1515.70 | 1488.87 | 1494.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1515.70 | 1488.87 | 1494.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1515.70 | 1488.87 | 1494.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 1527.60 | 1488.87 | 1494.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1530.80 | 1503.20 | 1500.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1587.60 | 1532.36 | 1516.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 10:15:00 | 1672.40 | 1679.27 | 1658.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 11:00:00 | 1672.40 | 1679.27 | 1658.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 1652.00 | 1673.82 | 1657.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:00:00 | 1652.00 | 1673.82 | 1657.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 1659.00 | 1670.85 | 1657.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:30:00 | 1653.10 | 1670.85 | 1657.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1680.90 | 1672.86 | 1659.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:15:00 | 1695.30 | 1672.86 | 1659.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1696.10 | 1678.71 | 1664.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1638.60 | 1669.71 | 1663.16 | SL hit (close<static) qty=1.00 sl=1657.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1618.00 | 1655.01 | 1658.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1596.10 | 1643.23 | 1652.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 1606.30 | 1605.99 | 1621.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 13:30:00 | 1607.20 | 1605.99 | 1621.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1613.00 | 1607.39 | 1620.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 1613.00 | 1607.39 | 1620.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1618.60 | 1609.64 | 1620.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 1630.00 | 1609.64 | 1620.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1627.10 | 1613.13 | 1620.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:00:00 | 1619.60 | 1614.42 | 1620.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:45:00 | 1619.20 | 1612.13 | 1615.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 12:15:00 | 1627.80 | 1618.51 | 1618.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 12:15:00 | 1627.80 | 1618.51 | 1618.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 1649.40 | 1627.53 | 1622.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 1625.30 | 1627.08 | 1622.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:00:00 | 1625.30 | 1627.08 | 1622.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1602.40 | 1622.15 | 1621.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 1602.40 | 1622.15 | 1621.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 1598.50 | 1617.42 | 1619.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 15:15:00 | 1589.00 | 1605.82 | 1612.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1614.00 | 1607.46 | 1612.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1614.00 | 1607.46 | 1612.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1614.00 | 1607.46 | 1612.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1614.00 | 1607.46 | 1612.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1633.30 | 1612.63 | 1614.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1633.30 | 1612.63 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 1631.50 | 1616.40 | 1616.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 1653.60 | 1623.84 | 1619.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1613.80 | 1638.69 | 1629.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1613.80 | 1638.69 | 1629.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1613.80 | 1638.69 | 1629.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1613.80 | 1638.69 | 1629.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1622.00 | 1635.36 | 1629.05 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1595.20 | 1622.29 | 1624.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1589.20 | 1615.67 | 1621.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1604.10 | 1602.16 | 1612.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 1604.10 | 1602.16 | 1612.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1638.00 | 1609.32 | 1614.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 1638.00 | 1609.32 | 1614.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1625.80 | 1612.62 | 1615.97 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1639.50 | 1621.75 | 1619.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 1668.90 | 1632.82 | 1625.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1623.80 | 1646.64 | 1635.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1623.80 | 1646.64 | 1635.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1623.80 | 1646.64 | 1635.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1623.80 | 1646.64 | 1635.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1610.40 | 1639.39 | 1633.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1610.40 | 1639.39 | 1633.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1606.90 | 1632.89 | 1631.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1614.00 | 1632.89 | 1631.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1612.80 | 1628.88 | 1629.47 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1667.20 | 1629.82 | 1628.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1709.10 | 1661.60 | 1645.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1663.70 | 1671.06 | 1654.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:00:00 | 1663.70 | 1671.06 | 1654.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 1759.70 | 1773.26 | 1757.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:45:00 | 1741.40 | 1773.26 | 1757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 1755.40 | 1769.68 | 1757.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 1716.20 | 1769.68 | 1757.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1685.70 | 1752.89 | 1750.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 1685.70 | 1752.89 | 1750.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 1696.10 | 1741.53 | 1745.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 1662.30 | 1701.65 | 1721.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1698.30 | 1670.09 | 1690.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1723.50 | 1680.77 | 1693.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1723.50 | 1680.77 | 1693.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 1726.90 | 1702.89 | 1701.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 1747.80 | 1715.34 | 1707.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1786.50 | 1788.65 | 1772.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1769.00 | 1788.65 | 1772.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1756.60 | 1782.24 | 1771.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1756.60 | 1782.24 | 1771.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1779.70 | 1781.73 | 1772.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1793.90 | 1781.73 | 1772.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 1780.90 | 1781.89 | 1774.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1781.60 | 1777.56 | 1774.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:15:00 | 1783.20 | 1777.25 | 1774.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1777.10 | 1777.22 | 1775.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:15:00 | 1771.10 | 1777.22 | 1775.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1762.00 | 1774.18 | 1773.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1762.00 | 1774.18 | 1773.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 1746.30 | 1768.60 | 1771.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1746.30 | 1768.60 | 1771.39 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1786.40 | 1773.32 | 1772.28 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 1751.70 | 1769.54 | 1771.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 1730.00 | 1754.00 | 1760.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 1782.30 | 1750.44 | 1753.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1785.60 | 1757.47 | 1756.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 1799.20 | 1765.81 | 1760.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 13:15:00 | 1911.30 | 1922.40 | 1893.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 1911.30 | 1922.40 | 1893.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 1901.00 | 1916.55 | 1895.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 1884.00 | 1916.55 | 1895.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1891.00 | 1911.44 | 1895.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1878.80 | 1911.44 | 1895.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1895.70 | 1908.29 | 1895.20 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 1873.30 | 1889.56 | 1889.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 1860.30 | 1881.06 | 1885.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 1890.60 | 1878.57 | 1881.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1895.00 | 1881.85 | 1883.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1902.50 | 1881.85 | 1883.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 1912.10 | 1887.90 | 1885.78 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1870.70 | 1883.61 | 1884.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1860.00 | 1875.43 | 1880.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 1874.00 | 1871.10 | 1876.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1893.20 | 1875.52 | 1877.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1893.20 | 1875.52 | 1877.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1892.00 | 1878.82 | 1879.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1869.90 | 1878.82 | 1879.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1909.10 | 1881.42 | 1879.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1909.10 | 1881.42 | 1879.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1949.80 | 1902.98 | 1890.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 1906.20 | 1934.85 | 1917.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1902.50 | 1928.38 | 1915.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 1902.50 | 1928.38 | 1915.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1888.10 | 1920.32 | 1913.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1887.10 | 1920.32 | 1913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 1886.80 | 1907.18 | 1908.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1873.30 | 1891.76 | 1899.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1879.30 | 1879.01 | 1889.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 1877.60 | 1879.01 | 1889.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1887.50 | 1880.71 | 1889.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1889.70 | 1880.71 | 1889.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1889.70 | 1882.50 | 1889.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 1892.60 | 1882.50 | 1889.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1891.20 | 1884.24 | 1889.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 1893.20 | 1884.24 | 1889.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1899.70 | 1887.33 | 1890.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1902.00 | 1887.33 | 1890.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1895.90 | 1891.27 | 1892.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1886.20 | 1891.27 | 1892.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1903.10 | 1892.63 | 1892.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1903.10 | 1892.63 | 1892.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1915.10 | 1900.17 | 1896.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 1888.20 | 1897.77 | 1895.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1880.30 | 1894.28 | 1894.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 1880.30 | 1894.28 | 1894.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1888.10 | 1893.04 | 1893.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 1869.50 | 1888.33 | 1891.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1833.60 | 1832.29 | 1848.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1833.60 | 1832.29 | 1848.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1863.90 | 1838.25 | 1848.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1863.90 | 1838.25 | 1848.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1852.80 | 1841.16 | 1848.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 12:00:00 | 1841.50 | 1841.23 | 1848.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 1845.40 | 1829.68 | 1828.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 1845.40 | 1829.68 | 1828.46 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1804.40 | 1823.08 | 1825.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1773.40 | 1802.79 | 1812.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 1798.00 | 1796.02 | 1805.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 1815.80 | 1796.02 | 1805.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1805.00 | 1797.81 | 1805.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1801.70 | 1797.81 | 1805.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1804.30 | 1799.11 | 1805.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1794.10 | 1799.11 | 1805.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:15:00 | 1802.20 | 1795.58 | 1801.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 1827.80 | 1802.02 | 1804.27 | SL hit (close>static) qty=1.00 sl=1810.20 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 1827.90 | 1807.20 | 1806.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 1835.00 | 1817.24 | 1811.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 1843.20 | 1845.46 | 1832.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:00:00 | 1843.20 | 1845.46 | 1832.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1859.10 | 1851.42 | 1839.08 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 1798.00 | 1827.56 | 1831.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1789.20 | 1819.89 | 1827.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 1808.80 | 1799.96 | 1809.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1812.60 | 1802.49 | 1810.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 1812.60 | 1802.49 | 1810.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1823.90 | 1806.77 | 1811.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1823.90 | 1806.77 | 1811.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1833.10 | 1812.04 | 1813.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 1833.70 | 1812.04 | 1813.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1834.00 | 1816.43 | 1815.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1852.00 | 1823.54 | 1818.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1840.50 | 1847.11 | 1835.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1838.20 | 1844.17 | 1836.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1833.00 | 1844.17 | 1836.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1828.50 | 1841.04 | 1835.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 1830.60 | 1841.04 | 1835.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1814.10 | 1835.65 | 1833.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 1814.10 | 1835.65 | 1833.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 1810.00 | 1830.52 | 1831.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1797.40 | 1821.06 | 1826.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1767.40 | 1766.77 | 1782.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 1767.40 | 1766.77 | 1782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1780.00 | 1769.42 | 1782.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1780.00 | 1769.42 | 1782.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1799.20 | 1775.37 | 1783.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 1799.20 | 1775.37 | 1783.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1786.40 | 1777.58 | 1783.94 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1797.50 | 1788.96 | 1788.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1823.80 | 1798.85 | 1793.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 14:15:00 | 1816.50 | 1817.29 | 1810.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:30:00 | 1817.30 | 1817.29 | 1810.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1766.50 | 1808.25 | 1807.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1766.50 | 1808.25 | 1807.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1791.70 | 1804.94 | 1805.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 1745.10 | 1773.87 | 1783.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 1782.70 | 1773.69 | 1781.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1790.00 | 1776.95 | 1782.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1769.90 | 1776.95 | 1782.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1772.10 | 1775.98 | 1781.48 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 1794.00 | 1785.14 | 1784.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1814.60 | 1791.03 | 1786.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 1782.40 | 1809.80 | 1801.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1792.10 | 1806.26 | 1800.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 1802.40 | 1804.83 | 1800.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1779.50 | 1802.10 | 1801.16 | SL hit (close<static) qty=1.00 sl=1780.50 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1780.80 | 1796.97 | 1798.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1778.40 | 1793.26 | 1797.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1776.70 | 1775.35 | 1783.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:45:00 | 1780.30 | 1775.35 | 1783.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1782.30 | 1776.74 | 1783.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1782.30 | 1776.74 | 1783.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1778.20 | 1777.03 | 1783.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1764.10 | 1777.03 | 1783.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 1786.30 | 1760.20 | 1757.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1786.30 | 1760.20 | 1757.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 1813.00 | 1779.53 | 1769.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 1824.60 | 1829.50 | 1807.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 1824.60 | 1829.50 | 1807.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1841.90 | 1850.22 | 1837.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 1847.10 | 1850.22 | 1837.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1840.60 | 1848.29 | 1837.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 1841.20 | 1848.29 | 1837.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1840.30 | 1846.69 | 1837.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1899.90 | 1839.56 | 1836.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1897.80 | 1903.58 | 1904.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 1897.80 | 1903.58 | 1904.15 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1910.30 | 1904.92 | 1904.71 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 1898.30 | 1903.83 | 1904.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1864.80 | 1896.03 | 1900.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1803.70 | 1792.55 | 1813.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 1804.10 | 1792.55 | 1813.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1818.00 | 1802.56 | 1810.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1818.00 | 1802.56 | 1810.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1812.00 | 1804.45 | 1810.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1805.90 | 1804.45 | 1810.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1818.90 | 1807.34 | 1811.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1819.80 | 1807.34 | 1811.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1831.80 | 1812.23 | 1813.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1831.80 | 1812.23 | 1813.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1830.60 | 1815.91 | 1815.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1834.50 | 1823.43 | 1819.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1831.20 | 1832.26 | 1825.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1853.00 | 1863.13 | 1852.06 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1833.20 | 1846.77 | 1846.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1821.40 | 1841.70 | 1844.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 1801.10 | 1797.48 | 1810.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 14:00:00 | 1801.10 | 1797.48 | 1810.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1804.00 | 1798.78 | 1809.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 1809.10 | 1798.78 | 1809.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1810.10 | 1801.05 | 1809.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1817.50 | 1801.05 | 1809.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1818.60 | 1804.56 | 1810.44 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1825.10 | 1815.13 | 1814.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 1837.90 | 1823.17 | 1818.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 1810.20 | 1820.58 | 1817.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1814.90 | 1819.44 | 1817.54 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1806.00 | 1814.94 | 1815.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1800.50 | 1809.67 | 1812.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 1804.40 | 1798.15 | 1804.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1812.80 | 1801.08 | 1805.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 1812.80 | 1801.08 | 1805.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1806.20 | 1802.11 | 1805.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1814.10 | 1802.11 | 1805.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1826.80 | 1807.35 | 1807.07 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1794.80 | 1810.05 | 1811.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 14:15:00 | 1793.30 | 1806.70 | 1810.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1791.10 | 1789.19 | 1796.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1800.80 | 1787.31 | 1792.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1800.80 | 1787.31 | 1792.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1810.50 | 1791.95 | 1794.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 1810.00 | 1791.95 | 1794.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1795.50 | 1794.48 | 1795.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1795.50 | 1794.48 | 1795.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1803.40 | 1796.26 | 1795.95 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1780.30 | 1793.75 | 1795.25 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 1801.40 | 1795.80 | 1795.65 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1785.70 | 1794.81 | 1795.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1777.00 | 1791.25 | 1793.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1753.20 | 1753.00 | 1769.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 11:00:00 | 1753.20 | 1753.00 | 1769.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1768.20 | 1753.50 | 1765.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1768.20 | 1753.50 | 1765.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1749.40 | 1752.68 | 1763.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1739.50 | 1752.85 | 1762.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1703.10 | 1695.71 | 1695.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1703.10 | 1695.71 | 1695.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 1707.10 | 1697.98 | 1696.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 1734.00 | 1745.06 | 1746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 1729.00 | 1742.24 | 1744.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1742.00 | 1740.16 | 1742.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1745.00 | 1741.13 | 1742.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1745.20 | 1741.13 | 1742.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1740.00 | 1740.90 | 1742.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1733.10 | 1740.90 | 1742.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1730.60 | 1737.50 | 1740.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 1735.00 | 1736.67 | 1739.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:15:00 | 1648.25 | 1679.64 | 1691.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:15:00 | 1646.44 | 1673.62 | 1687.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:15:00 | 1644.07 | 1673.62 | 1687.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 1651.20 | 1649.36 | 1663.63 | SL hit (close>ema200) qty=0.50 sl=1649.36 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 1675.40 | 1662.76 | 1661.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 12:15:00 | 1688.30 | 1671.74 | 1666.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1730.00 | 1756.28 | 1735.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1735.70 | 1752.16 | 1735.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 1724.90 | 1752.16 | 1735.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1712.40 | 1744.21 | 1733.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 1712.40 | 1744.21 | 1733.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1719.90 | 1739.35 | 1732.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1711.80 | 1739.35 | 1732.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1724.00 | 1730.68 | 1729.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1800.90 | 1730.68 | 1729.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1765.50 | 1805.50 | 1807.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1765.50 | 1805.50 | 1807.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 1740.30 | 1792.46 | 1801.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1762.60 | 1759.95 | 1778.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1762.60 | 1759.95 | 1778.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1789.60 | 1765.88 | 1779.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:30:00 | 1792.70 | 1765.88 | 1779.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1790.20 | 1770.75 | 1780.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1794.40 | 1770.75 | 1780.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1787.30 | 1776.32 | 1781.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1773.20 | 1776.32 | 1781.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1805.40 | 1783.05 | 1783.54 | SL hit (close>static) qty=1.00 sl=1793.60 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1799.20 | 1786.28 | 1784.96 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 1778.10 | 1785.38 | 1786.33 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1796.40 | 1787.26 | 1786.89 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1777.80 | 1785.91 | 1786.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 1752.90 | 1779.30 | 1783.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1741.10 | 1735.67 | 1750.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 1741.10 | 1735.67 | 1750.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1753.20 | 1742.02 | 1748.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1753.20 | 1742.02 | 1748.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1766.70 | 1746.96 | 1750.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:45:00 | 1771.50 | 1746.96 | 1750.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1802.10 | 1757.99 | 1755.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1819.20 | 1770.23 | 1761.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1831.00 | 1837.43 | 1817.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 1831.00 | 1837.43 | 1817.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1820.00 | 1840.07 | 1829.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 1811.70 | 1840.07 | 1829.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1810.20 | 1834.10 | 1827.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1810.20 | 1834.10 | 1827.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1805.40 | 1823.08 | 1823.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 1797.80 | 1809.04 | 1815.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 1801.00 | 1791.03 | 1790.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 11:15:00 | 1804.80 | 1793.79 | 1791.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 10:15:00 | 1802.60 | 1803.22 | 1798.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 11:15:00 | 1803.40 | 1803.22 | 1798.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 1803.70 | 1803.32 | 1798.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 1807.30 | 1803.26 | 1799.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1890.90 | 1924.64 | 1927.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1890.90 | 1924.64 | 1927.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1857.50 | 1905.71 | 1915.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 1791.00 | 1790.59 | 1825.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:00:00 | 1791.00 | 1790.59 | 1825.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1827.40 | 1799.09 | 1823.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1827.40 | 1799.09 | 1823.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1823.60 | 1803.99 | 1823.45 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1856.70 | 1832.65 | 1830.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 1874.20 | 1840.96 | 1834.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 1878.00 | 1878.30 | 1864.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 1884.90 | 1878.30 | 1864.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1907.90 | 1916.95 | 1906.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1907.90 | 1916.95 | 1906.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1915.10 | 1916.58 | 1907.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1916.80 | 1913.77 | 1908.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 1920.00 | 1912.82 | 1908.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1902.20 | 1911.61 | 1909.42 | SL hit (close<static) qty=1.00 sl=1903.70 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1903.00 | 1907.09 | 1907.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1864.60 | 1897.14 | 1902.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1841.70 | 1838.75 | 1855.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 1842.30 | 1838.75 | 1855.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1691.40 | 1689.77 | 1702.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1690.00 | 1689.77 | 1702.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1626.60 | 1642.46 | 1653.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 1617.20 | 1638.63 | 1650.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 1620.70 | 1635.89 | 1648.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:30:00 | 1621.00 | 1630.13 | 1643.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1617.50 | 1627.30 | 1639.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1639.30 | 1629.70 | 1639.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 1630.70 | 1629.70 | 1639.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1661.00 | 1635.96 | 1641.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 1661.00 | 1635.96 | 1641.71 | SL hit (close>static) qty=1.00 sl=1659.80 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 1673.70 | 1649.28 | 1647.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 1682.40 | 1660.10 | 1652.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 1641.50 | 1659.72 | 1654.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1642.70 | 1656.32 | 1652.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 1661.60 | 1654.82 | 1652.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 14:15:00 | 1667.50 | 1654.82 | 1652.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 15:15:00 | 1666.60 | 1655.60 | 1653.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1638.20 | 1652.44 | 1652.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1638.20 | 1652.44 | 1652.65 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 1660.00 | 1651.93 | 1651.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 1693.60 | 1660.27 | 1655.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1691.40 | 1693.20 | 1678.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1691.40 | 1693.20 | 1678.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1666.20 | 1687.80 | 1677.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1666.20 | 1687.80 | 1677.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1650.30 | 1680.30 | 1674.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1650.30 | 1680.30 | 1674.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1635.60 | 1671.36 | 1671.41 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 1712.00 | 1676.71 | 1673.66 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1641.10 | 1669.59 | 1670.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 1627.20 | 1661.11 | 1666.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1637.50 | 1636.92 | 1651.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 1637.50 | 1636.92 | 1651.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1643.50 | 1631.74 | 1643.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1643.50 | 1631.74 | 1643.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1641.30 | 1633.65 | 1643.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 1651.30 | 1633.65 | 1643.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 1630.60 | 1633.04 | 1642.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1614.70 | 1639.03 | 1641.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1653.10 | 1631.30 | 1634.99 | SL hit (close>static) qty=1.00 sl=1643.30 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 1648.60 | 1638.01 | 1637.60 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1610.10 | 1634.32 | 1636.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1594.30 | 1620.02 | 1627.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 1443.00 | 1438.07 | 1470.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 1443.00 | 1438.07 | 1470.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1583.80 | 1467.21 | 1481.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1583.80 | 1467.21 | 1481.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1559.20 | 1485.61 | 1488.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 1540.00 | 1485.61 | 1488.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 1487.00 | 1512.88 | 1515.80 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1545.90 | 1521.12 | 1517.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 1546.70 | 1526.23 | 1520.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1558.70 | 1541.78 | 1533.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 1551.70 | 1548.30 | 1538.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1551.20 | 1544.44 | 1539.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1509.60 | 1530.29 | 1534.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 1496.10 | 1495.40 | 1507.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 1496.10 | 1495.40 | 1507.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1507.00 | 1496.94 | 1505.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1510.80 | 1496.94 | 1505.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1508.00 | 1499.15 | 1505.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1505.60 | 1499.15 | 1505.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1499.00 | 1499.12 | 1505.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 1497.30 | 1499.12 | 1505.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 1493.50 | 1501.96 | 1504.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 1496.50 | 1500.86 | 1504.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1495.90 | 1500.97 | 1503.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1486.00 | 1497.98 | 1502.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1481.60 | 1495.10 | 1500.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:45:00 | 1477.60 | 1491.84 | 1498.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1546.20 | 1511.98 | 1504.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 1493.80 | 1519.09 | 1511.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1493.60 | 1513.99 | 1509.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1492.00 | 1513.99 | 1509.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1488.70 | 1506.20 | 1506.70 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1525.10 | 1506.47 | 1504.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 1530.20 | 1511.21 | 1507.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1513.70 | 1514.14 | 1509.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1514.70 | 1515.21 | 1510.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1527.00 | 1517.40 | 1512.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1523.70 | 1519.09 | 1514.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:45:00 | 1521.40 | 1519.82 | 1514.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1492.60 | 1515.18 | 1514.13 | SL hit (close<static) qty=1.00 sl=1507.10 alert=retest2 |

### Cycle 146 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1482.70 | 1508.69 | 1511.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1478.80 | 1498.68 | 1506.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 1474.00 | 1469.98 | 1482.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 09:30:00 | 1473.50 | 1469.48 | 1480.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1472.20 | 1468.78 | 1478.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 1476.50 | 1468.78 | 1478.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1482.60 | 1471.54 | 1478.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:00:00 | 1482.60 | 1471.54 | 1478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1482.60 | 1473.75 | 1479.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:30:00 | 1482.00 | 1473.75 | 1479.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1475.80 | 1475.87 | 1478.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1466.90 | 1475.87 | 1478.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1467.70 | 1471.87 | 1475.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 1466.40 | 1471.50 | 1475.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.56 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1394.32 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.08 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 1423.70 | 1420.84 | 1439.32 | SL hit (close>ema200) qty=0.50 sl=1420.84 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1464.90 | 1443.09 | 1442.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1470.00 | 1452.04 | 1446.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 1467.60 | 1468.88 | 1458.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 1467.60 | 1468.88 | 1458.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1460.70 | 1468.22 | 1460.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1460.70 | 1468.22 | 1460.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1456.50 | 1465.88 | 1460.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1449.60 | 1465.88 | 1460.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1452.30 | 1463.16 | 1459.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1457.70 | 1463.16 | 1459.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 1463.50 | 1461.78 | 1459.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:45:00 | 1457.90 | 1461.00 | 1459.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 1477.40 | 1458.13 | 1456.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 1497.00 | 1467.80 | 1461.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1509.50 | 1517.71 | 1500.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 1508.80 | 1517.71 | 1500.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1507.60 | 1513.67 | 1501.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1507.60 | 1513.67 | 1501.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1497.30 | 1510.40 | 1501.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1497.30 | 1510.40 | 1501.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1468.20 | 1501.96 | 1498.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 1480.00 | 1501.96 | 1498.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1505.50 | 1504.14 | 1500.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 1505.00 | 1504.14 | 1500.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1497.20 | 1503.18 | 1500.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1497.20 | 1503.18 | 1500.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1504.00 | 1503.34 | 1501.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1472.70 | 1503.34 | 1501.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1456.30 | 1493.93 | 1497.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1449.80 | 1485.11 | 1492.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1454.60 | 1451.89 | 1467.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 1454.60 | 1451.89 | 1467.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1466.60 | 1454.83 | 1467.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1466.60 | 1454.83 | 1467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1471.30 | 1458.13 | 1467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1478.40 | 1458.13 | 1467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1471.20 | 1460.74 | 1468.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1472.60 | 1460.74 | 1468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1461.30 | 1460.85 | 1467.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1461.60 | 1460.85 | 1467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1460.20 | 1460.72 | 1466.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1496.90 | 1460.72 | 1466.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1480.00 | 1464.58 | 1468.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 1474.30 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 1472.40 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1419.70 | 1461.52 | 1466.53 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1469.60 | 1452.06 | 1451.82 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1431.40 | 1448.68 | 1450.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1396.40 | 1435.39 | 1443.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1429.00 | 1423.94 | 1433.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1429.00 | 1423.94 | 1433.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1425.10 | 1424.17 | 1432.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1415.30 | 1424.17 | 1432.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1443.60 | 1428.06 | 1433.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 1443.60 | 1428.06 | 1433.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1437.20 | 1429.89 | 1434.22 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1463.70 | 1441.73 | 1438.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1469.60 | 1447.30 | 1441.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 1467.20 | 1448.52 | 1444.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:30:00 | 1460.40 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:00:00 | 1461.10 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1492.20 | 1452.58 | 1447.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1481.70 | 1488.67 | 1474.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 1481.70 | 1488.67 | 1474.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1465.70 | 1498.34 | 1493.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 1462.40 | 1481.71 | 1486.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:30:00 | 1471.60 | 1474.89 | 1478.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 1472.10 | 1474.75 | 1478.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 13:15:00 | 1528.60 | 1499.42 | 1490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1605.20 | 1606.70 | 1580.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1596.40 | 1606.70 | 1580.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1596.70 | 1604.70 | 1582.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 1622.30 | 1607.92 | 1587.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1698.40 | 1675.73 | 1673.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1711.00 | 1682.79 | 1676.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1684.60 | 1689.76 | 1681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1683.50 | 1688.51 | 1681.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 1686.00 | 1688.51 | 1681.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1680.50 | 1686.91 | 1681.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1672.50 | 1686.91 | 1681.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1654.50 | 1680.43 | 1679.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1654.50 | 1680.43 | 1679.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1650.00 | 1674.34 | 1676.69 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 1673.90 | 1670.87 | 1670.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1679.40 | 1672.84 | 1671.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1668.10 | 1673.52 | 1672.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1712.40 | 1681.29 | 1675.93 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 1645.50 | 1677.57 | 1678.91 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 1683.50 | 1679.24 | 1679.08 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1662.80 | 1676.88 | 1678.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 1657.60 | 1673.02 | 1676.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 09:15:00 | 1288.20 | 2024-05-21 15:15:00 | 1290.80 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-05-23 09:15:00 | 1277.35 | 2024-05-28 09:15:00 | 1213.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 10:30:00 | 1280.00 | 2024-05-28 09:15:00 | 1216.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 11:00:00 | 1281.60 | 2024-05-28 09:15:00 | 1217.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 09:15:00 | 1277.35 | 2024-05-30 10:15:00 | 1204.10 | STOP_HIT | 0.50 | 5.73% |
| SELL | retest2 | 2024-05-24 10:30:00 | 1280.00 | 2024-05-30 10:15:00 | 1204.10 | STOP_HIT | 0.50 | 5.93% |
| SELL | retest2 | 2024-05-24 11:00:00 | 1281.60 | 2024-05-30 10:15:00 | 1204.10 | STOP_HIT | 0.50 | 6.05% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1331.50 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-06-11 09:15:00 | 1316.40 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2024-06-11 11:00:00 | 1303.35 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1314.85 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-06-12 11:45:00 | 1308.30 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-06-12 13:00:00 | 1308.50 | 2024-06-21 10:15:00 | 1330.40 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-06-25 11:45:00 | 1296.30 | 2024-06-26 09:15:00 | 1338.90 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-06-25 14:45:00 | 1310.00 | 2024-06-26 09:15:00 | 1338.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1421.00 | 2024-07-22 14:15:00 | 1461.55 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1493.45 | 2024-07-30 10:15:00 | 1464.25 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-07-29 13:15:00 | 1492.00 | 2024-07-30 10:15:00 | 1464.25 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-07-29 14:00:00 | 1492.40 | 2024-07-30 10:15:00 | 1464.25 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-07-29 15:00:00 | 1491.05 | 2024-07-30 10:15:00 | 1464.25 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-08-05 13:15:00 | 1473.00 | 2024-08-06 14:15:00 | 1422.20 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-08-22 14:00:00 | 1698.45 | 2024-08-29 10:15:00 | 1725.35 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2024-08-23 10:15:00 | 1705.85 | 2024-08-29 10:15:00 | 1725.35 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2024-08-26 09:15:00 | 1709.80 | 2024-08-29 10:15:00 | 1725.35 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest1 | 2024-09-13 09:30:00 | 1822.05 | 2024-09-16 09:15:00 | 1789.25 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest1 | 2024-09-13 11:15:00 | 1820.25 | 2024-09-16 09:15:00 | 1789.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest1 | 2024-09-13 11:45:00 | 1820.05 | 2024-09-16 09:15:00 | 1789.25 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-20 10:30:00 | 1901.05 | 2024-09-24 12:15:00 | 1808.50 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2024-09-20 12:15:00 | 1900.90 | 2024-09-24 12:15:00 | 1808.50 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-09-20 13:45:00 | 1900.25 | 2024-09-24 12:15:00 | 1808.50 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-09-20 15:00:00 | 1930.00 | 2024-09-24 12:15:00 | 1808.50 | STOP_HIT | 1.00 | -6.30% |
| SELL | retest2 | 2024-09-27 09:15:00 | 1688.75 | 2024-09-27 10:15:00 | 1604.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 09:15:00 | 1688.75 | 2024-09-30 09:15:00 | 1648.70 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-09-30 09:45:00 | 1682.05 | 2024-10-01 09:15:00 | 1597.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:45:00 | 1682.05 | 2024-10-01 09:15:00 | 1651.60 | STOP_HIT | 0.50 | 1.81% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1727.65 | 2024-10-07 13:15:00 | 1688.05 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-11 10:30:00 | 1615.75 | 2024-10-14 09:15:00 | 1664.90 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-10-18 09:30:00 | 1691.40 | 2024-10-18 14:15:00 | 1681.65 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-10-18 11:00:00 | 1690.90 | 2024-10-21 09:15:00 | 1666.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-18 11:30:00 | 1692.00 | 2024-10-21 09:15:00 | 1666.55 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-10-18 12:15:00 | 1689.80 | 2024-10-21 09:15:00 | 1666.55 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-18 13:15:00 | 1698.65 | 2024-10-21 09:15:00 | 1666.55 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1659.40 | 2024-10-28 09:15:00 | 1659.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-10-23 14:30:00 | 1651.35 | 2024-10-28 09:15:00 | 1659.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-11-04 13:15:00 | 1714.10 | 2024-11-05 09:15:00 | 1668.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1715.00 | 2024-11-05 09:15:00 | 1668.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-11-06 14:45:00 | 1656.30 | 2024-11-07 09:15:00 | 1734.60 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2024-11-12 12:45:00 | 1675.95 | 2024-11-14 12:15:00 | 1695.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1675.70 | 2024-11-14 12:15:00 | 1695.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-11-26 13:45:00 | 1826.00 | 2024-12-02 15:15:00 | 2004.81 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2024-11-27 09:30:00 | 1823.90 | 2024-12-02 15:15:00 | 2003.10 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2024-11-27 11:30:00 | 1822.55 | 2024-12-03 09:15:00 | 2008.60 | TARGET_HIT | 1.00 | 10.21% |
| BUY | retest2 | 2024-11-27 12:00:00 | 1821.00 | 2024-12-03 09:15:00 | 2006.29 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2024-11-28 13:30:00 | 1825.40 | 2024-12-03 09:15:00 | 2007.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 13:30:00 | 2097.05 | 2024-12-16 15:15:00 | 2120.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-18 09:15:00 | 2156.60 | 2024-12-18 12:15:00 | 2105.10 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-12-24 09:30:00 | 2074.30 | 2024-12-30 14:15:00 | 2146.95 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-12-24 10:00:00 | 2073.60 | 2024-12-30 14:15:00 | 2146.95 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-01-21 10:15:00 | 1695.80 | 2025-01-22 09:15:00 | 1611.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 1695.80 | 2025-01-22 14:15:00 | 1612.10 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2025-01-27 13:30:00 | 1705.00 | 2025-01-28 09:15:00 | 1657.35 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1708.05 | 2025-02-03 09:15:00 | 1653.65 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-02-10 12:00:00 | 1688.00 | 2025-02-11 10:15:00 | 1603.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 12:00:00 | 1688.00 | 2025-02-11 13:15:00 | 1519.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-18 11:15:00 | 1526.00 | 2025-02-21 10:15:00 | 1547.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-02-18 12:30:00 | 1525.25 | 2025-02-21 11:15:00 | 1560.65 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-18 14:45:00 | 1525.00 | 2025-02-21 11:15:00 | 1560.65 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-02-18 15:15:00 | 1526.00 | 2025-02-21 11:15:00 | 1560.65 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-02-19 13:15:00 | 1492.55 | 2025-02-21 11:15:00 | 1560.65 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1454.80 | 2025-03-03 09:15:00 | 1382.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1454.80 | 2025-03-03 14:15:00 | 1454.30 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2025-03-17 15:15:00 | 1355.00 | 2025-03-18 09:15:00 | 1387.85 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1643.10 | 2025-03-26 15:15:00 | 1601.95 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-03-27 09:15:00 | 1644.50 | 2025-03-28 11:15:00 | 1603.35 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-03-27 10:15:00 | 1642.90 | 2025-03-28 11:15:00 | 1603.35 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-03-27 11:30:00 | 1645.00 | 2025-03-28 11:15:00 | 1603.35 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-04-04 09:15:00 | 1537.55 | 2025-04-07 09:15:00 | 1383.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-22 14:15:00 | 1695.30 | 2025-04-23 10:15:00 | 1638.60 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1696.10 | 2025-04-23 10:15:00 | 1638.60 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-04-29 11:00:00 | 1619.60 | 2025-04-30 12:15:00 | 1627.80 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-04-30 10:45:00 | 1619.20 | 2025-04-30 12:15:00 | 1627.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1793.90 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-05-27 14:15:00 | 1780.90 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1781.60 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-28 12:15:00 | 1783.20 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1869.90 | 2025-06-16 13:15:00 | 1909.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1886.20 | 2025-06-23 11:15:00 | 1903.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-27 12:00:00 | 1841.50 | 2025-07-02 11:15:00 | 1845.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-07 11:15:00 | 1794.10 | 2025-07-07 14:15:00 | 1827.80 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-07-07 14:15:00 | 1802.20 | 2025-07-07 14:15:00 | 1827.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-31 11:45:00 | 1802.40 | 2025-08-01 09:15:00 | 1779.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-01 10:45:00 | 1798.50 | 2025-08-01 11:15:00 | 1780.80 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-01 11:15:00 | 1802.00 | 2025-08-01 11:15:00 | 1780.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-05 09:15:00 | 1764.10 | 2025-08-07 15:15:00 | 1786.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1899.90 | 2025-08-25 10:15:00 | 1897.80 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1739.50 | 2025-10-06 11:15:00 | 1703.10 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1733.10 | 2025-10-17 10:15:00 | 1648.25 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-13 12:00:00 | 1730.60 | 2025-10-17 11:15:00 | 1646.44 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-10-13 13:45:00 | 1735.00 | 2025-10-17 11:15:00 | 1644.07 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1733.10 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2025-10-13 12:00:00 | 1730.60 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-10-13 13:45:00 | 1735.00 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1800.90 | 2025-11-06 11:15:00 | 1765.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1773.20 | 2025-11-10 10:15:00 | 1805.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-28 13:15:00 | 1807.30 | 2025-12-15 09:15:00 | 1890.90 | STOP_HIT | 1.00 | 4.63% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1916.80 | 2025-12-29 12:15:00 | 1902.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-29 10:15:00 | 1920.00 | 2025-12-29 12:15:00 | 1902.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-16 11:15:00 | 1617.20 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-01-16 12:15:00 | 1620.70 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-01-16 13:30:00 | 1621.00 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1617.50 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-01-20 13:30:00 | 1661.60 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-20 14:15:00 | 1667.50 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-20 15:15:00 | 1666.60 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1614.70 | 2026-01-30 13:15:00 | 1653.10 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-02-05 15:15:00 | 1540.00 | 2026-02-05 15:15:00 | 1540.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-12 12:15:00 | 1558.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1551.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1551.20 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-18 12:15:00 | 1497.30 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-19 09:15:00 | 1493.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-19 10:00:00 | 1496.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-19 11:15:00 | 1495.90 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-19 13:15:00 | 1481.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1477.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-02-26 10:30:00 | 1527.00 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1523.70 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 13:45:00 | 1521.40 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 09:15:00 | 1393.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 09:15:00 | 1394.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 09:15:00 | 1393.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1457.70 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-03-12 11:45:00 | 1463.50 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-03-12 12:45:00 | 1457.90 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-25 10:45:00 | 1474.30 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-03-25 11:15:00 | 1472.40 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-07 12:30:00 | 1467.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-07 14:30:00 | 1460.40 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-07 15:00:00 | 1461.10 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1492.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-15 12:30:00 | 1471.60 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-15 13:30:00 | 1472.10 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-21 11:45:00 | 1622.30 | 2026-04-28 13:15:00 | 1661.10 | STOP_HIT | 1.00 | 2.39% |
