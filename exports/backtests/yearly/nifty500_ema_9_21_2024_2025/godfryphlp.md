# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 2424.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 132 |
| ALERT1 | 100 |
| ALERT2 | 99 |
| ALERT2_SKIP | 54 |
| ALERT3 | 235 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 113 |
| PARTIAL | 30 |
| TARGET_HIT | 15 |
| STOP_HIT | 103 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 76 / 72
- **Target hits / Stop hits / Partials:** 15 / 103 / 30
- **Avg / median % per leg:** 1.43% / 0.09%
- **Sum % (uncompounded):** 211.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 12 | 24.5% | 6 | 42 | 1 | -0.31% | -15.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.17% | 4.7% |
| BUY @ 3rd Alert (retest2) | 45 | 10 | 22.2% | 6 | 39 | 0 | -0.44% | -19.9% |
| SELL (all) | 99 | 64 | 64.6% | 9 | 61 | 29 | 2.29% | 227.0% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.39% | -0.8% |
| SELL @ 3rd Alert (retest2) | 97 | 63 | 64.9% | 9 | 59 | 29 | 2.35% | 227.8% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 0.65% | 3.9% |
| retest2 (combined) | 142 | 73 | 51.4% | 15 | 98 | 29 | 1.46% | 207.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 14:15:00 | 1106.50 | 1113.97 | 1114.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1155.00 | 1120.42 | 1117.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 1200.00 | 1136.34 | 1124.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1181.22 | 1186.87 | 1167.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1181.22 | 1186.87 | 1167.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1284.17 | 1312.56 | 1295.64 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 1284.57 | 1300.10 | 1302.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 1278.30 | 1293.20 | 1298.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 1294.33 | 1293.43 | 1298.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 1294.33 | 1293.43 | 1298.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1294.33 | 1293.43 | 1298.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 1294.33 | 1293.43 | 1298.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1308.55 | 1296.45 | 1299.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 1322.33 | 1296.45 | 1299.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1303.18 | 1297.80 | 1299.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:30:00 | 1308.23 | 1297.80 | 1299.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 1320.68 | 1304.29 | 1302.24 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1295.05 | 1301.17 | 1301.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 1288.67 | 1296.52 | 1299.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 1302.75 | 1294.18 | 1296.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 1302.75 | 1294.18 | 1296.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1302.75 | 1294.18 | 1296.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 1302.75 | 1294.18 | 1296.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1293.67 | 1294.08 | 1296.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:30:00 | 1278.77 | 1288.61 | 1293.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:30:00 | 1278.08 | 1285.52 | 1291.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 1282.25 | 1284.87 | 1290.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 15:00:00 | 1282.67 | 1283.65 | 1288.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1252.82 | 1274.94 | 1283.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 1316.67 | 1281.20 | 1281.72 | SL hit (close>static) qty=1.00 sl=1305.95 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1273.73 | 1226.57 | 1224.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 1276.67 | 1262.11 | 1247.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1292.03 | 1295.52 | 1279.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1310.73 | 1295.52 | 1279.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 14:30:00 | 1307.98 | 1304.17 | 1292.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1286.77 | 1300.69 | 1291.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-11 15:15:00 | 1286.77 | 1300.69 | 1291.92 | SL hit (close<ema400) qty=1.00 sl=1291.92 alert=retest1 |

### Cycle 7 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 1285.02 | 1294.30 | 1295.39 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 1356.18 | 1306.68 | 1300.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 10:15:00 | 1378.35 | 1321.01 | 1307.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1388.15 | 1397.81 | 1371.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1388.15 | 1397.81 | 1371.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1388.15 | 1397.81 | 1371.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1383.07 | 1397.81 | 1371.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1393.33 | 1405.70 | 1388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 1402.67 | 1405.70 | 1388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1394.82 | 1403.52 | 1389.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:45:00 | 1428.00 | 1408.29 | 1393.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 1427.30 | 1461.00 | 1461.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1427.30 | 1461.00 | 1461.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 09:15:00 | 1386.67 | 1437.99 | 1449.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1416.38 | 1399.16 | 1419.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1416.38 | 1399.16 | 1419.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1416.38 | 1399.16 | 1419.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 1417.93 | 1399.16 | 1419.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1411.27 | 1401.58 | 1418.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 1405.18 | 1401.58 | 1418.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 1423.33 | 1417.90 | 1419.52 | SL hit (close>static) qty=1.00 sl=1422.30 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 1430.73 | 1421.14 | 1419.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1464.22 | 1433.97 | 1427.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 10:15:00 | 1447.02 | 1450.43 | 1441.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 10:15:00 | 1447.02 | 1450.43 | 1441.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1447.02 | 1450.43 | 1441.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 1447.02 | 1450.43 | 1441.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1441.68 | 1449.24 | 1442.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 1442.88 | 1449.24 | 1442.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1448.30 | 1449.05 | 1443.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1455.58 | 1447.50 | 1443.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 1431.27 | 1444.25 | 1442.35 | SL hit (close<static) qty=1.00 sl=1441.68 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 1418.17 | 1439.04 | 1440.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 1405.77 | 1432.38 | 1437.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 1359.33 | 1349.68 | 1369.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1359.33 | 1349.68 | 1369.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1359.33 | 1349.68 | 1369.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 1366.25 | 1349.68 | 1369.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1363.37 | 1358.43 | 1367.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 1362.22 | 1358.43 | 1367.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1373.33 | 1361.41 | 1367.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1380.13 | 1361.41 | 1367.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1396.13 | 1368.35 | 1370.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 1394.37 | 1368.35 | 1370.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 1475.35 | 1383.10 | 1375.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 1538.17 | 1414.11 | 1390.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 1515.77 | 1516.95 | 1467.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 15:15:00 | 1474.57 | 1499.79 | 1477.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1474.57 | 1499.79 | 1477.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 1450.90 | 1499.79 | 1477.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1436.02 | 1487.03 | 1473.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:00:00 | 1436.02 | 1487.03 | 1473.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1434.50 | 1476.53 | 1470.18 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1439.27 | 1463.75 | 1465.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1420.65 | 1451.75 | 1459.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1427.00 | 1417.19 | 1431.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1427.00 | 1417.19 | 1431.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1427.00 | 1417.19 | 1431.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1427.00 | 1417.19 | 1431.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1432.97 | 1420.25 | 1428.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 1432.97 | 1420.25 | 1428.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1427.33 | 1421.66 | 1428.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 1443.33 | 1421.66 | 1428.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1412.85 | 1397.33 | 1408.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1412.85 | 1397.33 | 1408.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1406.77 | 1399.21 | 1408.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:30:00 | 1391.38 | 1399.01 | 1406.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:15:00 | 1387.25 | 1397.95 | 1405.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 1444.32 | 1405.51 | 1407.16 | SL hit (close>static) qty=1.00 sl=1416.50 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 1455.80 | 1415.57 | 1411.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 11:15:00 | 1478.08 | 1428.07 | 1417.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 1431.60 | 1454.99 | 1434.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 1431.60 | 1454.99 | 1434.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1431.60 | 1454.99 | 1434.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 1446.25 | 1454.99 | 1434.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1426.67 | 1449.33 | 1434.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1416.03 | 1449.33 | 1434.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1427.78 | 1442.18 | 1433.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 1427.78 | 1442.18 | 1433.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 1435.07 | 1440.76 | 1433.50 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 1422.45 | 1430.52 | 1431.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 1414.52 | 1425.64 | 1429.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 14:15:00 | 1410.10 | 1408.26 | 1415.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 1410.10 | 1408.26 | 1415.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1405.33 | 1407.68 | 1414.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 1465.37 | 1407.68 | 1414.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 1468.58 | 1419.86 | 1419.10 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 1415.68 | 1418.56 | 1418.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 15:15:00 | 1414.00 | 1417.65 | 1418.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1419.97 | 1418.10 | 1418.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 1419.97 | 1418.10 | 1418.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 1419.97 | 1418.10 | 1418.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 11:45:00 | 1405.55 | 1415.63 | 1417.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 14:15:00 | 1410.25 | 1415.19 | 1416.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 1339.74 | 1400.17 | 1409.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:15:00 | 1335.27 | 1387.98 | 1402.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 14:15:00 | 1356.98 | 1346.71 | 1362.75 | SL hit (close>ema200) qty=0.50 sl=1346.71 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1391.67 | 1371.18 | 1370.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1444.25 | 1396.35 | 1387.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1472.13 | 1475.97 | 1454.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:00:00 | 1472.13 | 1475.97 | 1454.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1465.30 | 1467.53 | 1458.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1465.30 | 1467.53 | 1458.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1456.32 | 1464.08 | 1458.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 10:30:00 | 1467.23 | 1465.93 | 1459.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 1491.55 | 1466.06 | 1462.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-19 10:15:00 | 1613.95 | 1545.71 | 1510.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 1799.00 | 1819.77 | 1822.41 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1896.65 | 1835.14 | 1829.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 1917.63 | 1903.65 | 1883.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 09:15:00 | 2023.33 | 2027.14 | 1983.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 10:00:00 | 2023.33 | 2027.14 | 1983.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2151.12 | 2210.83 | 2175.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 2151.12 | 2210.83 | 2175.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 2119.23 | 2192.51 | 2170.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:45:00 | 2127.73 | 2192.51 | 2170.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 2122.25 | 2157.26 | 2158.71 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 13:15:00 | 2286.77 | 2167.78 | 2157.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 14:15:00 | 2411.60 | 2216.54 | 2180.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 2253.22 | 2271.19 | 2240.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 2253.22 | 2271.19 | 2240.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2253.22 | 2271.19 | 2240.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 2253.22 | 2271.19 | 2240.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 2250.82 | 2267.11 | 2241.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 11:30:00 | 2262.33 | 2264.80 | 2242.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 2261.65 | 2264.80 | 2242.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 2264.33 | 2263.34 | 2245.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 2269.70 | 2260.14 | 2247.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 2226.00 | 2253.31 | 2245.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 2226.00 | 2253.31 | 2245.40 | SL hit (close<static) qty=1.00 sl=2240.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 2212.33 | 2237.07 | 2239.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 2204.68 | 2230.59 | 2236.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 2244.92 | 2229.20 | 2234.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 2244.92 | 2229.20 | 2234.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 2244.92 | 2229.20 | 2234.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 2207.57 | 2222.68 | 2229.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 2206.67 | 2220.33 | 2227.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 2207.97 | 2224.68 | 2228.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 2350.73 | 2249.89 | 2239.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 2350.73 | 2249.89 | 2239.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 2442.40 | 2288.39 | 2257.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 2580.70 | 2591.01 | 2480.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 2580.70 | 2591.01 | 2480.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 2576.67 | 2582.91 | 2529.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 2682.25 | 2541.44 | 2528.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 2608.90 | 2579.59 | 2557.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:30:00 | 2636.90 | 2593.34 | 2565.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 2606.70 | 2604.93 | 2578.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 2528.62 | 2589.67 | 2574.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 2528.62 | 2589.67 | 2574.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 2536.67 | 2579.07 | 2570.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 2447.25 | 2579.07 | 2570.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 2505.25 | 2564.31 | 2564.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 2505.25 | 2564.31 | 2564.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 2396.67 | 2438.11 | 2473.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 2409.02 | 2396.53 | 2430.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 2409.02 | 2396.53 | 2430.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2409.02 | 2396.53 | 2430.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:00:00 | 2352.13 | 2387.65 | 2423.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 2368.32 | 2383.78 | 2418.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:45:00 | 2367.65 | 2378.69 | 2409.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 2351.00 | 2375.35 | 2400.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 2234.52 | 2275.72 | 2305.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 2249.90 | 2275.72 | 2305.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 2249.27 | 2275.72 | 2305.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 2233.45 | 2275.72 | 2305.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 2234.08 | 2229.34 | 2262.66 | SL hit (close>ema200) qty=0.50 sl=2229.34 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 2298.12 | 2209.00 | 2208.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 2388.33 | 2257.63 | 2231.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 2337.18 | 2339.48 | 2296.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 13:15:00 | 2296.00 | 2327.68 | 2304.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 2296.00 | 2327.68 | 2304.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:45:00 | 2297.17 | 2327.68 | 2304.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 2296.67 | 2321.48 | 2303.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 2293.33 | 2321.48 | 2303.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 2311.67 | 2319.52 | 2304.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:00:00 | 2312.65 | 2318.14 | 2305.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 2316.67 | 2327.69 | 2317.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2318.97 | 2316.07 | 2315.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 2332.42 | 2326.77 | 2322.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 2324.00 | 2326.21 | 2322.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 2380.00 | 2335.91 | 2327.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:15:00 | 2349.17 | 2329.93 | 2328.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 15:15:00 | 2313.27 | 2324.86 | 2326.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 2313.27 | 2324.86 | 2326.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2265.17 | 2312.92 | 2320.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 2159.08 | 2147.53 | 2189.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 12:00:00 | 2135.00 | 2145.02 | 2184.51 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 2184.50 | 2155.49 | 2182.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-23 13:15:00 | 2184.50 | 2155.49 | 2182.57 | SL hit (close>ema400) qty=1.00 sl=2182.57 alert=retest1 |

### Cycle 28 — BUY (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 15:15:00 | 2288.00 | 2199.97 | 2199.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 11:15:00 | 2299.62 | 2244.47 | 2221.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 14:15:00 | 2133.33 | 2227.68 | 2220.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 14:15:00 | 2133.33 | 2227.68 | 2220.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 2133.33 | 2227.68 | 2220.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 2133.33 | 2227.68 | 2220.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 2170.00 | 2216.15 | 2215.94 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 2126.33 | 2198.18 | 2207.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 14:15:00 | 2079.12 | 2129.42 | 2152.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 13:15:00 | 2114.83 | 2110.83 | 2131.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 14:00:00 | 2114.83 | 2110.83 | 2131.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 2124.00 | 2113.46 | 2130.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 2124.00 | 2113.46 | 2130.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 2130.88 | 2117.46 | 2129.37 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 2147.10 | 2130.48 | 2129.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 2226.57 | 2154.35 | 2140.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 2263.33 | 2268.49 | 2227.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 09:45:00 | 2273.33 | 2268.49 | 2227.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 2314.00 | 2328.32 | 2311.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 2308.53 | 2328.32 | 2311.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2298.77 | 2322.41 | 2310.04 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2290.07 | 2303.60 | 2303.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2229.15 | 2288.71 | 2296.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 13:15:00 | 2304.00 | 2247.10 | 2265.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 13:15:00 | 2304.00 | 2247.10 | 2265.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 2304.00 | 2247.10 | 2265.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:30:00 | 2296.40 | 2247.10 | 2265.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 2209.12 | 2239.50 | 2260.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 2175.52 | 2239.50 | 2260.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 2207.32 | 2224.69 | 2249.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 2198.52 | 2219.74 | 2239.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 2096.95 | 2149.70 | 2192.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 2066.74 | 2112.40 | 2159.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 2088.59 | 2112.40 | 2159.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-18 09:15:00 | 1986.59 | 2050.53 | 2100.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 32 — BUY (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 11:15:00 | 1940.80 | 1934.25 | 1933.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1975.13 | 1945.55 | 1939.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 1940.35 | 1954.19 | 1948.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 1940.35 | 1954.19 | 1948.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1940.35 | 1954.19 | 1948.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 1942.10 | 1954.19 | 1948.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1944.98 | 1952.35 | 1948.19 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 13:15:00 | 1931.23 | 1944.70 | 1945.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 14:15:00 | 1899.98 | 1935.75 | 1941.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 1922.65 | 1918.27 | 1929.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 1922.65 | 1918.27 | 1929.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1927.83 | 1920.18 | 1929.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:30:00 | 1925.55 | 1920.18 | 1929.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1921.62 | 1920.47 | 1928.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 1916.67 | 1920.26 | 1927.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 14:15:00 | 1893.33 | 1882.78 | 1881.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 1893.33 | 1882.78 | 1881.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 15:15:00 | 1896.32 | 1885.49 | 1883.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 14:15:00 | 1993.12 | 1993.73 | 1970.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 15:00:00 | 1993.12 | 1993.73 | 1970.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1975.28 | 1989.45 | 1972.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 1982.67 | 1989.45 | 1972.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1965.07 | 1984.57 | 1972.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 1973.33 | 1984.57 | 1972.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1980.00 | 1983.66 | 1972.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 2003.33 | 1986.95 | 1975.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 1968.33 | 1988.98 | 1990.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1968.33 | 1988.98 | 1990.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 1961.50 | 1983.48 | 1988.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 10:15:00 | 1734.53 | 1699.55 | 1732.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 10:15:00 | 1734.53 | 1699.55 | 1732.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1734.53 | 1699.55 | 1732.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 1734.53 | 1699.55 | 1732.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 1799.33 | 1719.51 | 1738.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 1790.78 | 1719.51 | 1738.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1818.02 | 1739.21 | 1745.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 1839.03 | 1739.21 | 1745.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 1863.30 | 1764.03 | 1756.19 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 1707.35 | 1760.61 | 1766.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 1699.57 | 1748.40 | 1760.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 09:15:00 | 1698.33 | 1680.27 | 1716.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:15:00 | 1710.80 | 1680.27 | 1716.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1718.23 | 1687.86 | 1716.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 1667.33 | 1719.64 | 1723.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 1727.33 | 1705.86 | 1703.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 1727.33 | 1705.86 | 1703.65 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1680.10 | 1701.68 | 1703.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1655.82 | 1692.51 | 1699.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1661.22 | 1654.39 | 1670.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 1661.22 | 1654.39 | 1670.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1629.45 | 1651.58 | 1664.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 1615.07 | 1637.70 | 1654.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1595.05 | 1630.17 | 1646.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1534.32 | 1578.12 | 1596.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:15:00 | 1515.30 | 1565.29 | 1588.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 15:15:00 | 1453.56 | 1507.16 | 1547.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 40 — BUY (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 15:15:00 | 1449.98 | 1441.36 | 1440.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 1500.00 | 1452.00 | 1445.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 1447.93 | 1456.61 | 1449.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 14:15:00 | 1447.93 | 1456.61 | 1449.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1447.93 | 1456.61 | 1449.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 1447.93 | 1456.61 | 1449.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 1451.67 | 1455.62 | 1449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 1427.42 | 1455.62 | 1449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1426.70 | 1449.84 | 1447.81 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1418.97 | 1441.63 | 1444.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1395.37 | 1433.77 | 1439.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1435.57 | 1427.00 | 1435.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1435.57 | 1427.00 | 1435.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1435.57 | 1427.00 | 1435.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 1435.57 | 1427.00 | 1435.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1471.00 | 1435.80 | 1438.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1471.00 | 1435.80 | 1438.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 1464.68 | 1441.58 | 1440.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1490.28 | 1456.40 | 1448.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1494.55 | 1500.08 | 1483.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 1494.55 | 1500.08 | 1483.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1495.40 | 1499.14 | 1484.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 1486.47 | 1499.14 | 1484.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1483.38 | 1495.76 | 1485.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:15:00 | 1507.33 | 1490.07 | 1484.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 1532.92 | 1511.02 | 1499.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 13:15:00 | 1658.06 | 1557.58 | 1522.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 15:15:00 | 1568.63 | 1587.59 | 1588.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 09:15:00 | 1563.07 | 1582.68 | 1585.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 1613.33 | 1555.47 | 1559.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1613.33 | 1555.47 | 1559.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1613.33 | 1555.47 | 1559.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 1614.47 | 1555.47 | 1559.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 1656.77 | 1575.73 | 1568.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 11:15:00 | 1702.08 | 1601.00 | 1580.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 14:15:00 | 1693.30 | 1763.59 | 1734.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 14:15:00 | 1693.30 | 1763.59 | 1734.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 1693.30 | 1763.59 | 1734.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 1693.30 | 1763.59 | 1734.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 1690.00 | 1748.87 | 1730.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:30:00 | 1638.52 | 1723.59 | 1720.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 10:15:00 | 1644.05 | 1707.69 | 1713.56 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 1796.67 | 1725.85 | 1720.82 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 1673.60 | 1725.44 | 1730.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 15:15:00 | 1658.33 | 1712.02 | 1724.24 | Break + close below crossover candle low |

### Cycle 48 — BUY (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 09:15:00 | 1920.40 | 1753.70 | 1742.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-14 14:15:00 | 1999.83 | 1887.63 | 1820.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 13:15:00 | 2285.33 | 2364.16 | 2207.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-18 14:00:00 | 2285.33 | 2364.16 | 2207.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 2261.47 | 2343.62 | 2212.83 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 2103.35 | 2186.79 | 2195.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 13:15:00 | 2042.57 | 2120.09 | 2159.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 1915.00 | 1912.71 | 1984.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 12:45:00 | 1903.60 | 1912.71 | 1984.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1944.70 | 1899.12 | 1953.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 1955.13 | 1899.12 | 1953.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1955.13 | 1910.32 | 1953.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 1955.13 | 1910.32 | 1953.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1939.60 | 1916.18 | 1952.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:30:00 | 1945.00 | 1916.18 | 1952.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1706.90 | 1707.39 | 1745.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1706.90 | 1707.39 | 1745.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1707.32 | 1696.52 | 1724.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 1707.32 | 1696.52 | 1724.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1695.25 | 1696.88 | 1719.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:15:00 | 1690.93 | 1696.88 | 1719.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 1755.92 | 1717.94 | 1719.53 | SL hit (close>static) qty=1.00 sl=1729.32 alert=retest2 |

### Cycle 50 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 1760.68 | 1726.48 | 1723.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 1788.93 | 1742.38 | 1731.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1778.47 | 1779.77 | 1758.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1778.47 | 1779.77 | 1758.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1750.85 | 1771.43 | 1762.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1750.85 | 1771.43 | 1762.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1754.15 | 1767.97 | 1762.07 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 1734.33 | 1757.76 | 1758.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1715.63 | 1746.50 | 1752.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 1750.93 | 1739.16 | 1747.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 1750.93 | 1739.16 | 1747.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1750.93 | 1739.16 | 1747.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 1750.93 | 1739.16 | 1747.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1753.00 | 1741.93 | 1747.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:15:00 | 1750.40 | 1741.93 | 1747.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1750.40 | 1743.62 | 1747.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 13:30:00 | 1745.55 | 1745.87 | 1748.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 15:00:00 | 1743.33 | 1745.36 | 1748.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 1787.20 | 1754.47 | 1751.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 1787.20 | 1754.47 | 1751.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 1811.58 | 1765.89 | 1757.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1998.00 | 2022.70 | 1975.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 1998.00 | 2022.70 | 1975.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 2071.42 | 2086.30 | 2071.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 2069.02 | 2086.30 | 2071.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 2071.67 | 2083.38 | 2071.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 2071.57 | 2083.38 | 2071.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 2080.77 | 2082.86 | 2072.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 2096.40 | 2080.59 | 2073.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 2059.75 | 2075.37 | 2071.89 | SL hit (close<static) qty=1.00 sl=2069.67 alert=retest2 |

### Cycle 53 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 2058.78 | 2067.84 | 2068.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 2050.60 | 2064.39 | 2067.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 2084.35 | 2065.81 | 2066.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 2084.35 | 2065.81 | 2066.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 2084.35 | 2065.81 | 2066.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 2085.98 | 2065.81 | 2066.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 2083.58 | 2069.36 | 2068.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 2100.37 | 2075.56 | 2071.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 2287.53 | 2327.37 | 2267.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:00:00 | 2287.53 | 2327.37 | 2267.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 2314.68 | 2324.84 | 2271.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 2325.65 | 2324.84 | 2271.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:45:00 | 2335.67 | 2325.84 | 2276.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:00:00 | 2349.58 | 2330.59 | 2283.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 2243.92 | 2305.61 | 2310.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 2243.92 | 2305.61 | 2310.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 2226.47 | 2289.78 | 2302.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2220.90 | 2160.41 | 2203.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2220.90 | 2160.41 | 2203.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2220.90 | 2160.41 | 2203.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 2220.90 | 2160.41 | 2203.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 2220.90 | 2172.51 | 2204.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:30:00 | 2220.90 | 2172.51 | 2204.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 2327.00 | 2229.44 | 2221.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 2379.67 | 2302.45 | 2268.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 2477.00 | 2495.00 | 2457.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 2477.00 | 2495.00 | 2457.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 2477.00 | 2495.00 | 2457.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:45:00 | 2476.00 | 2495.00 | 2457.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 2509.67 | 2513.24 | 2497.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 2546.33 | 2513.24 | 2497.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-02 13:15:00 | 2800.96 | 2757.75 | 2736.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 2822.00 | 2849.00 | 2851.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 09:15:00 | 2794.83 | 2825.92 | 2839.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 10:15:00 | 2826.00 | 2825.94 | 2838.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 2826.00 | 2825.94 | 2838.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 2826.00 | 2825.94 | 2838.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 2826.00 | 2825.94 | 2838.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 2840.50 | 2828.85 | 2838.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:45:00 | 2833.33 | 2828.85 | 2838.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 2841.83 | 2831.45 | 2838.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:30:00 | 2846.67 | 2831.45 | 2838.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 2796.17 | 2824.39 | 2834.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 2768.67 | 2810.45 | 2827.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 2630.24 | 2767.82 | 2804.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 2740.00 | 2734.02 | 2766.85 | SL hit (close>ema200) qty=0.50 sl=2734.02 alert=retest2 |

### Cycle 58 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2867.67 | 2798.60 | 2791.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2985.00 | 2868.50 | 2831.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2951.33 | 2958.78 | 2905.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 2951.33 | 2958.78 | 2905.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 2933.33 | 2953.69 | 2908.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 2932.50 | 2953.69 | 2908.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 2906.17 | 3005.11 | 2976.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 2906.17 | 3005.11 | 2976.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2906.17 | 2985.32 | 2969.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 2906.17 | 2985.32 | 2969.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 2906.17 | 2956.83 | 2958.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 2789.00 | 2903.48 | 2931.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 2907.33 | 2842.87 | 2876.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2905.83 | 2855.46 | 2878.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 2898.00 | 2855.46 | 2878.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2872.00 | 2873.70 | 2882.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 2869.67 | 2873.70 | 2882.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2886.67 | 2876.29 | 2882.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2888.67 | 2876.29 | 2882.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2903.17 | 2881.67 | 2884.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 2905.00 | 2881.67 | 2884.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2873.33 | 2880.00 | 2883.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:15:00 | 2858.00 | 2880.00 | 2883.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:15:00 | 2715.10 | 2738.05 | 2760.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 2736.17 | 2734.06 | 2752.83 | SL hit (close>ema200) qty=0.50 sl=2734.06 alert=retest2 |

### Cycle 60 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2771.33 | 2762.10 | 2761.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 2822.17 | 2776.23 | 2768.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 2828.33 | 2858.16 | 2824.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2848.67 | 2856.26 | 2826.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 2881.50 | 2824.15 | 2820.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 2903.67 | 2837.49 | 2827.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 2796.17 | 2829.22 | 2824.20 | SL hit (close<static) qty=1.00 sl=2820.50 alert=retest2 |

### Cycle 61 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 2813.00 | 2820.00 | 2820.51 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 2858.67 | 2822.04 | 2820.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 2885.50 | 2834.73 | 2826.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 2812.83 | 2830.35 | 2825.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 2820.67 | 2828.42 | 2825.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:15:00 | 2812.83 | 2828.42 | 2825.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 2810.33 | 2824.80 | 2823.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 2811.00 | 2824.80 | 2823.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 2810.33 | 2821.91 | 2822.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 11:15:00 | 2760.50 | 2806.35 | 2814.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 2740.50 | 2739.83 | 2761.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:15:00 | 2760.67 | 2739.83 | 2761.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 2766.50 | 2745.16 | 2761.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 2765.33 | 2745.16 | 2761.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 2767.50 | 2749.63 | 2762.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:30:00 | 2783.00 | 2749.63 | 2762.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 2775.33 | 2754.77 | 2763.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:30:00 | 2775.50 | 2754.77 | 2763.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2760.17 | 2762.58 | 2764.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:30:00 | 2751.67 | 2759.86 | 2763.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 2752.17 | 2759.49 | 2761.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 2789.83 | 2757.61 | 2758.88 | SL hit (close>static) qty=1.00 sl=2776.67 alert=retest2 |

### Cycle 64 — BUY (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 14:15:00 | 2778.17 | 2755.10 | 2752.48 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2729.17 | 2752.18 | 2754.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 2720.17 | 2742.04 | 2748.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:45:00 | 2702.83 | 2721.11 | 2732.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 2704.83 | 2709.29 | 2723.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 2666.50 | 2676.44 | 2695.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 2703.50 | 2680.34 | 2682.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 2740.50 | 2704.34 | 2694.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 2988.00 | 3041.78 | 2967.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 2988.00 | 3041.78 | 2967.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2955.00 | 3024.42 | 2966.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 3029.83 | 3011.36 | 2970.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 3004.67 | 3010.39 | 2976.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 2920.50 | 2963.08 | 2964.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2920.50 | 2963.08 | 2964.75 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 3041.17 | 2965.75 | 2960.53 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 2928.33 | 2959.64 | 2960.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 2888.33 | 2931.75 | 2946.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 2777.83 | 2767.62 | 2811.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 2777.83 | 2767.62 | 2811.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2808.00 | 2775.70 | 2810.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 2817.00 | 2775.70 | 2810.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2839.83 | 2788.52 | 2813.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 2839.83 | 2788.52 | 2813.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2833.33 | 2797.48 | 2815.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:15:00 | 2836.67 | 2797.48 | 2815.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 2830.33 | 2823.67 | 2823.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2845.33 | 2828.00 | 2825.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 2802.67 | 2826.44 | 2825.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 2798.33 | 2820.82 | 2823.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 2786.67 | 2810.26 | 2817.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 2840.50 | 2817.84 | 2819.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 2912.00 | 2836.67 | 2827.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 2978.33 | 2871.11 | 2845.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 3070.17 | 3092.67 | 3019.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 3070.17 | 3092.67 | 3019.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 3076.17 | 3094.41 | 3071.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:45:00 | 3074.50 | 3094.41 | 3071.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 3070.00 | 3089.53 | 3071.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 3070.00 | 3089.53 | 3071.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 3073.33 | 3086.29 | 3071.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 3102.00 | 3086.29 | 3071.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 3052.50 | 3087.18 | 3088.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 3052.50 | 3087.18 | 3088.12 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 3139.67 | 3091.01 | 3089.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 3168.33 | 3106.48 | 3096.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 3130.83 | 3136.86 | 3116.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 3130.83 | 3136.86 | 3116.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3123.33 | 3134.12 | 3118.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 3085.33 | 3134.12 | 3118.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 3107.00 | 3128.70 | 3117.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 3107.00 | 3128.70 | 3117.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3119.00 | 3126.76 | 3117.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 3126.83 | 3126.76 | 3117.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 3129.00 | 3126.91 | 3118.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:45:00 | 3130.33 | 3126.42 | 3119.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:30:00 | 3127.33 | 3126.51 | 3119.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 3050.00 | 3100.95 | 3109.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 2913.33 | 2910.10 | 2942.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 10:30:00 | 2919.67 | 2910.10 | 2942.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2958.50 | 2921.87 | 2942.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2958.50 | 2921.87 | 2942.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2955.67 | 2928.63 | 2943.76 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 3130.00 | 2984.29 | 2966.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 3155.67 | 3042.78 | 2997.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 3090.83 | 3097.03 | 3046.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 10:00:00 | 3090.83 | 3097.03 | 3046.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 3040.00 | 3077.79 | 3049.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 3040.00 | 3077.79 | 3049.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 3044.33 | 3071.09 | 3049.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 3045.67 | 3071.09 | 3049.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3042.83 | 3065.44 | 3048.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 3042.83 | 3065.44 | 3048.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 3046.33 | 3061.62 | 3048.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 3014.33 | 3061.62 | 3048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3029.50 | 3055.20 | 3046.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 2998.50 | 3055.20 | 3046.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 2964.67 | 3037.09 | 3039.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2931.17 | 3015.91 | 3029.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2963.17 | 2955.62 | 2982.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 2967.17 | 2955.62 | 2982.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 2986.17 | 2963.23 | 2981.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 2986.17 | 2963.23 | 2981.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 3033.33 | 2977.25 | 2985.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 3273.33 | 2977.25 | 2985.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 3272.67 | 3036.34 | 3011.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 3466.67 | 3273.01 | 3163.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3395.00 | 3496.76 | 3360.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 3395.00 | 3496.76 | 3360.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3347.67 | 3466.94 | 3359.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 3408.83 | 3448.89 | 3360.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 3454.33 | 3385.08 | 3359.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 3315.00 | 3362.13 | 3356.15 | SL hit (close<static) qty=1.00 sl=3316.67 alert=retest2 |

### Cycle 79 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 3325.00 | 3347.69 | 3350.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 3289.83 | 3336.12 | 3344.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 3247.83 | 3238.58 | 3270.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:15:00 | 3278.50 | 3238.58 | 3270.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 3279.67 | 3246.80 | 3271.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 3308.50 | 3246.80 | 3271.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 3258.67 | 3249.17 | 3270.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 3250.00 | 3249.17 | 3270.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 3287.33 | 3256.80 | 3271.72 | SL hit (close>static) qty=1.00 sl=3284.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 3337.50 | 3287.21 | 3283.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 3360.17 | 3326.36 | 3305.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 3226.67 | 3336.57 | 3323.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 3235.83 | 3316.42 | 3315.29 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 3199.17 | 3292.97 | 3304.73 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 3533.17 | 3311.60 | 3290.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 3607.83 | 3370.85 | 3318.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 3639.00 | 3639.77 | 3538.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 3639.00 | 3639.77 | 3538.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 3566.33 | 3639.91 | 3593.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:30:00 | 3553.50 | 3639.91 | 3593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3523.33 | 3616.59 | 3586.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3523.33 | 3616.59 | 3586.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 3487.67 | 3553.90 | 3562.62 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 3603.83 | 3568.11 | 3567.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 3614.83 | 3577.46 | 3572.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 3541.67 | 3582.53 | 3576.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3556.67 | 3577.36 | 3575.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 3555.83 | 3577.36 | 3575.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 3548.17 | 3573.25 | 3573.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3506.67 | 3555.81 | 3565.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 3613.33 | 3559.32 | 3564.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 3680.67 | 3583.59 | 3575.41 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 3524.00 | 3574.84 | 3575.23 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 3665.83 | 3584.01 | 3578.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 3710.00 | 3609.21 | 3590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 3641.67 | 3641.92 | 3617.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 3558.00 | 3624.08 | 3613.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 3558.00 | 3624.08 | 3613.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3557.00 | 3610.66 | 3608.08 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 13:15:00 | 3541.67 | 3596.87 | 3602.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 14:15:00 | 3524.67 | 3582.43 | 3595.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 3422.67 | 3412.24 | 3478.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 3523.67 | 3418.43 | 3444.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3523.67 | 3418.43 | 3444.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 3530.67 | 3418.43 | 3444.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 3479.67 | 3430.68 | 3447.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:45:00 | 3468.33 | 3438.54 | 3449.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 3476.33 | 3443.03 | 3450.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 3496.67 | 3459.14 | 3456.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 3496.67 | 3459.14 | 3456.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 3632.33 | 3493.78 | 3472.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 3629.00 | 3629.09 | 3566.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 3634.00 | 3629.09 | 3566.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 3561.67 | 3606.73 | 3571.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 3561.67 | 3606.73 | 3571.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 3550.00 | 3595.38 | 3569.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:30:00 | 3548.33 | 3595.38 | 3569.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 3547.00 | 3577.91 | 3565.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 3625.00 | 3577.91 | 3565.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3603.67 | 3583.06 | 3568.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 3661.00 | 3614.80 | 3596.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 3560.00 | 3590.11 | 3590.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 3560.00 | 3590.11 | 3590.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 3515.00 | 3575.09 | 3583.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 15:15:00 | 3413.33 | 3412.09 | 3467.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 3432.33 | 3412.09 | 3467.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3408.33 | 3411.34 | 3462.38 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 3670.00 | 3503.03 | 3480.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 3900.00 | 3610.74 | 3535.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 3518.00 | 3566.04 | 3566.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 3509.00 | 3554.63 | 3561.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 3544.00 | 3491.24 | 3511.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 3498.00 | 3492.59 | 3510.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 3460.00 | 3490.60 | 3504.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3287.00 | 3349.23 | 3375.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 3366.00 | 3257.62 | 3299.62 | SL hit (close>ema200) qty=0.50 sl=3257.62 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 3520.00 | 3342.00 | 3332.66 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3385.00 | 3409.09 | 3412.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 3346.00 | 3386.15 | 3398.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 3344.00 | 3328.07 | 3346.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 3338.00 | 3330.06 | 3345.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 3305.00 | 3330.06 | 3345.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 3325.10 | 3304.05 | 3319.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 3396.80 | 3316.74 | 3317.33 | SL hit (close>static) qty=1.00 sl=3346.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 3422.30 | 3337.86 | 3326.87 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 3316.00 | 3334.47 | 3335.73 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 3399.40 | 3339.61 | 3336.70 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 3231.80 | 3367.33 | 3373.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 3143.50 | 3322.57 | 3352.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 3142.60 | 3137.70 | 3187.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 12:15:00 | 3162.00 | 3137.70 | 3187.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3172.80 | 3154.49 | 3177.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 3164.00 | 3159.51 | 3175.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3165.90 | 3146.51 | 3153.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 3157.60 | 3146.99 | 3149.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 3005.80 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 3007.61 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 2999.72 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3044.30 | 3023.99 | 3051.66 | SL hit (close>ema200) qty=0.50 sl=3023.99 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 3028.40 | 3019.30 | 3019.12 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 3011.10 | 3017.77 | 3018.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 2998.90 | 3012.16 | 3015.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 2936.80 | 2909.16 | 2927.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2938.60 | 2915.04 | 2928.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 2943.00 | 2915.04 | 2928.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 2934.40 | 2922.75 | 2929.79 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 2962.80 | 2937.60 | 2935.13 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2903.70 | 2935.57 | 2936.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 2889.00 | 2926.26 | 2932.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 2901.00 | 2880.13 | 2896.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2900.00 | 2884.10 | 2897.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2878.00 | 2884.10 | 2897.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 2909.00 | 2892.17 | 2890.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2909.00 | 2892.17 | 2890.35 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 2875.50 | 2899.58 | 2899.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 2873.10 | 2889.30 | 2894.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 2863.30 | 2861.35 | 2876.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 2872.80 | 2863.64 | 2876.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 2874.60 | 2863.64 | 2876.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 2792.50 | 2800.95 | 2828.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 2773.10 | 2793.02 | 2804.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 2634.44 | 2665.61 | 2710.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 2673.10 | 2667.10 | 2707.16 | SL hit (close>ema200) qty=0.50 sl=2667.10 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 2768.00 | 2727.86 | 2725.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 2878.50 | 2757.99 | 2739.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 2799.00 | 2814.31 | 2782.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 2750.40 | 2814.31 | 2782.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2726.60 | 2796.77 | 2777.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 2720.00 | 2796.77 | 2777.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2724.70 | 2782.35 | 2772.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 2756.60 | 2782.35 | 2772.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 2801.00 | 2835.65 | 2839.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 2801.00 | 2835.65 | 2839.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 2792.10 | 2822.00 | 2832.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2787.40 | 2777.63 | 2796.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 11:00:00 | 2787.40 | 2777.63 | 2796.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2793.10 | 2780.73 | 2795.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2793.10 | 2780.73 | 2795.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2780.00 | 2780.58 | 2794.40 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2863.30 | 2812.10 | 2806.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 2873.20 | 2847.99 | 2828.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 2852.00 | 2857.38 | 2842.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:45:00 | 2850.60 | 2857.38 | 2842.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2853.50 | 2856.60 | 2843.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 2848.20 | 2856.60 | 2843.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2849.40 | 2853.78 | 2844.56 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 2823.70 | 2837.29 | 2838.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 2809.90 | 2831.82 | 2836.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 2841.10 | 2826.56 | 2832.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2833.20 | 2827.89 | 2832.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 2836.40 | 2827.89 | 2832.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 2834.80 | 2829.27 | 2832.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 2834.80 | 2829.27 | 2832.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2837.80 | 2830.98 | 2832.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 2837.80 | 2830.98 | 2832.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2821.30 | 2829.04 | 2831.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 2846.60 | 2829.04 | 2831.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2824.90 | 2827.63 | 2830.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 2828.80 | 2827.63 | 2830.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2734.20 | 2789.77 | 2808.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 2713.60 | 2772.22 | 2798.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 2713.10 | 2710.55 | 2750.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 2572.00 | 2744.13 | 2752.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 2577.92 | 2694.36 | 2728.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 2577.44 | 2694.36 | 2728.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-01 10:15:00 | 2442.24 | 2645.05 | 2703.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 110 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2221.20 | 2152.54 | 2143.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 2252.80 | 2222.66 | 2199.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 2237.00 | 2240.07 | 2220.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 2215.30 | 2240.07 | 2220.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2224.60 | 2236.97 | 2220.75 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 2192.20 | 2210.90 | 2212.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 2184.90 | 2202.79 | 2208.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2106.20 | 2093.01 | 2126.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2056.60 | 2090.68 | 2116.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2016.50 | 2017.82 | 2036.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 2005.90 | 2016.12 | 2027.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:00:00 | 2006.30 | 2014.16 | 2025.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 2024.80 | 2015.43 | 2021.23 | SL hit (close>ema400) qty=1.00 sl=2021.23 alert=retest1 |

### Cycle 112 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2069.00 | 2026.14 | 2025.58 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 2006.10 | 2029.55 | 2031.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2000.70 | 2023.78 | 2028.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 1940.00 | 1933.21 | 1965.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 1956.40 | 1933.21 | 1965.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1934.60 | 1933.49 | 1962.84 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 2015.10 | 1973.70 | 1971.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 2029.40 | 2000.61 | 1986.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1994.70 | 2011.28 | 1995.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1978.20 | 2004.67 | 1994.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1978.20 | 2004.67 | 1994.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1978.40 | 1999.41 | 1992.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:45:00 | 1976.20 | 1999.41 | 1992.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 1983.40 | 1989.32 | 1989.45 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 2103.90 | 2012.23 | 1999.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 2185.00 | 2046.79 | 2016.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2140.00 | 2178.10 | 2157.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2133.00 | 2169.08 | 2155.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 2133.00 | 2169.08 | 2155.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 2134.00 | 2145.78 | 2147.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2104.20 | 2137.47 | 2143.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 2126.00 | 2122.52 | 2132.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:30:00 | 2126.90 | 2122.52 | 2132.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2105.00 | 2049.92 | 2071.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 2105.00 | 2049.92 | 2071.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 2085.00 | 2056.94 | 2072.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 2059.70 | 2064.87 | 2072.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 2302.90 | 2111.85 | 2092.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 2302.90 | 2111.85 | 2092.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 2329.90 | 2185.93 | 2131.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 2415.00 | 2472.95 | 2377.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:30:00 | 2413.90 | 2472.95 | 2377.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2314.80 | 2428.31 | 2399.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 2284.00 | 2428.31 | 2399.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2295.10 | 2401.67 | 2389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 2300.00 | 2401.67 | 2389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 2275.70 | 2376.48 | 2379.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 2224.80 | 2329.89 | 2356.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 2135.70 | 2096.79 | 2123.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 2119.90 | 2101.42 | 2122.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 2125.70 | 2101.42 | 2122.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 2118.20 | 2104.77 | 2122.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 2112.00 | 2106.24 | 2121.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 2112.10 | 2106.24 | 2121.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2056.00 | 2108.79 | 2121.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2006.40 | 2101.47 | 2116.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2006.49 | 2101.47 | 2116.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 2017.50 | 2016.55 | 2044.20 | SL hit (close>ema200) qty=0.50 sl=2016.55 alert=retest2 |

### Cycle 120 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 2070.50 | 2046.89 | 2044.28 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1987.00 | 2041.63 | 2044.32 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 2054.00 | 2033.17 | 2030.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2063.90 | 2042.52 | 2035.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 2033.10 | 2074.08 | 2075.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2025.50 | 2058.35 | 2067.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 2028.60 | 2008.77 | 2029.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2031.30 | 2015.07 | 2028.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:00:00 | 2015.50 | 2015.16 | 2027.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 2010.50 | 2015.05 | 2023.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2039.00 | 2028.37 | 2027.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 2039.00 | 2028.37 | 2027.05 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2012.30 | 2024.97 | 2026.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1992.60 | 2014.58 | 2020.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 2002.00 | 2011.09 | 2016.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1998.60 | 2007.70 | 2013.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 1901.90 | 1951.53 | 1982.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 1898.67 | 1951.53 | 1982.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1895.70 | 1882.13 | 1918.92 | SL hit (close>ema200) qty=0.50 sl=1882.13 alert=retest2 |

### Cycle 126 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2063.50 | 1942.68 | 1934.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2126.90 | 1979.53 | 1952.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 2035.00 | 2036.53 | 1993.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 2006.30 | 2036.53 | 1993.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1995.60 | 2028.34 | 1993.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1995.60 | 2028.34 | 1993.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1961.50 | 2014.98 | 1990.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1961.50 | 2014.98 | 1990.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1935.30 | 1999.04 | 1985.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1935.30 | 1999.04 | 1985.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1904.00 | 1963.73 | 1971.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1890.10 | 1932.70 | 1953.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1965.50 | 1913.22 | 1931.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1948.80 | 1920.34 | 1932.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1941.20 | 1935.11 | 1937.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 1929.80 | 1920.04 | 1919.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1929.80 | 1920.04 | 1919.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 1935.30 | 1924.29 | 1921.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1927.00 | 1927.99 | 1924.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1986.70 | 1927.99 | 1924.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:15:00 | 2086.04 | 1998.88 | 1971.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2048.50 | 2072.20 | 2044.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2048.50 | 2072.20 | 2044.70 | SL hit (close<ema200) qty=0.50 sl=2072.20 alert=retest1 |

### Cycle 129 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 2133.30 | 2149.00 | 2149.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 2110.60 | 2141.32 | 2146.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 2163.20 | 2135.26 | 2140.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 2165.90 | 2141.39 | 2142.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 2177.20 | 2141.39 | 2142.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 2182.70 | 2149.65 | 2146.39 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 2112.90 | 2146.56 | 2150.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 2105.60 | 2133.00 | 2143.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2135.70 | 2114.22 | 2125.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2127.00 | 2116.78 | 2126.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 2130.00 | 2116.78 | 2126.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 2112.80 | 2115.98 | 2124.83 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2174.90 | 2133.45 | 2128.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 2219.10 | 2150.58 | 2136.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 2231.90 | 2245.64 | 2221.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 12:00:00 | 2231.90 | 2245.64 | 2221.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 2223.80 | 2241.27 | 2221.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 2224.00 | 2241.27 | 2221.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 2216.70 | 2236.35 | 2221.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 2218.40 | 2236.35 | 2221.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 2218.00 | 2232.68 | 2220.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:30:00 | 2212.70 | 2232.68 | 2220.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2204.60 | 2224.72 | 2219.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 2227.00 | 2218.49 | 2217.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 11:15:00 | 2449.70 | 2378.25 | 2331.68 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-30 09:30:00 | 1278.77 | 2024-05-31 15:15:00 | 1316.67 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-05-30 11:30:00 | 1278.08 | 2024-05-31 15:15:00 | 1316.67 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-05-30 13:00:00 | 1282.25 | 2024-05-31 15:15:00 | 1316.67 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-05-30 15:00:00 | 1282.67 | 2024-05-31 15:15:00 | 1316.67 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-06-03 09:15:00 | 1244.37 | 2024-06-04 10:15:00 | 1187.81 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2024-06-03 11:45:00 | 1250.33 | 2024-06-04 10:15:00 | 1187.50 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-06-03 14:00:00 | 1250.00 | 2024-06-04 11:15:00 | 1182.15 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2024-06-03 09:15:00 | 1244.37 | 2024-06-04 12:15:00 | 1119.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 11:45:00 | 1250.33 | 2024-06-04 12:15:00 | 1125.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:00:00 | 1250.00 | 2024-06-04 12:15:00 | 1125.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 09:45:00 | 1246.20 | 2024-06-06 10:15:00 | 1273.73 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest1 | 2024-06-11 09:15:00 | 1310.73 | 2024-06-11 15:15:00 | 1286.77 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2024-06-11 14:30:00 | 1307.98 | 2024-06-11 15:15:00 | 1286.77 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1308.45 | 2024-06-13 15:15:00 | 1285.02 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1305.02 | 2024-06-13 15:15:00 | 1285.02 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-06-13 14:30:00 | 1299.00 | 2024-06-13 15:15:00 | 1285.02 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-06-20 11:45:00 | 1428.00 | 2024-06-27 13:15:00 | 1427.30 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-07-01 11:15:00 | 1405.18 | 2024-07-02 12:15:00 | 1423.33 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-07-02 15:00:00 | 1406.52 | 2024-07-03 10:15:00 | 1431.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-07-08 09:15:00 | 1455.58 | 2024-07-08 09:15:00 | 1431.27 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-07-24 13:30:00 | 1391.38 | 2024-07-25 09:15:00 | 1444.32 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-07-24 15:15:00 | 1387.25 | 2024-07-25 09:15:00 | 1444.32 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2024-08-02 11:45:00 | 1405.55 | 2024-08-05 09:15:00 | 1339.74 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-08-02 14:15:00 | 1410.25 | 2024-08-05 10:15:00 | 1335.27 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-08-02 11:45:00 | 1405.55 | 2024-08-06 14:15:00 | 1356.98 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2024-08-02 14:15:00 | 1410.25 | 2024-08-06 14:15:00 | 1356.98 | STOP_HIT | 0.50 | 3.78% |
| BUY | retest2 | 2024-08-14 10:30:00 | 1467.23 | 2024-08-19 10:15:00 | 1613.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 09:15:00 | 1491.55 | 2024-08-19 10:15:00 | 1640.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-10 11:30:00 | 2262.33 | 2024-09-11 09:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-10 12:15:00 | 2261.65 | 2024-09-11 09:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-09-10 13:30:00 | 2264.33 | 2024-09-11 09:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-11 09:15:00 | 2269.70 | 2024-09-11 09:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-09-12 13:45:00 | 2207.57 | 2024-09-13 10:15:00 | 2350.73 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2024-09-12 15:15:00 | 2206.67 | 2024-09-13 10:15:00 | 2350.73 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2024-09-13 09:30:00 | 2207.97 | 2024-09-13 10:15:00 | 2350.73 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest2 | 2024-09-19 09:15:00 | 2682.25 | 2024-09-23 09:15:00 | 2505.25 | STOP_HIT | 1.00 | -6.60% |
| BUY | retest2 | 2024-09-20 09:30:00 | 2608.90 | 2024-09-23 09:15:00 | 2505.25 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2024-09-20 10:30:00 | 2636.90 | 2024-09-23 09:15:00 | 2505.25 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2024-09-20 14:15:00 | 2606.70 | 2024-09-23 09:15:00 | 2505.25 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-09-26 11:00:00 | 2352.13 | 2024-10-03 09:15:00 | 2234.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 12:00:00 | 2368.32 | 2024-10-03 09:15:00 | 2249.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 13:45:00 | 2367.65 | 2024-10-03 09:15:00 | 2249.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 09:30:00 | 2351.00 | 2024-10-03 09:15:00 | 2233.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 11:00:00 | 2352.13 | 2024-10-04 09:15:00 | 2234.08 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2024-09-26 12:00:00 | 2368.32 | 2024-10-04 09:15:00 | 2234.08 | STOP_HIT | 0.50 | 5.67% |
| SELL | retest2 | 2024-09-26 13:45:00 | 2367.65 | 2024-10-04 09:15:00 | 2234.08 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2024-09-27 09:30:00 | 2351.00 | 2024-10-04 09:15:00 | 2234.08 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2024-10-04 14:30:00 | 2227.45 | 2024-10-08 09:15:00 | 2116.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:30:00 | 2227.45 | 2024-10-08 10:15:00 | 2195.02 | STOP_HIT | 0.50 | 1.46% |
| BUY | retest2 | 2024-10-11 10:00:00 | 2312.65 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-10-14 10:00:00 | 2316.67 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-10-15 09:30:00 | 2318.97 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-10-16 12:00:00 | 2332.42 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-10-16 15:00:00 | 2380.00 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-10-17 13:15:00 | 2349.17 | 2024-10-17 15:15:00 | 2313.27 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest1 | 2024-10-23 12:00:00 | 2135.00 | 2024-10-23 13:15:00 | 2184.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-11-11 15:15:00 | 2175.52 | 2024-11-13 12:15:00 | 2096.95 | PARTIAL | 0.50 | 3.61% |
| SELL | retest2 | 2024-11-12 09:30:00 | 2207.32 | 2024-11-14 09:15:00 | 2066.74 | PARTIAL | 0.50 | 6.37% |
| SELL | retest2 | 2024-11-12 13:30:00 | 2198.52 | 2024-11-14 09:15:00 | 2088.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 2175.52 | 2024-11-18 09:15:00 | 1986.59 | TARGET_HIT | 0.50 | 8.68% |
| SELL | retest2 | 2024-11-12 09:30:00 | 2207.32 | 2024-11-18 09:15:00 | 1978.67 | TARGET_HIT | 0.50 | 10.36% |
| SELL | retest2 | 2024-11-12 13:30:00 | 2198.52 | 2024-11-19 09:15:00 | 2035.62 | STOP_HIT | 0.50 | 7.41% |
| SELL | retest2 | 2024-12-02 15:15:00 | 1916.67 | 2024-12-09 14:15:00 | 1893.33 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-12-13 12:45:00 | 2003.33 | 2024-12-17 12:15:00 | 1968.33 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1667.33 | 2025-01-02 12:15:00 | 1727.33 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-01-08 13:15:00 | 1615.07 | 2025-01-13 09:15:00 | 1534.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1595.05 | 2025-01-13 10:15:00 | 1515.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:15:00 | 1615.07 | 2025-01-13 15:15:00 | 1453.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1595.05 | 2025-01-14 10:15:00 | 1520.22 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2025-01-31 12:15:00 | 1507.33 | 2025-02-01 13:15:00 | 1658.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 12:00:00 | 1532.92 | 2025-02-01 13:15:00 | 1686.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-05 10:15:00 | 1690.93 | 2025-03-06 09:15:00 | 1755.92 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-03-11 13:30:00 | 1745.55 | 2025-03-12 09:15:00 | 1787.20 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-11 15:00:00 | 1743.33 | 2025-03-12 09:15:00 | 1787.20 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-03-26 09:15:00 | 2096.40 | 2025-03-26 10:15:00 | 2059.75 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-04-02 11:15:00 | 2325.65 | 2025-04-04 11:15:00 | 2243.92 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-04-02 11:45:00 | 2335.67 | 2025-04-04 11:15:00 | 2243.92 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-04-02 13:00:00 | 2349.58 | 2025-04-04 11:15:00 | 2243.92 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2025-04-23 09:15:00 | 2546.33 | 2025-05-02 13:15:00 | 2800.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 2768.67 | 2025-05-09 09:15:00 | 2630.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 2768.67 | 2025-05-09 15:15:00 | 2740.00 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2025-05-21 11:15:00 | 2858.00 | 2025-05-28 09:15:00 | 2715.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 11:15:00 | 2858.00 | 2025-05-28 12:15:00 | 2736.17 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2025-06-03 09:45:00 | 2881.50 | 2025-06-03 11:15:00 | 2796.17 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-06-03 10:45:00 | 2903.67 | 2025-06-03 11:15:00 | 2796.17 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-06-10 11:30:00 | 2751.67 | 2025-06-12 09:15:00 | 2789.83 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-11 12:15:00 | 2752.17 | 2025-06-12 09:15:00 | 2789.83 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-06-12 11:30:00 | 2742.50 | 2025-06-13 14:15:00 | 2778.17 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2750.00 | 2025-06-13 14:15:00 | 2778.17 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-19 11:45:00 | 2702.83 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-19 14:30:00 | 2704.83 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-06-20 15:00:00 | 2666.50 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-24 12:15:00 | 2703.50 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-07-01 09:15:00 | 3029.83 | 2025-07-02 09:15:00 | 2920.50 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-07-01 11:00:00 | 3004.67 | 2025-07-02 09:15:00 | 2920.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-07-18 09:15:00 | 3102.00 | 2025-07-21 13:15:00 | 3052.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-23 12:15:00 | 3126.83 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-23 12:45:00 | 3129.00 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-23 13:45:00 | 3130.33 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-23 14:30:00 | 3127.33 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-08-07 11:30:00 | 3408.83 | 2025-08-08 14:15:00 | 3315.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-08 10:45:00 | 3454.33 | 2025-08-08 14:15:00 | 3315.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-08-13 12:15:00 | 3250.00 | 2025-08-13 12:15:00 | 3287.33 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-04 11:45:00 | 3468.33 | 2025-09-04 15:15:00 | 3496.67 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-04 12:45:00 | 3476.33 | 2025-09-04 15:15:00 | 3496.67 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-10 10:45:00 | 3661.00 | 2025-09-11 09:15:00 | 3560.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-09-22 13:45:00 | 3460.00 | 2025-09-26 09:15:00 | 3287.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 3460.00 | 2025-09-29 12:15:00 | 3366.00 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-10-13 09:15:00 | 3305.00 | 2025-10-15 09:15:00 | 3396.80 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-10-14 10:00:00 | 3325.10 | 2025-10-15 09:15:00 | 3396.80 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-28 11:45:00 | 3164.00 | 2025-11-04 09:15:00 | 3005.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3165.90 | 2025-11-04 09:15:00 | 3007.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3157.60 | 2025-11-04 09:15:00 | 2999.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:45:00 | 3164.00 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3165.90 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3157.60 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2878.00 | 2025-11-27 09:15:00 | 2909.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2773.10 | 2025-12-09 09:15:00 | 2634.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2773.10 | 2025-12-09 10:15:00 | 2673.10 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2768.00 | 2025-12-09 15:15:00 | 2768.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-11 11:15:00 | 2756.60 | 2025-12-17 13:15:00 | 2801.00 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2025-12-30 10:45:00 | 2713.60 | 2026-01-01 09:15:00 | 2577.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 09:45:00 | 2713.10 | 2026-01-01 09:15:00 | 2577.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 10:45:00 | 2713.60 | 2026-01-01 10:15:00 | 2442.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 09:45:00 | 2713.10 | 2026-01-01 10:15:00 | 2441.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 2572.00 | 2026-01-01 10:15:00 | 2443.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 2572.00 | 2026-01-01 11:15:00 | 2314.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-01-22 13:15:00 | 2056.60 | 2026-01-29 15:15:00 | 2024.80 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2005.90 | 2026-01-30 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-01-29 11:00:00 | 2006.30 | 2026-01-30 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-02-17 15:00:00 | 2059.70 | 2026-02-18 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -11.81% |
| SELL | retest2 | 2026-02-27 14:30:00 | 2112.00 | 2026-03-02 09:15:00 | 2006.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2112.10 | 2026-03-02 09:15:00 | 2006.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 14:30:00 | 2112.00 | 2026-03-05 09:15:00 | 2017.50 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2112.10 | 2026-03-05 09:15:00 | 2017.50 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2056.00 | 2026-03-06 11:15:00 | 2070.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-17 11:00:00 | 2015.50 | 2026-03-18 13:15:00 | 2039.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-03-17 15:00:00 | 2010.50 | 2026-03-18 13:15:00 | 2039.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2002.00 | 2026-03-23 11:15:00 | 1901.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1998.60 | 2026-03-23 11:15:00 | 1898.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2002.00 | 2026-03-24 12:15:00 | 1895.70 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1998.60 | 2026-03-24 12:15:00 | 1895.70 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1941.20 | 2026-04-06 15:15:00 | 1929.80 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1986.70 | 2026-04-09 09:15:00 | 2086.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1986.70 | 2026-04-13 09:15:00 | 2048.50 | STOP_HIT | 0.50 | 3.11% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2063.90 | 2026-04-21 11:15:00 | 2133.30 | STOP_HIT | 1.00 | 3.36% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2102.30 | 2026-04-21 11:15:00 | 2133.30 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2026-05-05 14:15:00 | 2227.00 | 2026-05-08 11:15:00 | 2449.70 | TARGET_HIT | 1.00 | 10.00% |
