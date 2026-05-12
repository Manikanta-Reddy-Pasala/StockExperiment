# Bharti Airtel Ltd. (BHARTIARTL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1834.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 102 |
| ALERT2 | 100 |
| ALERT2_SKIP | 44 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 106 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 98
- **Target hits / Stop hits / Partials:** 0 / 110 / 5
- **Avg / median % per leg:** -0.36% / -0.76%
- **Sum % (uncompounded):** -41.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 6 | 12.0% | 0 | 50 | 0 | -0.64% | -31.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| BUY @ 3rd Alert (retest2) | 47 | 6 | 12.8% | 0 | 47 | 0 | -0.63% | -29.8% |
| SELL (all) | 65 | 11 | 16.9% | 0 | 60 | 5 | -0.15% | -9.7% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.34% | 1.3% |
| SELL @ 3rd Alert (retest2) | 64 | 10 | 15.6% | 0 | 59 | 5 | -0.17% | -11.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.18% | -0.7% |
| retest2 (combined) | 111 | 16 | 14.4% | 0 | 106 | 5 | -0.37% | -40.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 15:15:00 | 1288.00 | 1289.75 | 1289.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 14:15:00 | 1283.85 | 1287.15 | 1288.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 10:15:00 | 1293.10 | 1288.27 | 1288.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 10:15:00 | 1293.10 | 1288.27 | 1288.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1293.10 | 1288.27 | 1288.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:00:00 | 1293.10 | 1288.27 | 1288.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1290.75 | 1288.77 | 1288.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 1297.25 | 1291.26 | 1289.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 1349.20 | 1350.88 | 1342.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 1349.20 | 1350.88 | 1342.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1342.25 | 1349.15 | 1342.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 1342.25 | 1349.15 | 1342.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1338.70 | 1347.06 | 1342.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1335.50 | 1347.06 | 1342.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1337.10 | 1345.07 | 1341.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:15:00 | 1334.90 | 1345.07 | 1341.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1336.10 | 1343.28 | 1341.18 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 1333.10 | 1339.68 | 1339.81 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 1342.40 | 1340.22 | 1340.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 1347.70 | 1341.72 | 1340.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1387.25 | 1390.14 | 1380.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 1387.25 | 1390.14 | 1380.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1380.40 | 1388.19 | 1380.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 1375.95 | 1386.30 | 1380.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1378.30 | 1384.70 | 1379.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:45:00 | 1378.10 | 1384.70 | 1379.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1377.70 | 1383.30 | 1379.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:30:00 | 1379.85 | 1383.30 | 1379.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1376.95 | 1382.03 | 1379.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 1377.60 | 1382.03 | 1379.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1374.35 | 1380.49 | 1379.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 1374.35 | 1380.49 | 1379.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 1369.50 | 1376.78 | 1377.51 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 12:15:00 | 1377.50 | 1376.28 | 1376.14 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 1370.25 | 1376.12 | 1376.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 10:15:00 | 1356.30 | 1372.16 | 1374.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 1364.65 | 1361.30 | 1367.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 1364.65 | 1361.30 | 1367.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1364.65 | 1361.30 | 1367.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1364.65 | 1361.30 | 1367.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1370.05 | 1363.05 | 1367.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1418.10 | 1363.05 | 1367.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1406.20 | 1371.68 | 1371.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1298.00 | 1366.79 | 1372.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1261.25 | 1345.69 | 1362.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 1330.65 | 1315.61 | 1334.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 11:15:00 | 1330.65 | 1315.61 | 1334.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1330.65 | 1315.61 | 1334.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 1330.65 | 1315.61 | 1334.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1332.00 | 1318.89 | 1333.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 1339.25 | 1318.89 | 1333.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1339.60 | 1323.03 | 1334.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1342.55 | 1323.03 | 1334.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1346.10 | 1327.64 | 1335.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1346.10 | 1327.64 | 1335.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1362.70 | 1339.96 | 1339.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 1371.30 | 1349.98 | 1344.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1417.60 | 1420.26 | 1402.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 1414.40 | 1420.26 | 1402.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1428.35 | 1436.54 | 1428.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 1428.35 | 1436.54 | 1428.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1435.75 | 1436.38 | 1429.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1438.30 | 1429.24 | 1428.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 1425.35 | 1429.56 | 1429.21 | SL hit (close<static) qty=1.00 sl=1428.30 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 1419.95 | 1427.64 | 1428.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 1414.70 | 1425.26 | 1427.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 1410.25 | 1391.92 | 1400.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 1410.25 | 1391.92 | 1400.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1410.25 | 1391.92 | 1400.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 1410.25 | 1391.92 | 1400.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 1406.05 | 1394.75 | 1401.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 1412.95 | 1394.75 | 1401.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 1417.50 | 1405.26 | 1404.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 1424.15 | 1416.67 | 1414.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 10:15:00 | 1460.60 | 1463.16 | 1450.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:30:00 | 1453.00 | 1463.16 | 1450.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1455.90 | 1461.71 | 1450.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 1460.10 | 1461.71 | 1450.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1449.95 | 1459.36 | 1450.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 1449.95 | 1459.36 | 1450.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1457.60 | 1459.01 | 1451.14 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 1439.00 | 1450.38 | 1450.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 1429.40 | 1446.18 | 1448.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 12:15:00 | 1424.90 | 1421.74 | 1428.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 12:15:00 | 1424.90 | 1421.74 | 1428.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1424.90 | 1421.74 | 1428.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:30:00 | 1430.80 | 1421.74 | 1428.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1432.80 | 1423.95 | 1428.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 1433.30 | 1423.95 | 1428.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1422.80 | 1423.72 | 1428.00 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 10:15:00 | 1434.00 | 1428.87 | 1428.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 09:15:00 | 1441.85 | 1435.49 | 1433.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 1434.00 | 1435.19 | 1433.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 1434.00 | 1435.19 | 1433.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1434.00 | 1435.19 | 1433.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:45:00 | 1433.50 | 1435.19 | 1433.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1439.35 | 1436.02 | 1433.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:00:00 | 1445.60 | 1438.88 | 1435.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 1428.00 | 1436.74 | 1435.68 | SL hit (close<static) qty=1.00 sl=1433.20 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 1423.60 | 1434.29 | 1434.97 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 1440.60 | 1436.42 | 1435.88 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 1432.70 | 1435.26 | 1435.43 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 1438.75 | 1435.50 | 1435.42 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 13:15:00 | 1433.75 | 1435.33 | 1435.37 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 1436.90 | 1435.64 | 1435.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1465.95 | 1441.77 | 1438.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1467.60 | 1471.83 | 1462.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 1467.60 | 1471.83 | 1462.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1463.20 | 1470.11 | 1462.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 1466.60 | 1470.11 | 1462.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1458.35 | 1467.76 | 1462.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 1456.75 | 1467.76 | 1462.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1459.70 | 1466.14 | 1462.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 1453.35 | 1466.14 | 1462.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1462.55 | 1464.85 | 1462.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:30:00 | 1456.75 | 1464.85 | 1462.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 1463.05 | 1464.49 | 1462.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 1461.55 | 1464.49 | 1462.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1466.55 | 1464.90 | 1462.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 14:15:00 | 1469.15 | 1465.46 | 1463.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 1455.55 | 1462.22 | 1462.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 1455.55 | 1462.22 | 1462.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 1443.15 | 1458.40 | 1460.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 1458.55 | 1458.43 | 1460.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 1458.55 | 1458.43 | 1460.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1458.55 | 1458.43 | 1460.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1458.55 | 1458.43 | 1460.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1465.00 | 1459.75 | 1461.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 1468.10 | 1459.75 | 1461.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1465.40 | 1460.88 | 1461.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1465.55 | 1460.88 | 1461.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1456.20 | 1459.94 | 1460.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 1450.60 | 1457.91 | 1459.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 1443.10 | 1453.65 | 1456.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:45:00 | 1450.00 | 1452.22 | 1455.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 13:00:00 | 1450.45 | 1451.78 | 1454.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1449.75 | 1450.71 | 1453.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1478.55 | 1450.71 | 1453.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 1487.00 | 1457.97 | 1456.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1487.00 | 1457.97 | 1456.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1491.30 | 1464.64 | 1459.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 10:15:00 | 1492.30 | 1492.66 | 1479.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 12:15:00 | 1483.75 | 1490.10 | 1480.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1483.75 | 1490.10 | 1480.62 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1465.55 | 1477.22 | 1478.17 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 1491.65 | 1480.66 | 1479.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 13:15:00 | 1495.80 | 1485.74 | 1481.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1486.90 | 1498.17 | 1493.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1486.90 | 1498.17 | 1493.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1486.90 | 1498.17 | 1493.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 1484.65 | 1498.17 | 1493.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 1493.25 | 1497.19 | 1493.05 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1473.00 | 1490.56 | 1491.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 1460.95 | 1484.64 | 1488.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 14:15:00 | 1445.00 | 1442.49 | 1452.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 15:00:00 | 1445.00 | 1442.49 | 1452.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 1463.00 | 1446.59 | 1451.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:45:00 | 1461.90 | 1446.59 | 1451.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1452.40 | 1447.75 | 1451.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:15:00 | 1449.05 | 1449.08 | 1451.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1477.60 | 1454.78 | 1453.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1477.60 | 1454.78 | 1453.59 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 1452.00 | 1458.95 | 1459.11 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 15:15:00 | 1463.40 | 1459.88 | 1459.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 1469.60 | 1461.82 | 1460.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 1462.00 | 1462.95 | 1461.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 1462.00 | 1462.95 | 1461.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1462.00 | 1462.95 | 1461.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1462.00 | 1462.95 | 1461.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1463.55 | 1463.07 | 1461.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 1463.55 | 1463.07 | 1461.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1461.75 | 1462.81 | 1461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1461.75 | 1462.81 | 1461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1458.20 | 1461.89 | 1461.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 1459.55 | 1461.89 | 1461.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1458.75 | 1461.26 | 1460.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1466.50 | 1461.26 | 1460.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1466.50 | 1462.92 | 1461.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:45:00 | 1459.55 | 1462.92 | 1461.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1473.05 | 1479.88 | 1474.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 1473.05 | 1479.88 | 1474.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1470.55 | 1478.01 | 1474.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 1469.90 | 1478.01 | 1474.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 1467.60 | 1471.80 | 1472.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1455.00 | 1468.15 | 1470.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 11:15:00 | 1459.75 | 1455.19 | 1460.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 11:15:00 | 1459.75 | 1455.19 | 1460.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1459.75 | 1455.19 | 1460.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 1459.75 | 1455.19 | 1460.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1455.70 | 1455.29 | 1459.85 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1482.75 | 1464.18 | 1462.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 1494.00 | 1470.15 | 1465.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1481.50 | 1481.72 | 1474.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 1481.50 | 1481.72 | 1474.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1574.05 | 1581.01 | 1568.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 1578.95 | 1581.01 | 1568.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1572.95 | 1579.04 | 1570.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1572.20 | 1579.04 | 1570.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1569.25 | 1577.08 | 1570.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:45:00 | 1567.95 | 1577.08 | 1570.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1570.95 | 1575.85 | 1570.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 1566.85 | 1575.85 | 1570.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1566.45 | 1574.12 | 1570.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 1566.45 | 1574.12 | 1570.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1561.00 | 1571.49 | 1569.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 1561.00 | 1571.49 | 1569.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 1559.65 | 1566.88 | 1567.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1551.00 | 1562.45 | 1565.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 1563.30 | 1557.98 | 1561.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 14:15:00 | 1563.30 | 1557.98 | 1561.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 1563.30 | 1557.98 | 1561.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 1563.30 | 1557.98 | 1561.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 1561.30 | 1558.65 | 1561.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1553.70 | 1558.65 | 1561.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1566.10 | 1545.64 | 1545.84 | SL hit (close>static) qty=1.00 sl=1566.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1554.95 | 1547.50 | 1546.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 1567.90 | 1551.58 | 1548.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1576.20 | 1582.03 | 1571.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 1576.20 | 1582.03 | 1571.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1652.00 | 1658.30 | 1649.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 1652.55 | 1658.30 | 1649.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1653.10 | 1657.26 | 1649.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1672.75 | 1655.83 | 1649.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 12:15:00 | 1730.00 | 1745.54 | 1747.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 1730.00 | 1745.54 | 1747.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1711.10 | 1734.49 | 1741.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1717.20 | 1716.85 | 1726.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:30:00 | 1718.60 | 1716.85 | 1726.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1665.30 | 1660.79 | 1668.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:30:00 | 1658.95 | 1662.12 | 1666.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1683.65 | 1666.09 | 1667.37 | SL hit (close>static) qty=1.00 sl=1679.50 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1700.80 | 1673.03 | 1670.40 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 1673.75 | 1674.95 | 1675.02 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 1676.00 | 1675.16 | 1675.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 1688.55 | 1679.32 | 1677.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1718.05 | 1725.24 | 1715.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1718.05 | 1725.24 | 1715.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1718.05 | 1725.24 | 1715.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1718.05 | 1725.24 | 1715.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1716.70 | 1723.53 | 1715.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 1718.35 | 1723.53 | 1715.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1713.15 | 1721.46 | 1715.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 1713.15 | 1721.46 | 1715.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1714.25 | 1720.02 | 1715.45 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1701.95 | 1712.06 | 1712.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1689.30 | 1707.50 | 1710.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 1708.60 | 1705.17 | 1708.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 1708.60 | 1705.17 | 1708.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1708.60 | 1705.17 | 1708.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 1712.20 | 1705.17 | 1708.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1710.05 | 1706.15 | 1708.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:30:00 | 1706.75 | 1706.15 | 1708.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1707.10 | 1706.34 | 1708.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 1702.00 | 1706.34 | 1708.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 1703.50 | 1694.71 | 1698.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:00:00 | 1697.90 | 1695.85 | 1698.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 1616.90 | 1657.72 | 1667.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 1618.32 | 1657.72 | 1667.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 1613.01 | 1657.72 | 1667.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 1651.00 | 1641.75 | 1651.61 | SL hit (close>ema200) qty=0.50 sl=1641.75 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 1556.70 | 1537.26 | 1536.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 1559.75 | 1541.76 | 1538.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 1567.85 | 1567.87 | 1555.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 1567.85 | 1567.87 | 1555.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1562.60 | 1572.46 | 1567.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:30:00 | 1578.00 | 1571.07 | 1567.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 1577.35 | 1571.07 | 1567.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:30:00 | 1578.30 | 1571.85 | 1568.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 1578.70 | 1571.85 | 1568.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1581.05 | 1575.21 | 1570.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1577.70 | 1575.21 | 1570.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1565.15 | 1573.20 | 1570.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1568.90 | 1573.20 | 1570.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1564.15 | 1571.39 | 1569.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1564.15 | 1571.39 | 1569.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 1561.00 | 1567.95 | 1568.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 1561.00 | 1567.95 | 1568.51 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 1579.15 | 1569.71 | 1569.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 1613.65 | 1578.50 | 1573.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 1625.25 | 1632.49 | 1617.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 09:45:00 | 1623.80 | 1632.49 | 1617.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1621.70 | 1627.82 | 1619.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:45:00 | 1622.00 | 1627.82 | 1619.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1621.60 | 1626.58 | 1619.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:30:00 | 1614.95 | 1626.58 | 1619.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1618.90 | 1625.04 | 1619.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 1607.15 | 1625.04 | 1619.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1597.80 | 1619.59 | 1617.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 1597.80 | 1619.59 | 1617.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 10:15:00 | 1589.05 | 1613.48 | 1615.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1583.15 | 1597.61 | 1606.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 09:15:00 | 1598.65 | 1595.59 | 1603.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 1598.65 | 1595.59 | 1603.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1598.65 | 1595.59 | 1603.66 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 1613.25 | 1605.83 | 1605.51 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 1598.80 | 1605.42 | 1605.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 1597.40 | 1602.74 | 1604.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 1602.85 | 1600.71 | 1602.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 11:15:00 | 1602.85 | 1600.71 | 1602.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1602.85 | 1600.71 | 1602.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:30:00 | 1604.30 | 1600.71 | 1602.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1599.80 | 1600.53 | 1602.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1591.35 | 1602.02 | 1602.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:00:00 | 1593.10 | 1584.96 | 1587.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:45:00 | 1596.10 | 1587.85 | 1588.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 1603.40 | 1590.96 | 1589.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 1603.40 | 1590.96 | 1589.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 09:15:00 | 1634.85 | 1607.93 | 1599.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 1640.10 | 1657.38 | 1645.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 1640.10 | 1657.38 | 1645.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1640.10 | 1657.38 | 1645.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 1640.10 | 1657.38 | 1645.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1630.40 | 1651.98 | 1644.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 1636.30 | 1651.98 | 1644.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1612.30 | 1637.61 | 1638.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 1603.55 | 1619.85 | 1628.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 1605.85 | 1604.79 | 1614.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 12:00:00 | 1605.85 | 1604.79 | 1614.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1605.15 | 1603.32 | 1609.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 1620.55 | 1603.32 | 1609.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1609.45 | 1604.55 | 1609.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 1609.45 | 1604.55 | 1609.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1605.00 | 1604.64 | 1609.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 1599.00 | 1600.65 | 1607.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 10:30:00 | 1601.65 | 1595.27 | 1601.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:00:00 | 1601.00 | 1595.27 | 1601.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 1597.00 | 1594.66 | 1600.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1580.90 | 1587.43 | 1594.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-26 13:15:00 | 1603.15 | 1593.69 | 1593.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 1603.15 | 1593.69 | 1593.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 1618.30 | 1600.56 | 1596.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 13:15:00 | 1598.50 | 1604.52 | 1600.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 13:15:00 | 1598.50 | 1604.52 | 1600.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1598.50 | 1604.52 | 1600.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 1598.50 | 1604.52 | 1600.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1598.35 | 1603.29 | 1600.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 1597.35 | 1603.29 | 1600.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1605.55 | 1603.74 | 1600.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 1598.55 | 1602.87 | 1600.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1601.75 | 1602.65 | 1600.56 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 1581.40 | 1598.81 | 1599.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1566.10 | 1590.96 | 1595.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 1596.20 | 1589.70 | 1593.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 1596.20 | 1589.70 | 1593.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1596.20 | 1589.70 | 1593.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 1596.20 | 1589.70 | 1593.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1588.10 | 1589.38 | 1592.91 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 1597.95 | 1594.50 | 1594.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 1599.75 | 1595.55 | 1594.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 1599.20 | 1604.98 | 1601.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 1599.20 | 1604.98 | 1601.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1599.20 | 1604.98 | 1601.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:45:00 | 1600.10 | 1604.98 | 1601.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1604.60 | 1604.91 | 1601.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 11:15:00 | 1605.80 | 1604.91 | 1601.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 15:15:00 | 1594.00 | 1600.50 | 1600.42 | SL hit (close<static) qty=1.00 sl=1599.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1587.45 | 1598.29 | 1599.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1583.25 | 1595.28 | 1597.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 1590.30 | 1589.95 | 1593.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:00:00 | 1590.30 | 1589.95 | 1593.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1594.10 | 1590.78 | 1593.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 1594.10 | 1590.78 | 1593.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1593.70 | 1591.36 | 1593.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1593.70 | 1591.36 | 1593.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1591.30 | 1591.35 | 1593.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1581.10 | 1591.28 | 1592.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:00:00 | 1585.95 | 1590.21 | 1592.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1585.15 | 1590.55 | 1592.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 13:15:00 | 1603.60 | 1591.65 | 1592.17 | SL hit (close>static) qty=1.00 sl=1593.70 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 14:15:00 | 1599.30 | 1593.18 | 1592.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 12:15:00 | 1614.55 | 1601.39 | 1597.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1601.40 | 1603.38 | 1599.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 1601.40 | 1603.38 | 1599.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1601.40 | 1603.38 | 1599.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1601.40 | 1603.38 | 1599.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1611.50 | 1605.01 | 1600.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 1602.90 | 1605.01 | 1600.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1606.05 | 1610.29 | 1606.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 10:15:00 | 1610.90 | 1610.29 | 1606.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 13:15:00 | 1596.65 | 1603.83 | 1604.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 1596.65 | 1603.83 | 1604.05 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 09:15:00 | 1617.00 | 1604.13 | 1603.93 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 1586.80 | 1604.37 | 1604.72 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 1606.70 | 1604.40 | 1604.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1621.60 | 1608.79 | 1606.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 1627.55 | 1627.64 | 1621.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 14:45:00 | 1630.00 | 1627.64 | 1621.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1638.85 | 1629.35 | 1623.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 1620.20 | 1629.35 | 1623.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1638.10 | 1637.28 | 1631.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:15:00 | 1622.80 | 1637.28 | 1631.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1635.15 | 1636.85 | 1631.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1634.10 | 1636.85 | 1631.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1651.55 | 1641.67 | 1634.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 13:15:00 | 1653.15 | 1641.67 | 1634.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1626.45 | 1639.60 | 1635.08 | SL hit (close<static) qty=1.00 sl=1630.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 15:15:00 | 1631.00 | 1633.54 | 1633.77 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 1637.00 | 1634.24 | 1634.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 09:15:00 | 1641.35 | 1637.17 | 1635.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 15:15:00 | 1641.45 | 1643.77 | 1640.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-27 09:15:00 | 1628.60 | 1643.77 | 1640.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1634.50 | 1641.91 | 1640.04 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 1611.50 | 1635.83 | 1637.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 1602.00 | 1625.36 | 1632.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1613.95 | 1613.14 | 1621.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 11:30:00 | 1613.45 | 1613.14 | 1621.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1618.00 | 1614.47 | 1620.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 1618.50 | 1614.47 | 1620.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1619.60 | 1615.50 | 1620.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:45:00 | 1619.00 | 1615.50 | 1620.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1613.65 | 1615.66 | 1619.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 14:30:00 | 1607.05 | 1614.73 | 1618.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1608.30 | 1614.20 | 1617.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 1607.25 | 1615.20 | 1617.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 13:15:00 | 1625.45 | 1618.67 | 1618.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 13:15:00 | 1625.45 | 1618.67 | 1618.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 1641.95 | 1623.33 | 1620.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 1614.95 | 1624.72 | 1622.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 1614.95 | 1624.72 | 1622.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1614.95 | 1624.72 | 1622.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 1576.95 | 1624.72 | 1622.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1615.45 | 1622.87 | 1621.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:30:00 | 1611.70 | 1622.87 | 1621.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1625.90 | 1624.22 | 1622.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:30:00 | 1643.45 | 1624.40 | 1622.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1646.55 | 1628.50 | 1625.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 1641.00 | 1642.42 | 1637.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:30:00 | 1640.75 | 1649.16 | 1647.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 12:15:00 | 1632.00 | 1644.36 | 1645.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 1632.00 | 1644.36 | 1645.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 1620.40 | 1639.56 | 1643.57 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 1700.00 | 1646.02 | 1644.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 14:15:00 | 1711.00 | 1701.46 | 1692.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 1714.00 | 1714.34 | 1704.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 1714.00 | 1714.34 | 1704.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1712.00 | 1713.44 | 1707.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 1712.00 | 1713.44 | 1707.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1692.35 | 1709.74 | 1708.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:00:00 | 1692.35 | 1709.74 | 1708.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 10:15:00 | 1693.85 | 1706.57 | 1706.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 12:15:00 | 1688.65 | 1700.58 | 1703.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 1645.00 | 1639.05 | 1651.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 1645.00 | 1639.05 | 1651.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1651.80 | 1642.87 | 1651.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 1656.70 | 1642.87 | 1651.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1642.00 | 1642.70 | 1650.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 1648.75 | 1642.70 | 1650.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1627.70 | 1615.28 | 1626.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 1627.70 | 1615.28 | 1626.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1629.15 | 1618.05 | 1627.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 1629.95 | 1618.05 | 1627.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 1642.85 | 1632.21 | 1631.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 11:15:00 | 1650.40 | 1640.97 | 1636.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 1615.80 | 1639.57 | 1638.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 1615.80 | 1639.57 | 1638.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1615.80 | 1639.57 | 1638.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 1615.80 | 1639.57 | 1638.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 1605.75 | 1632.81 | 1635.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 1597.10 | 1625.66 | 1631.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 1592.30 | 1591.83 | 1607.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:30:00 | 1589.90 | 1591.83 | 1607.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1598.25 | 1583.86 | 1591.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 1598.25 | 1583.86 | 1591.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1603.85 | 1587.86 | 1592.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 1603.85 | 1587.86 | 1592.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1620.15 | 1598.66 | 1596.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 14:15:00 | 1626.05 | 1616.60 | 1609.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 1632.15 | 1636.10 | 1628.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 13:45:00 | 1632.05 | 1636.10 | 1628.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1631.40 | 1635.16 | 1628.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:15:00 | 1628.00 | 1635.16 | 1628.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1628.00 | 1633.73 | 1628.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1634.70 | 1633.73 | 1628.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1643.20 | 1635.62 | 1629.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:15:00 | 1644.75 | 1635.62 | 1629.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 1645.00 | 1653.10 | 1645.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:45:00 | 1644.75 | 1647.73 | 1644.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 1647.70 | 1647.73 | 1644.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1647.70 | 1647.72 | 1644.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:30:00 | 1650.90 | 1648.75 | 1645.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 1640.00 | 1646.98 | 1645.37 | SL hit (close<static) qty=1.00 sl=1640.35 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 1633.30 | 1642.61 | 1643.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-18 12:15:00 | 1629.85 | 1637.84 | 1639.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 1634.35 | 1634.29 | 1637.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 09:15:00 | 1634.35 | 1634.29 | 1637.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1634.35 | 1634.29 | 1637.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 11:30:00 | 1630.10 | 1633.85 | 1636.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 13:00:00 | 1630.05 | 1633.09 | 1635.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 1664.85 | 1640.28 | 1638.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 1664.85 | 1640.28 | 1638.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 1681.65 | 1653.53 | 1645.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 13:15:00 | 1719.90 | 1721.48 | 1703.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 14:00:00 | 1719.90 | 1721.48 | 1703.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1721.90 | 1719.69 | 1709.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1716.30 | 1719.69 | 1709.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1734.95 | 1735.41 | 1728.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 1731.40 | 1735.41 | 1728.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1728.05 | 1733.94 | 1728.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1728.05 | 1733.94 | 1728.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1728.15 | 1732.78 | 1728.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 1727.40 | 1732.78 | 1728.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1725.80 | 1731.39 | 1728.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1725.80 | 1731.39 | 1728.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1726.00 | 1730.31 | 1728.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 1730.00 | 1728.42 | 1727.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:00:00 | 1730.00 | 1728.73 | 1727.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1754.00 | 1729.64 | 1728.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1725.50 | 1729.06 | 1729.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 1725.50 | 1729.06 | 1729.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 1723.60 | 1727.97 | 1728.78 | Break + close below crossover candle low |

### Cycle 68 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1753.25 | 1733.03 | 1731.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1755.00 | 1744.15 | 1737.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1743.95 | 1745.23 | 1739.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1743.95 | 1745.23 | 1739.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1743.95 | 1745.23 | 1739.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 1741.45 | 1745.23 | 1739.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 1743.05 | 1749.76 | 1746.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 1742.85 | 1749.76 | 1746.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 1740.50 | 1747.90 | 1746.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 1720.60 | 1747.90 | 1746.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1708.40 | 1740.00 | 1742.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 10:15:00 | 1692.10 | 1730.42 | 1738.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1717.80 | 1704.00 | 1718.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1717.80 | 1704.00 | 1718.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1717.80 | 1704.00 | 1718.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1708.50 | 1706.90 | 1718.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1711.15 | 1706.90 | 1718.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 1733.35 | 1712.19 | 1719.57 | SL hit (close>static) qty=1.00 sl=1728.90 alert=retest2 |

### Cycle 70 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1760.20 | 1724.96 | 1722.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1800.70 | 1759.77 | 1743.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 1877.50 | 1877.62 | 1856.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 09:45:00 | 1869.60 | 1877.62 | 1856.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 1848.90 | 1869.10 | 1858.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:00:00 | 1848.90 | 1869.10 | 1858.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1849.10 | 1865.10 | 1857.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:30:00 | 1847.00 | 1865.10 | 1857.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1843.40 | 1856.46 | 1854.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1843.40 | 1856.46 | 1854.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1841.70 | 1853.51 | 1853.66 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1855.60 | 1853.92 | 1853.84 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 1845.70 | 1852.28 | 1853.10 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 1860.90 | 1854.00 | 1853.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 1882.30 | 1859.66 | 1856.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 1860.10 | 1862.84 | 1858.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 1860.10 | 1862.84 | 1858.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1860.10 | 1862.84 | 1858.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:30:00 | 1852.30 | 1862.84 | 1858.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1855.70 | 1861.42 | 1858.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 1855.30 | 1861.42 | 1858.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1851.30 | 1859.39 | 1857.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:45:00 | 1853.50 | 1859.39 | 1857.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1850.70 | 1857.65 | 1857.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 1849.90 | 1857.65 | 1857.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 1847.20 | 1855.56 | 1856.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1823.80 | 1846.14 | 1851.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 1828.90 | 1823.31 | 1831.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 14:00:00 | 1828.90 | 1823.31 | 1831.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1839.10 | 1825.83 | 1830.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 1839.70 | 1825.83 | 1830.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1825.80 | 1825.83 | 1829.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:30:00 | 1822.10 | 1826.52 | 1829.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 12:15:00 | 1835.80 | 1831.20 | 1830.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 12:15:00 | 1835.80 | 1831.20 | 1830.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 13:15:00 | 1855.00 | 1835.96 | 1832.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 1844.90 | 1849.39 | 1841.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:00:00 | 1844.90 | 1849.39 | 1841.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1838.50 | 1847.22 | 1840.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:30:00 | 1840.80 | 1847.22 | 1840.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1837.60 | 1845.29 | 1840.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:30:00 | 1837.20 | 1845.29 | 1840.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1887.00 | 1893.91 | 1887.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 1887.00 | 1893.91 | 1887.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 1885.80 | 1892.29 | 1887.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 1885.80 | 1892.29 | 1887.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1883.10 | 1890.45 | 1886.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:30:00 | 1882.70 | 1890.45 | 1886.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1877.00 | 1887.76 | 1885.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1877.00 | 1887.76 | 1885.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1864.50 | 1881.53 | 1883.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1856.40 | 1876.50 | 1880.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1871.50 | 1858.22 | 1866.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1871.50 | 1858.22 | 1866.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1871.50 | 1858.22 | 1866.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 1880.70 | 1858.22 | 1866.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1866.20 | 1859.81 | 1866.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 1854.00 | 1866.64 | 1867.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 09:45:00 | 1853.50 | 1839.23 | 1850.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 14:15:00 | 1868.10 | 1845.41 | 1844.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 1868.10 | 1845.41 | 1844.04 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 09:15:00 | 1816.70 | 1842.50 | 1843.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1802.70 | 1812.72 | 1819.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1821.50 | 1813.63 | 1818.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1817.90 | 1814.48 | 1818.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 1819.50 | 1814.48 | 1818.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1811.20 | 1813.83 | 1817.50 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1832.20 | 1821.03 | 1819.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1834.40 | 1823.71 | 1821.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1832.50 | 1834.49 | 1830.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1832.50 | 1834.49 | 1830.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1831.00 | 1833.79 | 1830.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1839.90 | 1833.79 | 1830.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 1844.70 | 1854.48 | 1855.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 1844.70 | 1854.48 | 1855.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 1838.40 | 1849.08 | 1852.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1846.80 | 1845.88 | 1849.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 1846.80 | 1845.88 | 1849.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1841.30 | 1844.97 | 1848.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 1846.00 | 1844.97 | 1848.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1872.80 | 1849.99 | 1850.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 1872.80 | 1849.99 | 1850.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1870.60 | 1854.12 | 1852.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 1873.70 | 1858.03 | 1854.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 12:15:00 | 1876.10 | 1876.51 | 1871.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:30:00 | 1877.30 | 1876.51 | 1871.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1870.00 | 1875.13 | 1871.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1870.00 | 1875.13 | 1871.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1870.10 | 1874.12 | 1871.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1854.00 | 1874.12 | 1871.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1855.00 | 1870.30 | 1870.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 1851.40 | 1870.30 | 1870.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1866.10 | 1869.46 | 1869.82 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 1872.80 | 1870.13 | 1870.09 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1862.60 | 1868.62 | 1869.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 1854.50 | 1863.70 | 1866.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 1859.30 | 1858.76 | 1862.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 1859.30 | 1858.76 | 1862.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1861.20 | 1859.02 | 1861.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1857.50 | 1859.02 | 1861.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 1863.20 | 1859.61 | 1861.31 | SL hit (close>static) qty=1.00 sl=1862.40 alert=retest2 |

### Cycle 86 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1864.60 | 1852.29 | 1851.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 09:15:00 | 1876.10 | 1865.90 | 1861.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 1942.70 | 1944.26 | 1927.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 12:30:00 | 1937.80 | 1944.26 | 1927.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 2017.20 | 2027.73 | 2024.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 2017.20 | 2027.73 | 2024.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 2019.00 | 2025.98 | 2023.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 2022.60 | 2025.98 | 2023.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 2009.00 | 2020.83 | 2021.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 2009.00 | 2020.83 | 2021.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2007.40 | 2018.14 | 2020.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2016.30 | 2015.28 | 2018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2018.30 | 2015.88 | 2018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2023.90 | 2015.88 | 2018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2023.00 | 2017.30 | 2018.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 2018.10 | 2017.30 | 2018.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2020.10 | 2017.86 | 2018.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 2026.00 | 2017.86 | 2018.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 2014.20 | 2017.55 | 2018.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:45:00 | 2010.70 | 2017.70 | 2018.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 2032.10 | 2020.58 | 2019.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 2032.10 | 2020.58 | 2019.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2036.20 | 2027.62 | 2024.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 2026.50 | 2028.38 | 2025.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:00:00 | 2026.50 | 2028.38 | 2025.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2029.00 | 2028.51 | 2025.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 2029.00 | 2028.51 | 2025.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2021.10 | 2027.02 | 2025.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 2014.70 | 2027.02 | 2025.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2019.20 | 2025.46 | 2024.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2006.30 | 2025.46 | 2024.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1987.40 | 2017.85 | 2021.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1983.40 | 2005.85 | 2015.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 1922.20 | 1919.96 | 1941.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1922.20 | 1919.96 | 1941.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1930.30 | 1922.53 | 1938.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1935.30 | 1922.53 | 1938.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1936.90 | 1925.40 | 1938.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1936.90 | 1925.40 | 1938.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1931.80 | 1926.68 | 1938.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 1941.50 | 1926.68 | 1938.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1934.80 | 1930.50 | 1936.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1930.60 | 1930.50 | 1936.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1932.60 | 1930.92 | 1936.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1905.00 | 1933.44 | 1935.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1924.50 | 1909.97 | 1909.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1924.50 | 1909.97 | 1909.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1929.70 | 1916.35 | 1912.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 1937.90 | 1939.96 | 1929.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 1937.90 | 1939.96 | 1929.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1942.20 | 1939.72 | 1932.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:00:00 | 1948.00 | 1941.37 | 1933.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1916.50 | 1933.12 | 1933.01 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1905.10 | 1927.51 | 1930.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 1894.00 | 1916.42 | 1924.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 1912.30 | 1905.26 | 1914.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 1912.30 | 1905.26 | 1914.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1921.80 | 1908.57 | 1915.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 1921.80 | 1908.57 | 1915.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1916.80 | 1910.22 | 1915.48 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1935.00 | 1918.14 | 1917.86 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1905.30 | 1919.65 | 1920.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1892.30 | 1908.81 | 1913.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1901.40 | 1898.65 | 1905.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 1901.40 | 1898.65 | 1905.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1910.00 | 1900.92 | 1906.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1910.00 | 1900.92 | 1906.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1919.50 | 1904.64 | 1907.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1919.50 | 1904.64 | 1907.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1914.60 | 1906.63 | 1907.99 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 1926.90 | 1912.04 | 1910.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 1933.70 | 1916.37 | 1912.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 1929.10 | 1931.54 | 1925.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 1929.10 | 1931.54 | 1925.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1932.00 | 1931.63 | 1925.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 1925.50 | 1931.63 | 1925.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1924.00 | 1930.11 | 1925.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1928.30 | 1930.11 | 1925.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1915.60 | 1927.20 | 1924.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1915.60 | 1927.20 | 1924.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1917.00 | 1925.16 | 1923.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 1917.00 | 1925.16 | 1923.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1910.40 | 1922.21 | 1922.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 1867.50 | 1908.93 | 1915.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 1860.90 | 1855.14 | 1864.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 1861.40 | 1855.14 | 1864.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1862.90 | 1856.78 | 1863.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 1862.90 | 1856.78 | 1863.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1861.20 | 1857.66 | 1863.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 1861.70 | 1857.66 | 1863.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1860.70 | 1858.27 | 1862.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1860.70 | 1858.27 | 1862.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1867.90 | 1860.20 | 1863.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 1864.90 | 1860.20 | 1863.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1870.90 | 1862.34 | 1864.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1872.60 | 1862.34 | 1864.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1875.00 | 1866.51 | 1865.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1888.90 | 1874.40 | 1870.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1905.90 | 1907.37 | 1895.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 1905.90 | 1907.37 | 1895.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1931.20 | 1930.40 | 1923.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1927.40 | 1930.40 | 1923.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1919.90 | 1928.30 | 1923.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1919.90 | 1928.30 | 1923.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1926.30 | 1927.90 | 1923.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 1941.70 | 1929.86 | 1925.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1915.70 | 1928.15 | 1925.62 | SL hit (close<static) qty=1.00 sl=1918.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1914.20 | 1925.66 | 1925.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1897.90 | 1910.14 | 1916.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1896.90 | 1890.03 | 1895.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1898.00 | 1891.62 | 1896.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1901.20 | 1891.62 | 1896.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1894.40 | 1892.18 | 1896.00 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1907.80 | 1898.90 | 1898.11 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1894.80 | 1897.56 | 1897.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1880.20 | 1894.08 | 1896.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 13:15:00 | 1886.70 | 1884.49 | 1889.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 1886.70 | 1884.49 | 1889.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1884.20 | 1884.43 | 1888.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1877.80 | 1883.08 | 1886.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 1879.20 | 1883.28 | 1886.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 1880.40 | 1883.60 | 1886.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 1881.40 | 1881.69 | 1884.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1884.80 | 1882.17 | 1884.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1884.80 | 1882.17 | 1884.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1891.70 | 1884.08 | 1885.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1891.70 | 1884.08 | 1885.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 1906.50 | 1888.56 | 1887.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1906.50 | 1888.56 | 1887.06 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 1886.80 | 1888.81 | 1888.96 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 1897.70 | 1890.59 | 1889.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 1901.60 | 1892.79 | 1890.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 1894.80 | 1894.26 | 1891.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1887.90 | 1893.99 | 1892.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1887.90 | 1893.99 | 1892.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1886.40 | 1892.47 | 1892.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1886.40 | 1892.47 | 1892.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 1886.80 | 1891.33 | 1891.60 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 1893.90 | 1891.92 | 1891.82 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 1890.30 | 1891.60 | 1891.68 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1895.00 | 1892.28 | 1891.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 1902.80 | 1894.71 | 1893.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 1908.50 | 1909.37 | 1903.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 1908.50 | 1909.37 | 1903.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1902.40 | 1907.98 | 1903.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 1902.40 | 1907.98 | 1903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1903.00 | 1906.98 | 1903.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1906.40 | 1906.98 | 1903.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1906.10 | 1906.81 | 1903.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1910.30 | 1906.81 | 1903.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1911.60 | 1904.80 | 1903.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 1943.80 | 1945.38 | 1945.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1943.80 | 1945.38 | 1945.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1939.20 | 1944.14 | 1944.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 1935.60 | 1933.30 | 1938.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 1935.60 | 1933.30 | 1938.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1936.00 | 1933.41 | 1937.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1939.20 | 1933.41 | 1937.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1939.30 | 1934.59 | 1937.45 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 13:15:00 | 1945.60 | 1939.45 | 1939.04 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1935.00 | 1938.31 | 1938.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1930.80 | 1936.81 | 1937.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1918.00 | 1917.50 | 1924.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1918.00 | 1917.50 | 1924.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1900.70 | 1913.96 | 1921.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1898.50 | 1909.28 | 1918.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1899.60 | 1888.92 | 1888.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 1899.60 | 1888.92 | 1888.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 1904.70 | 1893.64 | 1890.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 15:15:00 | 1939.00 | 1940.50 | 1928.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 09:15:00 | 1944.80 | 1940.50 | 1928.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1940.30 | 1940.75 | 1937.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 1938.90 | 1940.75 | 1937.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1958.40 | 1944.16 | 1939.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 1962.60 | 1952.20 | 1948.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 1962.60 | 1955.99 | 1951.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 2062.80 | 2075.41 | 2076.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 2062.80 | 2075.41 | 2076.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 2053.10 | 2064.74 | 2069.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 2120.00 | 2076.11 | 2071.10 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 2011.50 | 2081.94 | 2086.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 2002.70 | 2066.09 | 2078.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 2022.00 | 2019.41 | 2037.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 2022.00 | 2019.41 | 2037.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2049.00 | 2025.55 | 2035.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 2049.00 | 2025.55 | 2035.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2041.70 | 2028.78 | 2036.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 2049.30 | 2028.78 | 2036.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2065.80 | 2044.70 | 2041.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 2071.70 | 2056.29 | 2048.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 2085.00 | 2086.80 | 2074.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 2085.00 | 2086.80 | 2074.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2158.70 | 2158.55 | 2149.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 2171.10 | 2160.31 | 2152.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:15:00 | 2170.80 | 2161.29 | 2153.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 15:15:00 | 2145.00 | 2155.36 | 2154.99 | SL hit (close<static) qty=1.00 sl=2148.60 alert=retest2 |

### Cycle 115 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 2147.20 | 2153.73 | 2154.28 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 2165.50 | 2156.19 | 2155.25 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 2116.40 | 2150.30 | 2153.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 2092.20 | 2104.13 | 2113.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 2107.10 | 2097.87 | 2106.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2106.00 | 2099.50 | 2106.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 2108.70 | 2099.50 | 2106.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2098.40 | 2099.28 | 2105.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2093.80 | 2102.52 | 2105.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 2095.70 | 2090.74 | 2092.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 2099.90 | 2094.41 | 2094.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 2099.90 | 2094.41 | 2094.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 2108.00 | 2099.34 | 2096.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 2097.10 | 2100.56 | 2097.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2097.20 | 2099.89 | 2097.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 2094.70 | 2099.89 | 2097.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 2089.20 | 2097.75 | 2097.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 2089.20 | 2097.75 | 2097.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2089.00 | 2096.00 | 2096.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2081.60 | 2093.12 | 2094.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2089.50 | 2088.72 | 2091.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 2089.50 | 2088.72 | 2091.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2087.50 | 2088.48 | 2091.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 2083.50 | 2088.04 | 2090.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2083.20 | 2087.92 | 2090.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 2093.70 | 2089.07 | 2090.66 | SL hit (close>static) qty=1.00 sl=2091.40 alert=retest2 |

### Cycle 120 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2083.10 | 2070.44 | 2070.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 2092.50 | 2076.56 | 2073.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2093.70 | 2104.41 | 2099.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2092.20 | 2101.97 | 2098.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 2088.70 | 2101.97 | 2098.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2094.00 | 2098.89 | 2097.79 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 2088.70 | 2096.85 | 2096.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 2084.10 | 2094.30 | 2095.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 2098.00 | 2094.69 | 2095.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 2090.90 | 2093.93 | 2095.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 2115.10 | 2093.93 | 2095.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2131.30 | 2101.40 | 2098.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 2140.00 | 2109.12 | 2102.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2130.70 | 2133.00 | 2119.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 2130.10 | 2133.00 | 2119.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2125.80 | 2129.72 | 2121.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 2125.70 | 2129.72 | 2121.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2123.40 | 2128.04 | 2122.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 2123.40 | 2128.04 | 2122.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2123.00 | 2127.03 | 2122.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 2126.70 | 2127.03 | 2122.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2129.60 | 2127.54 | 2122.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 2132.30 | 2127.54 | 2122.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 2131.70 | 2129.60 | 2124.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 2107.90 | 2121.60 | 2122.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 2107.90 | 2121.60 | 2122.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 2100.30 | 2109.38 | 2115.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 2094.70 | 2092.69 | 2102.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:45:00 | 2090.50 | 2092.69 | 2102.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2099.40 | 2094.03 | 2102.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 2099.40 | 2094.03 | 2102.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2099.90 | 2095.86 | 2100.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2099.90 | 2095.86 | 2100.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2100.90 | 2096.87 | 2100.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 2098.80 | 2096.87 | 2100.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2099.40 | 2097.37 | 2100.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 2095.00 | 2097.37 | 2100.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2096.40 | 2097.18 | 2100.40 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 2105.00 | 2102.38 | 2102.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 2121.70 | 2114.37 | 2110.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 2098.10 | 2111.12 | 2109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2102.90 | 2109.48 | 2108.51 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 2105.10 | 2107.39 | 2107.65 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 10:15:00 | 2114.10 | 2108.06 | 2107.51 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2094.40 | 2105.33 | 2106.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 2083.30 | 2099.53 | 2102.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 2089.20 | 2088.46 | 2094.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 2089.20 | 2088.46 | 2094.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2068.70 | 2071.64 | 2081.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 2062.10 | 2069.17 | 2079.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:15:00 | 1958.99 | 1971.70 | 1981.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 1966.50 | 1958.18 | 1968.09 | SL hit (close>ema200) qty=0.50 sl=1958.18 alert=retest2 |

### Cycle 128 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1973.00 | 1967.54 | 1967.08 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1939.00 | 1961.58 | 1964.44 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1988.00 | 1963.87 | 1961.26 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 2012.10 | 2024.86 | 2024.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 2002.10 | 2018.17 | 2021.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 2017.80 | 2015.88 | 2019.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2014.90 | 2015.68 | 2019.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 2009.60 | 2017.26 | 2018.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2009.30 | 2017.26 | 2018.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 2007.60 | 2007.84 | 2012.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 13:15:00 | 2026.50 | 2013.56 | 2013.67 | SL hit (close>static) qty=1.00 sl=2020.40 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 2029.60 | 2016.77 | 2015.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 2030.00 | 2019.41 | 2016.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 2021.10 | 2024.42 | 2020.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 2020.70 | 2023.67 | 2020.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 2019.90 | 2023.67 | 2020.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 2022.00 | 2023.34 | 2020.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 2027.50 | 2022.87 | 2020.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 2027.00 | 2022.26 | 2021.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2015.30 | 2020.87 | 2020.81 | SL hit (close<static) qty=1.00 sl=2017.50 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2015.30 | 2019.76 | 2020.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 2005.70 | 2016.94 | 2018.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1992.40 | 1985.46 | 1995.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 10:15:00 | 1991.90 | 1985.46 | 1995.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1991.10 | 1986.59 | 1994.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1987.10 | 1986.59 | 1994.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1997.00 | 1991.32 | 1994.57 | SL hit (close>static) qty=1.00 sl=1996.30 alert=retest2 |

### Cycle 134 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1908.00 | 1897.73 | 1896.87 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 1887.40 | 1898.23 | 1899.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 12:15:00 | 1880.50 | 1892.82 | 1896.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 1855.70 | 1855.68 | 1865.14 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1839.50 | 1855.68 | 1865.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1814.90 | 1795.08 | 1801.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 1814.90 | 1795.08 | 1801.74 | SL hit (close>ema400) qty=1.00 sl=1801.74 alert=retest1 |

### Cycle 136 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1821.70 | 1805.93 | 1805.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 1833.00 | 1811.34 | 1808.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:45:00 | 1853.40 | 1847.90 | 1836.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:15:00 | 1853.80 | 1848.36 | 1837.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 1853.60 | 1847.53 | 1838.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:00:00 | 1854.10 | 1844.57 | 1839.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1843.90 | 1848.26 | 1843.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1843.90 | 1848.26 | 1843.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1844.80 | 1847.57 | 1843.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1812.00 | 1847.57 | 1843.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1800.10 | 1838.07 | 1839.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1800.10 | 1838.07 | 1839.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 14:15:00 | 1792.60 | 1812.32 | 1824.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1806.00 | 1802.98 | 1814.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 1803.80 | 1802.98 | 1814.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1812.50 | 1804.54 | 1812.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1812.70 | 1804.54 | 1812.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1819.40 | 1807.51 | 1813.05 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1840.10 | 1816.65 | 1816.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1841.70 | 1824.79 | 1820.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 1831.00 | 1831.42 | 1825.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:30:00 | 1832.10 | 1831.42 | 1825.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1825.20 | 1830.18 | 1825.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 1824.20 | 1830.18 | 1825.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1824.70 | 1829.08 | 1825.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 1824.70 | 1829.08 | 1825.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1846.10 | 1832.48 | 1827.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 1850.00 | 1832.48 | 1827.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1803.00 | 1829.39 | 1827.15 | SL hit (close<static) qty=1.00 sl=1818.40 alert=retest2 |

### Cycle 139 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1796.40 | 1822.79 | 1824.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 1790.90 | 1810.28 | 1817.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1800.10 | 1801.78 | 1810.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 1799.90 | 1801.12 | 1809.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1807.50 | 1789.79 | 1788.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1807.50 | 1789.79 | 1788.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1810.20 | 1793.87 | 1790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 1868.70 | 1865.89 | 1856.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1879.00 | 1871.50 | 1863.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 15:15:00 | 1857.50 | 1863.28 | 1863.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 15:15:00 | 1857.50 | 1863.28 | 1863.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 09:15:00 | 1840.50 | 1858.73 | 1861.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 11:15:00 | 1840.40 | 1839.48 | 1846.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:00:00 | 1840.40 | 1839.48 | 1846.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1846.40 | 1841.95 | 1846.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 1846.40 | 1841.95 | 1846.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1847.30 | 1843.02 | 1846.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 1848.80 | 1843.02 | 1846.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1848.50 | 1844.11 | 1846.56 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 1852.90 | 1848.70 | 1848.23 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1842.00 | 1846.86 | 1847.45 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1858.80 | 1849.25 | 1848.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1864.70 | 1852.34 | 1849.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1854.30 | 1856.53 | 1853.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1858.70 | 1856.97 | 1853.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1850.50 | 1856.97 | 1853.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1837.60 | 1853.09 | 1852.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 1837.60 | 1853.09 | 1852.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1841.30 | 1850.74 | 1851.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 1833.60 | 1844.17 | 1847.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 13:15:00 | 1837.40 | 1836.92 | 1841.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:15:00 | 1837.90 | 1836.92 | 1841.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1840.50 | 1837.64 | 1841.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1840.50 | 1837.64 | 1841.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1841.50 | 1838.41 | 1841.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1824.70 | 1838.41 | 1841.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1817.00 | 1834.13 | 1839.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1808.70 | 1822.71 | 1831.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1834.40 | 1828.49 | 1827.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1834.40 | 1828.49 | 1827.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 1851.00 | 1835.98 | 1831.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1869.80 | 1874.91 | 1859.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 1877.90 | 1874.25 | 1860.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:45:00 | 1879.60 | 1875.22 | 1862.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 13:00:00 | 1879.30 | 1876.03 | 1863.92 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1866.00 | 1880.65 | 1870.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1866.00 | 1880.65 | 1870.79 | SL hit (close<ema400) qty=1.00 sl=1870.79 alert=retest1 |

### Cycle 147 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 1840.20 | 1863.43 | 1864.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 1830.40 | 1856.83 | 1861.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1831.60 | 1823.21 | 1835.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:00:00 | 1831.60 | 1823.21 | 1835.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1834.10 | 1825.38 | 1835.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 1835.90 | 1825.38 | 1835.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1834.50 | 1827.21 | 1835.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 1834.90 | 1827.21 | 1835.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1833.50 | 1829.68 | 1835.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 1837.90 | 1829.68 | 1835.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1832.10 | 1830.16 | 1834.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1830.90 | 1830.89 | 1834.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1824.30 | 1829.71 | 1833.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:45:00 | 1829.50 | 1830.24 | 1833.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:15:00 | 1829.30 | 1830.24 | 1833.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1820.00 | 1827.42 | 1830.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 14:15:00 | 1834.40 | 1832.10 | 1832.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 1834.40 | 1832.10 | 1832.09 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-18 09:15:00 | 1438.30 | 2024-06-18 10:15:00 | 1425.35 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-07-10 15:00:00 | 1445.60 | 2024-07-11 10:15:00 | 1428.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-22 14:15:00 | 1469.15 | 2024-07-23 11:15:00 | 1455.55 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-07-24 12:15:00 | 1450.60 | 2024-07-26 09:15:00 | 1487.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-07-25 09:15:00 | 1443.10 | 2024-07-26 09:15:00 | 1487.00 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-07-25 10:45:00 | 1450.00 | 2024-07-26 09:15:00 | 1487.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-07-25 13:00:00 | 1450.45 | 2024-07-26 09:15:00 | 1487.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-08-08 15:15:00 | 1449.05 | 2024-08-09 09:15:00 | 1477.60 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-09-05 09:15:00 | 1553.70 | 2024-09-10 09:15:00 | 1566.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-09-19 09:15:00 | 1672.75 | 2024-09-27 12:15:00 | 1730.00 | STOP_HIT | 1.00 | 3.42% |
| SELL | retest2 | 2024-10-08 14:30:00 | 1658.95 | 2024-10-09 09:15:00 | 1683.65 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-10-18 15:15:00 | 1702.00 | 2024-10-29 09:15:00 | 1616.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1703.50 | 2024-10-29 09:15:00 | 1618.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 12:00:00 | 1697.90 | 2024-10-29 09:15:00 | 1613.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:15:00 | 1702.00 | 2024-10-30 10:15:00 | 1651.00 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1703.50 | 2024-10-30 10:15:00 | 1651.00 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-10-22 12:00:00 | 1697.90 | 2024-10-30 10:15:00 | 1651.00 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest2 | 2024-11-27 12:30:00 | 1578.00 | 2024-11-28 14:15:00 | 1561.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-11-27 13:00:00 | 1577.35 | 2024-11-28 14:15:00 | 1561.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-11-27 13:30:00 | 1578.30 | 2024-11-28 14:15:00 | 1561.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-11-27 14:15:00 | 1578.70 | 2024-11-28 14:15:00 | 1561.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-10 09:15:00 | 1591.35 | 2024-12-12 11:15:00 | 1603.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-12 10:00:00 | 1593.10 | 2024-12-12 11:15:00 | 1603.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-12 10:45:00 | 1596.10 | 2024-12-12 11:15:00 | 1603.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-12-20 12:30:00 | 1599.00 | 2024-12-26 13:15:00 | 1603.15 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-12-23 10:30:00 | 1601.65 | 2024-12-26 13:15:00 | 1603.15 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-12-23 11:00:00 | 1601.00 | 2024-12-26 13:15:00 | 1603.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-12-23 11:45:00 | 1597.00 | 2024-12-26 13:15:00 | 1603.15 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-01-03 11:15:00 | 1605.80 | 2025-01-03 15:15:00 | 1594.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-06 09:15:00 | 1605.30 | 2025-01-06 10:15:00 | 1587.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1581.10 | 2025-01-08 13:15:00 | 1603.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-01-08 10:00:00 | 1585.95 | 2025-01-08 13:15:00 | 1603.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-08 11:15:00 | 1585.15 | 2025-01-08 13:15:00 | 1603.60 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-01-13 10:15:00 | 1610.90 | 2025-01-13 13:15:00 | 1596.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-01-21 13:15:00 | 1653.15 | 2025-01-21 14:15:00 | 1626.45 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-01-29 14:30:00 | 1607.05 | 2025-01-30 13:15:00 | 1625.45 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-30 09:30:00 | 1608.30 | 2025-01-30 13:15:00 | 1625.45 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-01-30 11:30:00 | 1607.25 | 2025-01-30 13:15:00 | 1625.45 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-02-01 11:30:00 | 1643.45 | 2025-02-06 12:15:00 | 1632.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1646.55 | 2025-02-06 12:15:00 | 1632.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-04 11:15:00 | 1641.00 | 2025-02-06 12:15:00 | 1632.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-02-06 10:30:00 | 1640.75 | 2025-02-06 12:15:00 | 1632.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-03-11 10:15:00 | 1644.75 | 2025-03-13 11:15:00 | 1640.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-03-12 11:45:00 | 1645.00 | 2025-03-13 14:15:00 | 1633.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-03-12 14:45:00 | 1644.75 | 2025-03-13 14:15:00 | 1633.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-03-12 15:15:00 | 1647.70 | 2025-03-13 14:15:00 | 1633.30 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-13 09:30:00 | 1650.90 | 2025-03-13 14:15:00 | 1633.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-03-19 11:30:00 | 1630.10 | 2025-03-20 09:15:00 | 1664.85 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-03-19 13:00:00 | 1630.05 | 2025-03-20 09:15:00 | 1664.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-03-28 13:00:00 | 1730.00 | 2025-04-01 14:15:00 | 1725.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-03-28 14:00:00 | 1730.00 | 2025-04-01 14:15:00 | 1725.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-04-01 09:15:00 | 1754.00 | 2025-04-01 14:15:00 | 1725.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1708.50 | 2025-04-08 11:15:00 | 1733.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-04-08 11:15:00 | 1711.15 | 2025-04-08 11:15:00 | 1733.35 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-04-29 14:30:00 | 1822.10 | 2025-04-30 12:15:00 | 1835.80 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-05-13 09:15:00 | 1854.00 | 2025-05-15 14:15:00 | 1868.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-14 09:45:00 | 1853.50 | 2025-05-15 14:15:00 | 1868.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1839.90 | 2025-06-02 10:15:00 | 1844.70 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1857.50 | 2025-06-12 09:15:00 | 1863.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-12 10:30:00 | 1857.50 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1855.50 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-12 12:00:00 | 1855.00 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-04 09:15:00 | 2022.60 | 2025-07-04 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-07 13:45:00 | 2010.70 | 2025-07-07 14:15:00 | 2032.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1905.00 | 2025-07-23 10:15:00 | 1924.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-25 11:00:00 | 1948.00 | 2025-07-28 09:15:00 | 1916.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-08-22 13:30:00 | 1941.70 | 2025-08-25 09:15:00 | 1915.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-25 12:45:00 | 1929.80 | 2025-08-26 09:15:00 | 1914.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-25 14:30:00 | 1930.10 | 2025-08-26 09:15:00 | 1914.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1877.80 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-09-04 12:15:00 | 1879.20 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-04 14:45:00 | 1880.40 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-05 09:45:00 | 1881.40 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-15 10:15:00 | 1910.30 | 2025-09-23 13:15:00 | 1943.80 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1911.60 | 2025-09-23 13:15:00 | 1943.80 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1898.50 | 2025-10-06 10:15:00 | 1899.60 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-15 11:15:00 | 1962.60 | 2025-10-30 13:15:00 | 2062.80 | STOP_HIT | 1.00 | 5.11% |
| BUY | retest2 | 2025-10-15 12:30:00 | 1962.60 | 2025-10-30 13:15:00 | 2062.80 | STOP_HIT | 1.00 | 5.11% |
| BUY | retest2 | 2025-11-21 12:45:00 | 2171.10 | 2025-11-24 15:15:00 | 2145.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-21 14:15:00 | 2170.80 | 2025-11-24 15:15:00 | 2145.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2093.80 | 2025-12-05 10:15:00 | 2099.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-12-05 09:15:00 | 2095.70 | 2025-12-05 10:15:00 | 2099.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-09 14:15:00 | 2083.50 | 2025-12-10 09:15:00 | 2093.70 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-12-10 09:15:00 | 2083.20 | 2025-12-10 09:15:00 | 2093.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-12-10 11:30:00 | 2082.50 | 2025-12-12 14:15:00 | 2083.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-12-24 10:15:00 | 2132.30 | 2025-12-26 10:15:00 | 2107.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-24 12:45:00 | 2131.70 | 2025-12-26 10:15:00 | 2107.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-09 10:45:00 | 2062.10 | 2026-01-28 11:15:00 | 1958.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:45:00 | 2062.10 | 2026-01-29 11:15:00 | 1966.50 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2026-02-13 10:30:00 | 2009.60 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2009.30 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-16 10:15:00 | 2007.60 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-18 09:30:00 | 2027.50 | 2026-02-19 09:15:00 | 2015.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-19 09:15:00 | 2027.00 | 2026-02-19 09:15:00 | 2015.30 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1987.10 | 2026-02-23 14:15:00 | 1997.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1946.60 | 2026-03-02 09:15:00 | 1849.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1946.60 | 2026-03-04 10:15:00 | 1891.10 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest1 | 2026-03-11 09:15:00 | 1839.50 | 2026-03-17 09:15:00 | 1814.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2026-03-19 10:45:00 | 1853.40 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-03-19 12:15:00 | 1853.80 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-03-19 13:30:00 | 1853.60 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-03-20 11:00:00 | 1854.10 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2026-03-27 15:15:00 | 1850.00 | 2026-03-30 09:15:00 | 1803.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-04-01 11:00:00 | 1800.10 | 2026-04-07 09:15:00 | 1807.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-04-01 11:30:00 | 1799.90 | 2026-04-07 09:15:00 | 1807.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-04-13 11:00:00 | 1868.70 | 2026-04-15 15:15:00 | 1857.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1879.00 | 2026-04-15 15:15:00 | 1857.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1808.70 | 2026-04-28 11:15:00 | 1834.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest1 | 2026-04-30 11:15:00 | 1877.90 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-04-30 11:45:00 | 1879.60 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2026-04-30 13:00:00 | 1879.30 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-05-07 09:15:00 | 1830.90 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-05-07 09:45:00 | 1824.30 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-05-07 12:45:00 | 1829.50 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-05-07 13:15:00 | 1829.30 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.28% |
