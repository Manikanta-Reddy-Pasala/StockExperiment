# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1717.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 99 |
| ALERT2 | 98 |
| ALERT2_SKIP | 51 |
| ALERT3 | 264 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 140 |
| PARTIAL | 25 |
| TARGET_HIT | 3 |
| STOP_HIT | 148 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 176 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 116
- **Target hits / Stop hits / Partials:** 3 / 148 / 25
- **Avg / median % per leg:** 0.25% / -0.97%
- **Sum % (uncompounded):** 44.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 4 | 7.0% | 2 | 55 | 0 | -1.36% | -77.5% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.65% | -3.3% |
| BUY @ 3rd Alert (retest2) | 52 | 4 | 7.7% | 2 | 50 | 0 | -1.43% | -74.3% |
| SELL (all) | 119 | 56 | 47.1% | 1 | 93 | 25 | 1.03% | 122.3% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.54% | -15.3% |
| SELL @ 3rd Alert (retest2) | 113 | 56 | 49.6% | 1 | 87 | 25 | 1.22% | 137.5% |
| retest1 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.68% | -18.5% |
| retest2 (combined) | 165 | 60 | 36.4% | 3 | 137 | 25 | 0.38% | 63.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1369.20 | 1332.30 | 1332.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 1374.50 | 1359.03 | 1348.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 1359.80 | 1361.02 | 1351.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 1359.80 | 1361.02 | 1351.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1351.00 | 1359.87 | 1355.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 1352.15 | 1359.87 | 1355.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1346.20 | 1357.13 | 1354.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 1346.20 | 1357.13 | 1354.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 1336.70 | 1353.05 | 1353.32 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 1376.05 | 1356.50 | 1354.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 1408.40 | 1382.11 | 1371.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 1390.15 | 1403.30 | 1390.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 1390.15 | 1403.30 | 1390.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1390.15 | 1403.30 | 1390.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 1390.15 | 1403.30 | 1390.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1378.55 | 1398.35 | 1389.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 1377.20 | 1398.35 | 1389.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1385.65 | 1395.81 | 1388.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 1425.40 | 1386.88 | 1386.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 1408.00 | 1425.36 | 1415.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 1406.30 | 1412.46 | 1412.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 1406.30 | 1412.46 | 1412.68 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 09:15:00 | 1447.55 | 1414.33 | 1413.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 1458.65 | 1436.80 | 1427.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 1439.00 | 1453.03 | 1441.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 15:15:00 | 1439.00 | 1453.03 | 1441.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1439.00 | 1453.03 | 1441.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1439.65 | 1453.03 | 1441.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1412.60 | 1444.95 | 1438.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 1412.60 | 1444.95 | 1438.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1426.70 | 1441.30 | 1437.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 11:45:00 | 1464.70 | 1445.36 | 1439.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 1511.45 | 1445.21 | 1442.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 1300.25 | 1455.21 | 1469.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1300.25 | 1455.21 | 1469.52 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1469.05 | 1444.40 | 1443.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1492.00 | 1461.21 | 1452.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 1581.05 | 1588.61 | 1568.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 1581.05 | 1588.61 | 1568.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1592.00 | 1591.38 | 1578.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:00:00 | 1606.65 | 1599.42 | 1591.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 14:15:00 | 1586.00 | 1589.06 | 1589.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1586.00 | 1589.06 | 1589.34 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 1604.55 | 1591.49 | 1590.35 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 1577.25 | 1592.50 | 1592.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 1485.65 | 1570.17 | 1582.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 1506.85 | 1502.43 | 1523.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:30:00 | 1504.95 | 1502.43 | 1523.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1508.00 | 1506.96 | 1519.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:45:00 | 1498.00 | 1504.77 | 1516.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 1502.25 | 1506.07 | 1512.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 1502.35 | 1510.62 | 1513.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 1500.05 | 1511.18 | 1512.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 1474.30 | 1468.71 | 1479.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:30:00 | 1467.65 | 1468.71 | 1479.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1491.75 | 1474.15 | 1479.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 1488.00 | 1474.15 | 1479.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1494.30 | 1478.18 | 1481.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:45:00 | 1495.80 | 1478.18 | 1481.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 1495.00 | 1484.09 | 1483.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 1495.00 | 1484.09 | 1483.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 15:15:00 | 1498.00 | 1488.61 | 1485.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 10:15:00 | 1502.30 | 1507.45 | 1500.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 10:15:00 | 1502.30 | 1507.45 | 1500.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1502.30 | 1507.45 | 1500.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:45:00 | 1501.00 | 1507.45 | 1500.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1496.55 | 1504.73 | 1500.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 1496.55 | 1504.73 | 1500.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1497.70 | 1503.33 | 1500.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:30:00 | 1509.00 | 1504.59 | 1500.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 1474.15 | 1498.11 | 1498.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 1474.15 | 1498.11 | 1498.57 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 1521.85 | 1497.57 | 1494.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 15:15:00 | 1523.00 | 1511.33 | 1502.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 1524.00 | 1531.86 | 1520.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 1524.00 | 1531.86 | 1520.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1524.50 | 1530.39 | 1520.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 1521.45 | 1530.39 | 1520.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1520.00 | 1528.31 | 1520.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:45:00 | 1518.65 | 1528.31 | 1520.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1524.50 | 1527.55 | 1520.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:00:00 | 1528.20 | 1527.68 | 1521.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:30:00 | 1530.00 | 1528.31 | 1522.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 1512.55 | 1525.41 | 1522.10 | SL hit (close<static) qty=1.00 sl=1519.95 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 1510.00 | 1520.66 | 1520.69 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 1538.25 | 1521.25 | 1519.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 1539.95 | 1527.19 | 1522.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 12:15:00 | 1544.10 | 1557.39 | 1543.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 1544.10 | 1557.39 | 1543.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1544.10 | 1557.39 | 1543.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 1544.10 | 1557.39 | 1543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1552.60 | 1556.43 | 1544.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:30:00 | 1537.05 | 1556.43 | 1544.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1543.75 | 1553.89 | 1544.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 1534.25 | 1553.89 | 1544.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1547.00 | 1552.52 | 1544.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1563.05 | 1552.52 | 1544.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 1516.95 | 1537.91 | 1539.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1516.95 | 1537.91 | 1539.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 13:15:00 | 1509.80 | 1522.43 | 1529.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 1532.00 | 1520.68 | 1526.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 1532.00 | 1520.68 | 1526.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1532.00 | 1520.68 | 1526.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1481.55 | 1525.56 | 1527.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 1536.50 | 1521.56 | 1521.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1536.50 | 1521.56 | 1521.26 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 14:15:00 | 1520.00 | 1520.85 | 1520.96 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 1526.25 | 1521.65 | 1521.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 1564.35 | 1536.92 | 1529.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 10:15:00 | 1556.95 | 1557.48 | 1546.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 11:00:00 | 1556.95 | 1557.48 | 1546.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1814.30 | 1834.51 | 1808.45 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 1772.90 | 1793.48 | 1794.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 1750.00 | 1781.51 | 1788.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1795.55 | 1784.32 | 1789.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1795.55 | 1784.32 | 1789.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1795.55 | 1784.32 | 1789.38 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 1814.00 | 1793.71 | 1792.99 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1778.80 | 1792.57 | 1792.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1756.00 | 1785.26 | 1789.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 15:15:00 | 1789.95 | 1786.20 | 1789.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 15:15:00 | 1789.95 | 1786.20 | 1789.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1789.95 | 1786.20 | 1789.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:15:00 | 1777.40 | 1786.20 | 1789.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1767.00 | 1782.36 | 1787.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:00:00 | 1763.65 | 1775.40 | 1781.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 1762.30 | 1772.78 | 1779.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 1761.90 | 1770.93 | 1777.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1844.55 | 1779.84 | 1779.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1844.55 | 1779.84 | 1779.74 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 1762.35 | 1778.56 | 1780.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 1759.45 | 1774.74 | 1778.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 1726.00 | 1700.85 | 1716.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 1726.00 | 1700.85 | 1716.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1726.00 | 1700.85 | 1716.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 1726.00 | 1700.85 | 1716.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 1709.70 | 1702.62 | 1716.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 1699.95 | 1702.09 | 1714.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:45:00 | 1707.70 | 1698.63 | 1707.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 1743.00 | 1693.14 | 1697.39 | SL hit (close>static) qty=1.00 sl=1733.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 1765.55 | 1707.62 | 1703.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 1797.90 | 1745.57 | 1723.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 1756.95 | 1757.62 | 1735.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 1756.95 | 1757.62 | 1735.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1750.15 | 1756.13 | 1737.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:45:00 | 1739.00 | 1756.13 | 1737.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1732.20 | 1751.34 | 1736.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 1732.20 | 1751.34 | 1736.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1723.00 | 1745.67 | 1735.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:30:00 | 1723.40 | 1745.67 | 1735.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1713.45 | 1730.77 | 1730.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:15:00 | 1708.25 | 1730.77 | 1730.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 11:15:00 | 1708.10 | 1726.24 | 1728.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 1694.30 | 1719.85 | 1725.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 1660.10 | 1646.58 | 1664.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 10:15:00 | 1660.10 | 1646.58 | 1664.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1660.10 | 1646.58 | 1664.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 1660.10 | 1646.58 | 1664.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1659.90 | 1649.24 | 1664.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:30:00 | 1661.20 | 1649.24 | 1664.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1686.00 | 1656.59 | 1666.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:30:00 | 1671.80 | 1656.59 | 1666.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1679.90 | 1661.26 | 1667.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 1672.65 | 1667.28 | 1669.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 1688.30 | 1674.56 | 1672.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 1688.30 | 1674.56 | 1672.78 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1663.00 | 1671.85 | 1672.54 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 1704.85 | 1675.68 | 1672.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1746.05 | 1695.46 | 1682.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 09:15:00 | 1737.00 | 1739.05 | 1718.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 10:00:00 | 1737.00 | 1739.05 | 1718.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1725.10 | 1736.26 | 1719.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:45:00 | 1732.35 | 1736.26 | 1719.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1738.35 | 1736.67 | 1720.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:45:00 | 1720.00 | 1736.67 | 1720.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1712.45 | 1733.60 | 1723.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1712.45 | 1733.60 | 1723.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1712.50 | 1729.38 | 1722.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1688.50 | 1729.38 | 1722.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 1689.20 | 1714.58 | 1716.56 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 1738.80 | 1718.48 | 1717.42 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 1694.00 | 1714.03 | 1715.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 1690.90 | 1703.76 | 1710.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 14:15:00 | 1690.45 | 1686.70 | 1695.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 14:15:00 | 1690.45 | 1686.70 | 1695.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1690.45 | 1686.70 | 1695.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 1700.00 | 1686.70 | 1695.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 1686.00 | 1686.56 | 1694.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 1700.50 | 1686.56 | 1694.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1704.15 | 1690.08 | 1695.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:30:00 | 1678.40 | 1688.68 | 1693.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 1683.45 | 1680.92 | 1686.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1683.15 | 1686.74 | 1688.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 1694.75 | 1688.89 | 1688.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 1694.75 | 1688.89 | 1688.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 1721.75 | 1702.40 | 1696.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 1749.05 | 1751.23 | 1735.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 1749.05 | 1751.23 | 1735.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1774.00 | 1787.80 | 1774.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 1824.80 | 1787.80 | 1774.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 10:15:00 | 1866.05 | 1878.67 | 1880.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1866.05 | 1878.67 | 1880.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 1861.95 | 1875.33 | 1878.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1892.30 | 1875.17 | 1877.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 1892.30 | 1875.17 | 1877.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1892.30 | 1875.17 | 1877.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1892.30 | 1875.17 | 1877.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1884.00 | 1876.93 | 1877.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1904.70 | 1876.93 | 1877.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1914.80 | 1884.51 | 1881.23 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 1879.05 | 1887.85 | 1887.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 15:15:00 | 1874.15 | 1882.38 | 1885.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 15:15:00 | 1868.80 | 1868.33 | 1875.05 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1850.60 | 1868.33 | 1875.05 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:00:00 | 1850.85 | 1864.83 | 1872.85 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 1846.20 | 1862.79 | 1871.19 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 13:00:00 | 1850.50 | 1859.46 | 1868.16 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1844.95 | 1855.03 | 1863.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1887.30 | 1861.49 | 1865.31 | SL hit (close>ema400) qty=1.00 sl=1865.31 alert=retest1 |

### Cycle 37 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 1895.80 | 1868.35 | 1868.08 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1848.25 | 1868.37 | 1869.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 1815.35 | 1853.91 | 1862.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 1816.20 | 1815.65 | 1831.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:45:00 | 1815.20 | 1815.65 | 1831.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 39 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1964.55 | 1845.80 | 1842.11 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 1866.25 | 1881.98 | 1882.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 1854.30 | 1874.08 | 1878.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 1890.05 | 1877.27 | 1879.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 1890.05 | 1877.27 | 1879.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1890.05 | 1877.27 | 1879.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 1890.05 | 1877.27 | 1879.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1890.00 | 1879.82 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1960.05 | 1879.82 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 1945.55 | 1892.97 | 1886.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 09:15:00 | 1969.75 | 1947.87 | 1944.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 13:15:00 | 1966.55 | 1967.59 | 1960.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 14:00:00 | 1966.55 | 1967.59 | 1960.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1958.05 | 1965.34 | 1960.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1988.05 | 1965.34 | 1960.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 13:15:00 | 1947.40 | 1970.92 | 1967.17 | SL hit (close<static) qty=1.00 sl=1958.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 1944.00 | 1965.03 | 1965.11 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 2004.90 | 1968.43 | 1966.36 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 1949.10 | 1967.97 | 1969.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1914.40 | 1952.45 | 1961.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 1849.85 | 1847.11 | 1870.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 1849.85 | 1847.11 | 1870.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1849.85 | 1847.11 | 1870.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 1861.60 | 1847.11 | 1870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1843.00 | 1828.43 | 1838.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1843.00 | 1828.43 | 1838.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1783.95 | 1744.38 | 1763.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 1783.95 | 1744.38 | 1763.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 1785.10 | 1752.53 | 1765.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 1785.10 | 1752.53 | 1765.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1760.00 | 1766.17 | 1769.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1766.85 | 1766.17 | 1769.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1762.10 | 1765.36 | 1768.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1762.10 | 1765.36 | 1768.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1729.85 | 1753.59 | 1761.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 1693.80 | 1727.58 | 1745.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:30:00 | 1697.95 | 1711.77 | 1732.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:45:00 | 1697.05 | 1707.32 | 1727.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 1609.11 | 1644.11 | 1665.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 1613.05 | 1644.11 | 1665.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 1612.20 | 1644.11 | 1665.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 1576.70 | 1576.21 | 1605.65 | SL hit (close>ema200) qty=0.50 sl=1576.21 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1618.50 | 1553.64 | 1552.23 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 1525.80 | 1553.52 | 1556.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 1505.80 | 1525.56 | 1536.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 1527.50 | 1524.26 | 1533.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 1527.50 | 1524.26 | 1533.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1527.50 | 1524.26 | 1533.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 1551.40 | 1524.26 | 1533.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1521.30 | 1523.67 | 1532.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 1532.45 | 1523.67 | 1532.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1518.00 | 1523.14 | 1530.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 1515.00 | 1523.14 | 1530.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 15:00:00 | 1511.35 | 1520.48 | 1528.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1606.95 | 1535.62 | 1533.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1606.95 | 1535.62 | 1533.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 10:15:00 | 1608.35 | 1550.16 | 1540.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 1659.25 | 1659.63 | 1629.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:45:00 | 1660.70 | 1659.63 | 1629.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1664.95 | 1673.35 | 1655.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 1657.40 | 1673.35 | 1655.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1653.45 | 1666.97 | 1658.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:00:00 | 1653.45 | 1666.97 | 1658.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 1641.05 | 1661.79 | 1656.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:00:00 | 1641.05 | 1661.79 | 1656.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 1644.50 | 1651.82 | 1652.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 10:15:00 | 1636.00 | 1648.65 | 1651.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 1649.20 | 1648.76 | 1651.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 11:15:00 | 1649.20 | 1648.76 | 1651.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1649.20 | 1648.76 | 1651.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 1656.80 | 1648.76 | 1651.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1646.50 | 1648.31 | 1650.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:30:00 | 1653.85 | 1648.31 | 1650.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1647.75 | 1648.32 | 1650.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 1640.15 | 1648.32 | 1650.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1649.00 | 1648.45 | 1650.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1633.05 | 1648.45 | 1650.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1630.00 | 1644.76 | 1648.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 10:00:00 | 1614.55 | 1627.37 | 1636.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 1667.75 | 1640.71 | 1639.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1667.75 | 1640.71 | 1639.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 13:15:00 | 1687.00 | 1659.69 | 1650.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 14:15:00 | 1669.80 | 1670.24 | 1662.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 1669.80 | 1670.24 | 1662.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1669.80 | 1670.24 | 1662.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:45:00 | 1661.15 | 1670.24 | 1662.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1656.00 | 1667.51 | 1662.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 1656.00 | 1667.51 | 1662.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1648.75 | 1663.76 | 1661.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 1648.75 | 1663.76 | 1661.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1655.30 | 1662.07 | 1660.80 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 1648.05 | 1658.63 | 1659.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 14:15:00 | 1638.25 | 1654.55 | 1657.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 1610.65 | 1608.95 | 1623.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 15:00:00 | 1610.65 | 1608.95 | 1623.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1620.00 | 1609.64 | 1619.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:45:00 | 1625.65 | 1609.64 | 1619.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1614.60 | 1610.63 | 1619.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 1603.45 | 1617.07 | 1620.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 1523.28 | 1573.85 | 1597.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 1539.10 | 1530.35 | 1559.96 | SL hit (close>ema200) qty=0.50 sl=1530.35 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1491.20 | 1463.86 | 1463.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 1500.00 | 1484.16 | 1476.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1514.40 | 1516.94 | 1502.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 1514.40 | 1516.94 | 1502.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1498.35 | 1510.81 | 1505.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 1498.25 | 1510.81 | 1505.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1480.00 | 1504.65 | 1503.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1480.00 | 1504.65 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1475.15 | 1498.75 | 1500.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1465.95 | 1488.24 | 1495.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 1467.00 | 1464.86 | 1475.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:00:00 | 1467.00 | 1464.86 | 1475.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1352.00 | 1336.03 | 1365.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1349.70 | 1336.03 | 1365.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1367.05 | 1342.23 | 1365.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1367.05 | 1342.23 | 1365.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1364.90 | 1346.76 | 1365.23 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1404.65 | 1375.77 | 1372.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 1428.60 | 1396.35 | 1384.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 1475.65 | 1481.09 | 1456.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 1475.65 | 1481.09 | 1456.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1463.20 | 1477.34 | 1469.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:45:00 | 1463.50 | 1477.34 | 1469.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1443.80 | 1470.63 | 1466.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1443.80 | 1470.63 | 1466.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1444.40 | 1461.63 | 1463.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1440.15 | 1454.23 | 1459.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1444.60 | 1409.96 | 1425.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1444.60 | 1409.96 | 1425.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1444.60 | 1409.96 | 1425.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1444.60 | 1409.96 | 1425.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1444.55 | 1416.87 | 1427.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1444.00 | 1416.87 | 1427.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1442.25 | 1429.03 | 1430.04 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1467.50 | 1436.72 | 1433.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 1482.75 | 1445.93 | 1437.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1431.55 | 1453.40 | 1445.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 1431.55 | 1453.40 | 1445.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1431.55 | 1453.40 | 1445.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:30:00 | 1441.55 | 1453.40 | 1445.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1427.25 | 1448.17 | 1444.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 1425.60 | 1448.17 | 1444.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 1435.20 | 1442.14 | 1442.16 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 14:15:00 | 1454.70 | 1444.65 | 1443.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 10:15:00 | 1474.70 | 1453.43 | 1447.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 15:15:00 | 1467.15 | 1469.38 | 1459.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 15:15:00 | 1467.15 | 1469.38 | 1459.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 1467.15 | 1469.38 | 1459.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 09:45:00 | 1474.20 | 1469.77 | 1460.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 10:45:00 | 1475.20 | 1471.06 | 1461.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 15:00:00 | 1479.25 | 1474.77 | 1466.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 14:15:00 | 1452.80 | 1466.84 | 1466.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 1452.80 | 1466.84 | 1466.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 1450.40 | 1463.55 | 1465.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 1467.10 | 1464.26 | 1465.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 1467.10 | 1464.26 | 1465.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1467.10 | 1464.26 | 1465.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 11:00:00 | 1464.35 | 1464.28 | 1465.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 10:15:00 | 1472.50 | 1465.45 | 1464.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 1472.50 | 1465.45 | 1464.56 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1414.95 | 1454.92 | 1459.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 13:15:00 | 1408.50 | 1445.63 | 1455.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 1354.00 | 1351.68 | 1388.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:30:00 | 1320.45 | 1345.70 | 1379.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1366.00 | 1348.88 | 1367.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 1396.30 | 1348.88 | 1367.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1396.00 | 1358.30 | 1370.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 1396.00 | 1358.30 | 1370.29 | SL hit (close>ema400) qty=1.00 sl=1370.29 alert=retest1 |

### Cycle 61 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 1376.45 | 1366.17 | 1364.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 1379.20 | 1368.78 | 1366.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1385.50 | 1389.21 | 1379.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 09:30:00 | 1381.30 | 1389.21 | 1379.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1376.55 | 1386.68 | 1379.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1376.55 | 1386.68 | 1379.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1361.65 | 1381.67 | 1377.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 1361.65 | 1381.67 | 1377.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1355.00 | 1376.34 | 1375.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:30:00 | 1361.85 | 1376.34 | 1375.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 1349.50 | 1370.97 | 1373.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 1348.30 | 1366.44 | 1371.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1274.45 | 1272.95 | 1296.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 1276.95 | 1272.95 | 1296.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1241.15 | 1260.13 | 1278.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1231.35 | 1260.13 | 1278.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 1225.00 | 1247.79 | 1269.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 1235.85 | 1238.21 | 1258.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:45:00 | 1232.85 | 1228.04 | 1243.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 1234.75 | 1223.94 | 1232.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 1234.75 | 1223.94 | 1232.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 1233.50 | 1225.85 | 1232.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 1244.30 | 1225.85 | 1232.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1251.00 | 1230.88 | 1234.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1251.00 | 1230.88 | 1234.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1247.15 | 1234.13 | 1235.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1257.55 | 1234.13 | 1235.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 1249.05 | 1237.12 | 1236.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1249.05 | 1237.12 | 1236.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1258.45 | 1244.05 | 1240.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 1239.45 | 1244.64 | 1241.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 1239.45 | 1244.64 | 1241.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1239.45 | 1244.64 | 1241.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:00:00 | 1259.45 | 1247.60 | 1243.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 13:45:00 | 1260.15 | 1250.66 | 1245.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:15:00 | 1268.65 | 1250.66 | 1245.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:30:00 | 1266.00 | 1256.41 | 1250.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1274.45 | 1272.39 | 1262.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:15:00 | 1279.15 | 1272.39 | 1262.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:00:00 | 1279.25 | 1273.76 | 1263.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1244.80 | 1297.40 | 1299.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 1244.80 | 1297.40 | 1299.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 1234.35 | 1264.01 | 1277.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 1262.90 | 1258.40 | 1270.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 1268.55 | 1260.43 | 1270.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1268.55 | 1260.43 | 1270.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 1270.55 | 1260.43 | 1270.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 1263.05 | 1260.95 | 1269.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 1251.95 | 1260.95 | 1269.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:00:00 | 1259.00 | 1260.56 | 1268.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:00:00 | 1260.30 | 1260.51 | 1267.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 1257.00 | 1257.68 | 1265.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1268.50 | 1257.77 | 1264.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 1268.50 | 1257.77 | 1264.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1278.85 | 1261.99 | 1265.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 1278.85 | 1261.99 | 1265.60 | SL hit (close>static) qty=1.00 sl=1270.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1301.35 | 1271.46 | 1269.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 1313.95 | 1291.17 | 1280.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 1329.45 | 1329.84 | 1317.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1346.10 | 1329.84 | 1317.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1333.00 | 1350.61 | 1337.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1333.00 | 1350.61 | 1337.62 | SL hit (close<ema400) qty=1.00 sl=1337.62 alert=retest1 |

### Cycle 66 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1325.80 | 1334.37 | 1334.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 1315.35 | 1330.57 | 1332.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1325.45 | 1319.96 | 1325.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1325.45 | 1319.96 | 1325.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1325.45 | 1319.96 | 1325.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:30:00 | 1323.95 | 1319.96 | 1325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1313.20 | 1318.61 | 1324.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 1307.15 | 1318.61 | 1324.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:00:00 | 1310.00 | 1305.75 | 1311.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 1310.65 | 1306.73 | 1310.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1333.75 | 1313.35 | 1313.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1333.75 | 1313.35 | 1313.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 1356.60 | 1325.65 | 1319.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1512.05 | 1521.22 | 1494.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 1510.20 | 1521.22 | 1494.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1503.45 | 1515.82 | 1496.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1490.20 | 1515.82 | 1496.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1493.20 | 1511.29 | 1496.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1498.75 | 1511.29 | 1496.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1495.00 | 1508.04 | 1496.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:15:00 | 1481.05 | 1508.04 | 1496.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1495.00 | 1502.61 | 1495.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1511.40 | 1502.61 | 1495.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 1485.75 | 1498.51 | 1497.68 | SL hit (close<static) qty=1.00 sl=1490.40 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 10:15:00 | 1488.20 | 1496.45 | 1496.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 1479.50 | 1493.06 | 1495.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1503.25 | 1491.19 | 1493.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1503.25 | 1491.19 | 1493.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1503.25 | 1491.19 | 1493.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1503.25 | 1491.19 | 1493.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1498.00 | 1492.55 | 1493.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1506.85 | 1492.55 | 1493.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1520.10 | 1498.06 | 1496.22 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 1485.00 | 1496.11 | 1496.98 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 1499.45 | 1497.00 | 1496.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 1507.80 | 1499.48 | 1498.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 11:15:00 | 1501.50 | 1501.63 | 1499.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:45:00 | 1501.75 | 1501.63 | 1499.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1498.20 | 1500.95 | 1499.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 1497.60 | 1500.95 | 1499.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1496.50 | 1500.06 | 1499.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:45:00 | 1496.70 | 1500.06 | 1499.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1508.95 | 1501.83 | 1499.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1522.25 | 1502.53 | 1500.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 1517.70 | 1531.35 | 1521.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:15:00 | 1530.00 | 1528.09 | 1520.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1463.70 | 1518.19 | 1519.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1463.70 | 1518.19 | 1519.28 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 1521.50 | 1505.70 | 1505.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 1535.35 | 1512.36 | 1508.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 1607.90 | 1608.73 | 1583.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:00:00 | 1607.90 | 1608.73 | 1583.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1584.80 | 1599.82 | 1588.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 1584.80 | 1599.82 | 1588.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1587.80 | 1597.42 | 1588.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 1613.10 | 1597.61 | 1591.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 1600.10 | 1600.88 | 1594.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 1614.60 | 1596.97 | 1594.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:00:00 | 1599.90 | 1597.55 | 1594.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1599.90 | 1601.22 | 1597.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:45:00 | 1599.90 | 1601.22 | 1597.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 1589.70 | 1598.91 | 1597.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 1592.50 | 1598.91 | 1597.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 1588.00 | 1596.73 | 1596.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1599.40 | 1596.73 | 1596.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1575.90 | 1592.57 | 1594.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 1575.90 | 1592.57 | 1594.44 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 1605.30 | 1594.82 | 1594.73 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1563.10 | 1593.48 | 1596.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1541.90 | 1583.16 | 1591.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1582.30 | 1559.47 | 1572.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 1582.30 | 1559.47 | 1572.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1582.30 | 1559.47 | 1572.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 1582.30 | 1559.47 | 1572.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1559.40 | 1559.45 | 1571.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:15:00 | 1551.30 | 1559.71 | 1569.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:00:00 | 1554.10 | 1558.59 | 1568.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 1556.60 | 1564.98 | 1567.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:45:00 | 1554.60 | 1562.58 | 1566.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1557.00 | 1557.78 | 1562.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:00:00 | 1542.00 | 1553.01 | 1559.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:45:00 | 1539.80 | 1550.39 | 1557.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:30:00 | 1540.10 | 1543.32 | 1551.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 1539.30 | 1543.32 | 1551.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1533.80 | 1541.42 | 1549.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:45:00 | 1536.10 | 1541.42 | 1549.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1473.73 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1476.39 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1478.77 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1476.87 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1464.90 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1462.81 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1463.09 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 1462.33 | 1501.64 | 1523.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1487.90 | 1485.09 | 1502.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 1487.90 | 1485.09 | 1502.49 | SL hit (close>ema200) qty=0.50 sl=1485.09 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 1466.60 | 1442.05 | 1439.57 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1433.60 | 1439.95 | 1439.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 14:15:00 | 1425.50 | 1435.61 | 1437.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1451.60 | 1439.25 | 1438.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1457.70 | 1446.82 | 1442.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 1425.80 | 1440.14 | 1440.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 11:15:00 | 1412.50 | 1434.61 | 1438.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 1430.20 | 1429.37 | 1434.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1435.00 | 1430.50 | 1434.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1459.20 | 1430.50 | 1434.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1438.90 | 1432.18 | 1434.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 1443.20 | 1432.18 | 1434.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1443.00 | 1434.34 | 1435.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:45:00 | 1434.50 | 1435.69 | 1436.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 1441.00 | 1436.76 | 1436.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1441.00 | 1436.76 | 1436.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1465.10 | 1444.23 | 1440.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1433.60 | 1443.27 | 1441.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 1424.50 | 1439.52 | 1439.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 1419.20 | 1433.42 | 1436.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1412.00 | 1411.98 | 1422.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 1412.00 | 1411.98 | 1422.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1400.30 | 1397.19 | 1403.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 1399.40 | 1397.19 | 1403.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1400.20 | 1397.80 | 1402.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:45:00 | 1392.50 | 1398.13 | 1402.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1429.50 | 1407.09 | 1405.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1429.50 | 1407.09 | 1405.66 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 1405.20 | 1414.67 | 1415.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 1401.20 | 1411.97 | 1414.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1398.20 | 1396.98 | 1404.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1402.10 | 1397.80 | 1403.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1402.10 | 1397.80 | 1403.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1383.30 | 1395.09 | 1401.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1382.00 | 1395.09 | 1401.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 1381.30 | 1378.50 | 1387.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1381.50 | 1379.10 | 1387.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1425.40 | 1408.75 | 1401.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 1416.10 | 1416.57 | 1409.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:00:00 | 1416.10 | 1416.57 | 1409.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1413.00 | 1416.95 | 1412.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 1413.00 | 1416.95 | 1412.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1410.00 | 1415.56 | 1412.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:45:00 | 1408.10 | 1415.56 | 1412.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1412.00 | 1414.85 | 1412.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1414.60 | 1414.85 | 1412.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1418.90 | 1415.66 | 1412.64 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 1409.80 | 1411.24 | 1411.42 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1418.00 | 1412.59 | 1412.02 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1402.90 | 1410.65 | 1411.19 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1421.40 | 1412.54 | 1411.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 1427.00 | 1415.44 | 1412.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1424.50 | 1442.84 | 1433.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1432.90 | 1440.86 | 1433.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1434.00 | 1440.86 | 1433.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 1434.40 | 1438.27 | 1433.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1421.70 | 1434.95 | 1432.08 | SL hit (close<static) qty=1.00 sl=1422.50 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1419.70 | 1429.27 | 1429.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1403.00 | 1424.02 | 1427.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1416.00 | 1412.15 | 1417.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1416.00 | 1412.15 | 1417.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1408.40 | 1411.54 | 1416.32 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1423.10 | 1417.99 | 1417.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1435.10 | 1421.41 | 1419.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 1415.30 | 1420.19 | 1418.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1418.70 | 1419.89 | 1418.94 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 1412.60 | 1417.57 | 1417.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1408.20 | 1415.70 | 1417.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1383.70 | 1377.09 | 1387.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1383.60 | 1378.39 | 1386.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1385.20 | 1378.39 | 1386.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1388.00 | 1380.32 | 1386.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1388.00 | 1380.32 | 1386.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1385.50 | 1381.35 | 1386.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1386.00 | 1381.35 | 1386.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1389.80 | 1383.04 | 1387.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:30:00 | 1391.70 | 1383.04 | 1387.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1383.00 | 1383.03 | 1386.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1395.90 | 1386.53 | 1387.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1407.10 | 1390.64 | 1389.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1407.60 | 1394.03 | 1391.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 1465.00 | 1465.47 | 1453.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:30:00 | 1458.80 | 1465.47 | 1453.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1468.30 | 1475.66 | 1467.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1453.30 | 1475.66 | 1467.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1465.80 | 1473.69 | 1467.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1465.80 | 1473.69 | 1467.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1460.30 | 1471.01 | 1466.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:45:00 | 1459.90 | 1471.01 | 1466.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1466.50 | 1466.07 | 1465.25 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1452.40 | 1463.34 | 1464.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1448.00 | 1460.27 | 1462.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 1457.20 | 1457.15 | 1460.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 1457.20 | 1457.15 | 1460.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1450.00 | 1455.72 | 1459.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1459.20 | 1455.72 | 1459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1448.70 | 1454.32 | 1458.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 1448.00 | 1453.45 | 1457.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 1463.50 | 1455.12 | 1457.55 | SL hit (close>static) qty=1.00 sl=1463.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 1467.20 | 1460.54 | 1459.66 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 1451.30 | 1459.58 | 1459.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1444.50 | 1456.56 | 1458.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 1442.30 | 1441.24 | 1447.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1443.50 | 1441.69 | 1447.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1443.50 | 1441.69 | 1447.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1443.50 | 1441.69 | 1447.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1439.70 | 1441.66 | 1446.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 1435.90 | 1440.51 | 1445.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 1436.10 | 1440.30 | 1444.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1438.10 | 1440.77 | 1443.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 1436.00 | 1429.96 | 1430.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1429.20 | 1430.49 | 1431.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1430.40 | 1430.49 | 1431.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1416.20 | 1426.88 | 1429.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1411.60 | 1426.88 | 1429.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 1410.30 | 1418.54 | 1423.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 1399.00 | 1418.54 | 1423.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 1412.70 | 1414.67 | 1420.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1406.10 | 1398.24 | 1403.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1406.10 | 1398.24 | 1403.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1399.30 | 1398.45 | 1403.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1397.80 | 1398.45 | 1403.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.11 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.29 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1366.19 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.20 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1371.40 | 1371.06 | 1378.27 | SL hit (close>ema200) qty=0.50 sl=1371.06 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1356.00 | 1339.09 | 1336.83 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1316.00 | 1335.09 | 1336.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1311.50 | 1322.80 | 1329.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1308.00 | 1295.29 | 1304.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1311.40 | 1298.51 | 1305.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1311.40 | 1298.51 | 1305.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1315.80 | 1304.43 | 1306.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 1315.80 | 1304.43 | 1306.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1292.70 | 1302.08 | 1305.09 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1319.00 | 1305.88 | 1305.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 1320.60 | 1308.83 | 1306.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 1340.90 | 1344.12 | 1333.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:45:00 | 1340.30 | 1344.12 | 1333.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1339.50 | 1343.20 | 1334.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 1340.20 | 1343.20 | 1334.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1333.50 | 1341.66 | 1335.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1333.50 | 1341.66 | 1335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1330.60 | 1339.45 | 1335.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 1339.00 | 1339.45 | 1335.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1346.00 | 1340.76 | 1336.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 1356.80 | 1345.56 | 1341.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1355.00 | 1345.56 | 1341.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1355.80 | 1349.03 | 1344.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 1354.10 | 1351.06 | 1345.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1349.40 | 1356.13 | 1352.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 1335.80 | 1347.17 | 1349.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1301.70 | 1293.71 | 1301.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1295.50 | 1294.07 | 1300.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 1287.00 | 1295.93 | 1299.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1282.00 | 1258.72 | 1256.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1282.00 | 1258.72 | 1256.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1287.90 | 1264.56 | 1258.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 1295.20 | 1295.99 | 1282.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 1295.20 | 1295.99 | 1282.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1292.90 | 1305.46 | 1298.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1292.90 | 1305.46 | 1298.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1296.80 | 1303.73 | 1298.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1288.80 | 1303.73 | 1298.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1276.00 | 1298.18 | 1296.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1276.00 | 1298.18 | 1296.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1271.30 | 1292.81 | 1294.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1264.80 | 1287.20 | 1291.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1266.60 | 1261.79 | 1261.51 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 1261.60 | 1266.71 | 1267.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 13:15:00 | 1260.20 | 1265.41 | 1266.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1265.20 | 1263.06 | 1264.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1267.00 | 1263.85 | 1264.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1266.30 | 1263.85 | 1264.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1265.00 | 1264.08 | 1264.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 1264.20 | 1264.08 | 1264.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 1264.60 | 1264.18 | 1264.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1279.00 | 1267.09 | 1265.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 1279.00 | 1267.09 | 1265.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 1283.70 | 1270.42 | 1267.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 1273.50 | 1274.81 | 1270.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 1273.50 | 1274.81 | 1270.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1268.00 | 1273.45 | 1270.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1268.00 | 1273.45 | 1270.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1268.80 | 1272.52 | 1270.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1268.80 | 1272.52 | 1270.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1268.50 | 1271.71 | 1270.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1268.50 | 1271.71 | 1270.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1264.00 | 1270.17 | 1269.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1264.00 | 1270.17 | 1269.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1279.10 | 1273.73 | 1271.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1284.20 | 1274.42 | 1272.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 11:45:00 | 1283.80 | 1277.73 | 1274.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 1280.90 | 1278.18 | 1274.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1282.30 | 1279.19 | 1276.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1275.50 | 1278.45 | 1276.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1275.50 | 1278.45 | 1276.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1274.90 | 1277.74 | 1276.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1274.90 | 1277.74 | 1276.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1271.00 | 1276.39 | 1275.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1271.00 | 1276.39 | 1275.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1273.70 | 1275.85 | 1275.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1271.50 | 1275.85 | 1275.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1263.60 | 1272.47 | 1273.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 1231.90 | 1231.03 | 1238.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:45:00 | 1231.70 | 1231.03 | 1238.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1224.00 | 1218.45 | 1224.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 1212.00 | 1217.77 | 1222.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1211.90 | 1217.76 | 1220.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1213.50 | 1217.76 | 1220.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1243.80 | 1237.86 | 1232.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1238.30 | 1239.08 | 1234.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:45:00 | 1237.40 | 1239.08 | 1234.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1233.10 | 1237.40 | 1234.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1239.60 | 1238.98 | 1235.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1294.00 | 1315.11 | 1317.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1294.00 | 1315.11 | 1317.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1288.60 | 1307.66 | 1313.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1312.70 | 1296.26 | 1304.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1313.30 | 1299.67 | 1304.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1312.90 | 1299.67 | 1304.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 1316.40 | 1307.82 | 1307.64 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1305.80 | 1307.29 | 1307.48 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 1310.20 | 1308.05 | 1307.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1315.90 | 1309.62 | 1308.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1305.00 | 1309.75 | 1308.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1310.00 | 1309.80 | 1308.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1309.70 | 1309.80 | 1308.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1315.10 | 1310.86 | 1309.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 1327.20 | 1318.52 | 1313.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 1303.10 | 1321.06 | 1322.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 1303.10 | 1321.06 | 1322.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 15:15:00 | 1302.00 | 1314.49 | 1318.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1276.80 | 1275.71 | 1289.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:30:00 | 1277.60 | 1275.71 | 1289.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1279.60 | 1277.33 | 1287.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1282.80 | 1277.33 | 1287.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1290.20 | 1280.72 | 1286.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1286.90 | 1280.72 | 1286.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1291.40 | 1282.86 | 1287.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1291.40 | 1282.86 | 1287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1305.90 | 1290.05 | 1289.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1311.90 | 1294.42 | 1291.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 1289.90 | 1304.51 | 1306.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1287.60 | 1296.35 | 1300.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1298.00 | 1293.20 | 1296.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1306.60 | 1295.88 | 1297.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1314.40 | 1295.88 | 1297.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1320.90 | 1300.89 | 1299.69 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 1301.60 | 1306.71 | 1306.97 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1313.80 | 1307.78 | 1307.24 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1295.30 | 1306.30 | 1307.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1289.70 | 1296.86 | 1301.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 1297.40 | 1296.25 | 1300.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 1297.40 | 1296.25 | 1300.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1302.60 | 1297.52 | 1300.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1302.60 | 1297.52 | 1300.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1303.60 | 1298.74 | 1300.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:30:00 | 1301.10 | 1298.74 | 1300.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1300.00 | 1298.99 | 1300.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1294.60 | 1298.99 | 1300.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:30:00 | 1298.30 | 1299.12 | 1300.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1297.90 | 1299.47 | 1300.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1313.00 | 1299.97 | 1300.05 | SL hit (close>static) qty=1.00 sl=1307.10 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1314.00 | 1302.77 | 1301.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1316.10 | 1308.11 | 1304.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 14:15:00 | 1313.00 | 1313.84 | 1310.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:45:00 | 1314.00 | 1313.84 | 1310.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1320.00 | 1314.66 | 1311.05 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 1306.50 | 1311.31 | 1311.79 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1318.00 | 1312.49 | 1312.08 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1306.80 | 1311.85 | 1312.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1302.20 | 1309.92 | 1311.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 1312.00 | 1309.73 | 1310.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1309.70 | 1309.72 | 1310.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 1308.90 | 1309.72 | 1310.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1287.00 | 1302.95 | 1307.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:00:00 | 1283.00 | 1294.30 | 1299.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 1279.60 | 1291.71 | 1297.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 1282.80 | 1288.79 | 1294.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1291.60 | 1272.20 | 1269.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 1280.20 | 1280.56 | 1275.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1292.70 | 1280.56 | 1275.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1285.80 | 1280.91 | 1275.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1287.00 | 1281.78 | 1277.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 12:15:00 | 1287.20 | 1281.78 | 1277.23 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1283.70 | 1286.40 | 1281.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1283.70 | 1286.40 | 1281.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1291.90 | 1289.69 | 1285.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1280.80 | 1287.59 | 1285.50 | SL hit (close<ema400) qty=1.00 sl=1285.50 alert=retest1 |

### Cycle 124 — SELL (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 14:15:00 | 1275.00 | 1284.06 | 1285.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1258.70 | 1277.38 | 1281.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 1258.80 | 1258.29 | 1266.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 1258.80 | 1258.29 | 1266.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1276.00 | 1262.52 | 1267.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1276.00 | 1262.52 | 1267.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1279.90 | 1266.00 | 1268.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1279.40 | 1266.00 | 1268.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1288.20 | 1272.52 | 1271.06 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1277.80 | 1283.95 | 1284.02 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 1289.30 | 1285.06 | 1284.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 1294.80 | 1287.01 | 1285.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 1284.40 | 1293.15 | 1289.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1280.00 | 1290.52 | 1289.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1280.20 | 1290.52 | 1289.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1277.20 | 1286.08 | 1287.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1275.40 | 1283.14 | 1285.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 1289.10 | 1274.79 | 1279.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1290.00 | 1277.83 | 1280.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1291.80 | 1277.83 | 1280.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1308.20 | 1286.81 | 1283.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1315.60 | 1302.09 | 1293.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1395.10 | 1395.75 | 1374.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 1394.70 | 1395.75 | 1374.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1385.70 | 1396.84 | 1389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1383.50 | 1396.84 | 1389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1393.00 | 1396.07 | 1389.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 1397.10 | 1396.50 | 1390.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1398.40 | 1395.04 | 1390.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1380.70 | 1392.17 | 1389.98 | SL hit (close<static) qty=1.00 sl=1381.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1368.00 | 1384.43 | 1386.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1365.70 | 1380.68 | 1384.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1343.60 | 1337.37 | 1351.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1343.60 | 1337.37 | 1351.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1351.50 | 1342.75 | 1350.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1351.60 | 1342.75 | 1350.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1347.00 | 1343.60 | 1350.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1349.00 | 1343.60 | 1350.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1344.60 | 1343.80 | 1349.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1335.20 | 1342.74 | 1348.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1333.80 | 1341.04 | 1346.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 1336.60 | 1342.36 | 1346.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 1336.20 | 1338.72 | 1343.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1344.90 | 1339.96 | 1343.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1344.90 | 1339.96 | 1343.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1349.70 | 1341.90 | 1344.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 1349.70 | 1341.90 | 1344.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1348.30 | 1343.18 | 1344.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1348.30 | 1343.18 | 1344.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1344.70 | 1343.49 | 1344.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 1347.60 | 1343.49 | 1344.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1348.00 | 1344.39 | 1345.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1353.10 | 1344.39 | 1345.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1369.00 | 1352.98 | 1349.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1356.00 | 1357.31 | 1352.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1356.00 | 1357.31 | 1352.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1359.20 | 1357.69 | 1353.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1356.30 | 1357.69 | 1353.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1357.70 | 1357.69 | 1353.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 1364.80 | 1359.11 | 1354.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 1340.00 | 1352.69 | 1352.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 13:15:00 | 1340.00 | 1352.69 | 1352.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 1333.90 | 1348.94 | 1351.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1310.80 | 1309.82 | 1321.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1304.20 | 1308.56 | 1318.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1307.60 | 1308.37 | 1317.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1316.90 | 1308.37 | 1317.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1316.50 | 1310.00 | 1317.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1316.50 | 1310.00 | 1317.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1322.00 | 1312.40 | 1318.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 1322.00 | 1312.40 | 1318.16 | SL hit (close>ema400) qty=1.00 sl=1318.16 alert=retest1 |

### Cycle 133 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 1327.40 | 1305.22 | 1303.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 1343.70 | 1329.93 | 1319.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1357.60 | 1361.67 | 1347.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 1357.60 | 1361.67 | 1347.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1341.00 | 1370.53 | 1359.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1341.00 | 1370.53 | 1359.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1335.70 | 1363.56 | 1357.38 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1297.80 | 1343.82 | 1349.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1295.40 | 1334.14 | 1344.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1335.20 | 1325.75 | 1335.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1328.30 | 1326.26 | 1334.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 1326.10 | 1326.26 | 1334.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1343.80 | 1330.37 | 1335.17 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1352.80 | 1340.83 | 1339.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1358.50 | 1344.37 | 1340.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1403.10 | 1406.99 | 1391.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 1403.10 | 1406.99 | 1391.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1400.80 | 1455.01 | 1440.93 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 1396.40 | 1426.54 | 1430.02 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 1447.40 | 1431.14 | 1430.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 1459.70 | 1448.12 | 1440.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 13:15:00 | 1459.50 | 1461.23 | 1454.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 1459.50 | 1461.23 | 1454.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1505.00 | 1512.11 | 1500.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 1505.00 | 1512.11 | 1500.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1493.60 | 1508.41 | 1499.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1493.60 | 1508.41 | 1499.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1491.00 | 1504.93 | 1498.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1501.70 | 1504.93 | 1498.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1533.00 | 1535.33 | 1526.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1548.00 | 1537.73 | 1528.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:30:00 | 1551.00 | 1541.74 | 1531.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1544.00 | 1540.30 | 1533.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 1550.50 | 1560.98 | 1560.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1489.00 | 1541.20 | 1550.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1500.80 | 1494.67 | 1516.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 1500.80 | 1494.67 | 1516.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1502.60 | 1491.14 | 1506.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1509.50 | 1491.14 | 1506.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1502.70 | 1493.46 | 1505.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1492.40 | 1493.46 | 1505.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 1343.16 | 1475.20 | 1489.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 1497.30 | 1459.23 | 1456.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 1513.20 | 1470.02 | 1462.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 1475.70 | 1487.92 | 1475.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1479.80 | 1486.29 | 1476.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1492.70 | 1486.29 | 1476.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1466.10 | 1482.69 | 1476.33 | SL hit (close<static) qty=1.00 sl=1469.10 alert=retest2 |

### Cycle 140 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1457.20 | 1470.38 | 1471.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1429.50 | 1462.20 | 1468.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1479.20 | 1454.09 | 1459.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1456.90 | 1454.65 | 1459.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1445.80 | 1454.65 | 1459.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:00:00 | 1449.30 | 1446.25 | 1452.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:45:00 | 1453.00 | 1448.73 | 1452.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:30:00 | 1451.30 | 1453.58 | 1453.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1436.50 | 1450.70 | 1452.63 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1481.20 | 1455.16 | 1453.42 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 1449.30 | 1452.88 | 1453.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1400.70 | 1442.19 | 1448.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1406.70 | 1378.36 | 1392.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1416.00 | 1385.89 | 1394.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 1416.00 | 1385.89 | 1394.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1384.90 | 1387.76 | 1392.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 1372.00 | 1386.05 | 1391.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1303.40 | 1330.91 | 1354.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1333.80 | 1327.67 | 1349.06 | SL hit (close>ema200) qty=0.50 sl=1327.67 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1353.20 | 1337.57 | 1337.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1370.90 | 1347.83 | 1342.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1440.90 | 1444.52 | 1425.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 1440.90 | 1444.52 | 1425.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1490.20 | 1468.94 | 1451.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1494.50 | 1468.94 | 1451.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1495.80 | 1475.67 | 1463.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 1643.95 | 1624.27 | 1599.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1718.00 | 1742.05 | 1742.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1705.90 | 1734.82 | 1739.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1719.50 | 1719.28 | 1729.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1760.60 | 1727.54 | 1731.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 1760.60 | 1727.54 | 1731.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1740.10 | 1730.05 | 1732.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 1732.00 | 1730.44 | 1732.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 1733.40 | 1731.03 | 1732.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1754.20 | 1736.81 | 1735.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1754.20 | 1736.81 | 1735.08 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 1721.20 | 1731.78 | 1732.98 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 1734.90 | 1729.33 | 1729.07 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1720.00 | 1727.46 | 1728.25 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1736.50 | 1729.27 | 1729.00 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 1725.00 | 1729.16 | 1729.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 1718.00 | 1726.93 | 1728.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:15:00 | 1729.30 | 1722.98 | 1725.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1732.00 | 1724.78 | 1726.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 1730.90 | 1724.78 | 1726.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 1731.00 | 1727.07 | 1727.07 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 1725.40 | 1726.96 | 1727.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1717.50 | 1725.07 | 1726.17 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 09:15:00 | 1425.40 | 2024-05-27 13:15:00 | 1406.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-05-24 15:15:00 | 1408.00 | 2024-05-27 13:15:00 | 1406.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-05-30 11:45:00 | 1464.70 | 2024-06-04 11:15:00 | 1300.25 | STOP_HIT | 1.00 | -11.23% |
| BUY | retest2 | 2024-05-31 09:15:00 | 1511.45 | 2024-06-04 11:15:00 | 1300.25 | STOP_HIT | 1.00 | -13.97% |
| BUY | retest2 | 2024-06-18 15:00:00 | 1606.65 | 2024-06-19 14:15:00 | 1586.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-06-26 11:45:00 | 1498.00 | 2024-07-03 13:15:00 | 1495.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-06-27 09:15:00 | 1502.25 | 2024-07-03 13:15:00 | 1495.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-06-27 13:45:00 | 1502.35 | 2024-07-03 13:15:00 | 1495.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-06-28 11:00:00 | 1500.05 | 2024-07-03 13:15:00 | 1495.00 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-07-05 14:30:00 | 1509.00 | 2024-07-08 09:15:00 | 1474.15 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-07-11 14:00:00 | 1528.20 | 2024-07-12 09:15:00 | 1512.55 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-11 14:30:00 | 1530.00 | 2024-07-12 09:15:00 | 1512.55 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-07-19 09:15:00 | 1563.05 | 2024-07-19 12:15:00 | 1516.95 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1481.55 | 2024-07-24 12:15:00 | 1536.50 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-08-08 12:00:00 | 1763.65 | 2024-08-09 09:15:00 | 1844.55 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2024-08-08 13:00:00 | 1762.30 | 2024-08-09 09:15:00 | 1844.55 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2024-08-08 13:45:00 | 1761.90 | 2024-08-09 09:15:00 | 1844.55 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2024-08-16 13:00:00 | 1699.95 | 2024-08-20 10:15:00 | 1743.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-08-19 10:45:00 | 1707.70 | 2024-08-20 10:15:00 | 1743.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-27 14:45:00 | 1672.65 | 2024-08-28 09:15:00 | 1688.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-09 11:30:00 | 1678.40 | 2024-09-11 09:15:00 | 1694.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-10 10:15:00 | 1683.45 | 2024-09-11 09:15:00 | 1694.75 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-09-10 15:00:00 | 1683.15 | 2024-09-11 09:15:00 | 1694.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-18 09:15:00 | 1824.80 | 2024-09-26 10:15:00 | 1866.05 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest1 | 2024-10-03 09:15:00 | 1850.60 | 2024-10-04 10:15:00 | 1887.30 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest1 | 2024-10-03 10:00:00 | 1850.85 | 2024-10-04 10:15:00 | 1887.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest1 | 2024-10-03 11:15:00 | 1846.20 | 2024-10-04 10:15:00 | 1887.30 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest1 | 2024-10-03 13:00:00 | 1850.50 | 2024-10-04 10:15:00 | 1887.30 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-10-22 09:15:00 | 1988.05 | 2024-10-22 13:15:00 | 1947.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-11-08 14:00:00 | 1693.80 | 2024-11-14 09:15:00 | 1609.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:30:00 | 1697.95 | 2024-11-14 09:15:00 | 1613.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:45:00 | 1697.05 | 2024-11-14 09:15:00 | 1612.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 1693.80 | 2024-11-18 13:15:00 | 1576.70 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest2 | 2024-11-11 09:30:00 | 1697.95 | 2024-11-18 13:15:00 | 1576.70 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2024-11-11 11:45:00 | 1697.05 | 2024-11-18 13:15:00 | 1576.70 | STOP_HIT | 0.50 | 7.09% |
| SELL | retest2 | 2024-11-29 13:15:00 | 1515.00 | 2024-12-02 09:15:00 | 1606.95 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2024-11-29 15:00:00 | 1511.35 | 2024-12-02 09:15:00 | 1606.95 | STOP_HIT | 1.00 | -6.33% |
| SELL | retest2 | 2024-12-11 10:00:00 | 1614.55 | 2024-12-11 14:15:00 | 1667.75 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-12-20 10:00:00 | 1603.45 | 2024-12-20 14:15:00 | 1523.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:00:00 | 1603.45 | 2024-12-23 13:15:00 | 1539.10 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2025-01-29 09:45:00 | 1474.20 | 2025-01-30 14:15:00 | 1452.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-01-29 10:45:00 | 1475.20 | 2025-01-30 14:15:00 | 1452.80 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-01-29 15:00:00 | 1479.25 | 2025-01-30 14:15:00 | 1452.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-01-31 11:00:00 | 1464.35 | 2025-02-01 10:15:00 | 1472.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-02-04 10:30:00 | 1320.45 | 2025-02-05 09:15:00 | 1396.00 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-02-05 11:15:00 | 1374.95 | 2025-02-06 15:15:00 | 1376.45 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-02-06 11:15:00 | 1373.95 | 2025-02-06 15:15:00 | 1376.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-02-06 11:45:00 | 1372.55 | 2025-02-06 15:15:00 | 1376.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-02-06 15:00:00 | 1373.65 | 2025-02-06 15:15:00 | 1376.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1231.35 | 2025-02-19 11:15:00 | 1249.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-02-14 12:00:00 | 1225.00 | 2025-02-19 11:15:00 | 1249.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-02-14 15:00:00 | 1235.85 | 2025-02-19 11:15:00 | 1249.05 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-17 13:45:00 | 1232.85 | 2025-02-19 11:15:00 | 1249.05 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-02-20 11:00:00 | 1259.45 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-02-20 13:45:00 | 1260.15 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-02-20 14:15:00 | 1268.65 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-02-21 10:30:00 | 1266.00 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-02-24 10:15:00 | 1279.15 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-02-24 11:00:00 | 1279.25 | 2025-02-28 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-03-04 09:15:00 | 1251.95 | 2025-03-04 14:15:00 | 1278.85 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-04 10:00:00 | 1259.00 | 2025-03-04 14:15:00 | 1278.85 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-03-04 11:00:00 | 1260.30 | 2025-03-04 14:15:00 | 1278.85 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-04 11:30:00 | 1257.00 | 2025-03-04 14:15:00 | 1278.85 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest1 | 2025-03-10 09:15:00 | 1346.10 | 2025-03-10 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-03-11 10:30:00 | 1352.85 | 2025-03-12 09:15:00 | 1325.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-13 11:15:00 | 1307.15 | 2025-03-18 09:15:00 | 1333.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-03-17 14:00:00 | 1310.00 | 2025-03-18 09:15:00 | 1333.75 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-03-17 15:00:00 | 1310.65 | 2025-03-18 09:15:00 | 1333.75 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1511.40 | 2025-03-27 09:15:00 | 1485.75 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-04-03 09:15:00 | 1522.25 | 2025-04-07 09:15:00 | 1463.70 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-04-04 10:30:00 | 1517.70 | 2025-04-07 09:15:00 | 1463.70 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-04-04 12:15:00 | 1530.00 | 2025-04-07 09:15:00 | 1463.70 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-04-21 09:30:00 | 1613.10 | 2025-04-23 09:15:00 | 1575.90 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-04-21 12:00:00 | 1600.10 | 2025-04-23 09:15:00 | 1575.90 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-04-22 09:15:00 | 1614.60 | 2025-04-23 09:15:00 | 1575.90 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-04-22 10:00:00 | 1599.90 | 2025-04-23 09:15:00 | 1575.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1599.40 | 2025-04-23 09:15:00 | 1575.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-04-28 13:15:00 | 1551.30 | 2025-05-05 09:15:00 | 1473.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 14:00:00 | 1554.10 | 2025-05-05 09:15:00 | 1476.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1556.60 | 2025-05-05 09:15:00 | 1478.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 12:45:00 | 1554.60 | 2025-05-05 09:15:00 | 1476.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 12:00:00 | 1542.00 | 2025-05-05 09:15:00 | 1464.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 12:45:00 | 1539.80 | 2025-05-05 09:15:00 | 1462.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 09:30:00 | 1540.10 | 2025-05-05 09:15:00 | 1463.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 10:15:00 | 1539.30 | 2025-05-05 09:15:00 | 1462.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 13:15:00 | 1551.30 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-04-28 14:00:00 | 1554.10 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1556.60 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-04-29 12:45:00 | 1554.60 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-04-30 12:00:00 | 1542.00 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-04-30 12:45:00 | 1539.80 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-05-02 09:30:00 | 1540.10 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-05-02 10:15:00 | 1539.30 | 2025-05-06 09:15:00 | 1487.90 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-05-06 11:15:00 | 1475.80 | 2025-05-09 09:15:00 | 1402.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 10:45:00 | 1476.50 | 2025-05-09 09:15:00 | 1402.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 12:00:00 | 1479.00 | 2025-05-09 09:15:00 | 1405.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 1475.80 | 2025-05-12 09:15:00 | 1449.10 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-05-08 10:45:00 | 1476.50 | 2025-05-12 09:15:00 | 1449.10 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-05-08 12:00:00 | 1479.00 | 2025-05-12 09:15:00 | 1449.10 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2025-05-16 12:45:00 | 1434.50 | 2025-05-16 13:15:00 | 1441.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-23 14:45:00 | 1392.50 | 2025-05-26 09:15:00 | 1429.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1382.00 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-02 10:00:00 | 1381.30 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-06-02 11:00:00 | 1381.50 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-06-11 15:15:00 | 1434.00 | 2025-06-12 10:15:00 | 1421.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-12 09:45:00 | 1434.40 | 2025-06-12 10:15:00 | 1421.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-02 10:30:00 | 1448.00 | 2025-07-02 12:15:00 | 1463.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1435.90 | 2025-07-18 10:15:00 | 1364.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 14:45:00 | 1436.10 | 2025-07-18 10:15:00 | 1364.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1438.10 | 2025-07-18 10:15:00 | 1366.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1436.00 | 2025-07-18 10:15:00 | 1364.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1435.90 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2025-07-07 14:45:00 | 1436.10 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1438.10 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1436.00 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1411.60 | 2025-07-22 13:15:00 | 1341.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-11 14:30:00 | 1410.30 | 2025-07-22 13:15:00 | 1342.07 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-07-11 15:15:00 | 1399.00 | 2025-07-22 14:15:00 | 1339.78 | PARTIAL | 0.50 | 4.23% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1411.60 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-07-11 14:30:00 | 1410.30 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-07-11 15:15:00 | 1399.00 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-07-14 10:15:00 | 1412.70 | 2025-07-25 11:15:00 | 1329.05 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1397.80 | 2025-07-25 11:15:00 | 1327.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-14 10:15:00 | 1412.70 | 2025-07-28 09:15:00 | 1340.20 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1397.80 | 2025-07-28 09:15:00 | 1340.20 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-08-12 09:45:00 | 1356.80 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-12 10:15:00 | 1355.00 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1355.80 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-12 14:30:00 | 1354.10 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-21 15:15:00 | 1287.00 | 2025-09-02 09:15:00 | 1282.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-09-18 10:15:00 | 1264.20 | 2025-09-18 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-18 10:45:00 | 1264.60 | 2025-09-18 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-23 10:00:00 | 1284.20 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-09-23 11:45:00 | 1283.80 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-23 13:15:00 | 1280.90 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-24 09:15:00 | 1282.30 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-03 13:15:00 | 1212.00 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-10-06 09:30:00 | 1211.90 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-06 10:15:00 | 1213.50 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-10-13 10:45:00 | 1239.60 | 2025-10-28 10:15:00 | 1294.00 | STOP_HIT | 1.00 | 4.39% |
| BUY | retest2 | 2025-11-03 09:45:00 | 1327.20 | 2025-11-04 13:15:00 | 1303.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1294.60 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-25 10:30:00 | 1298.30 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1297.90 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-05 12:00:00 | 1283.00 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1279.60 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1282.80 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2025-12-12 09:15:00 | 1292.70 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest1 | 2025-12-12 10:15:00 | 1285.80 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-12 11:45:00 | 1287.00 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-12 12:15:00 | 1287.20 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-01-07 14:30:00 | 1397.10 | 2026-01-08 10:15:00 | 1380.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-08 10:15:00 | 1398.40 | 2026-01-08 10:15:00 | 1380.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1335.20 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1333.80 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-01-14 09:15:00 | 1336.60 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-01-14 11:00:00 | 1336.20 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-19 10:00:00 | 1364.80 | 2026-01-19 13:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest1 | 2026-01-22 11:30:00 | 1304.20 | 2026-01-22 14:15:00 | 1322.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-23 09:15:00 | 1311.70 | 2026-01-28 09:15:00 | 1327.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1548.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-24 11:30:00 | 1551.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-24 15:15:00 | 1544.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-02 11:30:00 | 1550.50 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1492.40 | 2026-03-09 09:15:00 | 1343.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-13 11:15:00 | 1492.70 | 2026-03-13 12:15:00 | 1466.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1445.80 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-18 10:00:00 | 1449.30 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1453.00 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-03-18 14:30:00 | 1451.30 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-03-27 14:15:00 | 1372.00 | 2026-03-30 14:15:00 | 1303.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 14:15:00 | 1372.00 | 2026-04-01 09:15:00 | 1333.80 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1494.50 | 2026-04-22 09:15:00 | 1643.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1495.80 | 2026-04-22 09:15:00 | 1645.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 14:00:00 | 1732.00 | 2026-05-04 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-04-30 15:00:00 | 1733.40 | 2026-05-04 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -1.20% |
