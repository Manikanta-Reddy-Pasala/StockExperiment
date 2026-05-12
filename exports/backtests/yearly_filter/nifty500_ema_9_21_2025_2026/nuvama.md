# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 1605.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 98 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 17 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 48 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 38
- **Target hits / Stop hits / Partials:** 1 / 52 / 0
- **Avg / median % per leg:** -0.50% / -1.40%
- **Sum % (uncompounded):** -26.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 10 | 66.7% | 1 | 14 | 0 | 1.58% | 23.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 10 | 66.7% | 1 | 14 | 0 | 1.58% | 23.7% |
| SELL (all) | 38 | 5 | 13.2% | 0 | 38 | 0 | -1.33% | -50.4% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.26% | -6.3% |
| SELL @ 3rd Alert (retest2) | 33 | 4 | 12.1% | 0 | 33 | 0 | -1.34% | -44.1% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.26% | -6.3% |
| retest2 (combined) | 48 | 14 | 29.2% | 1 | 47 | 0 | -0.42% | -20.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1360.00 | 1363.41 | 1363.45 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1365.10 | 1363.75 | 1363.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1370.00 | 1365.63 | 1364.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:00:00 | 1360.80 | 1365.15 | 1364.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1363.00 | 1364.72 | 1364.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 1361.30 | 1364.72 | 1364.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1370.20 | 1365.82 | 1364.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1375.00 | 1365.82 | 1364.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1377.20 | 1368.10 | 1366.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:00:00 | 1390.80 | 1378.25 | 1372.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 1389.70 | 1382.55 | 1375.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1392.40 | 1385.83 | 1379.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1389.50 | 1388.23 | 1382.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1387.40 | 1388.71 | 1384.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 1389.50 | 1388.71 | 1384.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1384.00 | 1387.77 | 1384.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1417.00 | 1387.77 | 1384.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 1393.20 | 1392.71 | 1388.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 1442.20 | 1451.77 | 1451.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1442.20 | 1451.77 | 1451.91 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1469.90 | 1454.08 | 1452.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1474.00 | 1464.58 | 1458.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 1500.00 | 1511.61 | 1495.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 1500.00 | 1511.61 | 1495.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1497.00 | 1508.69 | 1495.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 1493.40 | 1508.69 | 1495.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1500.20 | 1506.99 | 1495.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 1494.10 | 1506.99 | 1495.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1474.30 | 1502.46 | 1497.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1474.30 | 1502.46 | 1497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1491.20 | 1500.21 | 1496.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1475.20 | 1500.21 | 1496.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1478.80 | 1492.38 | 1493.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1462.80 | 1480.04 | 1486.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1418.00 | 1417.60 | 1434.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 1418.00 | 1417.60 | 1434.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1452.10 | 1422.72 | 1432.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 1452.10 | 1422.72 | 1432.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1446.10 | 1427.40 | 1433.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 1441.00 | 1431.62 | 1435.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 1460.30 | 1437.35 | 1437.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 1460.30 | 1437.35 | 1437.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 1475.40 | 1444.96 | 1440.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 1460.00 | 1461.99 | 1453.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 12:45:00 | 1459.80 | 1461.99 | 1453.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1450.90 | 1459.74 | 1453.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1450.90 | 1459.74 | 1453.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1457.50 | 1459.29 | 1454.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1458.80 | 1459.29 | 1454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1452.30 | 1457.89 | 1453.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1452.30 | 1457.89 | 1453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1431.60 | 1452.64 | 1451.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1431.60 | 1452.64 | 1451.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1424.00 | 1446.91 | 1449.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1403.40 | 1438.21 | 1445.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1398.20 | 1394.97 | 1411.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 1390.60 | 1394.97 | 1411.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1408.10 | 1397.60 | 1410.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1408.00 | 1397.60 | 1410.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1406.00 | 1399.00 | 1408.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 1401.60 | 1399.00 | 1408.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1412.00 | 1401.60 | 1408.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 1412.00 | 1401.60 | 1408.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1410.80 | 1403.44 | 1408.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1410.80 | 1403.44 | 1408.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1414.40 | 1405.63 | 1409.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1430.30 | 1405.63 | 1409.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1419.20 | 1408.35 | 1410.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 1417.50 | 1408.35 | 1410.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1435.70 | 1413.82 | 1412.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1468.00 | 1433.11 | 1423.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1640.00 | 1645.17 | 1611.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:15:00 | 1644.90 | 1645.17 | 1611.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1620.00 | 1638.82 | 1614.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1617.10 | 1638.82 | 1614.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1671.50 | 1649.86 | 1630.37 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 1601.70 | 1626.39 | 1626.98 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 1664.00 | 1633.08 | 1629.87 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 1561.80 | 1618.94 | 1624.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1478.20 | 1579.96 | 1605.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1509.50 | 1509.31 | 1554.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:45:00 | 1505.00 | 1509.31 | 1554.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1499.80 | 1498.87 | 1525.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 1489.50 | 1497.00 | 1522.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 13:15:00 | 1496.90 | 1495.22 | 1517.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1497.40 | 1495.87 | 1515.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 1491.80 | 1495.06 | 1513.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1486.00 | 1477.91 | 1492.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 1464.40 | 1473.24 | 1482.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1491.30 | 1482.31 | 1482.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1491.30 | 1482.31 | 1482.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1509.10 | 1490.68 | 1486.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 1509.20 | 1510.80 | 1500.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:45:00 | 1511.40 | 1510.80 | 1500.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1499.00 | 1508.44 | 1499.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 1502.70 | 1508.44 | 1499.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1502.40 | 1507.23 | 1500.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1546.60 | 1507.23 | 1500.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1516.00 | 1531.71 | 1532.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 1516.00 | 1531.71 | 1532.48 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1551.40 | 1533.31 | 1532.61 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1513.30 | 1542.45 | 1543.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1483.80 | 1530.72 | 1538.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 1447.80 | 1447.72 | 1466.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 1451.00 | 1447.72 | 1466.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1480.60 | 1455.60 | 1464.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 1480.60 | 1455.60 | 1464.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1481.60 | 1460.80 | 1465.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 1481.40 | 1460.80 | 1465.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1484.00 | 1468.77 | 1468.74 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1451.50 | 1465.32 | 1467.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 1446.30 | 1459.37 | 1463.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1465.60 | 1460.62 | 1463.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1444.20 | 1457.33 | 1462.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1408.00 | 1454.87 | 1460.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 1391.40 | 1376.03 | 1375.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1391.40 | 1376.03 | 1375.33 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1370.10 | 1374.50 | 1374.74 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 1380.70 | 1375.87 | 1375.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 1386.00 | 1378.87 | 1376.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 1376.40 | 1389.40 | 1383.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1374.30 | 1386.38 | 1382.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 1374.30 | 1386.38 | 1382.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1370.40 | 1383.18 | 1381.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 1370.40 | 1383.18 | 1381.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1395.00 | 1386.22 | 1383.01 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1377.00 | 1381.63 | 1381.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1346.40 | 1372.08 | 1376.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1354.20 | 1352.54 | 1360.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 1354.20 | 1352.54 | 1360.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1370.00 | 1356.03 | 1361.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:15:00 | 1375.40 | 1356.03 | 1361.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1375.40 | 1359.91 | 1362.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1407.80 | 1359.91 | 1362.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1413.40 | 1370.61 | 1367.37 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1376.90 | 1384.73 | 1385.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1362.60 | 1376.66 | 1381.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 1295.00 | 1292.52 | 1310.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:00:00 | 1295.00 | 1292.52 | 1310.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1315.20 | 1295.89 | 1300.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 1317.90 | 1295.89 | 1300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1306.90 | 1298.09 | 1300.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 1299.00 | 1299.42 | 1300.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 1307.10 | 1302.60 | 1302.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1307.10 | 1302.60 | 1302.12 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1289.80 | 1301.12 | 1301.99 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1318.40 | 1302.83 | 1302.51 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1284.70 | 1302.24 | 1302.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 1279.60 | 1290.67 | 1295.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 1286.70 | 1286.57 | 1292.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 1286.70 | 1286.57 | 1292.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1293.10 | 1283.91 | 1288.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1290.80 | 1283.91 | 1288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1293.00 | 1285.73 | 1289.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1293.00 | 1285.73 | 1289.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1284.40 | 1285.46 | 1288.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1275.00 | 1281.21 | 1285.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1268.40 | 1281.17 | 1285.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1301.30 | 1272.40 | 1272.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1301.30 | 1272.40 | 1272.13 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1269.90 | 1277.11 | 1277.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 1264.00 | 1273.94 | 1275.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1271.30 | 1270.04 | 1272.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1276.90 | 1271.41 | 1273.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1277.80 | 1271.41 | 1273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1282.90 | 1273.71 | 1274.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1282.90 | 1273.71 | 1274.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1288.00 | 1276.57 | 1275.41 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1263.60 | 1277.83 | 1277.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1254.90 | 1273.25 | 1275.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 1240.00 | 1239.85 | 1247.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 1240.00 | 1239.85 | 1247.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1232.00 | 1218.99 | 1229.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1232.00 | 1218.99 | 1229.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1238.20 | 1222.83 | 1229.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:30:00 | 1224.90 | 1222.87 | 1229.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 1227.90 | 1224.89 | 1229.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1242.00 | 1232.12 | 1231.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1242.00 | 1232.12 | 1231.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 1247.50 | 1237.05 | 1233.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1444.60 | 1444.71 | 1431.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 1444.60 | 1444.71 | 1431.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1438.80 | 1442.77 | 1432.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 1432.00 | 1442.77 | 1432.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1432.00 | 1440.27 | 1435.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 1432.20 | 1440.27 | 1435.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1426.60 | 1437.53 | 1434.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 1426.60 | 1437.53 | 1434.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1428.60 | 1435.75 | 1434.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 1428.60 | 1435.75 | 1434.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1438.00 | 1435.10 | 1434.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1434.80 | 1435.10 | 1434.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1445.20 | 1437.12 | 1435.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1439.90 | 1437.12 | 1435.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1445.20 | 1443.31 | 1440.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 1441.00 | 1443.31 | 1440.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1422.40 | 1440.04 | 1439.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 1422.40 | 1440.04 | 1439.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1427.60 | 1437.55 | 1438.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1414.00 | 1430.48 | 1435.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 1410.60 | 1406.15 | 1416.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 1410.60 | 1406.15 | 1416.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1413.00 | 1406.59 | 1415.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1421.70 | 1413.89 | 1417.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1450.00 | 1421.11 | 1420.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1467.70 | 1445.88 | 1439.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 1451.70 | 1461.35 | 1462.57 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 1480.00 | 1464.66 | 1463.70 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1456.60 | 1463.05 | 1463.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 1442.60 | 1457.24 | 1460.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1472.40 | 1453.58 | 1456.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1433.50 | 1449.56 | 1454.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1467.50 | 1449.56 | 1454.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1455.90 | 1431.14 | 1435.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1455.90 | 1431.14 | 1435.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1462.40 | 1437.39 | 1437.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1462.40 | 1437.39 | 1437.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 1451.20 | 1440.15 | 1438.87 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 1434.00 | 1438.02 | 1438.37 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 1468.30 | 1444.08 | 1441.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1486.90 | 1463.09 | 1456.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1463.00 | 1484.54 | 1474.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1452.50 | 1478.13 | 1472.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1452.50 | 1478.13 | 1472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1469.60 | 1472.32 | 1470.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1474.00 | 1471.36 | 1470.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1465.00 | 1470.09 | 1470.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1465.00 | 1470.09 | 1470.20 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 1471.70 | 1470.26 | 1470.21 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 1467.10 | 1469.63 | 1469.92 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1472.10 | 1470.12 | 1470.12 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1466.10 | 1469.32 | 1469.76 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1484.10 | 1472.27 | 1471.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1500.00 | 1479.34 | 1474.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1479.50 | 1481.98 | 1476.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1479.50 | 1481.98 | 1476.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1474.50 | 1480.48 | 1476.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1474.50 | 1480.48 | 1476.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1476.00 | 1479.58 | 1476.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1474.90 | 1479.58 | 1476.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1462.80 | 1476.23 | 1475.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1462.80 | 1476.23 | 1475.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1465.40 | 1474.06 | 1474.36 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 1474.40 | 1472.16 | 1471.94 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1460.30 | 1469.79 | 1470.88 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1470.10 | 1467.31 | 1467.28 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 1464.60 | 1466.77 | 1467.04 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1470.40 | 1467.50 | 1467.34 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 1465.60 | 1467.12 | 1467.18 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1469.60 | 1467.61 | 1467.40 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 1464.20 | 1466.88 | 1467.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1454.70 | 1464.44 | 1465.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1460.00 | 1459.74 | 1462.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 14:30:00 | 1457.10 | 1458.04 | 1461.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1451.30 | 1413.06 | 1423.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1451.30 | 1413.06 | 1423.91 | SL hit (close>ema400) qty=1.00 sl=1423.91 alert=retest1 |

### Cycle 56 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1460.90 | 1431.26 | 1430.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 1470.20 | 1457.41 | 1447.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 1493.50 | 1494.98 | 1483.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 14:45:00 | 1487.70 | 1494.98 | 1483.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1471.10 | 1489.63 | 1483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1471.10 | 1489.63 | 1483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1472.60 | 1486.22 | 1482.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 1472.60 | 1486.22 | 1482.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1464.40 | 1478.82 | 1479.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1462.10 | 1475.47 | 1477.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 1420.00 | 1417.59 | 1429.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:30:00 | 1425.20 | 1417.59 | 1429.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1441.40 | 1423.20 | 1429.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1441.40 | 1423.20 | 1429.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1433.30 | 1425.22 | 1430.00 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 1448.00 | 1434.37 | 1433.35 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1425.80 | 1432.57 | 1433.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1409.70 | 1427.99 | 1431.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 1423.20 | 1419.56 | 1425.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:30:00 | 1419.00 | 1419.56 | 1425.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1421.60 | 1419.17 | 1423.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1404.00 | 1416.44 | 1422.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1408.90 | 1414.93 | 1420.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 1401.40 | 1413.78 | 1419.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1434.70 | 1415.17 | 1418.35 | SL hit (close>static) qty=1.00 sl=1424.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 1442.40 | 1420.62 | 1420.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 1456.50 | 1427.80 | 1423.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1453.00 | 1456.57 | 1446.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 1454.00 | 1456.57 | 1446.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1451.80 | 1454.69 | 1450.29 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1436.20 | 1448.05 | 1448.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1435.80 | 1445.60 | 1447.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 1448.10 | 1445.89 | 1447.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1450.00 | 1446.71 | 1447.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 1440.40 | 1446.71 | 1447.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1462.60 | 1444.57 | 1445.45 | SL hit (close>static) qty=1.00 sl=1450.90 alert=retest2 |

### Cycle 62 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 1462.10 | 1448.07 | 1446.96 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 1426.30 | 1445.61 | 1447.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 11:15:00 | 1418.40 | 1440.17 | 1444.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1434.60 | 1432.39 | 1439.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1434.60 | 1432.39 | 1439.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1419.80 | 1430.11 | 1437.30 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1452.60 | 1436.38 | 1435.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 14:15:00 | 1456.10 | 1442.85 | 1438.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 1498.00 | 1502.18 | 1481.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1498.00 | 1502.18 | 1481.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1495.50 | 1497.57 | 1485.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 1490.00 | 1497.57 | 1485.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1439.00 | 1484.17 | 1481.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 1439.00 | 1484.17 | 1481.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1465.00 | 1478.07 | 1478.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 1455.50 | 1470.54 | 1474.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 1472.50 | 1470.93 | 1474.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1450.00 | 1470.93 | 1474.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:30:00 | 1454.00 | 1463.58 | 1470.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1455.00 | 1463.58 | 1470.42 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 14:15:00 | 1452.00 | 1457.40 | 1465.51 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1477.00 | 1459.94 | 1464.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1477.00 | 1459.94 | 1464.51 | SL hit (close>ema400) qty=1.00 sl=1464.51 alert=retest1 |

### Cycle 66 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1486.50 | 1469.26 | 1468.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 1492.50 | 1473.91 | 1470.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 1456.30 | 1473.63 | 1471.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 1456.00 | 1470.10 | 1470.41 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1496.80 | 1472.36 | 1469.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 1498.70 | 1477.63 | 1471.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1500.60 | 1487.71 | 1479.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1478.80 | 1485.87 | 1481.10 | SL hit (close<static) qty=1.00 sl=1479.70 alert=retest2 |

### Cycle 69 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1477.00 | 1479.47 | 1479.72 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 1487.70 | 1480.18 | 1479.88 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1467.10 | 1477.56 | 1478.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1444.90 | 1471.03 | 1475.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 11:15:00 | 1459.90 | 1459.52 | 1465.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:30:00 | 1461.00 | 1459.52 | 1465.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1421.00 | 1416.30 | 1425.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1439.40 | 1416.30 | 1425.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1443.90 | 1421.82 | 1427.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 1447.30 | 1421.82 | 1427.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1448.10 | 1427.08 | 1428.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 1447.20 | 1427.08 | 1428.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1447.80 | 1433.48 | 1431.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1490.70 | 1446.09 | 1438.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 12:15:00 | 1466.50 | 1470.52 | 1460.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 13:00:00 | 1466.50 | 1470.52 | 1460.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1465.00 | 1469.41 | 1460.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1465.00 | 1469.41 | 1460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1460.20 | 1467.57 | 1460.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 1463.30 | 1467.57 | 1460.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1461.90 | 1466.44 | 1460.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1432.20 | 1466.44 | 1460.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1442.50 | 1461.65 | 1459.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1423.40 | 1461.65 | 1459.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1433.70 | 1456.06 | 1456.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1429.90 | 1450.83 | 1454.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1398.60 | 1391.03 | 1412.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 1396.50 | 1391.03 | 1412.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1296.30 | 1287.52 | 1319.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1314.30 | 1287.52 | 1319.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1331.80 | 1303.23 | 1315.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1331.80 | 1303.23 | 1315.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1319.00 | 1306.39 | 1315.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1307.80 | 1309.51 | 1316.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 1299.50 | 1309.19 | 1315.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:15:00 | 1304.50 | 1310.15 | 1315.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 1333.00 | 1318.20 | 1317.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1333.00 | 1318.20 | 1317.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1351.30 | 1324.82 | 1320.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 1312.40 | 1334.17 | 1329.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1332.80 | 1333.89 | 1329.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1340.40 | 1330.77 | 1328.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1306.50 | 1325.92 | 1326.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1306.50 | 1325.92 | 1326.58 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1363.50 | 1304.03 | 1299.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1369.00 | 1317.02 | 1306.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 1369.70 | 1373.70 | 1356.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 1369.70 | 1373.70 | 1356.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1342.90 | 1369.04 | 1361.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1342.90 | 1369.04 | 1361.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1331.70 | 1361.57 | 1358.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1329.00 | 1361.57 | 1358.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1330.50 | 1351.15 | 1353.92 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1390.60 | 1359.69 | 1356.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1396.80 | 1367.11 | 1360.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 1372.10 | 1379.23 | 1371.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:45:00 | 1375.90 | 1379.23 | 1371.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1389.70 | 1381.33 | 1372.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1396.10 | 1385.02 | 1375.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1367.70 | 1381.46 | 1377.65 | SL hit (close<static) qty=1.00 sl=1368.00 alert=retest2 |

### Cycle 79 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 1361.30 | 1374.50 | 1374.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 1359.30 | 1371.46 | 1373.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1287.10 | 1286.03 | 1302.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1287.10 | 1286.03 | 1302.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1280.30 | 1281.23 | 1292.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1290.60 | 1281.23 | 1292.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1296.90 | 1285.71 | 1292.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1315.60 | 1285.71 | 1292.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1302.90 | 1289.14 | 1293.71 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 1312.50 | 1298.96 | 1297.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 1313.00 | 1301.77 | 1298.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 1293.10 | 1305.60 | 1302.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1308.80 | 1306.24 | 1303.20 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1294.40 | 1300.93 | 1301.50 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 1308.10 | 1302.37 | 1302.10 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 1285.30 | 1300.25 | 1301.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 1281.80 | 1296.56 | 1299.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 09:15:00 | 1277.40 | 1273.95 | 1283.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 10:00:00 | 1277.40 | 1273.95 | 1283.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1268.70 | 1273.77 | 1281.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 1266.20 | 1273.77 | 1281.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1266.70 | 1272.26 | 1277.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 1266.60 | 1270.77 | 1275.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 1267.30 | 1270.42 | 1275.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1274.90 | 1271.32 | 1275.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:30:00 | 1271.10 | 1271.32 | 1275.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1270.60 | 1271.17 | 1274.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1279.90 | 1271.17 | 1274.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1281.50 | 1273.24 | 1275.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 1274.20 | 1274.69 | 1275.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 1272.20 | 1274.69 | 1275.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 1280.00 | 1276.16 | 1276.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 1280.00 | 1276.16 | 1276.00 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1241.30 | 1269.19 | 1272.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1226.00 | 1247.76 | 1258.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 1222.40 | 1203.41 | 1216.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1220.30 | 1206.79 | 1216.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 1220.30 | 1206.79 | 1216.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 1243.00 | 1225.12 | 1222.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1249.60 | 1230.02 | 1225.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1227.30 | 1231.73 | 1227.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1229.70 | 1231.32 | 1227.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 1229.00 | 1231.32 | 1227.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1224.60 | 1229.98 | 1227.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1224.60 | 1229.98 | 1227.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1230.50 | 1230.08 | 1227.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 1180.00 | 1230.08 | 1227.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1187.10 | 1221.49 | 1223.98 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1227.10 | 1217.60 | 1216.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1236.60 | 1221.40 | 1218.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1222.10 | 1229.09 | 1224.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1220.00 | 1227.27 | 1224.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 1220.00 | 1227.27 | 1224.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1211.90 | 1224.20 | 1223.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1211.90 | 1224.20 | 1223.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1208.60 | 1221.08 | 1221.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1177.50 | 1212.36 | 1217.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1160.00 | 1154.98 | 1169.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1164.10 | 1154.98 | 1169.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1159.90 | 1156.12 | 1167.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1160.00 | 1156.12 | 1167.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1156.50 | 1156.20 | 1166.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1150.30 | 1156.40 | 1165.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1173.00 | 1160.43 | 1164.17 | SL hit (close>static) qty=1.00 sl=1170.50 alert=retest2 |

### Cycle 90 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1186.20 | 1168.08 | 1167.15 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1154.90 | 1169.05 | 1169.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1148.10 | 1159.94 | 1164.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1130.00 | 1118.89 | 1130.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1151.00 | 1125.31 | 1132.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1151.10 | 1125.31 | 1132.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1152.00 | 1130.65 | 1133.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1152.00 | 1130.65 | 1133.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 1148.50 | 1137.35 | 1136.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1161.30 | 1142.14 | 1138.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 1195.50 | 1196.18 | 1178.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:45:00 | 1192.00 | 1196.18 | 1178.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1162.60 | 1190.88 | 1179.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 1162.60 | 1190.88 | 1179.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1158.60 | 1184.42 | 1177.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 1158.60 | 1184.42 | 1177.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 1149.00 | 1173.11 | 1173.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1140.00 | 1162.66 | 1168.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 1162.40 | 1159.66 | 1165.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 1162.40 | 1159.66 | 1165.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1148.10 | 1157.74 | 1163.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1146.40 | 1157.74 | 1163.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 1146.20 | 1139.82 | 1148.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1167.00 | 1145.73 | 1149.44 | SL hit (close>static) qty=1.00 sl=1164.20 alert=retest2 |

### Cycle 94 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1166.30 | 1151.90 | 1151.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1176.80 | 1156.88 | 1154.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 1164.60 | 1168.70 | 1163.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:00:00 | 1164.60 | 1168.70 | 1163.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1176.80 | 1170.32 | 1164.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1179.00 | 1170.32 | 1164.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 1296.90 | 1263.10 | 1226.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1366.10 | 1376.40 | 1377.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1357.10 | 1370.76 | 1373.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1365.30 | 1358.17 | 1364.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1368.80 | 1360.30 | 1364.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1368.80 | 1360.30 | 1364.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1357.00 | 1359.64 | 1363.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 1355.50 | 1359.64 | 1363.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1377.10 | 1365.33 | 1364.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 1377.10 | 1365.33 | 1364.58 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1355.50 | 1364.54 | 1365.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1349.10 | 1361.46 | 1363.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1334.40 | 1330.42 | 1342.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 1334.50 | 1330.42 | 1342.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1333.20 | 1330.98 | 1341.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 1323.80 | 1330.76 | 1340.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1325.40 | 1330.76 | 1340.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 1323.80 | 1329.37 | 1338.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:45:00 | 1326.90 | 1333.11 | 1336.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1345.70 | 1335.63 | 1337.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 1353.20 | 1335.63 | 1337.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1351.10 | 1338.73 | 1338.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1351.10 | 1338.73 | 1338.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1381.90 | 1349.67 | 1343.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-11 10:15:00 | 1588.70 | 1591.13 | 1529.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-11 11:00:00 | 1588.70 | 1591.13 | 1529.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 10:15:00 | 1348.90 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-05-19 10:45:00 | 1354.40 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-05-22 13:15:00 | 1347.50 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-05-27 10:00:00 | 1390.80 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2025-05-27 12:15:00 | 1389.70 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.78% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1392.40 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1389.50 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.79% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1417.00 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-05-29 13:30:00 | 1393.20 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.52% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1441.00 | 2025-06-17 12:15:00 | 1460.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1489.50 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-07-08 13:15:00 | 1496.90 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-08 14:15:00 | 1497.40 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-08 15:00:00 | 1491.80 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-07-11 13:30:00 | 1464.40 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1546.60 | 2025-07-21 12:15:00 | 1516.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 09:15:00 | 1408.00 | 2025-08-12 12:15:00 | 1391.40 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-09-03 12:45:00 | 1299.00 | 2025-09-03 15:15:00 | 1307.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-10 14:45:00 | 1275.00 | 2025-09-12 12:15:00 | 1301.30 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1268.40 | 2025-09-12 12:15:00 | 1301.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-26 11:30:00 | 1224.90 | 2025-09-29 11:15:00 | 1242.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-26 13:15:00 | 1227.90 | 2025-09-29 11:15:00 | 1242.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1474.00 | 2025-11-12 09:15:00 | 1465.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-11-21 14:30:00 | 1457.10 | 2025-11-26 09:15:00 | 1451.30 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1404.00 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1408.90 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-10 14:15:00 | 1401.40 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-12-17 12:15:00 | 1440.40 | 2025-12-18 09:15:00 | 1462.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-12-30 09:15:00 | 1450.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2025-12-30 10:30:00 | 1454.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest1 | 2025-12-30 11:15:00 | 1455.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest1 | 2025-12-30 14:15:00 | 1452.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1500.60 | 2026-01-06 11:15:00 | 1478.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1307.80 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-29 10:30:00 | 1299.50 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-01-29 12:15:00 | 1304.50 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-01 11:30:00 | 1340.40 | 2026-02-01 12:15:00 | 1306.50 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-10 11:30:00 | 1396.10 | 2026-02-11 09:15:00 | 1367.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1266.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-25 10:30:00 | 1266.70 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1266.60 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-25 13:45:00 | 1267.30 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 1274.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-26 12:15:00 | 1272.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1150.30 | 2026-03-18 09:15:00 | 1173.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1146.40 | 2026-04-06 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-02 14:15:00 | 1146.20 | 2026-04-06 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1179.00 | 2026-04-09 09:15:00 | 1296.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1355.50 | 2026-04-28 13:15:00 | 1377.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-04 10:45:00 | 1323.80 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-05-04 11:15:00 | 1325.40 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-05-04 12:00:00 | 1323.80 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-05-05 10:45:00 | 1326.90 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -1.82% |
