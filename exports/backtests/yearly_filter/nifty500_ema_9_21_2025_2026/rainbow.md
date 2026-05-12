# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1311.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 23 |
| ALERT3 | 112 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 66 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 29 / 49
- **Target hits / Stop hits / Partials:** 1 / 70 / 7
- **Avg / median % per leg:** 0.24% / -0.82%
- **Sum % (uncompounded):** 18.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 8 | 26.7% | 0 | 30 | 0 | -0.65% | -19.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 6 | 0 | 0.83% | 5.0% |
| BUY @ 3rd Alert (retest2) | 24 | 5 | 20.8% | 0 | 24 | 0 | -1.02% | -24.5% |
| SELL (all) | 48 | 21 | 43.8% | 1 | 40 | 7 | 0.79% | 37.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 21 | 43.8% | 1 | 40 | 7 | 0.79% | 37.9% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 6 | 0 | 0.83% | 5.0% |
| retest2 (combined) | 72 | 26 | 36.1% | 1 | 64 | 7 | 0.19% | 13.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1348.60 | 1348.39 | 1348.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1354.50 | 1350.05 | 1349.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 1346.70 | 1353.69 | 1351.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1352.50 | 1353.45 | 1351.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 1358.20 | 1353.45 | 1351.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 1359.50 | 1372.13 | 1372.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1359.50 | 1372.13 | 1372.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1346.20 | 1354.84 | 1361.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 1339.50 | 1347.59 | 1354.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1362.90 | 1346.19 | 1345.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1362.90 | 1346.19 | 1345.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 1375.60 | 1360.45 | 1355.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 1403.00 | 1406.83 | 1393.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 1403.00 | 1406.83 | 1393.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1392.70 | 1404.29 | 1395.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 1392.70 | 1404.29 | 1395.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1399.40 | 1403.31 | 1396.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1412.50 | 1401.25 | 1395.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1403.80 | 1407.47 | 1403.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 1383.20 | 1404.52 | 1403.87 | SL hit (close<static) qty=1.00 sl=1392.40 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 1372.00 | 1398.01 | 1400.97 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1408.70 | 1402.70 | 1402.60 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 1400.10 | 1402.73 | 1402.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1390.90 | 1400.36 | 1401.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1394.00 | 1393.71 | 1397.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1410.00 | 1396.97 | 1398.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1410.00 | 1396.97 | 1398.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1421.30 | 1401.83 | 1400.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1429.90 | 1410.69 | 1405.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1414.00 | 1416.85 | 1410.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:00:00 | 1414.00 | 1416.85 | 1410.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1413.10 | 1415.39 | 1411.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 1417.00 | 1412.47 | 1411.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 1418.30 | 1415.69 | 1413.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1394.30 | 1411.21 | 1411.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1394.30 | 1411.21 | 1411.92 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 1434.30 | 1408.48 | 1406.73 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 1417.00 | 1428.40 | 1428.94 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 1438.60 | 1429.79 | 1429.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 1449.20 | 1433.68 | 1431.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 1454.40 | 1446.43 | 1442.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1471.80 | 1451.50 | 1447.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 1441.10 | 1452.68 | 1454.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 1441.10 | 1452.68 | 1454.15 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1476.70 | 1458.35 | 1456.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 1489.50 | 1464.58 | 1459.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 1536.00 | 1540.95 | 1526.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:45:00 | 1533.00 | 1540.95 | 1526.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1570.90 | 1544.04 | 1531.00 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1514.70 | 1524.72 | 1525.89 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1545.90 | 1526.29 | 1526.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1565.20 | 1539.92 | 1532.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 14:15:00 | 1589.20 | 1591.42 | 1569.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 1589.20 | 1591.42 | 1569.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1574.00 | 1589.77 | 1573.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1575.10 | 1589.77 | 1573.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1568.30 | 1585.48 | 1572.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1568.30 | 1585.48 | 1572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1558.70 | 1580.12 | 1571.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1558.70 | 1580.12 | 1571.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1566.70 | 1572.09 | 1569.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1561.20 | 1572.09 | 1569.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1554.80 | 1568.63 | 1568.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1557.00 | 1568.63 | 1568.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1544.60 | 1563.82 | 1566.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 1541.20 | 1559.30 | 1563.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1538.40 | 1528.67 | 1536.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1537.00 | 1530.33 | 1536.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 1536.60 | 1530.33 | 1536.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1531.40 | 1530.55 | 1536.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1553.90 | 1530.55 | 1536.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1571.20 | 1538.68 | 1539.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1571.20 | 1538.68 | 1539.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1544.00 | 1539.74 | 1539.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 1550.60 | 1539.74 | 1539.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1535.70 | 1538.93 | 1539.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 1532.10 | 1538.11 | 1539.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 1531.50 | 1537.29 | 1538.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1528.20 | 1537.29 | 1538.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1544.00 | 1534.76 | 1534.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1544.00 | 1534.76 | 1534.17 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1532.50 | 1533.78 | 1533.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 15:15:00 | 1521.00 | 1530.36 | 1532.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1542.40 | 1532.77 | 1533.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 1539.00 | 1534.02 | 1533.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 1550.10 | 1537.23 | 1535.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 1532.00 | 1537.47 | 1537.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1526.00 | 1535.18 | 1536.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 1533.90 | 1530.91 | 1533.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1529.50 | 1530.63 | 1533.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 1526.90 | 1529.88 | 1532.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1524.00 | 1515.76 | 1519.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1523.20 | 1517.25 | 1519.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1534.30 | 1521.44 | 1521.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1534.30 | 1521.44 | 1521.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1559.60 | 1534.52 | 1528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 1595.80 | 1597.09 | 1572.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:30:00 | 1595.90 | 1597.09 | 1572.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1565.90 | 1588.73 | 1572.45 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1551.80 | 1564.36 | 1564.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1537.30 | 1558.95 | 1562.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 1550.50 | 1545.17 | 1551.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1547.70 | 1545.68 | 1551.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:45:00 | 1556.40 | 1545.68 | 1551.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1525.00 | 1535.18 | 1544.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 1523.30 | 1531.83 | 1540.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:15:00 | 1521.00 | 1531.83 | 1540.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 1523.20 | 1529.17 | 1538.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 1523.30 | 1529.17 | 1538.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1523.30 | 1521.11 | 1527.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1523.30 | 1521.11 | 1527.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1510.20 | 1518.93 | 1526.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1526.70 | 1518.93 | 1526.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1528.50 | 1520.84 | 1526.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 1522.00 | 1524.14 | 1527.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 1518.00 | 1525.52 | 1527.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 1505.00 | 1483.55 | 1482.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1505.00 | 1483.55 | 1482.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1530.10 | 1500.89 | 1491.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1532.70 | 1542.45 | 1530.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1532.70 | 1542.45 | 1530.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1541.20 | 1542.20 | 1531.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 1550.00 | 1543.02 | 1532.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 1551.10 | 1542.61 | 1533.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:15:00 | 1547.60 | 1542.61 | 1533.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1555.60 | 1544.78 | 1536.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1586.20 | 1584.89 | 1577.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1566.50 | 1584.89 | 1577.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1580.10 | 1583.93 | 1578.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1579.60 | 1583.93 | 1578.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1580.60 | 1583.26 | 1578.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1558.80 | 1575.35 | 1576.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1558.80 | 1575.35 | 1576.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1540.50 | 1564.87 | 1571.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1501.90 | 1500.09 | 1512.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1501.90 | 1500.09 | 1512.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1506.70 | 1502.12 | 1511.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 1504.00 | 1502.12 | 1511.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1512.10 | 1504.12 | 1511.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 1508.60 | 1504.12 | 1511.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1503.00 | 1503.89 | 1510.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 1491.80 | 1503.89 | 1510.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 1502.00 | 1499.78 | 1505.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:00:00 | 1501.00 | 1500.03 | 1505.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1493.60 | 1501.20 | 1504.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1505.20 | 1502.00 | 1504.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 1505.20 | 1502.00 | 1504.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1505.40 | 1502.68 | 1504.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1505.10 | 1502.68 | 1504.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1507.00 | 1503.54 | 1505.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 1507.00 | 1503.54 | 1505.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 1505.10 | 1503.85 | 1505.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1505.00 | 1503.85 | 1505.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1510.00 | 1504.99 | 1505.14 | SL hit (close>static) qty=1.00 sl=1508.10 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1519.40 | 1507.87 | 1506.44 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1500.60 | 1511.09 | 1511.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1498.90 | 1508.65 | 1510.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 1511.70 | 1507.51 | 1509.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1506.20 | 1507.25 | 1509.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1481.80 | 1507.25 | 1509.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 09:15:00 | 1407.71 | 1424.34 | 1435.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 1416.60 | 1407.09 | 1415.37 | SL hit (close>ema200) qty=0.50 sl=1407.09 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1340.00 | 1327.66 | 1326.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1349.80 | 1340.52 | 1334.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1334.30 | 1339.56 | 1334.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1336.70 | 1338.99 | 1335.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 1337.70 | 1338.99 | 1335.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1332.00 | 1337.04 | 1335.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:15:00 | 1332.00 | 1337.04 | 1335.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1332.00 | 1336.03 | 1334.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:30:00 | 1335.30 | 1334.65 | 1334.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 1335.70 | 1334.65 | 1334.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1332.30 | 1333.67 | 1333.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1332.30 | 1333.67 | 1333.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 1327.90 | 1332.17 | 1333.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1336.00 | 1332.39 | 1333.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1328.50 | 1331.61 | 1332.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1338.40 | 1331.61 | 1332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1324.30 | 1326.35 | 1329.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 1319.10 | 1325.46 | 1328.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1355.30 | 1327.22 | 1327.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1355.30 | 1327.22 | 1327.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1360.00 | 1343.71 | 1339.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 1360.00 | 1360.12 | 1353.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 1360.00 | 1360.12 | 1353.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1354.50 | 1359.00 | 1353.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1352.90 | 1359.00 | 1353.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1365.00 | 1360.20 | 1354.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1384.50 | 1359.23 | 1357.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 1370.60 | 1366.28 | 1361.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1364.60 | 1373.90 | 1374.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1364.60 | 1373.90 | 1374.81 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1376.60 | 1373.43 | 1373.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1390.40 | 1377.07 | 1375.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1380.00 | 1386.26 | 1382.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1380.30 | 1385.06 | 1381.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1380.00 | 1385.06 | 1381.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1380.50 | 1384.15 | 1381.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1380.50 | 1384.15 | 1381.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1380.00 | 1383.32 | 1381.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1380.00 | 1383.32 | 1381.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1369.10 | 1380.48 | 1380.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1353.90 | 1370.52 | 1374.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 15:15:00 | 1353.90 | 1353.60 | 1362.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 1346.90 | 1353.60 | 1362.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1357.40 | 1354.36 | 1361.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1331.00 | 1355.16 | 1356.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 1335.20 | 1352.27 | 1354.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1337.60 | 1349.34 | 1353.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1305.00 | 1342.34 | 1347.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1300.40 | 1333.95 | 1343.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1292.50 | 1314.60 | 1328.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 1343.00 | 1323.24 | 1320.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1343.00 | 1323.24 | 1320.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1352.50 | 1336.40 | 1332.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1330.60 | 1336.32 | 1332.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1344.60 | 1337.98 | 1334.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 1349.50 | 1343.80 | 1339.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 1349.10 | 1345.62 | 1341.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 1349.40 | 1348.07 | 1344.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:30:00 | 1350.30 | 1348.16 | 1344.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1345.70 | 1350.91 | 1347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 1340.00 | 1350.91 | 1347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1351.30 | 1350.99 | 1348.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1334.80 | 1346.28 | 1347.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1334.80 | 1346.28 | 1347.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1324.40 | 1341.90 | 1345.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 12:15:00 | 1332.60 | 1331.62 | 1336.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:45:00 | 1333.20 | 1331.62 | 1336.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1336.80 | 1332.65 | 1336.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:45:00 | 1336.80 | 1332.65 | 1336.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1336.10 | 1333.34 | 1336.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 1335.10 | 1333.34 | 1336.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1335.10 | 1333.69 | 1336.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1360.00 | 1333.69 | 1336.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1351.40 | 1337.24 | 1337.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 1351.70 | 1337.24 | 1337.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 1348.10 | 1339.41 | 1338.77 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1338.40 | 1345.09 | 1345.15 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1349.50 | 1344.87 | 1344.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 1352.30 | 1347.71 | 1345.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 15:15:00 | 1347.50 | 1347.67 | 1346.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:15:00 | 1350.20 | 1347.67 | 1346.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1350.70 | 1348.27 | 1346.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:45:00 | 1366.00 | 1352.33 | 1349.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 1361.10 | 1356.11 | 1351.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1348.40 | 1365.74 | 1366.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 1348.40 | 1365.74 | 1366.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 1342.30 | 1356.88 | 1361.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1315.90 | 1315.44 | 1326.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 1315.90 | 1315.44 | 1326.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1321.50 | 1316.51 | 1325.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1321.50 | 1316.51 | 1325.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1320.80 | 1317.37 | 1324.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1318.00 | 1317.37 | 1324.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1316.50 | 1317.20 | 1324.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 1311.00 | 1315.35 | 1322.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1336.60 | 1320.21 | 1321.66 | SL hit (close>static) qty=1.00 sl=1335.10 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1352.50 | 1326.67 | 1324.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1358.50 | 1343.66 | 1334.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1356.90 | 1357.99 | 1351.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 14:45:00 | 1360.50 | 1358.39 | 1352.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 15:15:00 | 1361.90 | 1358.39 | 1352.19 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1346.00 | 1356.48 | 1352.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1346.00 | 1356.48 | 1352.43 | SL hit (close<ema400) qty=1.00 sl=1352.43 alert=retest1 |

### Cycle 40 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1337.00 | 1348.86 | 1349.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1331.70 | 1342.73 | 1346.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1320.30 | 1320.24 | 1326.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1320.30 | 1320.24 | 1326.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1313.50 | 1318.46 | 1323.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1309.80 | 1314.80 | 1319.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 1310.10 | 1314.44 | 1318.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1304.10 | 1318.23 | 1319.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:00:00 | 1307.00 | 1313.40 | 1315.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1285.20 | 1293.09 | 1300.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1270.90 | 1290.67 | 1298.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1270.00 | 1285.06 | 1292.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1244.31 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1244.59 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1241.65 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 1256.90 | 1256.21 | 1258.95 | SL hit (close>ema200) qty=0.50 sl=1256.21 alert=retest2 |

### Cycle 41 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1138.90 | 1124.46 | 1122.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1142.80 | 1128.13 | 1124.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1214.50 | 1214.73 | 1196.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1207.00 | 1214.34 | 1205.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1207.00 | 1214.34 | 1205.36 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1192.90 | 1200.88 | 1201.69 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1222.30 | 1201.37 | 1199.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1232.60 | 1207.62 | 1202.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 1208.50 | 1213.14 | 1207.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1216.30 | 1213.77 | 1208.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1212.80 | 1213.77 | 1208.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1209.00 | 1212.25 | 1209.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1207.20 | 1211.36 | 1209.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1202.40 | 1209.57 | 1208.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1202.40 | 1209.57 | 1208.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1212.80 | 1210.22 | 1209.07 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1192.40 | 1206.06 | 1207.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 1192.10 | 1200.14 | 1204.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 15:15:00 | 1200.00 | 1199.11 | 1202.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 1199.00 | 1199.11 | 1202.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1222.00 | 1203.83 | 1204.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1222.00 | 1203.83 | 1204.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1215.90 | 1206.24 | 1205.28 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1200.10 | 1206.51 | 1207.01 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1210.60 | 1207.12 | 1207.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1214.70 | 1209.12 | 1208.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 1206.40 | 1209.08 | 1208.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1207.70 | 1208.81 | 1208.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1196.70 | 1208.81 | 1208.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1204.90 | 1208.02 | 1208.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1187.80 | 1197.10 | 1202.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1190.80 | 1186.39 | 1193.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:00:00 | 1190.80 | 1186.39 | 1193.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1195.00 | 1188.11 | 1193.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 1195.00 | 1188.11 | 1193.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1201.00 | 1190.69 | 1194.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1190.00 | 1190.69 | 1194.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1199.40 | 1193.83 | 1195.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 1191.60 | 1193.83 | 1195.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 11:15:00 | 1212.30 | 1197.52 | 1196.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 1223.50 | 1205.08 | 1200.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 15:15:00 | 1204.40 | 1206.15 | 1201.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 1195.40 | 1206.15 | 1201.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1185.90 | 1202.10 | 1200.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 1185.90 | 1202.10 | 1200.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 1176.60 | 1197.00 | 1198.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 11:15:00 | 1170.10 | 1191.62 | 1195.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 1199.30 | 1192.32 | 1195.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1199.70 | 1193.79 | 1195.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:15:00 | 1206.60 | 1193.79 | 1195.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1210.00 | 1197.03 | 1197.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1210.00 | 1197.03 | 1197.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 1213.00 | 1200.23 | 1198.68 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 1179.30 | 1194.87 | 1196.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 11:15:00 | 1176.50 | 1191.20 | 1194.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1200.20 | 1188.38 | 1192.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1182.10 | 1187.12 | 1191.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1164.10 | 1187.12 | 1191.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 15:15:00 | 1176.30 | 1175.46 | 1181.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 1180.60 | 1175.79 | 1178.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:15:00 | 1180.90 | 1175.79 | 1178.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1183.40 | 1177.31 | 1179.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 1185.20 | 1177.31 | 1179.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1182.50 | 1178.35 | 1179.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1194.20 | 1178.35 | 1179.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1197.10 | 1182.10 | 1181.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1197.10 | 1182.10 | 1181.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 15:15:00 | 1206.10 | 1196.38 | 1189.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 1169.30 | 1190.97 | 1187.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1170.30 | 1186.83 | 1186.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1170.00 | 1186.83 | 1186.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 1172.00 | 1183.87 | 1184.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1142.00 | 1170.28 | 1177.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1126.70 | 1125.01 | 1139.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1126.70 | 1125.01 | 1139.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1120.10 | 1125.47 | 1132.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:00:00 | 1113.60 | 1123.09 | 1130.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1109.50 | 1120.02 | 1126.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 1114.00 | 1116.50 | 1118.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1142.00 | 1116.88 | 1114.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1142.00 | 1116.88 | 1114.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 1146.50 | 1128.65 | 1120.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1162.00 | 1164.12 | 1149.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 1169.80 | 1165.26 | 1151.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1154.50 | 1163.37 | 1156.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1154.50 | 1163.37 | 1156.00 | SL hit (close<ema400) qty=1.00 sl=1156.00 alert=retest1 |

### Cycle 56 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1130.50 | 1155.40 | 1158.18 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1166.80 | 1156.05 | 1155.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1193.10 | 1169.69 | 1163.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1211.10 | 1215.11 | 1192.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 1230.10 | 1217.71 | 1195.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:30:00 | 1229.20 | 1221.10 | 1199.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:00:00 | 1232.80 | 1245.44 | 1223.44 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1258.30 | 1250.36 | 1236.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1272.90 | 1253.33 | 1245.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 1265.70 | 1273.53 | 1266.92 | SL hit (close<ema400) qty=1.00 sl=1266.92 alert=retest1 |

### Cycle 58 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 1239.00 | 1258.84 | 1261.50 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1273.90 | 1256.18 | 1254.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 1278.70 | 1260.69 | 1256.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 13:15:00 | 1286.10 | 1288.41 | 1276.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 1286.10 | 1288.41 | 1276.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1300.00 | 1292.16 | 1281.17 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 1266.10 | 1277.69 | 1278.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1250.90 | 1272.33 | 1275.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1256.20 | 1246.47 | 1257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1261.10 | 1249.39 | 1257.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1263.40 | 1249.39 | 1257.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1256.80 | 1250.87 | 1257.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:30:00 | 1256.50 | 1250.87 | 1257.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1257.00 | 1252.10 | 1257.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 1250.20 | 1253.86 | 1257.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 1261.00 | 1249.73 | 1251.32 | SL hit (close>static) qty=1.00 sl=1259.30 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1262.60 | 1254.09 | 1253.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 1265.80 | 1258.10 | 1255.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1251.30 | 1254.30 | 1254.37 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1268.00 | 1256.09 | 1255.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 1248.50 | 1254.70 | 1255.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 1241.20 | 1251.67 | 1253.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 1257.90 | 1250.75 | 1252.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1260.00 | 1252.60 | 1253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1270.00 | 1252.60 | 1253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1275.10 | 1257.10 | 1255.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1282.30 | 1266.22 | 1260.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 1290.50 | 1290.98 | 1280.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 1289.70 | 1290.98 | 1280.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1297.90 | 1292.37 | 1282.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1301.00 | 1292.37 | 1282.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 13:00:00 | 1349.50 | 2025-05-12 13:15:00 | 1348.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-13 14:15:00 | 1358.20 | 2025-05-19 09:15:00 | 1359.50 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-05-21 12:00:00 | 1339.50 | 2025-05-23 09:15:00 | 1362.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1412.50 | 2025-06-02 14:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1403.80 | 2025-06-02 14:15:00 | 1383.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-10 09:45:00 | 1417.00 | 2025-06-11 09:15:00 | 1394.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-10 13:45:00 | 1418.30 | 2025-06-11 09:15:00 | 1394.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-20 09:30:00 | 1454.40 | 2025-06-24 13:15:00 | 1441.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1471.80 | 2025-06-24 13:15:00 | 1441.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-07-11 12:30:00 | 1532.10 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-11 13:30:00 | 1531.50 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-11 14:15:00 | 1528.20 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-18 15:00:00 | 1526.90 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1524.00 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1523.20 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-30 11:30:00 | 1523.30 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-07-30 12:15:00 | 1521.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-07-30 13:45:00 | 1523.20 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-07-30 14:15:00 | 1523.30 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-08-01 11:30:00 | 1522.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-08-04 09:30:00 | 1518.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-08-14 11:45:00 | 1550.00 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-08-14 12:30:00 | 1551.10 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-08-14 13:15:00 | 1547.60 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1555.60 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-08-29 15:15:00 | 1491.80 | 2025-09-03 09:15:00 | 1510.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-01 13:00:00 | 1502.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-01 14:00:00 | 1501.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1493.60 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1505.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1481.80 | 2025-09-22 09:15:00 | 1407.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1481.80 | 2025-09-23 14:15:00 | 1416.60 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2025-10-14 09:30:00 | 1335.30 | 2025-10-14 11:15:00 | 1332.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-10-14 10:15:00 | 1335.70 | 2025-10-14 11:15:00 | 1332.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-16 11:15:00 | 1319.10 | 2025-10-17 09:15:00 | 1355.30 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1384.50 | 2025-10-31 11:15:00 | 1364.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-28 12:45:00 | 1370.60 | 2025-10-31 11:15:00 | 1364.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-14 09:15:00 | 1331.00 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-14 10:15:00 | 1335.20 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-14 11:00:00 | 1337.60 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1305.00 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1292.50 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-11-26 11:15:00 | 1349.50 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-26 12:45:00 | 1349.10 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-27 11:00:00 | 1349.40 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-27 11:30:00 | 1350.30 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-10 14:45:00 | 1366.00 | 2025-12-15 12:15:00 | 1348.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-11 10:00:00 | 1361.10 | 2025-12-15 12:15:00 | 1348.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-19 11:30:00 | 1311.00 | 2025-12-22 09:15:00 | 1336.60 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-12-24 14:45:00 | 1360.50 | 2025-12-26 09:15:00 | 1346.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2025-12-24 15:15:00 | 1361.90 | 2025-12-26 09:15:00 | 1346.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1309.80 | 2026-01-16 09:15:00 | 1244.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1310.10 | 2026-01-16 09:15:00 | 1244.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1304.10 | 2026-01-16 09:15:00 | 1241.65 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1309.80 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1310.10 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1304.10 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2026-01-06 11:00:00 | 1307.00 | 2026-01-20 09:15:00 | 1238.89 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1270.90 | 2026-01-20 13:15:00 | 1207.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1270.00 | 2026-01-20 13:15:00 | 1206.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:00:00 | 1307.00 | 2026-01-22 14:15:00 | 1173.69 | TARGET_HIT | 0.50 | 10.20% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1270.90 | 2026-01-22 15:15:00 | 1194.00 | STOP_HIT | 0.50 | 6.05% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1270.00 | 2026-01-22 15:15:00 | 1194.00 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1164.10 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-03-10 13:45:00 | 1180.60 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-10 14:15:00 | 1180.90 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1113.60 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1109.50 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2026-03-20 15:00:00 | 1114.00 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2026-03-27 11:00:00 | 1169.80 | 2026-03-27 15:15:00 | 1154.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-30 11:45:00 | 1148.90 | 2026-04-02 09:15:00 | 1123.60 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-30 14:45:00 | 1157.00 | 2026-04-02 09:15:00 | 1123.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2026-04-08 09:45:00 | 1230.10 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest1 | 2026-04-08 10:30:00 | 1229.20 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest1 | 2026-04-09 10:00:00 | 1232.80 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1272.90 | 2026-04-16 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-04-27 15:15:00 | 1250.20 | 2026-04-29 10:15:00 | 1261.00 | STOP_HIT | 1.00 | -0.86% |
