# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1447.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 96 |
| ALERT2 | 96 |
| ALERT2_SKIP | 54 |
| ALERT3 | 227 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 114 |
| PARTIAL | 33 |
| TARGET_HIT | 6 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 78
- **Target hits / Stop hits / Partials:** 6 / 111 / 33
- **Avg / median % per leg:** 1.12% / -0.28%
- **Sum % (uncompounded):** 168.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 12 | 21.1% | 1 | 56 | 0 | -1.07% | -61.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| BUY @ 3rd Alert (retest2) | 54 | 12 | 22.2% | 1 | 53 | 0 | -1.09% | -58.9% |
| SELL (all) | 93 | 60 | 64.5% | 5 | 55 | 33 | 2.47% | 229.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 93 | 60 | 64.5% | 5 | 55 | 33 | 2.47% | 229.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| retest2 (combined) | 147 | 72 | 49.0% | 6 | 108 | 33 | 1.16% | 170.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 957.13 | 933.09 | 932.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 965.38 | 947.54 | 940.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1544.00 | 1544.81 | 1490.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:45:00 | 1536.50 | 1544.81 | 1490.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1455.00 | 1527.68 | 1491.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 1455.00 | 1527.68 | 1491.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1448.00 | 1511.74 | 1487.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 1448.00 | 1511.74 | 1487.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 1454.00 | 1473.65 | 1475.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 1487.90 | 1477.07 | 1476.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 1534.50 | 1488.56 | 1481.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 09:15:00 | 1504.90 | 1535.97 | 1521.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 1504.90 | 1535.97 | 1521.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1504.90 | 1535.97 | 1521.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 1504.90 | 1535.97 | 1521.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1516.00 | 1531.98 | 1520.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:15:00 | 1534.95 | 1526.94 | 1520.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1436.90 | 1544.32 | 1547.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1436.90 | 1544.32 | 1547.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1293.20 | 1436.29 | 1486.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1387.55 | 1344.87 | 1402.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1387.55 | 1344.87 | 1402.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1387.55 | 1344.87 | 1402.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1396.25 | 1344.87 | 1402.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1394.00 | 1354.69 | 1401.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1396.70 | 1354.69 | 1401.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1385.80 | 1368.05 | 1388.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 1390.70 | 1368.05 | 1388.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 1381.55 | 1370.75 | 1387.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:30:00 | 1391.35 | 1370.75 | 1387.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 1397.75 | 1377.68 | 1388.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:00:00 | 1397.75 | 1377.68 | 1388.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 1431.80 | 1388.50 | 1392.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 14:00:00 | 1431.80 | 1388.50 | 1392.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 14:15:00 | 1434.15 | 1397.63 | 1395.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1448.65 | 1412.97 | 1403.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 1424.05 | 1424.11 | 1411.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:45:00 | 1427.45 | 1424.11 | 1411.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1419.90 | 1423.49 | 1413.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1453.25 | 1422.77 | 1414.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 1426.80 | 1424.62 | 1418.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1429.90 | 1424.16 | 1419.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:45:00 | 1431.05 | 1426.58 | 1422.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1412.80 | 1423.82 | 1421.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 1413.95 | 1423.82 | 1421.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1417.20 | 1422.50 | 1420.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 15:15:00 | 1423.00 | 1422.00 | 1420.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 1412.70 | 1419.62 | 1419.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 10:15:00 | 1412.70 | 1419.62 | 1419.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 11:15:00 | 1406.70 | 1417.03 | 1418.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 13:15:00 | 1437.50 | 1419.30 | 1419.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 13:15:00 | 1437.50 | 1419.30 | 1419.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 1437.50 | 1419.30 | 1419.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 1437.50 | 1419.30 | 1419.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 14:15:00 | 1490.30 | 1433.50 | 1425.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 15:15:00 | 1496.75 | 1446.15 | 1432.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1565.75 | 1599.02 | 1558.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1565.75 | 1599.02 | 1558.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1565.75 | 1599.02 | 1558.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1572.00 | 1599.02 | 1558.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1559.95 | 1591.21 | 1558.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1556.00 | 1591.21 | 1558.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1573.60 | 1587.69 | 1559.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:15:00 | 1584.80 | 1570.83 | 1562.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1544.00 | 1564.73 | 1562.99 | SL hit (close<static) qty=1.00 sl=1558.65 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 1541.20 | 1560.02 | 1561.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 1528.00 | 1549.71 | 1555.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 1551.65 | 1543.26 | 1550.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 1551.65 | 1543.26 | 1550.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1551.65 | 1543.26 | 1550.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 1551.65 | 1543.26 | 1550.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1556.00 | 1545.81 | 1550.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 1561.20 | 1545.81 | 1550.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1557.95 | 1548.24 | 1551.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:30:00 | 1557.85 | 1548.24 | 1551.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1553.30 | 1549.25 | 1551.53 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1558.10 | 1552.70 | 1552.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 10:15:00 | 1581.00 | 1558.36 | 1555.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 1554.55 | 1558.66 | 1555.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 1554.55 | 1558.66 | 1555.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1554.55 | 1558.66 | 1555.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 1554.55 | 1558.66 | 1555.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1561.15 | 1559.16 | 1556.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 1586.90 | 1555.55 | 1555.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 10:15:00 | 1745.59 | 1696.29 | 1662.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 1660.25 | 1684.83 | 1687.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1639.40 | 1672.90 | 1680.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 1638.00 | 1636.50 | 1654.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 1648.00 | 1636.50 | 1654.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1649.50 | 1639.10 | 1654.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 1649.50 | 1639.10 | 1654.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1649.85 | 1642.09 | 1650.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 1649.85 | 1642.09 | 1650.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1663.95 | 1646.46 | 1652.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 1647.85 | 1650.93 | 1653.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 15:15:00 | 1668.00 | 1653.00 | 1652.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 15:15:00 | 1668.00 | 1653.00 | 1652.74 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 1649.50 | 1652.30 | 1652.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 1631.80 | 1645.13 | 1648.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 1497.00 | 1490.77 | 1520.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 13:00:00 | 1497.00 | 1490.77 | 1520.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1501.25 | 1494.28 | 1512.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1427.00 | 1499.93 | 1512.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 1355.65 | 1491.67 | 1507.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 1431.00 | 1419.31 | 1439.52 | SL hit (close>ema200) qty=0.50 sl=1419.31 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 1452.00 | 1436.98 | 1436.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 1465.60 | 1442.70 | 1438.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 1463.80 | 1469.57 | 1459.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 1463.80 | 1469.57 | 1459.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1463.80 | 1469.57 | 1459.53 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 1437.95 | 1455.19 | 1457.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 1434.10 | 1450.97 | 1455.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 1454.00 | 1448.87 | 1453.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 14:15:00 | 1454.00 | 1448.87 | 1453.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1454.00 | 1448.87 | 1453.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1454.00 | 1448.87 | 1453.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1445.40 | 1448.17 | 1452.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 1425.10 | 1448.17 | 1452.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 1353.84 | 1415.11 | 1431.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 1365.35 | 1362.30 | 1390.41 | SL hit (close>ema200) qty=0.50 sl=1362.30 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1404.95 | 1380.42 | 1378.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1426.90 | 1389.72 | 1382.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 15:15:00 | 1430.30 | 1431.56 | 1417.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:15:00 | 1318.30 | 1431.56 | 1417.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 16 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1315.35 | 1408.32 | 1408.56 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 1317.55 | 1316.62 | 1316.52 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 1314.00 | 1316.61 | 1316.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 1312.75 | 1315.87 | 1316.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1312.90 | 1312.67 | 1314.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1312.90 | 1312.67 | 1314.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1312.90 | 1312.67 | 1314.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 10:15:00 | 1309.00 | 1312.67 | 1314.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 1311.25 | 1313.92 | 1314.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 12:15:00 | 1317.20 | 1314.72 | 1314.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 1317.20 | 1314.72 | 1314.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 1325.10 | 1316.79 | 1315.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 09:15:00 | 1302.60 | 1315.42 | 1315.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 1302.60 | 1315.42 | 1315.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1302.60 | 1315.42 | 1315.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 1302.60 | 1315.42 | 1315.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1286.45 | 1309.63 | 1312.72 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1321.50 | 1309.92 | 1308.52 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 1293.45 | 1305.15 | 1306.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 1288.55 | 1300.90 | 1304.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 14:15:00 | 1305.00 | 1298.88 | 1302.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 14:15:00 | 1305.00 | 1298.88 | 1302.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1305.00 | 1298.88 | 1302.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 1305.00 | 1298.88 | 1302.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1302.00 | 1299.50 | 1302.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 1347.80 | 1299.50 | 1302.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 1335.00 | 1306.60 | 1305.23 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1312.05 | 1320.32 | 1320.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1292.75 | 1311.61 | 1316.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1250.75 | 1250.28 | 1270.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 1251.25 | 1254.08 | 1263.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1251.25 | 1254.08 | 1263.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:45:00 | 1252.25 | 1254.08 | 1263.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1244.35 | 1241.25 | 1247.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 09:45:00 | 1232.15 | 1240.48 | 1244.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:15:00 | 1170.54 | 1189.13 | 1206.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 11:15:00 | 1108.94 | 1147.57 | 1175.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1168.50 | 1161.10 | 1160.57 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 1136.75 | 1157.40 | 1159.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 1124.30 | 1141.23 | 1148.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 1126.90 | 1123.72 | 1133.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:30:00 | 1128.80 | 1123.72 | 1133.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1123.00 | 1123.18 | 1129.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:45:00 | 1124.80 | 1123.18 | 1129.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1129.05 | 1124.49 | 1129.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 1124.70 | 1124.49 | 1129.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1135.75 | 1126.74 | 1129.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 1138.25 | 1126.74 | 1129.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1140.00 | 1129.39 | 1130.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 1140.00 | 1129.39 | 1130.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 1140.55 | 1133.06 | 1132.27 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1124.25 | 1130.90 | 1131.43 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 1164.15 | 1136.94 | 1133.60 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 1125.00 | 1133.33 | 1134.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 1111.25 | 1128.91 | 1132.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 15:15:00 | 1124.50 | 1123.91 | 1127.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:15:00 | 1100.60 | 1123.91 | 1127.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1119.80 | 1123.09 | 1126.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:45:00 | 1085.75 | 1108.48 | 1117.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1139.45 | 1105.79 | 1106.37 | SL hit (close>static) qty=1.00 sl=1131.05 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 1130.00 | 1110.63 | 1108.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1165.90 | 1138.32 | 1124.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1206.00 | 1214.82 | 1199.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 1206.00 | 1214.82 | 1199.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1206.00 | 1214.82 | 1199.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 1206.00 | 1214.82 | 1199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1197.95 | 1209.89 | 1199.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:00:00 | 1197.95 | 1209.89 | 1199.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 1194.70 | 1206.85 | 1198.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:30:00 | 1196.00 | 1206.85 | 1198.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1191.00 | 1203.68 | 1198.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:45:00 | 1190.65 | 1203.68 | 1198.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 1172.55 | 1193.21 | 1194.45 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1207.95 | 1197.53 | 1196.29 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 1190.00 | 1197.89 | 1198.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1172.65 | 1191.71 | 1195.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1074.80 | 1070.14 | 1093.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1074.80 | 1070.14 | 1093.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1036.10 | 1031.21 | 1042.75 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1048.00 | 1045.47 | 1045.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1077.75 | 1052.01 | 1048.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 1058.95 | 1059.12 | 1052.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 1058.95 | 1059.12 | 1052.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1054.45 | 1058.19 | 1053.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 1054.45 | 1058.19 | 1053.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1064.80 | 1059.51 | 1054.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:15:00 | 1068.50 | 1059.81 | 1054.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 15:15:00 | 1055.00 | 1067.78 | 1068.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 1055.00 | 1067.78 | 1068.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1038.05 | 1061.83 | 1066.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1066.15 | 1045.18 | 1052.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1066.15 | 1045.18 | 1052.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1066.15 | 1045.18 | 1052.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 1066.00 | 1045.18 | 1052.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1070.50 | 1050.25 | 1054.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 1070.50 | 1050.25 | 1054.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1084.60 | 1057.12 | 1056.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1089.00 | 1063.49 | 1059.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 1075.70 | 1079.65 | 1071.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 1075.70 | 1079.65 | 1071.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1074.00 | 1078.03 | 1072.41 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1055.25 | 1067.66 | 1069.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1041.50 | 1054.69 | 1061.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1060.80 | 1055.91 | 1061.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 1060.80 | 1055.91 | 1061.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1050.50 | 1054.83 | 1060.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 1046.05 | 1054.83 | 1060.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:00:00 | 1046.35 | 1052.53 | 1058.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 1042.80 | 1050.58 | 1057.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 1044.25 | 1050.07 | 1056.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1020.05 | 1033.37 | 1044.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:45:00 | 1039.35 | 1033.37 | 1044.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 993.75 | 1023.66 | 1037.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 994.03 | 1023.66 | 1037.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 990.66 | 1023.66 | 1037.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 992.04 | 1023.66 | 1037.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 1001.10 | 997.18 | 1014.48 | SL hit (close>ema200) qty=0.50 sl=997.18 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 993.95 | 958.76 | 954.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1027.10 | 991.05 | 974.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 1141.25 | 1147.50 | 1108.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1161.60 | 1150.15 | 1128.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1161.60 | 1150.15 | 1128.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1185.30 | 1151.73 | 1145.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 1197.95 | 1210.44 | 1210.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 1197.95 | 1210.44 | 1210.47 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1213.00 | 1207.97 | 1207.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 09:15:00 | 1232.00 | 1213.42 | 1210.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 10:15:00 | 1220.50 | 1223.72 | 1218.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 10:15:00 | 1220.50 | 1223.72 | 1218.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1220.50 | 1223.72 | 1218.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:00:00 | 1220.50 | 1223.72 | 1218.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1228.80 | 1224.74 | 1219.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:15:00 | 1232.05 | 1226.06 | 1221.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 12:45:00 | 1239.90 | 1272.32 | 1266.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 1228.05 | 1261.17 | 1262.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 1228.05 | 1261.17 | 1262.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 1203.50 | 1226.38 | 1239.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 14:15:00 | 1256.20 | 1226.13 | 1236.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 1256.20 | 1226.13 | 1236.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1256.20 | 1226.13 | 1236.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1256.20 | 1226.13 | 1236.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1265.55 | 1234.01 | 1239.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 1225.90 | 1234.01 | 1239.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 14:15:00 | 1237.75 | 1229.86 | 1234.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:15:00 | 1175.86 | 1195.67 | 1206.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 12:15:00 | 1199.30 | 1196.13 | 1204.76 | SL hit (close>ema200) qty=0.50 sl=1196.13 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 1158.95 | 1140.84 | 1140.61 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1119.90 | 1141.65 | 1142.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1104.80 | 1134.28 | 1139.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 12:15:00 | 1140.00 | 1134.49 | 1138.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 12:15:00 | 1140.00 | 1134.49 | 1138.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 1140.00 | 1134.49 | 1138.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:45:00 | 1144.95 | 1134.49 | 1138.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1134.75 | 1134.54 | 1137.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 15:00:00 | 1120.00 | 1131.63 | 1136.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 15:15:00 | 1143.95 | 1134.10 | 1137.02 | SL hit (close>static) qty=1.00 sl=1141.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 1157.00 | 1138.88 | 1138.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 1173.15 | 1145.73 | 1141.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 1163.45 | 1164.75 | 1155.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 12:00:00 | 1163.45 | 1164.75 | 1155.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1163.30 | 1185.62 | 1177.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1163.30 | 1185.62 | 1177.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1170.60 | 1182.62 | 1176.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 1180.60 | 1182.62 | 1176.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 10:15:00 | 1147.90 | 1179.43 | 1179.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 1147.90 | 1179.43 | 1179.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 1141.45 | 1171.83 | 1176.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1138.90 | 1137.74 | 1153.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1138.90 | 1137.74 | 1153.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1150.00 | 1140.76 | 1149.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 1149.85 | 1140.76 | 1149.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1150.00 | 1142.61 | 1149.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 1150.00 | 1142.61 | 1149.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1138.55 | 1141.80 | 1148.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:15:00 | 1133.20 | 1140.74 | 1146.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 1133.15 | 1139.22 | 1145.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 1189.55 | 1152.49 | 1149.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1189.55 | 1152.49 | 1149.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 1203.50 | 1178.66 | 1164.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1291.65 | 1309.92 | 1274.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 1291.65 | 1309.92 | 1274.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1282.15 | 1296.33 | 1280.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1282.15 | 1296.33 | 1280.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1275.55 | 1292.18 | 1280.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1257.40 | 1292.18 | 1280.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1241.60 | 1282.06 | 1276.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1249.55 | 1282.06 | 1276.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1222.95 | 1270.24 | 1271.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1209.00 | 1257.99 | 1266.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1252.75 | 1237.40 | 1250.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1252.75 | 1237.40 | 1250.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1252.75 | 1237.40 | 1250.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1242.60 | 1237.40 | 1250.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1293.00 | 1248.52 | 1254.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1293.00 | 1248.52 | 1254.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1296.20 | 1258.06 | 1258.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 1290.15 | 1258.06 | 1258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1280.00 | 1262.45 | 1260.09 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1251.15 | 1260.90 | 1261.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1244.95 | 1257.71 | 1260.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1216.60 | 1192.38 | 1213.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1216.60 | 1192.38 | 1213.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1216.60 | 1192.38 | 1213.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 1216.60 | 1192.38 | 1213.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1241.20 | 1202.15 | 1215.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1241.20 | 1202.15 | 1215.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1227.25 | 1207.17 | 1216.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 1214.10 | 1207.93 | 1216.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1244.00 | 1216.28 | 1218.62 | SL hit (close>static) qty=1.00 sl=1243.90 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 1241.40 | 1221.30 | 1220.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1265.05 | 1235.71 | 1228.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 1287.90 | 1305.32 | 1287.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1287.90 | 1305.32 | 1287.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1287.90 | 1305.32 | 1287.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1280.50 | 1305.32 | 1287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1202.60 | 1284.78 | 1279.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1202.60 | 1284.78 | 1279.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 1229.05 | 1273.63 | 1275.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1170.95 | 1249.60 | 1263.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1194.05 | 1193.39 | 1221.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1215.05 | 1196.75 | 1208.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1215.05 | 1196.75 | 1208.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1220.50 | 1196.75 | 1208.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1221.05 | 1201.61 | 1210.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:15:00 | 1223.15 | 1201.61 | 1210.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1217.80 | 1212.06 | 1213.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 1217.80 | 1212.06 | 1213.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 1215.00 | 1212.65 | 1213.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 1208.75 | 1212.65 | 1213.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1213.05 | 1212.73 | 1213.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 1213.05 | 1212.73 | 1213.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1199.15 | 1210.01 | 1212.02 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 12:15:00 | 1231.45 | 1214.35 | 1213.65 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1206.60 | 1212.80 | 1213.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1195.20 | 1208.65 | 1211.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 11:15:00 | 1195.75 | 1195.12 | 1201.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 11:15:00 | 1195.75 | 1195.12 | 1201.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1195.75 | 1195.12 | 1201.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:45:00 | 1217.75 | 1195.12 | 1201.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1196.00 | 1193.74 | 1198.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 1196.00 | 1193.74 | 1198.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1193.00 | 1193.59 | 1198.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1174.50 | 1193.59 | 1198.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 13:15:00 | 1115.77 | 1166.67 | 1182.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 1057.05 | 1136.38 | 1164.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 1187.45 | 1161.45 | 1158.03 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1129.45 | 1153.74 | 1156.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 1116.15 | 1146.22 | 1152.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 1083.75 | 1039.93 | 1064.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 1083.75 | 1039.93 | 1064.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1083.75 | 1039.93 | 1064.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1083.75 | 1039.93 | 1064.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1074.85 | 1046.91 | 1065.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1088.85 | 1046.91 | 1065.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1066.50 | 1050.83 | 1065.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 1078.25 | 1050.83 | 1065.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1021.50 | 1035.79 | 1046.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 1042.85 | 1035.79 | 1046.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1026.20 | 1014.95 | 1026.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 1030.90 | 1014.95 | 1026.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1026.05 | 1017.17 | 1026.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 13:30:00 | 1019.85 | 1016.81 | 1025.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 1032.20 | 1017.81 | 1022.91 | SL hit (close>static) qty=1.00 sl=1029.55 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 1032.90 | 989.68 | 986.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 1050.30 | 1018.10 | 1002.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 1119.25 | 1130.45 | 1111.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 1119.25 | 1130.45 | 1111.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1119.25 | 1130.45 | 1111.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1112.25 | 1130.45 | 1111.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1119.00 | 1126.92 | 1113.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 1120.00 | 1126.92 | 1113.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1108.00 | 1123.14 | 1112.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1108.00 | 1123.14 | 1112.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1120.80 | 1122.67 | 1113.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 1111.85 | 1122.67 | 1113.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1118.50 | 1121.84 | 1114.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 1118.50 | 1121.84 | 1114.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1118.00 | 1121.07 | 1114.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1129.00 | 1121.07 | 1114.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 10:30:00 | 1120.85 | 1119.44 | 1114.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 1098.90 | 1115.33 | 1113.32 | SL hit (close<static) qty=1.00 sl=1114.05 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 13:15:00 | 1096.75 | 1108.96 | 1110.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 15:15:00 | 1094.00 | 1103.77 | 1107.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1130.80 | 1109.18 | 1109.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1130.80 | 1109.18 | 1109.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1130.80 | 1109.18 | 1109.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 1130.80 | 1109.18 | 1109.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 1132.00 | 1113.74 | 1111.92 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1106.00 | 1111.25 | 1111.42 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1120.65 | 1113.13 | 1112.26 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 15:15:00 | 1106.00 | 1112.53 | 1112.79 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1124.25 | 1114.87 | 1113.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1163.45 | 1130.23 | 1122.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1309.10 | 1331.75 | 1294.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1309.10 | 1331.75 | 1294.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1331.00 | 1320.43 | 1305.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 1318.10 | 1320.43 | 1305.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1339.10 | 1325.70 | 1312.53 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 1286.80 | 1310.67 | 1312.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 1256.00 | 1275.77 | 1289.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 1292.35 | 1260.50 | 1270.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 1292.35 | 1260.50 | 1270.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1292.35 | 1260.50 | 1270.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1292.35 | 1260.50 | 1270.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1314.00 | 1271.20 | 1274.59 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1328.00 | 1282.56 | 1279.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1338.15 | 1293.68 | 1284.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1284.80 | 1323.54 | 1308.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1284.80 | 1323.54 | 1308.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1284.80 | 1323.54 | 1308.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1284.80 | 1323.54 | 1308.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1300.90 | 1319.02 | 1307.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 1304.15 | 1319.02 | 1307.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 1320.30 | 1311.98 | 1306.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1227.05 | 1298.03 | 1301.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1227.05 | 1298.03 | 1301.47 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1288.65 | 1278.03 | 1277.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1323.85 | 1288.63 | 1282.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 1400.40 | 1402.42 | 1375.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 1400.40 | 1402.42 | 1375.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1382.00 | 1397.31 | 1377.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1382.80 | 1397.31 | 1377.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 1409.00 | 1417.33 | 1411.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 1409.00 | 1417.33 | 1411.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 1409.80 | 1415.82 | 1411.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1431.60 | 1415.82 | 1411.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1391.90 | 1411.04 | 1409.57 | SL hit (close<static) qty=1.00 sl=1405.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1398.30 | 1408.49 | 1408.54 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 1420.10 | 1410.54 | 1409.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 1423.50 | 1413.13 | 1410.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 1419.70 | 1423.07 | 1417.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:15:00 | 1420.00 | 1423.07 | 1417.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1417.00 | 1421.86 | 1417.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:45:00 | 1417.10 | 1421.86 | 1417.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1420.90 | 1421.67 | 1417.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 1414.90 | 1421.67 | 1417.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1403.10 | 1419.53 | 1417.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1403.10 | 1419.53 | 1417.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1386.00 | 1412.83 | 1415.01 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1485.00 | 1423.57 | 1418.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 1500.00 | 1449.95 | 1431.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1473.60 | 1511.57 | 1488.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1473.60 | 1511.57 | 1488.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1473.60 | 1511.57 | 1488.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 1473.60 | 1511.57 | 1488.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1505.40 | 1510.34 | 1490.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 11:30:00 | 1511.90 | 1510.27 | 1492.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1509.00 | 1514.82 | 1500.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 1483.80 | 1499.38 | 1499.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 1483.80 | 1499.38 | 1499.39 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1503.50 | 1500.20 | 1499.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 1533.00 | 1507.99 | 1503.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1529.90 | 1535.08 | 1520.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 1529.90 | 1535.08 | 1520.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1526.00 | 1533.26 | 1521.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 12:00:00 | 1539.00 | 1534.41 | 1522.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 1539.00 | 1534.45 | 1526.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 1498.20 | 1527.20 | 1524.15 | SL hit (close<static) qty=1.00 sl=1515.10 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 1494.80 | 1520.72 | 1521.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 11:15:00 | 1486.20 | 1513.81 | 1518.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1493.40 | 1480.07 | 1496.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 1493.40 | 1480.07 | 1496.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1493.40 | 1480.07 | 1496.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 1493.40 | 1480.07 | 1496.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1474.70 | 1479.00 | 1494.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 1467.10 | 1474.95 | 1488.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 1451.00 | 1470.24 | 1485.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1556.60 | 1484.59 | 1489.22 | SL hit (close>static) qty=1.00 sl=1497.80 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 1539.10 | 1495.49 | 1493.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1614.70 | 1562.39 | 1538.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1733.00 | 1747.90 | 1697.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 1736.00 | 1747.90 | 1697.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1812.00 | 1825.94 | 1807.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1812.00 | 1825.94 | 1807.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1780.00 | 1815.84 | 1807.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 1750.10 | 1815.84 | 1807.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1774.30 | 1807.53 | 1804.08 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1767.50 | 1799.53 | 1800.76 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1813.10 | 1802.24 | 1801.88 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1798.50 | 1801.15 | 1801.42 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 1845.90 | 1808.64 | 1804.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 1881.40 | 1843.96 | 1824.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 1886.90 | 1895.61 | 1869.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:45:00 | 1886.10 | 1895.61 | 1869.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1894.00 | 1908.41 | 1891.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 1890.50 | 1908.41 | 1891.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1906.00 | 1907.92 | 1892.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 1917.00 | 1907.92 | 1892.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 1917.60 | 1909.86 | 1895.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:45:00 | 1914.50 | 1910.47 | 1896.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 1960.60 | 1910.24 | 1900.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1959.40 | 1920.07 | 1905.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 2003.00 | 1947.67 | 1935.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:45:00 | 1992.50 | 2010.85 | 1987.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 1993.00 | 2005.70 | 1987.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 2011.00 | 1988.38 | 1984.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1991.40 | 1993.61 | 1988.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:45:00 | 1989.10 | 1993.61 | 1988.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1971.80 | 1989.25 | 1986.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1971.80 | 1989.25 | 1986.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1975.00 | 1986.40 | 1985.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 1971.70 | 1986.40 | 1985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-03 14:15:00 | 1978.70 | 1984.86 | 1985.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1978.70 | 1984.86 | 1985.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1953.90 | 1977.66 | 1981.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 15:15:00 | 1932.00 | 1947.14 | 1954.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 1926.50 | 1943.23 | 1950.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 1932.00 | 1940.08 | 1948.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 1957.10 | 1949.54 | 1949.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 12:15:00 | 1957.10 | 1949.54 | 1949.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 1969.10 | 1954.36 | 1951.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1937.80 | 1953.07 | 1951.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 1934.10 | 1949.28 | 1950.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 1924.00 | 1940.90 | 1945.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 1923.00 | 1892.74 | 1908.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1921.10 | 1898.41 | 1909.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 1923.90 | 1898.41 | 1909.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1894.00 | 1898.60 | 1908.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1872.90 | 1899.85 | 1906.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 1917.20 | 1889.07 | 1892.74 | SL hit (close>static) qty=1.00 sl=1909.70 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1937.50 | 1898.76 | 1896.81 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1876.70 | 1902.42 | 1904.06 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1915.00 | 1898.62 | 1898.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1947.00 | 1912.66 | 1904.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 1858.90 | 1913.26 | 1909.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 1862.20 | 1903.05 | 1905.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 1839.20 | 1867.07 | 1884.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1827.00 | 1823.11 | 1840.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 14:45:00 | 1829.00 | 1823.11 | 1840.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1854.00 | 1829.89 | 1840.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1846.60 | 1829.89 | 1840.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1863.00 | 1836.51 | 1842.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 1868.90 | 1836.51 | 1842.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 1866.90 | 1847.10 | 1846.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 1876.00 | 1852.88 | 1849.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1971.60 | 1971.99 | 1942.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:30:00 | 1957.00 | 1971.99 | 1942.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1969.30 | 1972.52 | 1952.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 1975.00 | 1973.02 | 1954.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 1976.70 | 1968.88 | 1959.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1993.20 | 1967.99 | 1961.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1944.80 | 1966.72 | 1967.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1944.80 | 1966.72 | 1967.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1938.00 | 1953.10 | 1959.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 1956.40 | 1950.88 | 1956.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1958.80 | 1952.46 | 1956.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1961.90 | 1952.46 | 1956.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1973.50 | 1956.67 | 1958.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1973.50 | 1956.67 | 1958.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1970.00 | 1959.34 | 1959.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1983.00 | 1964.07 | 1961.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1927.10 | 1963.85 | 1965.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1918.60 | 1954.80 | 1961.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1688.20 | 1686.98 | 1723.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 1688.20 | 1686.98 | 1723.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1717.50 | 1699.59 | 1720.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1717.50 | 1699.59 | 1720.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1722.00 | 1704.07 | 1720.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1721.00 | 1704.07 | 1720.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1729.60 | 1709.18 | 1721.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 1701.30 | 1717.85 | 1721.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 1704.30 | 1709.39 | 1716.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1696.80 | 1704.11 | 1711.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1616.23 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1619.08 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1611.96 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 1628.00 | 1619.94 | 1639.45 | SL hit (close>ema200) qty=0.50 sl=1619.94 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1559.40 | 1526.46 | 1522.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 1587.00 | 1538.57 | 1528.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1567.80 | 1593.03 | 1580.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1569.00 | 1588.23 | 1579.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 1564.00 | 1588.23 | 1579.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 1551.00 | 1573.32 | 1574.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 1549.80 | 1565.03 | 1570.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1543.50 | 1540.46 | 1551.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 1544.40 | 1540.46 | 1551.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1523.60 | 1522.54 | 1530.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1518.40 | 1523.57 | 1528.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1499.60 | 1523.04 | 1527.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 1442.48 | 1460.52 | 1482.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1424.62 | 1449.87 | 1473.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1449.80 | 1442.55 | 1457.11 | SL hit (close>ema200) qty=0.50 sl=1442.55 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1476.50 | 1461.59 | 1460.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1486.90 | 1466.65 | 1462.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1464.80 | 1469.73 | 1465.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1461.00 | 1467.98 | 1465.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:00:00 | 1461.00 | 1467.98 | 1465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1467.00 | 1465.54 | 1464.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1459.70 | 1465.54 | 1464.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 1456.00 | 1463.63 | 1464.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 1446.50 | 1460.20 | 1462.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1445.80 | 1438.23 | 1445.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1441.40 | 1438.86 | 1445.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 1450.40 | 1438.86 | 1445.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1450.40 | 1441.17 | 1445.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 1443.70 | 1441.17 | 1445.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1452.40 | 1443.42 | 1446.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1439.20 | 1444.98 | 1446.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1437.90 | 1443.56 | 1445.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 1481.10 | 1447.39 | 1443.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 1481.10 | 1447.39 | 1443.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 1485.50 | 1455.01 | 1447.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 1477.00 | 1482.50 | 1468.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 1477.00 | 1482.50 | 1468.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1617.40 | 1617.01 | 1602.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1631.50 | 1611.79 | 1604.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 1621.00 | 1617.39 | 1608.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1621.90 | 1618.99 | 1611.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1600.00 | 1620.79 | 1618.93 | SL hit (close<static) qty=1.00 sl=1601.20 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1583.80 | 1613.39 | 1615.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1580.00 | 1606.71 | 1612.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 1599.00 | 1575.14 | 1586.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1598.00 | 1579.72 | 1587.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1598.00 | 1579.72 | 1587.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1584.60 | 1580.69 | 1587.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 1582.70 | 1580.77 | 1586.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1503.57 | 1528.90 | 1552.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1495.00 | 1493.59 | 1509.39 | SL hit (close>ema200) qty=0.50 sl=1493.59 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1541.40 | 1515.97 | 1514.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1567.90 | 1531.00 | 1522.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1554.80 | 1555.15 | 1543.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:45:00 | 1552.80 | 1555.15 | 1543.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1544.40 | 1554.15 | 1547.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 1544.40 | 1554.15 | 1547.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1538.00 | 1550.92 | 1546.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1538.00 | 1550.92 | 1546.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1529.50 | 1546.63 | 1545.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 1529.50 | 1546.63 | 1545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 1532.60 | 1543.83 | 1543.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1509.60 | 1531.53 | 1537.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1504.00 | 1500.77 | 1511.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1504.00 | 1500.77 | 1511.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1519.00 | 1504.42 | 1512.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1519.00 | 1504.42 | 1512.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1524.10 | 1508.36 | 1513.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1519.00 | 1508.36 | 1513.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1532.00 | 1517.30 | 1517.06 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1502.10 | 1517.33 | 1518.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1484.30 | 1505.01 | 1510.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1506.80 | 1496.61 | 1502.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1511.80 | 1499.65 | 1503.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1513.60 | 1499.65 | 1503.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1502.60 | 1503.25 | 1504.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1496.10 | 1501.82 | 1503.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1546.70 | 1511.57 | 1507.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1546.70 | 1511.57 | 1507.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1558.00 | 1520.86 | 1511.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 10:15:00 | 1534.10 | 1535.28 | 1525.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 12:30:00 | 1542.20 | 1537.69 | 1528.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1547.30 | 1538.30 | 1531.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1544.60 | 1538.64 | 1532.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1540.60 | 1539.03 | 1532.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 1534.00 | 1538.90 | 1535.01 | SL hit (close<ema400) qty=1.00 sl=1535.01 alert=retest1 |

### Cycle 102 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1515.00 | 1529.88 | 1531.32 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1550.20 | 1533.95 | 1533.04 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1526.80 | 1536.01 | 1536.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1521.60 | 1531.65 | 1534.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 1512.30 | 1519.66 | 1523.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 1511.00 | 1518.04 | 1521.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 1511.70 | 1517.54 | 1521.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1536.40 | 1521.05 | 1522.29 | SL hit (close>static) qty=1.00 sl=1529.40 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1531.00 | 1523.62 | 1523.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 1543.10 | 1527.51 | 1525.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1526.10 | 1527.23 | 1525.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 1526.10 | 1527.23 | 1525.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1534.10 | 1528.60 | 1525.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1522.10 | 1528.60 | 1525.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1518.60 | 1526.60 | 1525.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1524.50 | 1526.60 | 1525.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1520.20 | 1525.32 | 1524.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1517.80 | 1525.32 | 1524.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1517.70 | 1523.80 | 1524.19 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1536.80 | 1526.46 | 1525.30 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1506.70 | 1522.79 | 1524.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1501.60 | 1518.55 | 1522.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1446.60 | 1441.03 | 1458.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 1453.40 | 1441.03 | 1458.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1502.80 | 1454.18 | 1461.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1500.30 | 1454.18 | 1461.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1505.60 | 1464.46 | 1465.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1500.00 | 1464.46 | 1465.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1509.10 | 1473.39 | 1469.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1517.90 | 1488.21 | 1476.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 1543.00 | 1535.33 | 1524.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1521.00 | 1528.49 | 1523.68 | SL hit (close<static) qty=1.00 sl=1521.20 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 1558.00 | 1572.54 | 1574.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1546.00 | 1564.68 | 1570.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1563.20 | 1548.91 | 1557.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1561.00 | 1551.33 | 1557.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 1558.20 | 1551.33 | 1557.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 1559.20 | 1554.85 | 1558.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 1560.00 | 1555.14 | 1557.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1482.00 | 1513.90 | 1531.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1480.29 | 1495.34 | 1516.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1481.24 | 1495.34 | 1516.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1479.00 | 1474.82 | 1490.75 | SL hit (close>ema200) qty=0.50 sl=1474.82 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 1495.80 | 1492.20 | 1491.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 15:15:00 | 1507.00 | 1498.01 | 1494.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 1530.00 | 1530.68 | 1520.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 1530.00 | 1530.68 | 1520.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1522.80 | 1529.10 | 1520.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 1525.00 | 1529.10 | 1520.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1522.60 | 1527.80 | 1520.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 1529.00 | 1528.04 | 1521.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1505.40 | 1522.50 | 1520.17 | SL hit (close<static) qty=1.00 sl=1517.30 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1491.30 | 1516.26 | 1517.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 1484.80 | 1509.97 | 1514.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1496.40 | 1495.71 | 1504.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 1498.50 | 1495.71 | 1504.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1501.30 | 1496.83 | 1504.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 1502.90 | 1496.83 | 1504.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1513.50 | 1500.16 | 1505.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1513.50 | 1500.16 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1514.00 | 1502.93 | 1505.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 1516.30 | 1502.93 | 1505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 1526.40 | 1509.32 | 1508.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 15:15:00 | 1537.90 | 1515.03 | 1511.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1500.00 | 1514.83 | 1511.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1522.10 | 1516.28 | 1512.31 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1472.20 | 1505.94 | 1509.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 1461.30 | 1497.01 | 1504.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1430.20 | 1428.89 | 1452.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1430.20 | 1428.89 | 1452.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1414.50 | 1411.03 | 1422.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 1425.30 | 1411.03 | 1422.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1426.20 | 1415.06 | 1421.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 1418.50 | 1414.67 | 1420.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 1347.58 | 1362.37 | 1381.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 1333.70 | 1330.39 | 1349.64 | SL hit (close>ema200) qty=0.50 sl=1330.39 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1373.60 | 1353.26 | 1351.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1387.00 | 1363.17 | 1356.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 1486.30 | 1487.93 | 1463.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:00:00 | 1486.30 | 1487.93 | 1463.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1467.90 | 1484.64 | 1474.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1467.90 | 1484.64 | 1474.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1470.20 | 1481.75 | 1473.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1478.00 | 1481.75 | 1473.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1456.70 | 1476.14 | 1472.58 | SL hit (close<static) qty=1.00 sl=1462.40 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1441.30 | 1469.17 | 1469.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1430.80 | 1461.50 | 1466.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1475.00 | 1458.57 | 1462.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1465.10 | 1459.87 | 1463.01 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1469.90 | 1464.98 | 1464.86 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1451.30 | 1462.54 | 1463.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 1447.70 | 1456.20 | 1460.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1484.50 | 1461.86 | 1462.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 1485.60 | 1466.61 | 1464.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1497.20 | 1472.73 | 1467.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 12:15:00 | 1541.70 | 1542.91 | 1522.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:45:00 | 1547.60 | 1542.91 | 1522.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1538.90 | 1540.75 | 1527.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 1557.20 | 1543.15 | 1534.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1560.40 | 1543.15 | 1534.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1563.30 | 1544.61 | 1539.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 1520.00 | 1539.34 | 1539.09 | SL hit (close<static) qty=1.00 sl=1526.50 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 1522.00 | 1535.87 | 1537.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1508.30 | 1527.82 | 1533.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 1530.60 | 1524.25 | 1529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1530.20 | 1525.44 | 1529.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1541.50 | 1525.44 | 1529.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1535.00 | 1527.35 | 1529.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1531.30 | 1527.35 | 1529.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1524.00 | 1517.54 | 1522.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 1524.00 | 1517.54 | 1522.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1520.20 | 1518.07 | 1522.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1529.40 | 1518.07 | 1522.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1520.40 | 1518.54 | 1522.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1510.70 | 1522.44 | 1523.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1513.80 | 1520.71 | 1522.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1511.80 | 1519.67 | 1521.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 1511.30 | 1519.88 | 1521.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1517.10 | 1516.24 | 1519.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1517.00 | 1516.24 | 1519.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1512.50 | 1515.28 | 1518.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 1502.00 | 1509.43 | 1513.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1435.16 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1438.11 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1436.21 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1435.73 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1426.90 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1454.10 | 1437.41 | 1457.12 | SL hit (close>ema200) qty=0.50 sl=1437.41 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 1465.40 | 1442.78 | 1442.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 1479.40 | 1450.10 | 1445.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 13:15:00 | 1537.60 | 1538.22 | 1512.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 14:00:00 | 1537.60 | 1538.22 | 1512.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1547.50 | 1537.61 | 1518.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 1551.10 | 1540.31 | 1521.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 1550.40 | 1542.33 | 1524.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1440.20 | 1509.29 | 1516.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1440.20 | 1509.29 | 1516.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1397.60 | 1474.99 | 1498.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 1342.80 | 1333.47 | 1378.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:30:00 | 1341.80 | 1333.47 | 1378.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1279.30 | 1267.79 | 1281.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1280.90 | 1267.79 | 1281.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1285.40 | 1271.32 | 1282.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 1286.60 | 1271.32 | 1282.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1290.00 | 1275.05 | 1282.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 1290.00 | 1275.05 | 1282.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1286.30 | 1277.30 | 1283.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:15:00 | 1288.50 | 1277.30 | 1283.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1295.90 | 1281.02 | 1284.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 1293.60 | 1281.02 | 1284.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1304.00 | 1289.07 | 1287.62 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1278.60 | 1289.38 | 1290.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 1273.70 | 1286.24 | 1288.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1283.40 | 1282.64 | 1286.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 1283.40 | 1282.64 | 1286.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1280.60 | 1274.60 | 1279.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1255.00 | 1273.70 | 1278.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1268.10 | 1259.95 | 1258.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1268.10 | 1259.95 | 1258.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1276.80 | 1265.25 | 1261.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1273.50 | 1284.80 | 1278.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1270.00 | 1281.84 | 1277.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1285.40 | 1281.84 | 1277.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1262.90 | 1288.80 | 1288.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 1262.90 | 1288.80 | 1288.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1252.00 | 1273.51 | 1280.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1253.80 | 1251.75 | 1263.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 1253.80 | 1251.75 | 1263.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1252.70 | 1246.38 | 1254.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 1258.00 | 1246.38 | 1254.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1262.30 | 1249.56 | 1255.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1262.30 | 1249.56 | 1255.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1263.00 | 1252.25 | 1255.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 1259.10 | 1252.25 | 1255.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1258.80 | 1253.68 | 1256.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 1273.00 | 1259.51 | 1258.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1273.00 | 1259.51 | 1258.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 09:15:00 | 1292.90 | 1268.97 | 1263.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:45:00 | 1265.60 | 1270.46 | 1266.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 1268.70 | 1270.11 | 1266.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 1253.00 | 1270.11 | 1266.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1270.50 | 1270.19 | 1266.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1267.90 | 1270.19 | 1266.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1270.50 | 1270.25 | 1267.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1269.30 | 1270.25 | 1267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1278.50 | 1271.90 | 1268.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 1303.80 | 1277.99 | 1272.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:15:00 | 1296.50 | 1280.33 | 1274.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1324.60 | 1283.73 | 1278.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1302.90 | 1338.23 | 1342.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1302.90 | 1338.23 | 1342.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1293.20 | 1315.37 | 1327.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1290.10 | 1287.34 | 1303.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 1303.50 | 1290.73 | 1299.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1303.50 | 1290.73 | 1299.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1303.50 | 1290.73 | 1299.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1298.90 | 1292.37 | 1299.66 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1313.00 | 1302.96 | 1302.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 1321.10 | 1308.27 | 1305.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1282.60 | 1304.37 | 1303.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1283.00 | 1300.10 | 1302.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1272.60 | 1291.45 | 1297.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1179.80 | 1175.32 | 1204.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1180.90 | 1175.32 | 1204.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1196.60 | 1179.67 | 1197.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1199.10 | 1179.67 | 1197.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1163.80 | 1177.89 | 1188.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 10:15:00 | 1158.00 | 1177.89 | 1188.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 1160.10 | 1170.10 | 1182.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 1154.50 | 1166.98 | 1180.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 13:15:00 | 1102.09 | 1125.00 | 1148.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1100.10 | 1119.98 | 1144.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1096.77 | 1119.98 | 1144.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1193.90 | 1130.29 | 1144.24 | SL hit (close>ema200) qty=0.50 sl=1130.29 alert=retest2 |

### Cycle 131 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1207.30 | 1152.25 | 1152.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1221.40 | 1198.48 | 1185.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 1338.60 | 1336.27 | 1319.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1336.30 | 1336.28 | 1320.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1338.90 | 1335.94 | 1321.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1344.80 | 1334.94 | 1323.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1348.50 | 1352.59 | 1345.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1353.80 | 1353.18 | 1346.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 12:15:00 | 1389.00 | 1394.73 | 1395.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 1389.00 | 1394.73 | 1395.19 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1401.30 | 1395.18 | 1395.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1410.40 | 1398.22 | 1396.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1392.50 | 1400.37 | 1398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1392.50 | 1398.80 | 1397.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 1385.50 | 1398.80 | 1397.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1363.50 | 1391.71 | 1394.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1359.70 | 1385.30 | 1391.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1373.00 | 1370.97 | 1380.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 1368.60 | 1370.97 | 1380.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1382.00 | 1373.18 | 1381.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1382.00 | 1373.18 | 1381.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1385.00 | 1375.54 | 1381.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1385.00 | 1375.54 | 1381.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1371.80 | 1374.79 | 1380.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1387.70 | 1374.79 | 1380.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1377.00 | 1373.42 | 1377.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 1379.50 | 1376.30 | 1378.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1378.80 | 1376.80 | 1378.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1375.30 | 1376.80 | 1378.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1396.50 | 1381.43 | 1380.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1396.50 | 1381.43 | 1380.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1413.60 | 1394.20 | 1387.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1444.00 | 1445.63 | 1427.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:45:00 | 1443.10 | 1445.63 | 1427.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-31 13:15:00 | 1534.95 | 2024-06-04 10:15:00 | 1436.90 | STOP_HIT | 1.00 | -6.39% |
| BUY | retest2 | 2024-06-11 09:15:00 | 1453.25 | 2024-06-13 10:15:00 | 1412.70 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-06-11 14:15:00 | 1426.80 | 2024-06-13 10:15:00 | 1412.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1429.90 | 2024-06-13 10:15:00 | 1412.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-06-12 11:45:00 | 1431.05 | 2024-06-13 10:15:00 | 1412.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-06-12 15:15:00 | 1423.00 | 2024-06-13 10:15:00 | 1412.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-06-20 11:15:00 | 1584.80 | 2024-06-21 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-06-26 09:15:00 | 1586.90 | 2024-07-05 10:15:00 | 1745.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-12 10:30:00 | 1647.85 | 2024-07-12 15:15:00 | 1668.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1427.00 | 2024-07-23 12:15:00 | 1355.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1427.00 | 2024-07-26 09:15:00 | 1431.00 | STOP_HIT | 0.50 | -0.28% |
| SELL | retest2 | 2024-08-02 09:15:00 | 1425.10 | 2024-08-05 09:15:00 | 1353.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 09:15:00 | 1425.10 | 2024-08-06 09:15:00 | 1365.35 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2024-08-27 10:15:00 | 1309.00 | 2024-08-28 12:15:00 | 1317.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-08-27 14:45:00 | 1311.25 | 2024-08-28 12:15:00 | 1317.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-09-16 09:45:00 | 1232.15 | 2024-09-18 11:15:00 | 1170.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 09:45:00 | 1232.15 | 2024-09-19 11:15:00 | 1108.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-07 09:45:00 | 1085.75 | 2024-10-08 10:15:00 | 1139.45 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2024-10-31 09:15:00 | 1068.50 | 2024-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-11-11 12:15:00 | 1046.05 | 2024-11-13 09:15:00 | 993.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 14:00:00 | 1046.35 | 2024-11-13 09:15:00 | 994.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1042.80 | 2024-11-13 09:15:00 | 990.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1044.25 | 2024-11-13 09:15:00 | 992.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 12:15:00 | 1046.05 | 2024-11-14 09:15:00 | 1001.10 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2024-11-11 14:00:00 | 1046.35 | 2024-11-14 09:15:00 | 1001.10 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1042.80 | 2024-11-14 09:15:00 | 1001.10 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1044.25 | 2024-11-14 09:15:00 | 1001.10 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2024-11-18 10:15:00 | 962.20 | 2024-11-18 15:15:00 | 914.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-18 11:30:00 | 957.65 | 2024-11-18 15:15:00 | 909.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-18 10:15:00 | 962.20 | 2024-11-19 09:15:00 | 977.85 | STOP_HIT | 0.50 | -1.63% |
| SELL | retest2 | 2024-11-18 11:30:00 | 957.65 | 2024-11-19 09:15:00 | 977.85 | STOP_HIT | 0.50 | -2.11% |
| SELL | retest2 | 2024-11-19 15:00:00 | 965.85 | 2024-11-22 09:15:00 | 917.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 15:00:00 | 965.85 | 2024-11-22 11:15:00 | 936.90 | STOP_HIT | 0.50 | 3.00% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1185.30 | 2024-12-10 10:15:00 | 1197.95 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-12-13 14:15:00 | 1232.05 | 2024-12-19 09:15:00 | 1228.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-12-18 12:45:00 | 1239.90 | 2024-12-19 09:15:00 | 1228.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1225.90 | 2024-12-27 10:15:00 | 1175.86 | PARTIAL | 0.50 | 4.08% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1225.90 | 2024-12-27 12:15:00 | 1199.30 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2024-12-23 14:15:00 | 1237.75 | 2024-12-30 12:15:00 | 1164.61 | PARTIAL | 0.50 | 5.91% |
| SELL | retest2 | 2024-12-23 14:15:00 | 1237.75 | 2024-12-30 14:15:00 | 1103.31 | TARGET_HIT | 0.50 | 10.86% |
| SELL | retest2 | 2025-01-06 15:00:00 | 1120.00 | 2025-01-06 15:15:00 | 1143.95 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-01-07 09:45:00 | 1120.25 | 2025-01-07 11:15:00 | 1157.00 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-01-10 11:15:00 | 1180.60 | 2025-01-13 10:15:00 | 1147.90 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-01-15 14:15:00 | 1133.20 | 2025-01-16 10:15:00 | 1189.55 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2025-01-15 15:00:00 | 1133.15 | 2025-01-16 10:15:00 | 1189.55 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-01-28 14:45:00 | 1214.10 | 2025-01-29 09:15:00 | 1244.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1174.50 | 2025-02-11 13:15:00 | 1115.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1174.50 | 2025-02-12 09:15:00 | 1057.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 1186.25 | 2025-02-13 12:15:00 | 1187.45 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-02-24 13:30:00 | 1019.85 | 2025-02-25 10:15:00 | 1032.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-02-25 13:00:00 | 1020.55 | 2025-02-28 12:15:00 | 969.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1017.95 | 2025-02-28 14:15:00 | 967.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 13:00:00 | 1020.55 | 2025-03-03 10:15:00 | 918.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1017.95 | 2025-03-03 11:15:00 | 916.16 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-12 09:15:00 | 1129.00 | 2025-03-12 11:15:00 | 1098.90 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-03-12 10:30:00 | 1120.85 | 2025-03-12 11:15:00 | 1098.90 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-04-04 11:15:00 | 1304.15 | 2025-04-07 09:15:00 | 1227.05 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest2 | 2025-04-04 14:45:00 | 1320.30 | 2025-04-07 09:15:00 | 1227.05 | STOP_HIT | 1.00 | -7.06% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1431.60 | 2025-04-23 09:15:00 | 1391.90 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-04-30 11:30:00 | 1511.90 | 2025-05-02 15:15:00 | 1483.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1509.00 | 2025-05-02 15:15:00 | 1483.80 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-06 12:00:00 | 1539.00 | 2025-05-07 09:15:00 | 1498.20 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-05-07 09:15:00 | 1539.00 | 2025-05-07 09:15:00 | 1498.20 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1467.10 | 2025-05-09 09:15:00 | 1556.60 | STOP_HIT | 1.00 | -6.10% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1451.00 | 2025-05-09 09:15:00 | 1556.60 | STOP_HIT | 1.00 | -7.28% |
| BUY | retest2 | 2025-05-26 11:15:00 | 1917.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-05-26 12:00:00 | 1917.60 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-05-26 12:45:00 | 1914.50 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2025-05-27 09:15:00 | 1960.60 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-05-30 09:15:00 | 2003.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-02 09:45:00 | 1992.50 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-02 10:45:00 | 1993.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-03 09:15:00 | 2011.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-06 15:15:00 | 1932.00 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-09 10:45:00 | 1926.50 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-06-09 13:00:00 | 1932.00 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1872.90 | 2025-06-17 09:15:00 | 1917.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-07-02 15:00:00 | 1975.00 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-07-03 12:30:00 | 1976.70 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1993.20 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-07-23 10:15:00 | 1701.30 | 2025-07-28 13:15:00 | 1616.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 13:00:00 | 1704.30 | 2025-07-28 13:15:00 | 1619.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1696.80 | 2025-07-28 13:15:00 | 1611.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:15:00 | 1701.30 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2025-07-23 13:00:00 | 1704.30 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1696.80 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1518.40 | 2025-08-28 14:15:00 | 1442.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1499.60 | 2025-08-29 09:15:00 | 1424.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1518.40 | 2025-09-01 09:15:00 | 1449.80 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1499.60 | 2025-09-01 09:15:00 | 1449.80 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1439.20 | 2025-09-10 11:15:00 | 1481.10 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1437.90 | 2025-09-10 11:15:00 | 1481.10 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1631.50 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-19 11:15:00 | 1621.00 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-19 13:45:00 | 1621.90 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-26 14:15:00 | 1503.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-30 15:15:00 | 1495.00 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2025-10-16 11:00:00 | 1496.10 | 2025-10-17 09:15:00 | 1546.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest1 | 2025-10-20 12:30:00 | 1542.20 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-10-21 13:45:00 | 1547.30 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2025-10-23 09:15:00 | 1544.60 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-30 12:45:00 | 1512.30 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-10-30 13:30:00 | 1511.00 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1511.70 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-13 09:30:00 | 1543.00 | 2025-11-13 12:15:00 | 1521.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-11-14 09:15:00 | 1619.30 | 2025-11-18 14:15:00 | 1558.00 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-11-20 11:15:00 | 1558.20 | 2025-11-24 10:15:00 | 1482.00 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1559.20 | 2025-11-24 14:15:00 | 1480.29 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1560.00 | 2025-11-24 14:15:00 | 1481.24 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-11-20 11:15:00 | 1558.20 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1559.20 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1560.00 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.19% |
| BUY | retest2 | 2025-12-02 15:00:00 | 1529.00 | 2025-12-03 09:15:00 | 1505.40 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1418.50 | 2025-12-17 09:15:00 | 1347.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1418.50 | 2025-12-18 11:15:00 | 1333.70 | STOP_HIT | 0.50 | 5.98% |
| BUY | retest2 | 2025-12-29 15:15:00 | 1478.00 | 2025-12-30 09:15:00 | 1456.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1557.20 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-01-08 10:00:00 | 1560.40 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1563.30 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1510.70 | 2026-01-21 09:15:00 | 1435.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1513.80 | 2026-01-21 09:15:00 | 1438.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1511.80 | 2026-01-21 09:15:00 | 1436.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1511.30 | 2026-01-21 09:15:00 | 1435.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1502.00 | 2026-01-21 09:15:00 | 1426.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1510.70 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1513.80 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1511.80 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1511.30 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1502.00 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.19% |
| BUY | retest2 | 2026-01-30 11:00:00 | 1551.10 | 2026-02-01 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2026-01-30 12:00:00 | 1550.40 | 2026-02-01 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1255.00 | 2026-02-18 09:15:00 | 1268.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-20 09:15:00 | 1285.40 | 2026-02-23 11:15:00 | 1262.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-02-26 12:15:00 | 1259.10 | 2026-02-26 14:15:00 | 1273.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1258.80 | 2026-02-26 14:15:00 | 1273.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-05 11:15:00 | 1303.80 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-03-05 12:15:00 | 1296.50 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2026-03-06 09:15:00 | 1324.60 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-27 10:15:00 | 1158.00 | 2026-03-30 13:15:00 | 1102.09 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-03-27 12:15:00 | 1160.10 | 2026-03-30 14:15:00 | 1100.10 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-27 13:00:00 | 1154.50 | 2026-03-30 14:15:00 | 1096.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:15:00 | 1158.00 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -3.10% |
| SELL | retest2 | 2026-03-27 12:15:00 | 1160.10 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -2.91% |
| SELL | retest2 | 2026-03-27 13:00:00 | 1154.50 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -3.41% |
| BUY | retest2 | 2026-04-13 11:30:00 | 1338.60 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1336.30 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1338.90 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.74% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1344.80 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1353.80 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 2.60% |
| SELL | retest2 | 2026-05-05 11:15:00 | 1375.30 | 2026-05-05 12:15:00 | 1396.50 | STOP_HIT | 1.00 | -1.54% |
