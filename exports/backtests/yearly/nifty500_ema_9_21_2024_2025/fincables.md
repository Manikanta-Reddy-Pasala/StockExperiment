# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1144.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 92 |
| ALERT2 | 89 |
| ALERT2_SKIP | 42 |
| ALERT3 | 228 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 109 |
| PARTIAL | 27 |
| TARGET_HIT | 11 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 71
- **Target hits / Stop hits / Partials:** 11 / 101 / 27
- **Avg / median % per leg:** 1.37% / -0.12%
- **Sum % (uncompounded):** 190.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 12 | 30.8% | 8 | 30 | 1 | 1.07% | 41.5% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.93% | 7.7% |
| BUY @ 3rd Alert (retest2) | 35 | 9 | 25.7% | 8 | 27 | 0 | 0.97% | 33.8% |
| SELL (all) | 100 | 56 | 56.0% | 3 | 71 | 26 | 1.49% | 149.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 100 | 56 | 56.0% | 3 | 71 | 26 | 1.49% | 149.4% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.93% | 7.7% |
| retest2 (combined) | 135 | 65 | 48.1% | 11 | 98 | 26 | 1.36% | 183.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1052.20 | 1028.46 | 1025.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 12:15:00 | 1097.00 | 1064.45 | 1058.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1455.00 | 1461.74 | 1422.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 10:00:00 | 1455.00 | 1461.74 | 1422.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1436.10 | 1456.95 | 1439.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 1436.10 | 1456.95 | 1439.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1425.70 | 1450.70 | 1438.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 1426.75 | 1450.70 | 1438.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1425.00 | 1445.56 | 1437.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 1425.00 | 1445.56 | 1437.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1444.00 | 1444.49 | 1438.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:15:00 | 1435.00 | 1444.49 | 1438.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1435.05 | 1442.60 | 1437.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 1440.00 | 1442.60 | 1437.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1440.00 | 1442.08 | 1438.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1471.00 | 1442.08 | 1438.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1445.50 | 1442.76 | 1438.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:15:00 | 1439.35 | 1442.76 | 1438.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 1451.80 | 1444.57 | 1439.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 1459.80 | 1447.63 | 1441.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 1453.60 | 1452.17 | 1444.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 1467.65 | 1452.17 | 1444.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1412.05 | 1448.55 | 1445.39 | SL hit (close<static) qty=1.00 sl=1430.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1385.20 | 1435.88 | 1439.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1361.00 | 1420.91 | 1432.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 1375.00 | 1374.62 | 1398.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:30:00 | 1375.00 | 1374.62 | 1398.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1405.00 | 1381.21 | 1395.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1405.00 | 1381.21 | 1395.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1383.55 | 1381.68 | 1394.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1443.15 | 1381.68 | 1394.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1456.00 | 1396.54 | 1400.02 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1485.15 | 1414.26 | 1407.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 1492.65 | 1429.94 | 1415.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 1469.00 | 1478.47 | 1456.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 1469.00 | 1478.47 | 1456.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1568.20 | 1586.31 | 1574.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 1568.20 | 1586.31 | 1574.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1567.05 | 1582.46 | 1573.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:45:00 | 1556.20 | 1582.46 | 1573.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1576.40 | 1581.25 | 1573.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:00:00 | 1580.40 | 1581.08 | 1574.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:45:00 | 1581.25 | 1580.90 | 1574.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 14:30:00 | 1584.45 | 1579.24 | 1574.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1557.75 | 1574.06 | 1573.02 | SL hit (close<static) qty=1.00 sl=1562.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 1554.70 | 1570.18 | 1571.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 1530.70 | 1559.06 | 1565.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 1565.60 | 1554.74 | 1559.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 14:15:00 | 1565.60 | 1554.74 | 1559.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1565.60 | 1554.74 | 1559.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:45:00 | 1570.10 | 1554.74 | 1559.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1563.00 | 1556.39 | 1560.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 1575.20 | 1556.39 | 1560.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1586.00 | 1565.28 | 1563.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 1616.05 | 1575.44 | 1568.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 1565.70 | 1578.48 | 1572.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 14:15:00 | 1565.70 | 1578.48 | 1572.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1565.70 | 1578.48 | 1572.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1565.70 | 1578.48 | 1572.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1550.00 | 1572.79 | 1570.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 1575.40 | 1572.79 | 1570.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1585.00 | 1615.49 | 1615.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 1585.00 | 1615.49 | 1615.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 1581.85 | 1608.77 | 1612.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 1578.05 | 1577.01 | 1589.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 1578.05 | 1577.01 | 1589.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1578.05 | 1577.01 | 1589.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 1585.45 | 1577.01 | 1589.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1595.95 | 1575.19 | 1584.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 1595.95 | 1575.19 | 1584.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1577.10 | 1575.57 | 1584.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 1586.00 | 1575.57 | 1584.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 1605.00 | 1581.46 | 1585.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 1605.00 | 1581.46 | 1585.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 1612.45 | 1587.66 | 1588.33 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 1622.00 | 1594.52 | 1591.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 1645.00 | 1604.62 | 1596.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 1649.90 | 1657.04 | 1643.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 11:00:00 | 1649.90 | 1657.04 | 1643.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1642.00 | 1652.94 | 1646.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:30:00 | 1648.75 | 1652.94 | 1646.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1647.65 | 1651.88 | 1646.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1657.00 | 1651.88 | 1646.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 1625.05 | 1650.47 | 1651.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1625.05 | 1650.47 | 1651.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 15:15:00 | 1625.00 | 1639.23 | 1645.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 1621.10 | 1597.03 | 1607.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1621.10 | 1597.03 | 1607.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1621.10 | 1597.03 | 1607.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 1621.10 | 1597.03 | 1607.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1634.20 | 1604.47 | 1609.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 1634.20 | 1604.47 | 1609.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1619.25 | 1613.65 | 1613.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 1625.00 | 1619.60 | 1616.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 1608.70 | 1617.42 | 1615.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 1608.70 | 1617.42 | 1615.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1608.70 | 1617.42 | 1615.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 1608.70 | 1617.42 | 1615.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 1612.25 | 1616.38 | 1615.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 1601.30 | 1616.38 | 1615.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 1600.00 | 1613.11 | 1614.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 1584.90 | 1607.47 | 1611.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 1607.90 | 1600.59 | 1606.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 1607.90 | 1600.59 | 1606.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1607.90 | 1600.59 | 1606.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 1607.90 | 1600.59 | 1606.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1614.50 | 1603.37 | 1607.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 1618.00 | 1603.37 | 1607.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1613.25 | 1605.35 | 1607.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 1613.25 | 1605.35 | 1607.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1618.00 | 1607.88 | 1608.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 1618.00 | 1607.88 | 1608.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1603.30 | 1606.96 | 1608.36 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1657.40 | 1616.42 | 1612.37 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1595.00 | 1618.27 | 1621.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1575.00 | 1604.94 | 1614.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 1550.00 | 1549.12 | 1569.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:45:00 | 1563.75 | 1549.12 | 1569.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1525.00 | 1542.92 | 1557.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:30:00 | 1523.25 | 1535.24 | 1551.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 1447.09 | 1529.83 | 1547.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 1540.75 | 1523.88 | 1538.32 | SL hit (close>ema200) qty=0.50 sl=1523.88 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 1555.10 | 1537.55 | 1535.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 1559.45 | 1548.20 | 1542.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 1580.25 | 1585.74 | 1574.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:30:00 | 1585.65 | 1585.74 | 1574.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1539.10 | 1574.38 | 1571.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 1544.30 | 1574.38 | 1571.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 1539.05 | 1567.32 | 1568.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 1534.95 | 1552.23 | 1560.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1528.35 | 1526.82 | 1537.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 10:45:00 | 1528.95 | 1526.82 | 1537.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1531.20 | 1526.43 | 1533.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 15:00:00 | 1531.20 | 1526.43 | 1533.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1522.75 | 1525.70 | 1532.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1476.45 | 1525.70 | 1532.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 1500.40 | 1491.44 | 1490.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 1500.40 | 1491.44 | 1490.42 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1465.00 | 1487.34 | 1490.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 1443.65 | 1462.06 | 1473.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 1450.45 | 1444.12 | 1456.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 1450.45 | 1444.12 | 1456.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1469.20 | 1449.14 | 1457.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 1483.25 | 1449.14 | 1457.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 1474.25 | 1454.16 | 1459.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 1474.25 | 1454.16 | 1459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 1474.80 | 1458.29 | 1460.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 1477.40 | 1458.29 | 1460.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 1480.00 | 1462.63 | 1462.39 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 1455.90 | 1461.72 | 1462.12 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1463.40 | 1462.50 | 1462.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 1465.05 | 1463.01 | 1462.67 | Break + close above crossover candle high |

### Cycle 20 — SELL (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 09:15:00 | 1437.75 | 1457.96 | 1460.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 10:15:00 | 1428.50 | 1452.07 | 1457.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 14:15:00 | 1448.00 | 1444.80 | 1451.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 1448.00 | 1444.80 | 1451.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1448.00 | 1444.80 | 1451.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 1448.00 | 1444.80 | 1451.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1475.05 | 1451.52 | 1453.54 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 1478.70 | 1456.96 | 1455.83 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 1447.80 | 1458.13 | 1458.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 14:15:00 | 1439.95 | 1452.69 | 1456.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 1463.10 | 1452.74 | 1455.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 1463.10 | 1452.74 | 1455.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1463.10 | 1452.74 | 1455.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 13:00:00 | 1446.45 | 1452.76 | 1454.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 14:00:00 | 1449.50 | 1452.11 | 1454.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 1506.90 | 1464.03 | 1459.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 09:15:00 | 1506.90 | 1464.03 | 1459.36 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 1454.95 | 1466.71 | 1467.59 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 1490.00 | 1468.06 | 1467.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 13:15:00 | 1468.25 | 1473.04 | 1473.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 1464.00 | 1471.23 | 1472.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1496.05 | 1459.75 | 1463.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1496.05 | 1459.75 | 1463.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1496.05 | 1459.75 | 1463.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1496.05 | 1459.75 | 1463.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 1496.85 | 1467.17 | 1466.16 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 1448.00 | 1465.31 | 1466.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 15:15:00 | 1440.05 | 1448.87 | 1454.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1404.95 | 1397.71 | 1408.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1404.95 | 1397.71 | 1408.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1404.95 | 1397.71 | 1408.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 1407.70 | 1397.71 | 1408.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1378.70 | 1382.71 | 1390.52 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 1416.60 | 1398.04 | 1395.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1421.10 | 1404.56 | 1399.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 1412.10 | 1414.71 | 1408.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:30:00 | 1414.00 | 1414.71 | 1408.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1407.00 | 1413.17 | 1408.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 1407.00 | 1413.17 | 1408.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1411.95 | 1412.92 | 1408.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:30:00 | 1411.00 | 1412.92 | 1408.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1408.40 | 1412.02 | 1408.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 1408.40 | 1412.02 | 1408.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1410.70 | 1411.75 | 1408.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:15:00 | 1409.80 | 1411.75 | 1408.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1409.80 | 1411.36 | 1408.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1398.70 | 1411.36 | 1408.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1392.00 | 1407.49 | 1407.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1392.00 | 1407.49 | 1407.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 1394.35 | 1404.86 | 1406.23 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1422.75 | 1408.20 | 1407.08 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1406.00 | 1407.38 | 1407.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1397.45 | 1405.39 | 1406.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1398.40 | 1396.78 | 1401.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 1398.40 | 1396.78 | 1401.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1392.10 | 1395.85 | 1400.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 1396.55 | 1395.85 | 1400.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1383.25 | 1393.33 | 1398.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:45:00 | 1382.00 | 1388.32 | 1394.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1381.00 | 1388.32 | 1394.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:45:00 | 1381.80 | 1387.25 | 1393.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 1432.00 | 1398.32 | 1397.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 1432.00 | 1398.32 | 1397.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 1467.00 | 1419.21 | 1407.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 1488.30 | 1494.05 | 1467.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 1488.30 | 1494.05 | 1467.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1466.90 | 1488.29 | 1469.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1466.90 | 1488.29 | 1469.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1472.00 | 1485.04 | 1469.87 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1455.20 | 1462.82 | 1463.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 1428.10 | 1454.38 | 1459.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1458.75 | 1454.99 | 1459.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 1458.75 | 1454.99 | 1459.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1458.75 | 1454.99 | 1459.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1458.75 | 1454.99 | 1459.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1457.00 | 1455.39 | 1458.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 1449.95 | 1454.64 | 1458.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 1446.80 | 1452.62 | 1456.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 1447.80 | 1453.76 | 1457.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 1445.00 | 1450.51 | 1454.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1425.75 | 1429.20 | 1438.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 1418.30 | 1429.20 | 1438.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:00:00 | 1420.00 | 1426.46 | 1434.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:15:00 | 1377.45 | 1404.69 | 1420.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1374.46 | 1395.28 | 1412.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1375.41 | 1395.28 | 1412.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1372.75 | 1395.28 | 1412.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1347.38 | 1384.83 | 1403.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1349.00 | 1384.83 | 1403.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 15:15:00 | 1381.95 | 1375.81 | 1389.93 | SL hit (close>ema200) qty=0.50 sl=1375.81 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1312.65 | 1305.99 | 1305.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 1323.05 | 1309.40 | 1307.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1330.45 | 1336.67 | 1328.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1330.45 | 1336.67 | 1328.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1330.45 | 1336.67 | 1328.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1330.45 | 1336.67 | 1328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1326.60 | 1334.66 | 1328.57 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 1314.35 | 1324.29 | 1325.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1304.35 | 1318.58 | 1322.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 1315.85 | 1315.51 | 1319.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:00:00 | 1315.85 | 1315.51 | 1319.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1320.20 | 1316.45 | 1319.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 1320.70 | 1316.45 | 1319.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1320.10 | 1317.18 | 1319.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 1320.10 | 1317.18 | 1319.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1324.95 | 1318.73 | 1320.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1327.00 | 1318.73 | 1320.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1308.95 | 1316.78 | 1319.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 1295.95 | 1309.52 | 1315.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1231.15 | 1269.80 | 1284.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 1280.40 | 1271.92 | 1283.97 | SL hit (close>ema200) qty=0.50 sl=1271.92 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1223.15 | 1210.69 | 1210.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1244.55 | 1218.95 | 1214.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1203.85 | 1219.62 | 1215.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1203.85 | 1219.62 | 1215.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1203.85 | 1219.62 | 1215.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1203.85 | 1219.62 | 1215.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1199.05 | 1215.50 | 1214.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1199.05 | 1215.50 | 1214.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1204.50 | 1211.91 | 1212.68 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1220.00 | 1211.10 | 1210.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1227.15 | 1214.31 | 1211.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 1223.70 | 1237.16 | 1231.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 1223.70 | 1237.16 | 1231.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1223.70 | 1237.16 | 1231.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 1225.55 | 1237.16 | 1231.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1214.25 | 1232.57 | 1229.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 1214.25 | 1232.57 | 1229.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1212.00 | 1225.71 | 1227.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 1206.30 | 1219.90 | 1224.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 1141.30 | 1141.03 | 1159.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:45:00 | 1141.35 | 1141.03 | 1159.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1123.50 | 1137.48 | 1151.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1097.00 | 1124.67 | 1132.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 1140.00 | 1118.69 | 1117.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1140.00 | 1118.69 | 1117.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1162.15 | 1141.59 | 1134.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1143.00 | 1145.18 | 1138.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 1143.00 | 1145.18 | 1138.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1147.05 | 1145.48 | 1140.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 1171.20 | 1145.79 | 1141.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 1150.95 | 1145.79 | 1141.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 15:00:00 | 1150.40 | 1144.95 | 1142.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:00:00 | 1150.20 | 1146.21 | 1144.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1151.20 | 1153.28 | 1149.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 1151.20 | 1153.28 | 1149.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1156.00 | 1153.83 | 1150.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1178.90 | 1153.83 | 1150.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 11:15:00 | 1266.05 | 1224.05 | 1197.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 1287.05 | 1300.10 | 1301.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1266.65 | 1287.60 | 1294.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 1284.45 | 1284.07 | 1291.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 1284.45 | 1284.07 | 1291.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1267.40 | 1274.60 | 1283.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 1260.95 | 1271.87 | 1281.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 1197.90 | 1212.14 | 1226.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 09:15:00 | 1134.86 | 1164.21 | 1185.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 12:15:00 | 1215.20 | 1182.08 | 1180.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 13:15:00 | 1246.00 | 1194.86 | 1186.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 12:15:00 | 1209.95 | 1217.47 | 1204.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 13:00:00 | 1209.95 | 1217.47 | 1204.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1208.20 | 1215.61 | 1205.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 1210.55 | 1215.61 | 1205.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1203.25 | 1213.14 | 1204.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1203.25 | 1213.14 | 1204.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1195.00 | 1209.51 | 1204.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1192.00 | 1209.51 | 1204.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1191.30 | 1205.87 | 1202.90 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 1190.60 | 1199.45 | 1200.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 1149.70 | 1183.41 | 1191.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 1194.00 | 1170.59 | 1179.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 1194.00 | 1170.59 | 1179.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1194.00 | 1170.59 | 1179.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 1194.00 | 1170.59 | 1179.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1185.40 | 1173.55 | 1180.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 1198.15 | 1173.55 | 1180.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1177.35 | 1175.34 | 1179.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 1177.35 | 1175.34 | 1179.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1175.80 | 1175.43 | 1179.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:45:00 | 1171.05 | 1174.91 | 1178.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 15:00:00 | 1170.25 | 1173.83 | 1177.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 1171.15 | 1174.69 | 1176.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:45:00 | 1171.95 | 1173.84 | 1175.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 13:15:00 | 1112.50 | 1133.80 | 1150.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 13:15:00 | 1111.74 | 1133.80 | 1150.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 13:15:00 | 1112.59 | 1133.80 | 1150.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 13:15:00 | 1113.35 | 1133.80 | 1150.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 12:15:00 | 1123.90 | 1121.47 | 1135.67 | SL hit (close>ema200) qty=0.50 sl=1121.47 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1037.05 | 1024.31 | 1024.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 11:15:00 | 1072.40 | 1042.80 | 1034.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 13:15:00 | 1034.50 | 1041.53 | 1035.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 13:15:00 | 1034.50 | 1041.53 | 1035.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1034.50 | 1041.53 | 1035.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 1034.50 | 1041.53 | 1035.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1025.25 | 1038.27 | 1034.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:00:00 | 1025.25 | 1038.27 | 1034.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1029.90 | 1036.60 | 1034.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1029.95 | 1036.60 | 1034.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1034.95 | 1036.17 | 1034.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 1038.00 | 1036.17 | 1034.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1027.75 | 1034.49 | 1033.82 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 1030.10 | 1033.10 | 1033.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1026.00 | 1031.43 | 1032.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 15:15:00 | 1027.00 | 1025.49 | 1028.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 09:15:00 | 1004.20 | 1025.49 | 1028.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1005.00 | 1021.39 | 1026.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 12:00:00 | 991.80 | 1012.45 | 1021.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 991.00 | 1005.85 | 1011.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 966.75 | 998.70 | 1005.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 988.95 | 995.88 | 998.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 995.85 | 994.44 | 997.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 995.85 | 994.44 | 997.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 999.00 | 995.36 | 997.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 999.00 | 995.36 | 997.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 971.00 | 990.48 | 994.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 968.00 | 984.53 | 991.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 998.55 | 990.05 | 989.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 998.55 | 990.05 | 989.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 1005.50 | 995.22 | 991.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 15:15:00 | 1000.00 | 1000.27 | 996.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:15:00 | 1019.85 | 1000.27 | 996.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1005.00 | 1006.08 | 1000.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1001.20 | 1006.08 | 1000.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1000.60 | 1004.98 | 1000.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 1000.60 | 1004.98 | 1000.61 | SL hit (close<ema400) qty=1.00 sl=1000.61 alert=retest1 |

### Cycle 48 — SELL (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 15:15:00 | 986.80 | 999.53 | 1000.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 12:15:00 | 980.45 | 992.39 | 996.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 994.00 | 991.16 | 994.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 15:15:00 | 994.00 | 991.16 | 994.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 994.00 | 991.16 | 994.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 990.00 | 991.16 | 994.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 997.50 | 992.43 | 995.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1002.05 | 992.43 | 995.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 995.35 | 993.01 | 995.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 999.00 | 993.01 | 995.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 992.00 | 992.81 | 994.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:30:00 | 992.70 | 992.81 | 994.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 982.50 | 978.98 | 984.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 982.50 | 978.98 | 984.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 985.00 | 980.19 | 984.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 977.45 | 980.19 | 984.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 976.65 | 979.48 | 984.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 970.00 | 979.48 | 984.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 970.00 | 978.59 | 981.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 921.50 | 950.72 | 964.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 921.50 | 950.72 | 964.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 09:15:00 | 873.00 | 896.83 | 918.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 987.65 | 939.26 | 933.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 15:15:00 | 1000.00 | 951.41 | 939.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 11:15:00 | 949.25 | 951.77 | 942.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 11:30:00 | 949.00 | 951.77 | 942.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 932.00 | 947.82 | 941.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 932.00 | 947.82 | 941.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 912.90 | 940.84 | 938.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 912.90 | 940.84 | 938.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 15:15:00 | 918.80 | 935.33 | 936.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 902.85 | 928.84 | 933.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 918.00 | 912.27 | 920.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 15:15:00 | 918.00 | 912.27 | 920.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 918.00 | 912.27 | 920.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 896.25 | 912.27 | 920.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 898.80 | 909.58 | 918.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 876.50 | 900.73 | 912.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 887.10 | 898.05 | 908.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 932.75 | 914.25 | 912.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 932.75 | 914.25 | 912.61 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 904.85 | 916.92 | 918.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 853.85 | 893.70 | 902.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 813.80 | 811.22 | 830.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 813.80 | 811.22 | 830.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 827.05 | 812.14 | 825.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 827.05 | 812.14 | 825.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 821.80 | 814.07 | 825.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:45:00 | 814.50 | 813.96 | 824.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 15:15:00 | 828.05 | 822.99 | 822.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 828.05 | 822.99 | 822.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 837.80 | 825.95 | 824.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 837.95 | 838.07 | 832.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 846.70 | 838.07 | 832.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 833.80 | 838.80 | 834.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:45:00 | 835.85 | 838.80 | 834.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 834.50 | 837.94 | 834.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 832.50 | 837.94 | 834.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 839.10 | 838.17 | 835.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:15:00 | 836.00 | 838.17 | 835.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 836.00 | 837.74 | 835.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 821.00 | 837.74 | 835.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 827.00 | 835.59 | 834.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 824.15 | 835.59 | 834.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 827.00 | 833.87 | 833.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 832.40 | 833.87 | 833.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 824.85 | 832.07 | 832.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 824.85 | 832.07 | 832.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 823.95 | 830.44 | 832.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 808.70 | 804.49 | 813.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 802.30 | 804.49 | 813.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 797.00 | 802.99 | 811.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 794.40 | 797.85 | 803.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 793.00 | 797.39 | 802.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:15:00 | 794.40 | 798.17 | 801.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 790.80 | 797.23 | 800.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 818.45 | 796.52 | 798.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 818.45 | 796.52 | 798.28 | SL hit (close>static) qty=1.00 sl=812.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 815.30 | 800.27 | 799.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 827.20 | 805.66 | 802.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 833.00 | 856.11 | 841.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 833.00 | 856.11 | 841.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 833.00 | 856.11 | 841.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 833.00 | 856.11 | 841.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 834.50 | 851.78 | 841.28 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 09:15:00 | 829.75 | 836.18 | 836.80 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 844.00 | 836.06 | 836.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 905.10 | 849.87 | 842.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 929.10 | 930.15 | 900.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:45:00 | 930.65 | 930.15 | 900.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 918.00 | 929.92 | 910.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 895.15 | 925.19 | 909.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 914.60 | 923.07 | 910.11 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 896.95 | 905.98 | 906.11 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 909.55 | 905.89 | 905.88 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 899.25 | 904.56 | 905.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 890.30 | 898.73 | 901.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 917.05 | 900.78 | 902.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 917.05 | 900.78 | 902.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 917.05 | 900.78 | 902.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 917.05 | 900.78 | 902.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 917.00 | 904.03 | 903.54 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 898.15 | 902.93 | 903.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 886.25 | 897.24 | 900.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 894.45 | 891.26 | 895.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 894.45 | 891.26 | 895.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 894.45 | 891.26 | 895.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 894.45 | 891.26 | 895.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 896.00 | 892.21 | 895.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 901.65 | 892.21 | 895.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 901.20 | 894.01 | 896.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 901.20 | 894.01 | 896.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 904.95 | 896.20 | 897.10 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 905.00 | 897.96 | 897.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 910.65 | 901.13 | 899.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 886.85 | 903.09 | 901.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 886.85 | 903.09 | 901.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 886.85 | 903.09 | 901.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 886.85 | 903.09 | 901.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 886.35 | 899.75 | 900.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 878.40 | 892.18 | 896.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 849.80 | 845.66 | 864.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 849.80 | 845.66 | 864.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 851.05 | 846.74 | 863.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 876.65 | 846.74 | 863.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 871.05 | 851.60 | 864.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 860.20 | 866.87 | 867.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:15:00 | 861.95 | 863.77 | 865.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 856.15 | 861.73 | 864.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 880.85 | 868.24 | 866.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 880.85 | 868.24 | 866.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 881.70 | 870.93 | 868.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 896.65 | 896.93 | 890.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 896.10 | 896.93 | 890.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 892.90 | 898.35 | 894.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 892.90 | 898.35 | 894.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 892.50 | 897.18 | 894.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 897.50 | 897.80 | 894.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 15:15:00 | 897.75 | 898.99 | 896.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 893.00 | 898.86 | 898.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 893.00 | 898.86 | 898.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 10:15:00 | 891.85 | 897.46 | 898.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 893.95 | 891.93 | 894.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 10:00:00 | 893.95 | 891.93 | 894.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 888.10 | 891.16 | 894.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 11:15:00 | 883.15 | 891.16 | 894.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 880.80 | 876.25 | 875.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 880.80 | 876.25 | 875.96 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 868.45 | 875.57 | 875.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 865.30 | 873.08 | 874.61 | Break + close below crossover candle low |

### Cycle 69 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 906.40 | 870.16 | 870.07 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 882.00 | 895.82 | 896.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 869.45 | 890.54 | 893.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 882.00 | 878.97 | 885.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 912.70 | 878.97 | 885.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 913.00 | 885.77 | 887.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 908.40 | 885.77 | 887.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 913.55 | 891.33 | 890.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 917.65 | 900.24 | 894.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 917.90 | 918.41 | 909.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 932.60 | 918.82 | 910.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:00:00 | 922.20 | 921.12 | 913.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 923.80 | 921.24 | 914.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 914.95 | 921.24 | 914.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:15:00 | 968.31 | 953.97 | 942.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 957.95 | 961.63 | 950.41 | SL hit (close<ema200) qty=0.50 sl=961.63 alert=retest1 |

### Cycle 72 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 938.75 | 949.37 | 950.44 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 955.15 | 949.96 | 949.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 959.55 | 954.17 | 951.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 967.05 | 968.58 | 961.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 967.05 | 968.58 | 961.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 978.45 | 984.53 | 981.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 978.45 | 984.53 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 972.00 | 982.03 | 980.52 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 968.20 | 979.26 | 979.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 959.35 | 975.28 | 977.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 966.00 | 964.80 | 970.21 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 978.55 | 972.30 | 971.97 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 966.75 | 971.60 | 971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 12:15:00 | 964.80 | 970.24 | 971.10 | Break + close below crossover candle low |

### Cycle 77 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 995.00 | 971.13 | 970.75 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 961.75 | 970.71 | 971.15 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 969.80 | 968.88 | 968.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 980.25 | 971.15 | 969.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 985.90 | 992.19 | 987.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 985.80 | 992.19 | 987.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 991.70 | 992.09 | 987.86 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 970.95 | 985.52 | 986.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 963.45 | 977.98 | 982.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 963.00 | 961.83 | 969.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 963.00 | 961.83 | 969.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 964.70 | 958.13 | 963.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 969.00 | 958.13 | 963.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 963.95 | 959.29 | 963.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 968.65 | 959.29 | 963.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 959.25 | 959.28 | 963.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 961.20 | 959.28 | 963.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 957.70 | 958.18 | 961.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:45:00 | 962.10 | 958.18 | 961.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 965.85 | 959.20 | 961.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 965.85 | 959.20 | 961.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 964.70 | 960.30 | 961.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 957.00 | 961.03 | 962.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 961.00 | 961.59 | 962.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 960.15 | 961.74 | 962.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 961.00 | 961.82 | 962.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 961.00 | 961.66 | 962.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 958.40 | 961.66 | 962.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 956.95 | 960.72 | 961.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 952.25 | 960.72 | 961.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.95 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.14 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 912.95 | 929.30 | 940.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 15:15:00 | 909.15 | 927.44 | 938.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 932.80 | 928.51 | 938.01 | SL hit (close>ema200) qty=0.50 sl=928.51 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 939.05 | 937.22 | 937.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 955.50 | 942.75 | 940.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 13:15:00 | 980.00 | 980.49 | 971.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 980.00 | 980.49 | 971.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 973.55 | 978.56 | 973.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 975.00 | 978.56 | 973.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 971.50 | 977.15 | 973.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 972.30 | 977.15 | 973.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 972.00 | 976.12 | 972.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 971.95 | 976.12 | 972.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 978.40 | 974.96 | 973.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 973.30 | 974.96 | 973.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 979.20 | 980.73 | 977.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 977.90 | 980.73 | 977.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 976.80 | 979.95 | 977.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 976.80 | 979.95 | 977.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 971.35 | 978.23 | 976.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 970.00 | 978.23 | 976.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 973.35 | 977.25 | 976.60 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 971.40 | 976.08 | 976.13 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 978.65 | 976.59 | 976.36 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 967.80 | 975.20 | 975.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 964.20 | 970.47 | 973.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 963.35 | 962.20 | 965.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 955.25 | 961.13 | 963.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:30:00 | 958.00 | 959.22 | 961.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 957.65 | 960.51 | 961.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 953.00 | 957.31 | 959.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 953.00 | 956.45 | 958.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 951.95 | 956.45 | 958.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 951.50 | 955.07 | 957.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 950.50 | 953.57 | 956.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 951.20 | 951.50 | 954.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 954.00 | 952.06 | 954.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 954.00 | 952.06 | 954.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 950.50 | 951.75 | 953.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 949.40 | 951.75 | 953.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 948.00 | 951.54 | 953.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:45:00 | 948.25 | 950.71 | 952.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 955.50 | 951.67 | 952.97 | SL hit (close>static) qty=1.00 sl=954.25 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 959.45 | 954.16 | 953.84 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 948.10 | 953.78 | 954.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 942.20 | 949.36 | 951.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 15:15:00 | 923.00 | 922.76 | 928.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 922.70 | 922.76 | 928.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 923.20 | 922.85 | 927.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 919.60 | 922.85 | 927.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 921.85 | 921.08 | 924.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 921.65 | 921.33 | 924.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 875.76 | 883.24 | 887.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 875.57 | 883.24 | 887.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 873.62 | 880.53 | 886.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 877.95 | 877.75 | 883.32 | SL hit (close>ema200) qty=0.50 sl=877.75 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 832.20 | 829.50 | 829.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 833.75 | 830.35 | 829.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 841.55 | 843.04 | 839.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:45:00 | 841.60 | 843.04 | 839.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 850.80 | 844.27 | 840.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 872.15 | 845.74 | 841.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 841.25 | 845.02 | 845.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 841.25 | 845.02 | 845.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 838.45 | 843.71 | 844.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 820.80 | 820.63 | 826.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:15:00 | 821.85 | 820.63 | 826.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 825.10 | 821.95 | 826.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 825.90 | 821.95 | 826.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 830.65 | 823.30 | 825.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 833.65 | 823.30 | 825.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 835.15 | 825.67 | 826.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 835.55 | 825.67 | 826.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 838.95 | 828.33 | 827.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 840.50 | 830.76 | 828.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 855.00 | 858.22 | 850.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 855.00 | 858.22 | 850.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 851.50 | 856.39 | 851.77 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 841.35 | 849.77 | 850.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 838.55 | 843.76 | 846.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 840.15 | 839.18 | 842.77 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 844.50 | 842.61 | 842.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 846.75 | 844.27 | 843.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 843.10 | 844.18 | 843.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 843.10 | 844.18 | 843.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 841.00 | 843.54 | 843.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 840.70 | 843.54 | 843.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 844.50 | 843.79 | 843.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 848.20 | 843.79 | 843.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 850.00 | 845.03 | 844.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 854.95 | 846.27 | 845.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 854.00 | 850.96 | 848.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 845.00 | 849.47 | 849.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 845.00 | 849.47 | 849.63 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 850.85 | 849.65 | 849.56 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 847.60 | 849.24 | 849.38 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 851.95 | 849.78 | 849.62 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 848.65 | 849.43 | 849.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 843.05 | 847.45 | 848.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 840.70 | 837.39 | 841.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 842.85 | 837.39 | 841.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 838.90 | 837.69 | 841.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 835.75 | 837.60 | 840.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 822.00 | 815.81 | 814.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 822.00 | 815.81 | 814.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 823.85 | 819.86 | 817.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 825.00 | 825.04 | 821.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 824.00 | 825.04 | 821.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 821.00 | 824.23 | 821.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 821.00 | 824.23 | 821.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 823.20 | 824.02 | 821.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 824.65 | 824.34 | 822.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 810.00 | 822.63 | 822.47 | SL hit (close<static) qty=1.00 sl=820.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 817.05 | 821.51 | 821.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 808.90 | 813.60 | 816.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 15:15:00 | 801.00 | 799.15 | 803.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 802.50 | 799.15 | 803.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 798.00 | 798.92 | 802.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 795.70 | 798.92 | 802.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 793.40 | 792.34 | 793.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 798.50 | 788.81 | 788.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 798.50 | 788.81 | 788.64 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 788.00 | 790.21 | 790.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 786.50 | 789.46 | 789.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 794.10 | 789.04 | 789.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 795.50 | 789.04 | 789.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 796.00 | 790.43 | 790.02 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 786.25 | 790.73 | 790.99 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 795.40 | 790.87 | 790.44 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 786.50 | 790.57 | 790.83 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 804.10 | 791.09 | 790.47 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 786.30 | 791.67 | 792.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 774.35 | 785.46 | 788.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 792.10 | 773.51 | 775.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 794.10 | 773.51 | 775.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 795.20 | 777.85 | 777.68 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 765.85 | 780.39 | 781.17 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 789.00 | 781.16 | 781.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 795.55 | 784.89 | 782.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 789.95 | 791.53 | 788.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 789.95 | 791.53 | 788.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 789.95 | 791.22 | 788.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 786.65 | 791.22 | 788.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 781.85 | 789.34 | 788.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 781.85 | 789.34 | 788.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 787.20 | 788.91 | 788.08 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 782.25 | 787.41 | 787.53 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 792.55 | 787.60 | 787.31 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 786.95 | 788.57 | 788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 785.00 | 787.73 | 788.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 776.90 | 776.46 | 781.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 776.90 | 776.46 | 781.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 772.80 | 775.34 | 779.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 771.10 | 774.09 | 778.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 768.20 | 772.49 | 776.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 732.54 | 735.84 | 739.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 729.79 | 735.84 | 739.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 723.85 | 721.31 | 727.73 | SL hit (close>ema200) qty=0.50 sl=721.31 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 739.45 | 730.38 | 729.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 745.00 | 738.86 | 736.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 767.50 | 770.44 | 762.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 767.50 | 770.44 | 762.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 770.00 | 769.29 | 763.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 761.45 | 769.30 | 764.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 784.15 | 775.54 | 770.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 790.05 | 780.91 | 775.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 772.00 | 779.05 | 779.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 772.00 | 779.05 | 779.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 767.00 | 775.07 | 777.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 776.05 | 774.06 | 776.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 776.05 | 774.06 | 776.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 775.00 | 774.25 | 776.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:00:00 | 772.00 | 773.97 | 775.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:15:00 | 733.40 | 744.43 | 755.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 737.60 | 736.75 | 745.80 | SL hit (close>ema200) qty=0.50 sl=736.75 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 770.85 | 751.00 | 748.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 785.45 | 757.89 | 751.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 779.45 | 781.70 | 773.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 781.40 | 782.05 | 777.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 781.40 | 782.05 | 777.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 785.85 | 782.68 | 781.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 770.70 | 779.24 | 780.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 770.70 | 779.24 | 780.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 768.65 | 774.72 | 777.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 761.50 | 759.69 | 765.44 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 779.35 | 767.36 | 765.77 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 759.50 | 769.21 | 770.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 757.80 | 766.92 | 769.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 728.25 | 720.07 | 732.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 728.25 | 720.07 | 732.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 729.20 | 721.89 | 728.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 729.20 | 721.89 | 728.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 729.40 | 723.39 | 728.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 734.50 | 723.39 | 728.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 722.60 | 723.24 | 727.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 720.25 | 722.38 | 727.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 706.30 | 720.32 | 723.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 726.00 | 717.92 | 717.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 726.00 | 717.92 | 717.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 727.00 | 719.74 | 718.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 719.30 | 719.65 | 718.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 719.30 | 719.65 | 718.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 716.80 | 719.08 | 718.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 716.40 | 719.08 | 718.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 717.95 | 718.85 | 718.55 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 717.00 | 718.18 | 718.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 710.50 | 716.64 | 717.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 721.90 | 717.10 | 717.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 721.70 | 717.10 | 717.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 726.00 | 718.88 | 718.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 736.80 | 724.13 | 721.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 724.95 | 726.02 | 723.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 724.95 | 726.02 | 723.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 720.65 | 724.95 | 722.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 720.65 | 724.95 | 722.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 723.00 | 724.56 | 722.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 711.65 | 724.56 | 722.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 713.00 | 722.25 | 722.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 710.45 | 722.25 | 722.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 704.40 | 718.68 | 720.47 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 739.45 | 720.72 | 719.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 751.05 | 746.10 | 741.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 14:15:00 | 818.20 | 818.29 | 805.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 818.20 | 818.29 | 805.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 821.35 | 818.85 | 807.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 827.85 | 819.47 | 809.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 824.40 | 819.78 | 810.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 825.45 | 820.44 | 811.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 824.25 | 820.44 | 811.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 801.60 | 816.60 | 812.55 | SL hit (close<static) qty=1.00 sl=805.50 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 807.75 | 811.59 | 811.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 804.80 | 808.86 | 810.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 814.50 | 809.21 | 810.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 814.35 | 809.21 | 810.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 809.45 | 809.26 | 810.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 808.15 | 809.26 | 810.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 816.85 | 811.21 | 811.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 816.85 | 811.21 | 811.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 819.90 | 814.76 | 812.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 815.05 | 817.37 | 815.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 812.90 | 817.37 | 815.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 813.65 | 816.62 | 815.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 813.85 | 816.62 | 815.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 809.00 | 813.99 | 814.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 807.85 | 812.76 | 813.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 807.45 | 799.86 | 804.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 807.45 | 799.86 | 804.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 808.20 | 801.53 | 804.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 804.90 | 802.42 | 804.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 823.00 | 807.68 | 806.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 823.00 | 807.68 | 806.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 842.20 | 825.58 | 819.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 900.50 | 902.31 | 877.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 925.70 | 902.31 | 877.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 924.00 | 901.64 | 900.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 911.70 | 907.74 | 904.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 911.80 | 907.37 | 904.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 913.35 | 908.57 | 905.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 921.00 | 908.57 | 905.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 909.60 | 939.76 | 943.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 902.25 | 932.25 | 939.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 902.25 | 900.11 | 915.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 902.25 | 900.11 | 915.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 853.35 | 849.19 | 863.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 844.35 | 848.77 | 861.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 844.15 | 848.61 | 859.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 873.95 | 863.56 | 862.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 873.95 | 863.56 | 862.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 877.60 | 866.37 | 864.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 851.00 | 866.84 | 865.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 851.95 | 866.84 | 865.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 848.10 | 863.09 | 863.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 845.55 | 854.93 | 859.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 856.00 | 855.14 | 858.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:15:00 | 880.55 | 855.14 | 858.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 872.85 | 858.68 | 860.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 880.00 | 858.68 | 860.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 872.90 | 861.53 | 861.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 886.00 | 867.95 | 864.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 842.35 | 869.06 | 866.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 842.35 | 869.06 | 866.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 836.45 | 862.54 | 864.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 824.80 | 854.99 | 860.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 826.95 | 826.45 | 839.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 826.95 | 826.45 | 839.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 814.95 | 826.68 | 836.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 820.50 | 826.68 | 836.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 834.70 | 826.41 | 834.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 833.45 | 826.41 | 834.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 834.30 | 827.98 | 834.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 821.50 | 831.75 | 834.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 780.42 | 794.48 | 807.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 813.40 | 794.99 | 805.34 | SL hit (close>ema200) qty=0.50 sl=794.99 alert=retest2 |

### Cycle 133 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 804.70 | 799.46 | 799.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 810.50 | 801.67 | 800.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 802.05 | 803.08 | 801.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 802.60 | 803.08 | 801.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 848.00 | 850.87 | 843.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 844.85 | 850.87 | 843.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 831.70 | 846.90 | 843.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 841.15 | 846.90 | 843.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 925.27 | 879.92 | 865.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 990.30 | 997.61 | 997.67 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1010.50 | 997.59 | 997.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 1013.40 | 1002.98 | 999.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1009.00 | 1009.42 | 1004.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 1009.00 | 1009.42 | 1004.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1039.00 | 1015.34 | 1007.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1049.40 | 1015.34 | 1007.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1049.55 | 1039.32 | 1031.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 09:15:00 | 1154.34 | 1091.02 | 1066.72 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 11:30:00 | 1459.80 | 2024-06-04 09:15:00 | 1412.05 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-06-03 13:30:00 | 1453.60 | 2024-06-04 09:15:00 | 1412.05 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-06-03 14:00:00 | 1467.65 | 2024-06-04 09:15:00 | 1412.05 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2024-06-18 13:00:00 | 1580.40 | 2024-06-19 09:15:00 | 1557.75 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-06-18 13:45:00 | 1581.25 | 2024-06-19 09:15:00 | 1557.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-18 14:30:00 | 1584.45 | 2024-06-19 09:15:00 | 1557.75 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-06-24 09:15:00 | 1575.40 | 2024-06-27 09:15:00 | 1585.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-07-05 09:15:00 | 1657.00 | 2024-07-08 11:15:00 | 1625.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-07-23 11:30:00 | 1523.25 | 2024-07-23 12:15:00 | 1447.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 11:30:00 | 1523.25 | 2024-07-24 09:15:00 | 1540.75 | STOP_HIT | 0.50 | -1.15% |
| SELL | retest2 | 2024-07-25 11:15:00 | 1523.00 | 2024-07-26 10:15:00 | 1555.10 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-07-25 13:00:00 | 1523.15 | 2024-07-26 10:15:00 | 1555.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-07-25 14:30:00 | 1522.75 | 2024-07-26 10:15:00 | 1555.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1476.45 | 2024-08-08 12:15:00 | 1500.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-08-22 13:00:00 | 1446.45 | 2024-08-23 09:15:00 | 1506.90 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2024-08-22 14:00:00 | 1449.50 | 2024-08-23 09:15:00 | 1506.90 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-09-20 13:45:00 | 1382.00 | 2024-09-23 09:15:00 | 1432.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-09-20 14:15:00 | 1381.00 | 2024-09-23 09:15:00 | 1432.00 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-09-20 14:45:00 | 1381.80 | 2024-09-23 09:15:00 | 1432.00 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-09-27 09:30:00 | 1449.95 | 2024-10-03 11:15:00 | 1377.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1446.80 | 2024-10-03 13:15:00 | 1374.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 12:15:00 | 1447.80 | 2024-10-03 13:15:00 | 1375.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:45:00 | 1445.00 | 2024-10-03 13:15:00 | 1372.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 1418.30 | 2024-10-04 09:15:00 | 1347.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1420.00 | 2024-10-04 09:15:00 | 1349.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 09:30:00 | 1449.95 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1446.80 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2024-09-27 12:15:00 | 1447.80 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2024-09-27 14:45:00 | 1445.00 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2024-10-01 10:15:00 | 1418.30 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1420.00 | 2024-10-04 15:15:00 | 1381.95 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1295.95 | 2024-10-23 09:15:00 | 1231.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1295.95 | 2024-10-23 10:15:00 | 1280.40 | STOP_HIT | 0.50 | 1.20% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1097.00 | 2024-11-25 09:15:00 | 1140.00 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2024-11-29 10:45:00 | 1171.20 | 2024-12-05 11:15:00 | 1266.05 | TARGET_HIT | 1.00 | 8.10% |
| BUY | retest2 | 2024-11-29 11:15:00 | 1150.95 | 2024-12-05 11:15:00 | 1265.44 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2024-11-29 15:00:00 | 1150.40 | 2024-12-05 11:15:00 | 1265.22 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2024-12-02 15:00:00 | 1150.20 | 2024-12-05 12:15:00 | 1288.32 | TARGET_HIT | 1.00 | 12.01% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1178.90 | 2024-12-05 12:15:00 | 1296.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 1260.95 | 2024-12-19 09:15:00 | 1197.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 1260.95 | 2024-12-23 09:15:00 | 1134.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 12:45:00 | 1171.05 | 2025-01-06 13:15:00 | 1112.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 15:00:00 | 1170.25 | 2025-01-06 13:15:00 | 1111.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 15:15:00 | 1171.15 | 2025-01-06 13:15:00 | 1112.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 10:45:00 | 1171.95 | 2025-01-06 13:15:00 | 1113.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 12:45:00 | 1171.05 | 2025-01-07 12:15:00 | 1123.90 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-01-01 15:00:00 | 1170.25 | 2025-01-07 12:15:00 | 1123.90 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-01-02 15:15:00 | 1171.15 | 2025-01-07 12:15:00 | 1123.90 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-01-03 10:45:00 | 1171.95 | 2025-01-07 12:15:00 | 1123.90 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-01-22 12:00:00 | 991.80 | 2025-01-30 12:15:00 | 998.55 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-01-24 12:00:00 | 991.00 | 2025-01-30 12:15:00 | 998.55 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-27 09:15:00 | 966.75 | 2025-01-30 12:15:00 | 998.55 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-01-28 09:15:00 | 988.95 | 2025-01-30 12:15:00 | 998.55 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-01-29 09:15:00 | 968.00 | 2025-01-30 12:15:00 | 998.55 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest1 | 2025-02-01 09:15:00 | 1019.85 | 2025-02-01 12:15:00 | 1000.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-02-07 10:15:00 | 970.00 | 2025-02-11 11:15:00 | 921.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 970.00 | 2025-02-11 11:15:00 | 921.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 970.00 | 2025-02-13 09:15:00 | 873.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 970.00 | 2025-02-13 09:15:00 | 873.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-18 12:45:00 | 876.50 | 2025-02-19 14:15:00 | 932.75 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2025-02-18 15:15:00 | 887.10 | 2025-02-19 14:15:00 | 932.75 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest2 | 2025-03-04 11:45:00 | 814.50 | 2025-03-05 15:15:00 | 828.05 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-03-10 11:15:00 | 832.40 | 2025-03-10 11:15:00 | 824.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-03-13 11:45:00 | 794.40 | 2025-03-18 09:15:00 | 818.45 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-03-13 14:15:00 | 793.00 | 2025-03-18 09:15:00 | 818.45 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-03-17 11:15:00 | 794.40 | 2025-03-18 09:15:00 | 818.45 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-03-17 11:45:00 | 790.80 | 2025-03-18 09:15:00 | 818.45 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-04-09 09:15:00 | 860.20 | 2025-04-11 11:15:00 | 880.85 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-04-09 13:15:00 | 861.95 | 2025-04-11 11:15:00 | 880.85 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-04-09 15:00:00 | 856.15 | 2025-04-11 11:15:00 | 880.85 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-04-21 09:30:00 | 897.50 | 2025-04-23 09:15:00 | 893.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-04-21 15:15:00 | 897.75 | 2025-04-23 09:15:00 | 893.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-04-24 11:15:00 | 883.15 | 2025-04-30 09:15:00 | 880.80 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest1 | 2025-05-14 09:15:00 | 932.60 | 2025-05-19 09:15:00 | 968.31 | PARTIAL | 0.50 | 3.83% |
| BUY | retest1 | 2025-05-14 09:15:00 | 932.60 | 2025-05-19 13:15:00 | 957.95 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2025-05-14 11:00:00 | 922.20 | 2025-05-20 12:15:00 | 950.50 | STOP_HIT | 1.00 | 3.07% |
| SELL | retest2 | 2025-06-18 11:45:00 | 957.00 | 2025-06-20 14:15:00 | 912.95 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-06-18 12:30:00 | 961.00 | 2025-06-20 14:15:00 | 912.14 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-06-18 14:15:00 | 960.15 | 2025-06-20 14:15:00 | 912.95 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-06-18 15:15:00 | 961.00 | 2025-06-20 15:15:00 | 909.15 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2025-06-18 11:45:00 | 957.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-06-18 12:30:00 | 961.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-06-18 14:15:00 | 960.15 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2025-06-18 15:15:00 | 961.00 | 2025-06-23 09:15:00 | 932.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-06-19 10:15:00 | 952.25 | 2025-06-25 10:15:00 | 939.05 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-07-10 11:15:00 | 955.25 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-10 13:30:00 | 958.00 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-07-11 10:15:00 | 957.65 | 2025-07-16 10:15:00 | 955.50 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-07-11 15:15:00 | 953.00 | 2025-07-16 12:15:00 | 956.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-14 09:15:00 | 951.95 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-14 11:00:00 | 951.50 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-14 12:30:00 | 950.50 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-15 09:30:00 | 951.20 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-15 14:15:00 | 949.40 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-15 15:15:00 | 948.00 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-16 09:45:00 | 948.25 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-16 10:45:00 | 950.00 | 2025-07-16 13:15:00 | 959.45 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-23 10:15:00 | 919.60 | 2025-08-01 15:15:00 | 875.76 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-07-23 15:00:00 | 921.85 | 2025-08-01 15:15:00 | 875.57 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-07-24 09:15:00 | 921.65 | 2025-08-04 09:15:00 | 873.62 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-07-23 10:15:00 | 919.60 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-07-23 15:00:00 | 921.85 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-07-24 09:15:00 | 921.65 | 2025-08-04 12:15:00 | 877.95 | STOP_HIT | 0.50 | 4.74% |
| BUY | retest2 | 2025-08-22 09:15:00 | 872.15 | 2025-08-25 13:15:00 | 841.25 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-09-15 09:15:00 | 854.95 | 2025-09-16 14:15:00 | 845.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-15 12:30:00 | 854.00 | 2025-09-16 14:15:00 | 845.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-22 13:45:00 | 835.75 | 2025-10-01 09:15:00 | 822.00 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-10-06 13:30:00 | 824.65 | 2025-10-07 11:15:00 | 810.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-14 10:15:00 | 795.70 | 2025-10-21 13:15:00 | 798.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-16 11:00:00 | 793.40 | 2025-10-21 13:15:00 | 798.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-20 13:15:00 | 771.10 | 2025-12-08 09:15:00 | 732.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:45:00 | 768.20 | 2025-12-08 09:15:00 | 729.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 771.10 | 2025-12-09 11:15:00 | 723.85 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2025-11-20 14:45:00 | 768.20 | 2025-12-09 11:15:00 | 723.85 | STOP_HIT | 0.50 | 5.77% |
| BUY | retest2 | 2025-12-19 15:15:00 | 790.05 | 2025-12-23 13:15:00 | 772.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-12-24 15:00:00 | 772.00 | 2025-12-30 10:15:00 | 733.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 15:00:00 | 772.00 | 2025-12-31 09:15:00 | 737.60 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2026-01-08 09:30:00 | 785.85 | 2026-01-08 14:15:00 | 770.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-23 10:45:00 | 720.25 | 2026-01-28 14:15:00 | 726.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-27 09:15:00 | 706.30 | 2026-01-28 14:15:00 | 726.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-02-12 11:15:00 | 827.85 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-02-12 12:15:00 | 824.40 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-02-12 12:45:00 | 825.45 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-12 13:15:00 | 824.25 | 2026-02-13 09:15:00 | 801.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-02-13 11:45:00 | 808.70 | 2026-02-16 11:15:00 | 807.75 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2026-02-17 11:15:00 | 808.15 | 2026-02-17 12:15:00 | 816.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-02-23 11:30:00 | 804.90 | 2026-02-23 13:15:00 | 823.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-03-02 10:15:00 | 925.70 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-05 09:15:00 | 924.00 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-03-05 12:30:00 | 911.70 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-03-05 13:30:00 | 911.80 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-03-05 15:15:00 | 921.00 | 2026-03-11 09:15:00 | 909.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-17 12:15:00 | 844.35 | 2026-03-18 12:15:00 | 873.95 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-03-17 12:45:00 | 844.15 | 2026-03-18 12:15:00 | 873.95 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-03-27 09:15:00 | 821.50 | 2026-03-30 14:15:00 | 780.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 821.50 | 2026-04-01 09:15:00 | 813.40 | STOP_HIT | 0.50 | 0.99% |
| BUY | retest2 | 2026-04-13 10:15:00 | 841.15 | 2026-04-16 09:15:00 | 925.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 11:15:00 | 1049.40 | 2026-05-08 09:15:00 | 1154.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1049.55 | 2026-05-08 09:15:00 | 1154.51 | TARGET_HIT | 1.00 | 10.00% |
