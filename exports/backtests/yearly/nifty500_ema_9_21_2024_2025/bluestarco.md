# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1691.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 138 |
| ALERT1 | 97 |
| ALERT2 | 95 |
| ALERT2_SKIP | 55 |
| ALERT3 | 269 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 123 |
| PARTIAL | 25 |
| TARGET_HIT | 17 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 64 / 83
- **Target hits / Stop hits / Partials:** 17 / 105 / 25
- **Avg / median % per leg:** 1.54% / -0.25%
- **Sum % (uncompounded):** 227.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 12 | 24.5% | 8 | 41 | 0 | 0.59% | 28.9% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.11% | -0.5% |
| BUY @ 3rd Alert (retest2) | 45 | 11 | 24.4% | 8 | 37 | 0 | 0.65% | 29.3% |
| SELL (all) | 98 | 52 | 53.1% | 9 | 64 | 25 | 2.02% | 198.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 98 | 52 | 53.1% | 9 | 64 | 25 | 2.02% | 198.1% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.11% | -0.5% |
| retest2 (combined) | 143 | 63 | 44.1% | 17 | 101 | 25 | 1.59% | 227.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 1437.00 | 1418.08 | 1417.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1461.80 | 1429.73 | 1423.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1484.45 | 1494.22 | 1474.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1484.45 | 1494.22 | 1474.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1560.00 | 1577.97 | 1559.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1557.80 | 1577.97 | 1559.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1557.30 | 1573.84 | 1559.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:30:00 | 1556.00 | 1573.84 | 1559.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1591.10 | 1577.29 | 1562.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 1556.90 | 1577.29 | 1562.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1578.60 | 1576.80 | 1564.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 1557.10 | 1576.80 | 1564.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1563.60 | 1574.03 | 1565.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1563.95 | 1574.03 | 1565.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1548.05 | 1568.84 | 1563.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 1548.05 | 1568.84 | 1563.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1554.55 | 1565.98 | 1563.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 1545.50 | 1565.98 | 1563.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 1545.40 | 1559.86 | 1560.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 14:15:00 | 1528.25 | 1553.54 | 1557.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 1498.75 | 1498.38 | 1510.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 13:45:00 | 1499.55 | 1498.38 | 1510.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1501.95 | 1495.43 | 1505.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 1514.10 | 1495.43 | 1505.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1498.05 | 1495.95 | 1504.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 1487.05 | 1493.77 | 1503.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1516.70 | 1500.94 | 1504.30 | SL hit (close>static) qty=1.00 sl=1510.45 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 1528.05 | 1509.89 | 1508.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1484.20 | 1509.03 | 1510.15 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 1541.55 | 1510.51 | 1507.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 1612.90 | 1530.99 | 1517.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1568.15 | 1594.12 | 1575.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1567.35 | 1588.77 | 1574.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:30:00 | 1585.10 | 1588.63 | 1575.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 11:15:00 | 1534.30 | 1567.79 | 1570.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 11:15:00 | 1534.30 | 1567.79 | 1570.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 12:15:00 | 1525.00 | 1559.24 | 1566.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 1524.40 | 1545.02 | 1553.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:00:00 | 1526.05 | 1541.22 | 1551.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 1567.55 | 1550.10 | 1551.42 | SL hit (close>static) qty=1.00 sl=1565.20 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 1566.85 | 1553.45 | 1552.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1600.75 | 1567.10 | 1559.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 1654.65 | 1666.37 | 1642.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 1650.00 | 1666.37 | 1642.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1652.75 | 1663.65 | 1643.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 1673.00 | 1664.43 | 1645.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 15:15:00 | 1700.00 | 1720.84 | 1721.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 1700.00 | 1720.84 | 1721.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 1696.00 | 1715.87 | 1719.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 1687.25 | 1684.32 | 1698.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 11:00:00 | 1687.25 | 1684.32 | 1698.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1690.30 | 1685.60 | 1695.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 1692.30 | 1685.60 | 1695.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1670.05 | 1683.60 | 1692.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1702.25 | 1683.60 | 1692.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1719.15 | 1690.71 | 1695.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 1719.15 | 1690.71 | 1695.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1715.00 | 1695.57 | 1697.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:00:00 | 1708.30 | 1698.12 | 1698.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 1709.85 | 1700.46 | 1699.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 1709.85 | 1700.46 | 1699.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 1691.10 | 1697.54 | 1698.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 1684.50 | 1694.93 | 1696.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 1660.40 | 1643.97 | 1651.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1649.05 | 1644.99 | 1651.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:30:00 | 1645.00 | 1644.75 | 1649.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 1657.00 | 1615.81 | 1616.29 | SL hit (close>static) qty=1.00 sl=1656.15 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 1702.40 | 1633.13 | 1624.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 1749.90 | 1656.48 | 1635.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 1673.40 | 1684.56 | 1661.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 1673.40 | 1684.56 | 1661.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1657.40 | 1679.13 | 1661.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1657.40 | 1679.13 | 1661.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1653.15 | 1673.93 | 1660.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:00:00 | 1653.15 | 1673.93 | 1660.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1652.05 | 1669.55 | 1659.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1652.05 | 1669.55 | 1659.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1661.50 | 1667.94 | 1660.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:15:00 | 1657.20 | 1667.94 | 1660.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1657.20 | 1665.80 | 1659.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 1682.00 | 1665.80 | 1659.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1685.50 | 1669.74 | 1662.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 1704.00 | 1678.30 | 1667.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-09 15:15:00 | 1874.40 | 1764.43 | 1715.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 1701.80 | 1722.80 | 1724.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 1692.25 | 1710.12 | 1716.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1711.20 | 1700.28 | 1708.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1704.60 | 1701.14 | 1708.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1694.00 | 1701.14 | 1708.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 12:15:00 | 1709.90 | 1700.25 | 1706.67 | SL hit (close>static) qty=1.00 sl=1709.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 15:15:00 | 1730.00 | 1713.41 | 1711.59 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1688.40 | 1708.41 | 1709.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1664.75 | 1699.68 | 1705.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:45:00 | 1709.65 | 1701.25 | 1705.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1706.60 | 1702.32 | 1705.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1706.60 | 1702.32 | 1705.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1693.90 | 1700.63 | 1704.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1671.85 | 1700.11 | 1703.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 1588.26 | 1639.68 | 1649.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 1646.05 | 1640.95 | 1649.06 | SL hit (close>ema200) qty=0.50 sl=1640.95 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 1684.75 | 1656.76 | 1655.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1689.40 | 1667.00 | 1660.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 1658.00 | 1665.20 | 1660.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 1654.50 | 1663.06 | 1659.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 1676.20 | 1663.06 | 1659.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1654.55 | 1661.36 | 1659.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:30:00 | 1682.95 | 1671.27 | 1665.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 13:15:00 | 1719.00 | 1731.27 | 1732.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 1719.00 | 1731.27 | 1732.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1710.90 | 1725.11 | 1729.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1604.30 | 1584.36 | 1615.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1621.95 | 1591.88 | 1615.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 1621.95 | 1591.88 | 1615.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1624.10 | 1598.33 | 1616.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 1641.35 | 1598.33 | 1616.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1600.00 | 1604.41 | 1614.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 1618.75 | 1604.41 | 1614.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1618.70 | 1607.26 | 1614.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 1618.70 | 1607.26 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1619.70 | 1609.75 | 1615.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1619.70 | 1609.75 | 1615.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1618.40 | 1611.48 | 1615.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 1619.50 | 1611.48 | 1615.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1603.50 | 1609.89 | 1614.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1589.90 | 1609.67 | 1612.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 1586.45 | 1603.15 | 1608.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 1702.95 | 1627.41 | 1617.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 1702.95 | 1627.41 | 1617.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 11:15:00 | 1722.75 | 1658.16 | 1633.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 09:15:00 | 1706.55 | 1707.49 | 1685.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:00:00 | 1706.55 | 1707.49 | 1685.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1713.90 | 1718.00 | 1702.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:00:00 | 1721.05 | 1718.61 | 1703.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:00:00 | 1722.25 | 1719.34 | 1705.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:45:00 | 1721.80 | 1726.06 | 1715.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 1726.00 | 1727.66 | 1717.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1692.10 | 1720.55 | 1715.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 1692.10 | 1720.55 | 1715.11 | SL hit (close<static) qty=1.00 sl=1700.05 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 1690.75 | 1708.29 | 1710.09 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1740.80 | 1709.26 | 1707.52 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 1702.85 | 1719.66 | 1720.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 1697.30 | 1712.03 | 1716.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 1705.00 | 1702.28 | 1710.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 1704.95 | 1702.28 | 1710.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1720.00 | 1705.82 | 1711.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 1719.15 | 1705.82 | 1711.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1735.90 | 1711.84 | 1713.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 1735.30 | 1711.84 | 1713.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 1745.00 | 1718.47 | 1716.41 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1697.35 | 1721.31 | 1723.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 15:15:00 | 1692.20 | 1708.45 | 1716.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 1709.05 | 1701.22 | 1709.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1700.65 | 1701.10 | 1708.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1700.65 | 1701.10 | 1708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1674.75 | 1671.69 | 1685.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 1676.95 | 1671.69 | 1685.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1684.40 | 1675.16 | 1681.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1684.40 | 1675.16 | 1681.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1680.00 | 1676.13 | 1681.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1724.00 | 1676.13 | 1681.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1723.30 | 1685.56 | 1685.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 09:15:00 | 1736.80 | 1719.86 | 1710.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 1800.10 | 1800.31 | 1780.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:15:00 | 1795.85 | 1800.31 | 1780.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1795.30 | 1799.14 | 1783.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 1773.95 | 1799.14 | 1783.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1775.20 | 1794.35 | 1782.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1800.65 | 1795.72 | 1784.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-16 09:15:00 | 1980.72 | 1871.13 | 1838.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1881.20 | 1885.50 | 1885.66 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1889.25 | 1885.85 | 1885.77 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1869.05 | 1883.16 | 1884.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 1856.60 | 1870.80 | 1877.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1884.15 | 1873.47 | 1877.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1904.10 | 1879.60 | 1880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 1915.70 | 1879.60 | 1880.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1882.05 | 1878.77 | 1879.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:00:00 | 1882.05 | 1878.77 | 1879.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 1934.15 | 1889.85 | 1884.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 1966.95 | 1924.00 | 1905.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 11:15:00 | 2122.30 | 2124.90 | 2100.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:00:00 | 2122.30 | 2124.90 | 2100.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 2099.20 | 2119.76 | 2100.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:45:00 | 2097.60 | 2119.76 | 2100.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 2089.00 | 2113.61 | 2099.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 2089.00 | 2113.61 | 2099.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 2049.90 | 2100.87 | 2095.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 2049.90 | 2100.87 | 2095.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 2068.00 | 2094.29 | 2092.69 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 2071.00 | 2089.63 | 2090.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 2050.00 | 2078.57 | 2085.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 2075.30 | 2068.60 | 2078.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 2100.00 | 2074.88 | 2080.13 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 12:15:00 | 2121.90 | 2087.74 | 2084.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 14:15:00 | 2150.85 | 2101.09 | 2091.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 2116.50 | 2120.56 | 2107.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 2052.95 | 2107.04 | 2102.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 2052.95 | 2107.04 | 2102.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 2067.60 | 2099.15 | 2099.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 2025.00 | 2084.32 | 2092.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 1986.65 | 1934.72 | 1970.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1978.00 | 1943.38 | 1970.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:15:00 | 2013.10 | 1943.38 | 1970.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 2005.00 | 1984.93 | 1983.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 14:15:00 | 2020.00 | 2001.27 | 1993.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 2004.65 | 2010.37 | 2000.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:45:00 | 2002.55 | 2010.37 | 2000.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 2017.20 | 2013.33 | 2004.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 2038.30 | 2015.05 | 2006.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 2024.85 | 2017.01 | 2007.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 2024.00 | 2020.82 | 2013.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2026.00 | 2022.74 | 2015.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 2047.20 | 2041.32 | 2030.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 2011.90 | 2024.19 | 2025.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 2011.90 | 2024.19 | 2025.24 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 2045.00 | 2025.02 | 2022.51 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 2009.45 | 2022.00 | 2023.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 2005.00 | 2016.98 | 2020.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 1888.00 | 1879.44 | 1905.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 09:15:00 | 1875.50 | 1879.44 | 1905.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1847.15 | 1872.98 | 1899.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 1835.65 | 1865.90 | 1894.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 1829.40 | 1851.00 | 1873.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 10:00:00 | 1829.80 | 1851.00 | 1873.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:15:00 | 1832.00 | 1848.24 | 1865.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 1850.85 | 1833.37 | 1846.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 1850.85 | 1833.37 | 1846.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1850.00 | 1836.70 | 1847.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:00:00 | 1836.05 | 1838.70 | 1846.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:00:00 | 1839.60 | 1823.20 | 1832.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 13:15:00 | 1844.20 | 1837.64 | 1837.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 13:15:00 | 1844.20 | 1837.64 | 1837.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 1870.00 | 1844.11 | 1840.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1820.20 | 1863.72 | 1853.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1833.50 | 1857.68 | 1851.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 1838.00 | 1857.68 | 1851.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 1823.90 | 1847.38 | 1847.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1823.90 | 1847.38 | 1847.47 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1855.70 | 1843.74 | 1843.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1887.85 | 1853.56 | 1848.06 | Break + close above crossover candle high |

### Cycle 38 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 1753.75 | 1849.49 | 1852.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 1742.60 | 1828.11 | 1842.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 1817.25 | 1797.50 | 1817.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1829.90 | 1803.98 | 1818.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 1824.75 | 1803.98 | 1818.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1815.00 | 1806.18 | 1817.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 12:30:00 | 1809.95 | 1806.05 | 1816.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 13:30:00 | 1806.75 | 1802.74 | 1814.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 1814.00 | 1784.31 | 1781.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 11:15:00 | 1814.00 | 1784.31 | 1781.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 09:15:00 | 1843.90 | 1813.98 | 1806.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1826.20 | 1833.33 | 1823.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1832.55 | 1832.97 | 1824.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 1823.20 | 1832.97 | 1824.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1847.05 | 1847.47 | 1841.24 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 1802.20 | 1835.09 | 1836.95 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 1853.30 | 1836.23 | 1834.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 15:15:00 | 1854.55 | 1839.89 | 1836.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1836.40 | 1840.54 | 1837.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1838.65 | 1840.16 | 1837.69 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 1831.95 | 1835.56 | 1835.95 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 1856.85 | 1840.15 | 1837.92 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 1818.00 | 1838.63 | 1839.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 1814.60 | 1833.82 | 1837.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:30:00 | 1844.00 | 1835.09 | 1837.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1845.25 | 1837.13 | 1837.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1861.30 | 1837.13 | 1837.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1872.10 | 1844.12 | 1841.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1900.00 | 1868.72 | 1856.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 2021.60 | 2040.85 | 1991.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 12:00:00 | 2086.70 | 2051.81 | 2005.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 14:15:00 | 2092.00 | 2059.87 | 2017.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2107.00 | 2097.88 | 2071.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 2118.00 | 2103.94 | 2076.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 2090.25 | 2111.25 | 2098.42 | SL hit (close<ema400) qty=1.00 sl=2098.42 alert=retest1 |

### Cycle 46 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 2048.70 | 2087.61 | 2089.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 2046.20 | 2079.33 | 2085.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 2060.80 | 2059.87 | 2072.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 2060.80 | 2059.87 | 2072.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 2090.75 | 2066.05 | 2074.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 2090.75 | 2066.05 | 2074.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2090.00 | 2070.84 | 2075.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 2094.45 | 2070.84 | 2075.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 2136.65 | 2085.71 | 2081.90 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 2070.00 | 2099.23 | 2100.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 2065.75 | 2092.54 | 2097.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 2030.90 | 2029.84 | 2049.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:45:00 | 2029.55 | 2029.84 | 2049.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2048.90 | 2033.65 | 2049.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 2048.90 | 2033.65 | 2049.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2033.00 | 2033.52 | 2047.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2039.95 | 2033.52 | 2047.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2039.80 | 2034.78 | 2047.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 2024.15 | 2035.63 | 2044.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 2033.45 | 2004.87 | 2003.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 2033.45 | 2004.87 | 2003.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2045.00 | 2012.90 | 2007.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 2115.55 | 2120.48 | 2087.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 11:15:00 | 2122.95 | 2117.92 | 2091.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 2122.95 | 2117.92 | 2091.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 2099.95 | 2117.92 | 2091.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 2160.55 | 2138.36 | 2113.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:15:00 | 2168.00 | 2138.36 | 2113.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-01 14:15:00 | 2384.80 | 2209.49 | 2162.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 2231.00 | 2262.14 | 2264.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 2175.00 | 2244.71 | 2256.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 2138.80 | 2122.99 | 2166.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 2138.80 | 2122.99 | 2166.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1881.85 | 1875.64 | 1907.80 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1928.00 | 1917.72 | 1917.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 1934.95 | 1921.17 | 1918.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 1921.55 | 1928.08 | 1923.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1926.95 | 1927.85 | 1923.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 1922.00 | 1927.85 | 1923.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1925.85 | 1927.44 | 1924.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 1924.30 | 1927.44 | 1924.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1921.95 | 1926.34 | 1924.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 1923.95 | 1926.34 | 1924.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1926.65 | 1926.40 | 1924.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1917.80 | 1926.40 | 1924.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1899.85 | 1921.09 | 1922.21 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1934.15 | 1922.29 | 1922.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 1946.95 | 1927.22 | 1924.53 | Break + close above crossover candle high |

### Cycle 54 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1886.50 | 1920.26 | 1922.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1872.85 | 1910.78 | 1917.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1879.85 | 1866.37 | 1882.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1885.00 | 1870.10 | 1882.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1850.00 | 1870.10 | 1882.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 1922.70 | 1883.69 | 1886.54 | SL hit (close>static) qty=1.00 sl=1892.90 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 1920.45 | 1891.04 | 1889.63 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1878.75 | 1892.67 | 1893.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1859.35 | 1886.01 | 1890.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1818.40 | 1778.90 | 1797.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1833.90 | 1789.90 | 1801.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1833.90 | 1789.90 | 1801.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1829.75 | 1812.06 | 1809.72 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 1806.20 | 1810.49 | 1810.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 1800.85 | 1808.56 | 1809.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1818.15 | 1805.97 | 1807.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 1820.90 | 1808.96 | 1808.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 1835.90 | 1814.35 | 1811.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 14:15:00 | 1815.55 | 1816.80 | 1813.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 1815.55 | 1816.80 | 1813.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1823.10 | 1818.06 | 1813.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 1866.10 | 1818.06 | 1813.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 14:15:00 | 2052.71 | 1921.46 | 1870.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1971.05 | 1983.63 | 1985.27 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 2002.05 | 1986.24 | 1985.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 11:15:00 | 2029.95 | 1994.98 | 1989.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 1985.60 | 2010.35 | 2001.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1969.45 | 2002.17 | 1998.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1969.45 | 2002.17 | 1998.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1972.05 | 1996.15 | 1996.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1888.95 | 1962.69 | 1979.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 1940.25 | 1900.12 | 1929.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1964.90 | 1913.08 | 1932.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1964.90 | 1913.08 | 1932.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1985.10 | 1927.48 | 1937.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1985.10 | 1927.48 | 1937.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 1997.20 | 1948.72 | 1945.78 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 1897.85 | 1948.03 | 1953.57 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 1911.15 | 1904.39 | 1904.29 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1885.25 | 1905.04 | 1905.67 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 1925.00 | 1900.26 | 1896.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 10:15:00 | 1943.15 | 1921.17 | 1910.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1906.85 | 1925.34 | 1916.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1938.95 | 1928.06 | 1918.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 1947.80 | 1928.49 | 1919.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-04 09:15:00 | 2142.58 | 2032.05 | 1982.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 2098.65 | 2105.16 | 2105.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 2092.50 | 2102.62 | 2104.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2121.60 | 2049.52 | 2062.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 2097.00 | 2059.01 | 2065.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 2105.25 | 2059.01 | 2065.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 2051.65 | 2056.68 | 2063.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 2051.65 | 2056.68 | 2063.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 2024.20 | 2046.96 | 2056.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 2010.80 | 2029.14 | 2038.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 2099.55 | 2041.76 | 2042.48 | SL hit (close>static) qty=1.00 sl=2066.15 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 2127.50 | 2058.91 | 2050.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 2138.95 | 2074.92 | 2058.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 2162.50 | 2185.27 | 2153.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 12:00:00 | 2162.50 | 2185.27 | 2153.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2156.55 | 2192.65 | 2179.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2156.55 | 2192.65 | 2179.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2174.00 | 2188.92 | 2178.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2219.65 | 2188.92 | 2178.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 2191.00 | 2195.07 | 2195.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 2191.00 | 2195.07 | 2195.40 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 2229.80 | 2201.52 | 2197.66 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 2192.05 | 2196.08 | 2196.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 2168.45 | 2191.14 | 2194.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 2066.20 | 2083.72 | 2100.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1859.58 | 2016.88 | 2055.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 2013.80 | 1970.47 | 1968.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 2023.70 | 1992.00 | 1979.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2023.60 | 2033.11 | 2014.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 12:15:00 | 2020.00 | 2027.81 | 2016.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 2020.00 | 2027.81 | 2016.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:30:00 | 2017.50 | 2027.81 | 2016.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 2021.70 | 2026.59 | 2016.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 2030.60 | 2018.25 | 2015.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1949.90 | 2020.26 | 2024.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1949.90 | 2020.26 | 2024.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 1926.10 | 1974.01 | 1999.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1785.30 | 1778.10 | 1841.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 1797.00 | 1778.10 | 1841.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1682.10 | 1680.05 | 1700.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1691.60 | 1680.05 | 1700.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1689.70 | 1679.01 | 1691.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1697.90 | 1679.01 | 1691.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1702.00 | 1683.61 | 1692.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:30:00 | 1709.20 | 1683.61 | 1692.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1717.40 | 1690.36 | 1695.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 1717.40 | 1690.36 | 1695.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1739.20 | 1700.13 | 1699.02 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 1688.00 | 1699.49 | 1699.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1648.90 | 1689.37 | 1694.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1672.20 | 1674.63 | 1682.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1655.90 | 1670.88 | 1680.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 1623.40 | 1656.90 | 1671.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 1637.70 | 1651.40 | 1667.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1542.23 | 1621.99 | 1648.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1555.82 | 1621.99 | 1648.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1600.00 | 1596.16 | 1620.98 | SL hit (close>ema200) qty=0.50 sl=1596.16 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1618.90 | 1575.74 | 1575.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1630.80 | 1586.75 | 1580.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1601.40 | 1607.84 | 1596.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1612.20 | 1608.71 | 1597.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 1618.40 | 1608.71 | 1597.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:30:00 | 1620.50 | 1609.53 | 1601.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1630.50 | 1610.43 | 1602.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1580.00 | 1604.74 | 1602.14 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 1580.00 | 1599.79 | 1600.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 1568.00 | 1587.08 | 1592.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1569.80 | 1569.08 | 1578.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1568.10 | 1568.88 | 1577.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 1564.20 | 1568.88 | 1577.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1572.80 | 1557.65 | 1557.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1572.80 | 1557.65 | 1557.54 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 1551.70 | 1558.37 | 1558.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 1548.70 | 1555.26 | 1556.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 1556.20 | 1547.11 | 1551.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1566.10 | 1550.91 | 1553.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 1564.50 | 1550.91 | 1553.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1564.90 | 1553.70 | 1554.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:15:00 | 1567.50 | 1553.70 | 1554.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1548.40 | 1551.08 | 1552.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1540.30 | 1551.08 | 1552.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1535.80 | 1548.02 | 1551.18 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1555.00 | 1549.13 | 1548.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1562.00 | 1551.71 | 1549.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 1547.10 | 1551.53 | 1550.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1549.60 | 1551.14 | 1550.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 1556.00 | 1552.38 | 1550.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 1628.20 | 1649.48 | 1649.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1628.20 | 1649.48 | 1649.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1606.00 | 1634.29 | 1642.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1602.70 | 1601.48 | 1617.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:30:00 | 1604.00 | 1601.48 | 1617.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1610.50 | 1603.35 | 1613.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1610.50 | 1603.35 | 1613.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1606.80 | 1604.04 | 1612.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 1616.80 | 1604.04 | 1612.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1616.00 | 1606.43 | 1612.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 1605.10 | 1608.08 | 1611.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1634.70 | 1615.82 | 1614.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1634.70 | 1615.82 | 1614.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1640.10 | 1623.38 | 1618.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1631.40 | 1631.48 | 1625.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1650.30 | 1661.93 | 1652.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1650.30 | 1661.93 | 1652.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1645.00 | 1658.54 | 1651.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 1646.70 | 1658.54 | 1651.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1620.50 | 1650.93 | 1648.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1620.50 | 1650.93 | 1648.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1632.00 | 1647.15 | 1647.22 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 1682.20 | 1649.94 | 1646.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 12:15:00 | 1688.50 | 1661.82 | 1652.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1830.80 | 1832.37 | 1787.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 1830.80 | 1832.37 | 1787.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1805.50 | 1827.31 | 1807.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1805.50 | 1827.31 | 1807.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1810.00 | 1823.85 | 1807.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:30:00 | 1804.10 | 1823.85 | 1807.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1805.00 | 1818.24 | 1809.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 1806.10 | 1818.24 | 1809.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1805.10 | 1815.61 | 1809.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 1807.30 | 1815.61 | 1809.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1805.80 | 1813.65 | 1809.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1805.80 | 1813.65 | 1809.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1811.00 | 1813.12 | 1809.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:45:00 | 1812.20 | 1811.85 | 1809.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1813.90 | 1811.85 | 1809.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 1814.90 | 1811.50 | 1809.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 1813.00 | 1812.61 | 1810.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1808.70 | 1811.83 | 1810.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 1810.60 | 1811.83 | 1810.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1810.10 | 1811.48 | 1810.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 1808.00 | 1811.48 | 1810.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1810.00 | 1811.18 | 1810.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 1803.50 | 1811.18 | 1810.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 1810.00 | 1810.95 | 1810.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 1811.70 | 1810.95 | 1810.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1800.60 | 1808.88 | 1809.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1800.60 | 1808.88 | 1809.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1799.10 | 1806.92 | 1808.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 1801.40 | 1798.34 | 1801.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1800.70 | 1798.81 | 1801.78 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1803.00 | 1802.84 | 1802.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1825.50 | 1808.18 | 1805.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1860.20 | 1867.56 | 1846.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 1860.70 | 1867.56 | 1846.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1901.10 | 1872.86 | 1858.91 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1840.10 | 1863.94 | 1864.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 1838.00 | 1851.38 | 1857.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 1755.20 | 1753.70 | 1773.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 1756.20 | 1753.70 | 1773.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1769.80 | 1757.55 | 1770.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1770.40 | 1757.55 | 1770.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1766.80 | 1759.40 | 1770.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 1765.00 | 1759.40 | 1770.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1782.00 | 1764.82 | 1770.83 | SL hit (close>static) qty=1.00 sl=1770.20 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1755.30 | 1747.62 | 1747.10 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1739.30 | 1746.37 | 1746.65 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1757.20 | 1746.39 | 1746.13 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1737.40 | 1744.59 | 1745.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1725.50 | 1737.45 | 1741.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1718.70 | 1717.16 | 1728.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:45:00 | 1719.40 | 1717.16 | 1728.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1732.50 | 1719.38 | 1726.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1732.50 | 1719.38 | 1726.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1742.90 | 1724.09 | 1728.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1742.90 | 1724.09 | 1728.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1748.00 | 1730.89 | 1730.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1773.00 | 1739.31 | 1734.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 1747.40 | 1749.34 | 1741.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 1747.40 | 1749.34 | 1741.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1731.60 | 1746.42 | 1742.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1731.60 | 1746.42 | 1742.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1736.80 | 1744.50 | 1741.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1786.30 | 1746.01 | 1743.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 1760.00 | 1777.89 | 1780.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 1760.00 | 1777.89 | 1780.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1746.20 | 1765.56 | 1773.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 14:15:00 | 1745.40 | 1745.14 | 1758.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:45:00 | 1748.20 | 1745.14 | 1758.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1761.80 | 1749.25 | 1757.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 1768.00 | 1749.25 | 1757.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1760.80 | 1751.56 | 1758.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 1760.80 | 1751.56 | 1758.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1764.30 | 1754.11 | 1758.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 1760.00 | 1754.11 | 1758.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1765.80 | 1756.45 | 1759.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 1765.80 | 1756.45 | 1759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1758.30 | 1759.24 | 1760.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1750.50 | 1759.24 | 1760.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1753.40 | 1758.21 | 1759.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 1770.00 | 1761.18 | 1760.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 1770.00 | 1761.18 | 1760.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 1791.90 | 1767.32 | 1763.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 1928.80 | 1930.33 | 1901.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 1928.80 | 1930.33 | 1901.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1906.10 | 1925.20 | 1906.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1907.50 | 1925.20 | 1906.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1906.10 | 1921.38 | 1906.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1907.00 | 1921.38 | 1906.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1897.00 | 1916.50 | 1905.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 1895.10 | 1916.50 | 1905.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1912.70 | 1915.74 | 1906.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 1917.00 | 1915.49 | 1906.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1922.30 | 1910.56 | 1905.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 1914.50 | 1910.65 | 1907.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 1913.10 | 1915.17 | 1911.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1917.00 | 1918.10 | 1914.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1913.70 | 1918.10 | 1914.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1904.00 | 1915.28 | 1913.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1904.00 | 1915.28 | 1913.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1904.90 | 1913.20 | 1912.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1888.60 | 1913.20 | 1912.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1886.10 | 1907.78 | 1910.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1886.10 | 1907.78 | 1910.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1863.40 | 1886.06 | 1897.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 1904.10 | 1883.75 | 1892.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1901.40 | 1887.28 | 1892.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:45:00 | 1893.10 | 1888.74 | 1893.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:15:00 | 1888.60 | 1888.74 | 1893.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1890.40 | 1887.89 | 1890.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 1914.30 | 1893.88 | 1891.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1914.30 | 1893.88 | 1891.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 1920.50 | 1899.21 | 1894.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 1953.50 | 1964.31 | 1951.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 1953.50 | 1964.31 | 1951.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1947.50 | 1960.95 | 1951.16 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1933.50 | 1946.32 | 1946.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 1910.30 | 1936.27 | 1941.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 1892.90 | 1889.77 | 1902.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:15:00 | 1896.80 | 1889.77 | 1902.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1887.20 | 1889.26 | 1900.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1876.10 | 1885.47 | 1889.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 1879.00 | 1886.07 | 1887.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 1924.10 | 1895.38 | 1891.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1924.10 | 1895.38 | 1891.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1931.60 | 1907.68 | 1898.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1911.80 | 1914.86 | 1905.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:00:00 | 1911.80 | 1914.86 | 1905.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1943.70 | 1954.14 | 1941.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1943.70 | 1954.14 | 1941.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1952.10 | 1953.26 | 1943.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 1947.40 | 1953.26 | 1943.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1976.90 | 1962.84 | 1951.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1993.30 | 1962.84 | 1951.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1943.30 | 1963.28 | 1964.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1943.30 | 1963.28 | 1964.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1913.00 | 1935.85 | 1945.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1881.00 | 1898.18 | 1909.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 1889.90 | 1886.90 | 1898.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 1898.00 | 1880.67 | 1880.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 1898.00 | 1880.67 | 1880.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 1923.40 | 1895.48 | 1888.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1881.00 | 1891.97 | 1893.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1869.50 | 1887.48 | 1891.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1888.00 | 1884.15 | 1888.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1888.00 | 1884.15 | 1888.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1890.60 | 1885.44 | 1888.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 1890.40 | 1885.44 | 1888.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1896.00 | 1887.55 | 1889.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1924.80 | 1887.55 | 1889.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1939.30 | 1897.90 | 1893.69 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1892.90 | 1907.63 | 1909.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1886.00 | 1903.31 | 1906.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1897.00 | 1892.82 | 1899.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1898.00 | 1893.86 | 1899.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1897.70 | 1893.86 | 1899.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1896.70 | 1894.43 | 1899.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1896.10 | 1894.43 | 1899.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1899.90 | 1895.52 | 1899.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 1899.90 | 1895.52 | 1899.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1896.60 | 1895.74 | 1899.25 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 1920.00 | 1904.18 | 1902.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1943.50 | 1912.05 | 1906.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1961.10 | 1961.84 | 1947.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 1964.00 | 1961.84 | 1947.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1959.00 | 1964.53 | 1956.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1959.00 | 1964.53 | 1956.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1969.10 | 1965.45 | 1958.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 1979.60 | 1967.48 | 1960.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 1975.00 | 1971.00 | 1963.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1962.10 | 1983.82 | 1985.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1962.10 | 1983.82 | 1985.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 1959.00 | 1978.85 | 1983.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1965.60 | 1958.31 | 1966.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1965.30 | 1959.71 | 1965.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1949.00 | 1960.73 | 1965.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 1958.70 | 1956.44 | 1960.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:45:00 | 1954.70 | 1959.27 | 1960.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1952.00 | 1924.33 | 1930.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1944.00 | 1928.26 | 1931.82 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1851.55 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1860.76 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1856.96 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1854.40 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1925.40 | 1906.17 | 1921.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 12:15:00 | 1829.13 | 1892.14 | 1913.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-07 09:15:00 | 1754.10 | 1822.21 | 1870.44 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 107 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1800.00 | 1786.80 | 1786.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 1805.00 | 1795.52 | 1791.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1790.50 | 1795.25 | 1791.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1787.70 | 1793.74 | 1791.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1795.30 | 1793.74 | 1791.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1785.90 | 1792.17 | 1790.95 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1788.86 | 1789.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1784.21 | 1787.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 1790.60 | 1779.42 | 1783.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1780.30 | 1779.60 | 1783.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 1785.90 | 1779.60 | 1783.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1779.00 | 1779.48 | 1783.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1785.10 | 1779.48 | 1783.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1777.70 | 1779.12 | 1782.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 1783.80 | 1779.12 | 1782.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1786.40 | 1780.90 | 1782.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1786.40 | 1780.90 | 1782.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1787.10 | 1782.14 | 1783.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1779.00 | 1782.14 | 1783.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 1780.00 | 1777.84 | 1779.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 1796.70 | 1780.58 | 1779.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 1796.70 | 1780.58 | 1779.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 1802.30 | 1788.45 | 1784.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1778.50 | 1790.82 | 1787.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1771.40 | 1786.93 | 1786.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 1769.00 | 1786.93 | 1786.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1779.70 | 1785.49 | 1785.50 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 1790.00 | 1783.52 | 1782.98 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 1765.30 | 1779.87 | 1781.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 1757.70 | 1775.44 | 1779.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1763.90 | 1761.65 | 1769.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1771.10 | 1763.54 | 1769.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1774.70 | 1763.54 | 1769.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1768.80 | 1764.59 | 1769.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 1757.50 | 1769.72 | 1770.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1762.00 | 1764.32 | 1766.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1765.00 | 1761.40 | 1763.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 1766.10 | 1762.34 | 1763.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1766.00 | 1763.50 | 1764.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1771.80 | 1763.50 | 1764.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1754.70 | 1761.77 | 1763.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1771.10 | 1763.54 | 1763.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 1771.10 | 1763.54 | 1763.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 1776.60 | 1767.35 | 1765.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 1758.80 | 1767.31 | 1765.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1755.30 | 1764.90 | 1764.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 1755.30 | 1764.90 | 1764.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 1754.90 | 1762.90 | 1763.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 1743.60 | 1759.04 | 1762.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 1759.50 | 1749.42 | 1755.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1755.90 | 1750.71 | 1755.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 1750.00 | 1750.71 | 1755.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1755.40 | 1751.65 | 1755.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1755.40 | 1751.65 | 1755.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1752.70 | 1751.86 | 1754.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1739.30 | 1751.86 | 1754.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 1749.80 | 1748.32 | 1751.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 1749.20 | 1749.40 | 1751.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1738.00 | 1750.52 | 1752.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1735.60 | 1747.53 | 1750.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 1722.60 | 1735.14 | 1741.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 1710.00 | 1723.66 | 1732.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1723.00 | 1722.02 | 1729.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1741.90 | 1734.61 | 1733.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1741.90 | 1734.61 | 1733.71 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 1726.00 | 1732.89 | 1733.01 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1739.80 | 1732.60 | 1732.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1745.50 | 1735.18 | 1733.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1799.50 | 1800.84 | 1785.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1806.70 | 1803.31 | 1787.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1813.20 | 1803.31 | 1787.69 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1805.00 | 1834.96 | 1827.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 1805.00 | 1834.96 | 1827.12 | SL hit (close<ema400) qty=1.00 sl=1827.12 alert=retest1 |

### Cycle 118 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 1773.00 | 1814.42 | 1818.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 1752.10 | 1769.10 | 1785.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 1768.60 | 1764.03 | 1777.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1768.60 | 1764.03 | 1777.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1779.10 | 1767.55 | 1776.52 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 1791.60 | 1777.16 | 1777.15 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1774.50 | 1776.62 | 1776.91 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 1781.60 | 1777.62 | 1777.34 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1773.00 | 1776.70 | 1776.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1763.30 | 1773.31 | 1775.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1722.10 | 1715.83 | 1731.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 1722.10 | 1715.83 | 1731.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1735.00 | 1719.67 | 1731.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1735.00 | 1719.67 | 1731.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1737.90 | 1723.31 | 1732.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 1729.50 | 1727.28 | 1732.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1748.70 | 1731.92 | 1734.02 | SL hit (close>static) qty=1.00 sl=1739.90 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1774.90 | 1740.52 | 1737.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 1782.90 | 1753.98 | 1744.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 1817.20 | 1835.26 | 1816.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1811.20 | 1830.45 | 1816.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1811.20 | 1830.45 | 1816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1825.40 | 1829.44 | 1816.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 1839.10 | 1821.19 | 1817.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1832.40 | 1824.44 | 1822.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 1790.80 | 1815.84 | 1818.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 1790.80 | 1815.84 | 1818.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 13:15:00 | 1787.20 | 1801.68 | 1809.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 1795.90 | 1786.06 | 1794.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1788.70 | 1786.59 | 1794.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1797.90 | 1786.59 | 1794.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1787.00 | 1786.67 | 1793.63 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1813.70 | 1799.69 | 1798.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1817.10 | 1806.24 | 1801.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 1797.80 | 1807.95 | 1804.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1805.80 | 1807.52 | 1804.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1810.00 | 1807.52 | 1804.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1800.00 | 1806.02 | 1804.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1814.30 | 1806.02 | 1804.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1809.90 | 1806.79 | 1804.61 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1793.80 | 1802.39 | 1802.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1776.80 | 1795.43 | 1799.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1733.20 | 1720.80 | 1740.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1735.20 | 1723.68 | 1739.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1721.20 | 1723.18 | 1738.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:00:00 | 1721.60 | 1722.87 | 1736.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 1722.00 | 1722.44 | 1729.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 1721.80 | 1692.69 | 1692.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1721.80 | 1692.69 | 1692.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1746.90 | 1703.54 | 1697.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 1793.90 | 1803.75 | 1775.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 13:30:00 | 1793.20 | 1803.75 | 1775.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1795.50 | 1798.82 | 1780.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 1803.50 | 1796.98 | 1782.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 1800.10 | 1796.98 | 1782.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:45:00 | 1802.90 | 1799.78 | 1785.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 09:15:00 | 1983.85 | 1927.09 | 1899.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1976.20 | 1990.97 | 1992.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1965.00 | 1983.06 | 1988.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 1987.00 | 1974.33 | 1978.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1968.70 | 1973.21 | 1977.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1959.80 | 1973.21 | 1977.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 1963.10 | 1970.12 | 1975.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 1963.10 | 1968.92 | 1974.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 1961.30 | 1967.39 | 1973.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1967.00 | 1957.09 | 1962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1955.40 | 1957.09 | 1962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1974.80 | 1960.63 | 1964.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1974.80 | 1960.63 | 1964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1965.40 | 1961.58 | 1964.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 1971.10 | 1961.58 | 1964.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1956.90 | 1960.31 | 1963.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 1957.00 | 1960.31 | 1963.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1959.50 | 1951.69 | 1956.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1959.50 | 1951.69 | 1956.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1950.20 | 1951.39 | 1956.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 1944.30 | 1952.34 | 1955.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 1945.80 | 1952.85 | 1955.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:00:00 | 1946.60 | 1951.60 | 1954.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 1941.30 | 1947.62 | 1951.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1864.94 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1864.94 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1863.23 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1861.81 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1847.08 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1848.51 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1849.27 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1844.23 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1889.10 | 1871.23 | 1891.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 1889.10 | 1871.23 | 1891.27 | SL hit (close>ema200) qty=0.50 sl=1871.23 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 1942.90 | 1904.79 | 1902.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1962.90 | 1924.03 | 1911.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 1916.00 | 1922.42 | 1912.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:45:00 | 1918.60 | 1922.42 | 1912.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1914.10 | 1920.76 | 1912.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1914.10 | 1920.76 | 1912.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1937.00 | 1924.01 | 1914.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 1943.80 | 1924.01 | 1914.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 10:15:00 | 1897.90 | 1927.08 | 1921.51 | SL hit (close<static) qty=1.00 sl=1908.70 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1866.00 | 1914.87 | 1916.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 1862.80 | 1898.08 | 1908.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1971.90 | 1912.01 | 1908.51 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1880.90 | 1919.28 | 1924.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1835.60 | 1893.73 | 1911.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 1802.90 | 1799.27 | 1827.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 1802.90 | 1799.27 | 1827.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1822.80 | 1807.54 | 1824.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1841.60 | 1807.54 | 1824.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1817.90 | 1809.61 | 1824.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:45:00 | 1811.60 | 1810.92 | 1822.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 1815.70 | 1812.45 | 1821.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1771.70 | 1815.62 | 1821.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 1721.02 | 1747.58 | 1774.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 1724.91 | 1747.58 | 1774.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 1630.44 | 1708.90 | 1742.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1745.80 | 1682.62 | 1681.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1753.20 | 1713.58 | 1697.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 1701.70 | 1719.93 | 1705.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1696.40 | 1715.23 | 1704.42 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1673.00 | 1695.57 | 1698.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1633.80 | 1683.22 | 1692.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 1633.60 | 1623.53 | 1649.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 13:15:00 | 1551.92 | 1588.93 | 1626.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-02 09:15:00 | 1470.24 | 1553.29 | 1599.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1611.70 | 1551.34 | 1551.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 1643.00 | 1612.42 | 1589.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 1894.20 | 1895.24 | 1869.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 12:15:00 | 1888.80 | 1895.24 | 1869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1865.60 | 1886.62 | 1870.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1865.60 | 1886.62 | 1870.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1867.50 | 1882.80 | 1869.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1864.60 | 1882.80 | 1869.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1872.60 | 1879.51 | 1870.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 1884.20 | 1879.51 | 1870.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1870.60 | 1877.73 | 1870.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 1891.80 | 1881.40 | 1874.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:30:00 | 1894.00 | 1882.08 | 1875.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 15:15:00 | 1895.00 | 1882.08 | 1875.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1854.40 | 1878.61 | 1874.83 | SL hit (close<static) qty=1.00 sl=1870.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1850.00 | 1870.49 | 1871.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 1830.80 | 1859.29 | 1866.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1826.90 | 1825.51 | 1842.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 1826.90 | 1825.51 | 1842.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1873.50 | 1832.98 | 1841.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1892.30 | 1832.98 | 1841.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1890.00 | 1844.38 | 1845.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1891.00 | 1844.38 | 1845.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1901.70 | 1855.84 | 1850.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1917.90 | 1883.44 | 1866.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1891.00 | 1891.38 | 1874.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1891.00 | 1891.38 | 1874.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1886.00 | 1890.45 | 1880.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1884.00 | 1890.45 | 1880.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1868.80 | 1886.12 | 1879.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 1868.80 | 1886.12 | 1879.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1863.70 | 1881.64 | 1878.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 1861.10 | 1881.64 | 1878.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1859.40 | 1872.66 | 1874.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1841.00 | 1863.40 | 1869.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1803.10 | 1796.35 | 1821.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 1790.60 | 1795.30 | 1816.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1790.60 | 1795.30 | 1816.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 1802.20 | 1795.30 | 1816.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1804.80 | 1800.74 | 1811.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1780.60 | 1799.69 | 1806.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1780.90 | 1792.85 | 1800.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1785.00 | 1798.34 | 1801.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 1785.20 | 1788.57 | 1795.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1752.40 | 1781.33 | 1791.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 1747.00 | 1776.07 | 1787.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1691.57 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1691.86 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1695.75 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1695.94 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 11:30:00 | 1487.05 | 2024-05-28 14:15:00 | 1516.70 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-06-04 13:30:00 | 1585.10 | 2024-06-05 11:15:00 | 1534.30 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-06-06 13:00:00 | 1524.40 | 2024-06-07 11:15:00 | 1567.55 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-06-06 14:00:00 | 1526.05 | 2024-06-07 11:15:00 | 1567.55 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-06-13 12:15:00 | 1673.00 | 2024-06-19 15:15:00 | 1700.00 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2024-06-24 12:00:00 | 1708.30 | 2024-06-24 12:15:00 | 1709.85 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-07-01 13:30:00 | 1645.00 | 2024-07-05 09:15:00 | 1657.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-09 11:45:00 | 1704.00 | 2024-07-09 15:15:00 | 1874.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-16 11:15:00 | 1694.00 | 2024-07-16 12:15:00 | 1709.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1671.85 | 2024-07-23 12:15:00 | 1588.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1671.85 | 2024-07-23 13:15:00 | 1646.05 | STOP_HIT | 0.50 | 1.54% |
| BUY | retest2 | 2024-07-26 10:30:00 | 1682.95 | 2024-08-02 13:15:00 | 1719.00 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1589.90 | 2024-08-13 09:15:00 | 1702.95 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2024-08-12 11:15:00 | 1586.45 | 2024-08-13 09:15:00 | 1702.95 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest2 | 2024-08-19 10:00:00 | 1721.05 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-08-19 11:00:00 | 1722.25 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-20 09:45:00 | 1721.80 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-20 10:30:00 | 1726.00 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-09-12 09:30:00 | 1800.65 | 2024-09-16 09:15:00 | 1980.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 09:15:00 | 2038.30 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-10-14 10:00:00 | 2024.85 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-10-14 15:00:00 | 2024.00 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-15 09:30:00 | 2026.00 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1835.65 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-28 09:30:00 | 1829.40 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-10-28 10:00:00 | 1829.80 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-28 14:15:00 | 1832.00 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-10-30 10:00:00 | 1836.05 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-10-31 10:00:00 | 1839.60 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1838.00 | 2024-11-04 12:15:00 | 1823.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-11-08 12:30:00 | 1809.95 | 2024-11-14 11:15:00 | 1814.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-11-08 13:30:00 | 1806.75 | 2024-11-14 11:15:00 | 1814.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-09 12:00:00 | 2086.70 | 2024-12-12 11:15:00 | 2090.25 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest1 | 2024-12-09 14:15:00 | 2092.00 | 2024-12-12 11:15:00 | 2090.25 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-12-11 10:30:00 | 2118.00 | 2024-12-12 14:15:00 | 2048.70 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-20 13:00:00 | 2024.15 | 2024-12-26 14:15:00 | 2033.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-01-01 10:15:00 | 2168.00 | 2025-01-01 14:15:00 | 2384.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1850.00 | 2025-01-23 10:15:00 | 1922.70 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-02-01 09:15:00 | 1866.10 | 2025-02-01 14:15:00 | 2052.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-03 10:15:00 | 1947.80 | 2025-03-04 09:15:00 | 2142.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-17 15:00:00 | 2010.80 | 2025-03-18 09:15:00 | 2099.55 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-03-24 09:15:00 | 2219.65 | 2025-03-26 10:15:00 | 2191.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-04-04 09:30:00 | 2066.20 | 2025-04-07 09:15:00 | 1859.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 13:00:00 | 2030.60 | 2025-04-23 10:15:00 | 1949.90 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1623.40 | 2025-05-09 09:15:00 | 1542.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1637.70 | 2025-05-09 09:15:00 | 1555.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1623.40 | 2025-05-09 15:15:00 | 1600.00 | STOP_HIT | 0.50 | 1.44% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1637.70 | 2025-05-09 15:15:00 | 1600.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-05-12 10:00:00 | 1631.20 | 2025-05-16 09:15:00 | 1549.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 10:00:00 | 1631.20 | 2025-05-16 12:15:00 | 1569.80 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-05-12 10:30:00 | 1626.80 | 2025-05-19 09:15:00 | 1618.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-05-13 10:15:00 | 1618.00 | 2025-05-19 09:15:00 | 1618.90 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-05-20 11:15:00 | 1618.40 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-05-20 14:30:00 | 1620.50 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1630.50 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-05-23 13:15:00 | 1564.20 | 2025-05-29 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-05 13:30:00 | 1556.00 | 2025-06-18 12:15:00 | 1628.20 | STOP_HIT | 1.00 | 4.64% |
| SELL | retest2 | 2025-06-23 13:00:00 | 1605.10 | 2025-06-24 10:15:00 | 1634.70 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-08 13:45:00 | 1812.20 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-08 14:15:00 | 1813.90 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-08 15:15:00 | 1814.90 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-09 11:30:00 | 1813.00 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-24 15:15:00 | 1765.00 | 2025-07-25 09:15:00 | 1782.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-25 12:15:00 | 1765.00 | 2025-07-30 14:15:00 | 1755.30 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-08-06 15:15:00 | 1786.30 | 2025-08-11 12:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-13 15:15:00 | 1750.50 | 2025-08-14 11:15:00 | 1770.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-14 09:45:00 | 1753.40 | 2025-08-14 11:15:00 | 1770.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-21 13:30:00 | 1917.00 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1922.30 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-22 13:00:00 | 1914.50 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-25 10:00:00 | 1913.10 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-28 13:45:00 | 1893.10 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-28 14:15:00 | 1888.60 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-08-29 12:15:00 | 1890.40 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-12 10:45:00 | 1876.10 | 2025-09-16 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-09-15 12:30:00 | 1879.00 | 2025-09-16 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-22 10:15:00 | 1993.30 | 2025-09-24 09:15:00 | 1943.30 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1881.00 | 2025-10-03 15:15:00 | 1898.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-30 14:45:00 | 1889.90 | 2025-10-03 15:15:00 | 1898.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-23 12:15:00 | 1979.60 | 2025-10-28 09:15:00 | 1962.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-23 15:15:00 | 1975.00 | 2025-10-28 09:15:00 | 1962.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1949.00 | 2025-11-06 11:15:00 | 1851.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1958.70 | 2025-11-06 11:15:00 | 1860.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:45:00 | 1954.70 | 2025-11-06 11:15:00 | 1856.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1952.00 | 2025-11-06 11:15:00 | 1854.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:30:00 | 1925.40 | 2025-11-06 12:15:00 | 1829.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1949.00 | 2025-11-07 09:15:00 | 1754.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1958.70 | 2025-11-07 09:15:00 | 1762.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 10:45:00 | 1954.70 | 2025-11-07 09:15:00 | 1759.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1952.00 | 2025-11-07 09:15:00 | 1756.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 11:30:00 | 1925.40 | 2025-11-07 09:15:00 | 1732.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1779.00 | 2025-11-19 14:15:00 | 1796.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-18 13:45:00 | 1780.00 | 2025-11-19 14:15:00 | 1796.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-27 09:15:00 | 1757.50 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-27 14:15:00 | 1762.00 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-28 13:00:00 | 1765.00 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-28 13:45:00 | 1766.10 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-12-04 09:15:00 | 1739.30 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-04 13:30:00 | 1749.80 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-04 14:45:00 | 1749.20 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1738.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-12-08 10:15:00 | 1722.60 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-08 15:15:00 | 1710.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-09 10:45:00 | 1723.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-12-16 10:30:00 | 1806.70 | 2025-12-19 10:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest1 | 2025-12-16 11:00:00 | 1813.20 | 2025-12-19 10:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-31 15:15:00 | 1729.50 | 2026-01-01 09:15:00 | 1748.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-01-07 14:30:00 | 1839.10 | 2026-01-09 13:15:00 | 1790.80 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1832.40 | 2026-01-09 13:15:00 | 1790.80 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1721.20 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-01-22 13:00:00 | 1721.60 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-23 12:00:00 | 1722.00 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-02-02 11:30:00 | 1803.50 | 2026-02-10 09:15:00 | 1983.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 12:00:00 | 1800.10 | 2026-02-10 09:15:00 | 1980.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 12:45:00 | 1802.90 | 2026-02-10 09:15:00 | 1983.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1959.80 | 2026-03-02 09:15:00 | 1864.94 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2026-02-23 13:15:00 | 1963.10 | 2026-03-02 09:15:00 | 1864.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1963.10 | 2026-03-02 09:15:00 | 1863.23 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1961.30 | 2026-03-04 09:15:00 | 1861.81 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-27 09:45:00 | 1944.30 | 2026-03-04 09:15:00 | 1847.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:30:00 | 1945.80 | 2026-03-04 09:15:00 | 1848.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:00:00 | 1946.60 | 2026-03-04 09:15:00 | 1849.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1941.30 | 2026-03-04 09:15:00 | 1844.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1959.80 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-02-23 13:15:00 | 1963.10 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1963.10 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1961.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2026-02-27 09:45:00 | 1944.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-02-27 10:30:00 | 1945.80 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-02-27 12:00:00 | 1946.60 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1941.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2026-03-06 13:15:00 | 1943.80 | 2026-03-09 10:15:00 | 1897.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1811.60 | 2026-03-20 10:15:00 | 1721.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 1815.70 | 2026-03-20 10:15:00 | 1724.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1811.60 | 2026-03-23 09:15:00 | 1630.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 1815.70 | 2026-03-23 09:15:00 | 1634.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1771.70 | 2026-03-23 09:15:00 | 1683.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1771.70 | 2026-03-24 10:15:00 | 1638.60 | STOP_HIT | 0.50 | 7.51% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1633.60 | 2026-04-01 13:15:00 | 1551.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1633.60 | 2026-04-02 09:15:00 | 1470.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-22 14:00:00 | 1891.80 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-22 14:30:00 | 1894.00 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-04-22 15:15:00 | 1895.00 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-05-06 09:15:00 | 1780.60 | 2026-05-08 09:15:00 | 1691.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-06 12:30:00 | 1780.90 | 2026-05-08 09:15:00 | 1691.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-07 09:15:00 | 1785.00 | 2026-05-08 09:15:00 | 1695.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-07 12:15:00 | 1785.20 | 2026-05-08 09:15:00 | 1695.94 | PARTIAL | 0.50 | 5.00% |
