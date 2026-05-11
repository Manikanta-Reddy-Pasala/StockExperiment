# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1020.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 95 |
| ALERT2 | 94 |
| ALERT2_SKIP | 55 |
| ALERT3 | 276 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 152 |
| PARTIAL | 23 |
| TARGET_HIT | 10 |
| STOP_HIT | 146 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 103
- **Target hits / Stop hits / Partials:** 10 / 145 / 23
- **Avg / median % per leg:** 1.08% / -0.52%
- **Sum % (uncompounded):** 192.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 24 | 39.3% | 10 | 50 | 1 | 1.27% | 77.8% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.00% | 6.0% |
| BUY @ 3rd Alert (retest2) | 58 | 22 | 37.9% | 10 | 48 | 0 | 1.24% | 71.8% |
| SELL (all) | 117 | 51 | 43.6% | 0 | 95 | 22 | 0.98% | 114.8% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.74% | 11.5% |
| SELL @ 3rd Alert (retest2) | 115 | 49 | 42.6% | 0 | 94 | 21 | 0.90% | 103.3% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 3.49% | 17.5% |
| retest2 (combined) | 173 | 71 | 41.0% | 10 | 142 | 21 | 1.01% | 175.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1495.05 | 1480.07 | 1479.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1506.30 | 1485.31 | 1481.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 1484.50 | 1489.65 | 1485.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1483.95 | 1488.51 | 1485.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 1479.30 | 1488.51 | 1485.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 1477.10 | 1484.77 | 1483.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:45:00 | 1478.15 | 1484.77 | 1483.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1485.00 | 1484.37 | 1483.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:30:00 | 1483.00 | 1484.37 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1689.00 | 1714.11 | 1688.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1664.00 | 1714.11 | 1688.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1650.00 | 1701.29 | 1684.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1661.00 | 1701.29 | 1684.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1641.45 | 1689.32 | 1680.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 1641.45 | 1689.32 | 1680.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 1633.55 | 1669.06 | 1672.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 1713.60 | 1677.97 | 1676.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 1734.85 | 1689.34 | 1681.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 1696.20 | 1702.74 | 1691.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1705.00 | 1703.19 | 1692.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 1699.00 | 1703.19 | 1692.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1700.00 | 1701.07 | 1693.14 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 1668.15 | 1689.46 | 1691.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1654.95 | 1682.56 | 1687.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:45:00 | 1642.95 | 1605.89 | 1617.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1624.90 | 1609.69 | 1618.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:30:00 | 1633.75 | 1609.69 | 1618.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 1608.90 | 1609.53 | 1617.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 1599.15 | 1607.45 | 1615.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:00:00 | 1598.85 | 1602.05 | 1609.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:30:00 | 1600.20 | 1601.37 | 1606.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 1600.20 | 1601.37 | 1606.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1660.15 | 1612.91 | 1610.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1660.15 | 1612.91 | 1610.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 1670.00 | 1650.34 | 1633.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1616.50 | 1648.21 | 1634.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1593.25 | 1637.22 | 1630.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1593.25 | 1637.22 | 1630.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1544.40 | 1618.66 | 1622.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 1516.95 | 1582.24 | 1603.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 1562.10 | 1549.73 | 1575.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 1556.20 | 1549.73 | 1575.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1559.05 | 1551.90 | 1572.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 1559.50 | 1551.90 | 1572.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1567.55 | 1555.03 | 1571.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1575.00 | 1555.03 | 1571.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1583.00 | 1560.63 | 1572.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1583.00 | 1560.63 | 1572.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1583.05 | 1565.11 | 1573.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 1586.55 | 1565.11 | 1573.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 1575.45 | 1567.18 | 1574.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 1572.85 | 1568.31 | 1573.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 1573.00 | 1569.33 | 1573.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1571.00 | 1572.03 | 1574.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 1590.30 | 1575.68 | 1575.83 | SL hit (close>static) qty=1.00 sl=1584.45 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 1588.55 | 1576.95 | 1576.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 1624.45 | 1586.45 | 1580.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 1639.00 | 1642.14 | 1625.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1708.95 | 1642.14 | 1625.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:15:00 | 1794.40 | 1757.34 | 1733.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 1750.05 | 1755.88 | 1734.80 | SL hit (close<ema200) qty=0.50 sl=1755.88 alert=retest1 |

### Cycle 8 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 1724.65 | 1763.96 | 1767.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 1718.40 | 1748.18 | 1759.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1685.05 | 1698.75 | 1708.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1699.65 | 1698.93 | 1707.84 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1737.00 | 1697.91 | 1694.54 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 1688.90 | 1701.04 | 1702.17 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 1758.65 | 1709.43 | 1705.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 11:15:00 | 1776.75 | 1732.22 | 1716.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1710.00 | 1728.96 | 1718.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1701.90 | 1723.55 | 1716.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 1701.90 | 1723.55 | 1716.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1706.35 | 1718.67 | 1715.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 1706.35 | 1718.67 | 1715.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1700.35 | 1715.01 | 1714.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 1698.50 | 1715.01 | 1714.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 1700.40 | 1712.09 | 1713.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 1694.30 | 1708.53 | 1711.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 1687.90 | 1682.08 | 1691.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1699.35 | 1685.54 | 1691.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1699.35 | 1685.54 | 1691.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1692.40 | 1686.91 | 1691.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:00:00 | 1682.95 | 1686.12 | 1691.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:00:00 | 1685.15 | 1685.92 | 1690.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 1702.20 | 1691.00 | 1691.86 | SL hit (close>static) qty=1.00 sl=1700.95 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 1700.00 | 1693.84 | 1693.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 1740.95 | 1703.28 | 1697.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 15:15:00 | 1726.50 | 1737.89 | 1725.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1715.00 | 1733.31 | 1724.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1715.00 | 1733.31 | 1724.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1715.30 | 1733.31 | 1724.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1706.60 | 1727.97 | 1723.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 1735.50 | 1727.97 | 1723.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 1720.00 | 1724.68 | 1724.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 1720.00 | 1723.75 | 1723.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1720.00 | 1723.75 | 1723.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 1691.00 | 1715.60 | 1719.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1694.15 | 1694.06 | 1703.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:30:00 | 1699.35 | 1694.06 | 1703.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1701.00 | 1683.37 | 1692.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1701.00 | 1683.37 | 1692.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1702.95 | 1687.29 | 1693.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1702.95 | 1687.29 | 1693.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 1706.25 | 1691.08 | 1694.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:00:00 | 1706.25 | 1691.08 | 1694.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 1706.25 | 1694.11 | 1695.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 1706.00 | 1694.11 | 1695.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 1717.25 | 1698.74 | 1697.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 1727.10 | 1713.11 | 1707.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1702.00 | 1712.44 | 1707.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1701.45 | 1710.24 | 1706.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:00:00 | 1725.10 | 1713.21 | 1708.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 1719.50 | 1733.25 | 1734.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 1719.50 | 1733.25 | 1734.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 1709.80 | 1722.55 | 1727.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1627.10 | 1619.68 | 1650.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 1631.05 | 1624.86 | 1641.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 1631.05 | 1624.86 | 1641.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:15:00 | 1620.10 | 1624.86 | 1641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1620.10 | 1623.91 | 1639.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:30:00 | 1603.35 | 1619.09 | 1634.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:30:00 | 1608.80 | 1615.10 | 1628.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:45:00 | 1604.25 | 1612.06 | 1625.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 1528.36 | 1564.90 | 1588.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 14:15:00 | 1523.18 | 1539.62 | 1565.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 14:15:00 | 1524.04 | 1539.62 | 1565.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 1534.45 | 1531.36 | 1548.93 | SL hit (close>ema200) qty=0.50 sl=1531.36 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 1557.60 | 1550.48 | 1550.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1607.80 | 1561.70 | 1555.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 1569.90 | 1577.97 | 1568.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1550.65 | 1572.51 | 1567.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 1540.05 | 1572.51 | 1567.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1554.15 | 1568.84 | 1566.09 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1550.35 | 1562.60 | 1563.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 13:15:00 | 1545.00 | 1555.52 | 1559.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1571.80 | 1553.60 | 1557.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1569.95 | 1556.87 | 1558.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:15:00 | 1578.20 | 1556.87 | 1558.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 1574.00 | 1560.29 | 1559.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 1587.80 | 1568.84 | 1563.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 1570.05 | 1579.07 | 1572.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1571.80 | 1577.62 | 1572.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 1560.00 | 1577.62 | 1572.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1576.95 | 1577.48 | 1572.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 1570.00 | 1577.48 | 1572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1578.00 | 1577.59 | 1573.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1577.00 | 1577.59 | 1573.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1573.15 | 1576.70 | 1573.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 1570.15 | 1576.70 | 1573.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1568.25 | 1575.01 | 1572.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 1565.30 | 1575.01 | 1572.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1564.00 | 1572.81 | 1571.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 1565.55 | 1572.81 | 1571.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1575.60 | 1576.81 | 1574.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 1575.60 | 1576.81 | 1574.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1556.80 | 1572.80 | 1572.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1550.70 | 1572.80 | 1572.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 1549.00 | 1568.04 | 1570.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 1533.50 | 1554.35 | 1563.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 1555.90 | 1548.40 | 1556.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1550.00 | 1548.72 | 1556.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 1562.15 | 1548.72 | 1556.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1549.95 | 1549.58 | 1555.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 1550.00 | 1549.58 | 1555.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1538.85 | 1544.87 | 1551.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:00:00 | 1529.90 | 1541.87 | 1549.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 1524.00 | 1534.38 | 1543.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 1532.30 | 1519.33 | 1518.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1532.30 | 1519.33 | 1518.08 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1511.60 | 1525.41 | 1525.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1481.45 | 1514.79 | 1520.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 1495.75 | 1494.78 | 1506.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 1495.75 | 1494.78 | 1506.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1511.90 | 1497.91 | 1506.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 1496.15 | 1500.86 | 1506.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 1525.80 | 1508.64 | 1508.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1525.80 | 1508.64 | 1508.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 1530.35 | 1512.98 | 1510.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 1517.55 | 1526.28 | 1520.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1524.00 | 1525.83 | 1520.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 1529.00 | 1525.83 | 1520.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 1526.80 | 1526.34 | 1521.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 1505.85 | 1520.02 | 1520.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 1505.85 | 1520.02 | 1520.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 15:15:00 | 1500.00 | 1510.40 | 1515.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 12:15:00 | 1519.40 | 1508.56 | 1507.41 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1499.10 | 1509.06 | 1510.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1488.00 | 1503.36 | 1507.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 1508.00 | 1494.97 | 1499.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1504.10 | 1496.80 | 1500.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:15:00 | 1504.10 | 1496.80 | 1500.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1499.40 | 1497.32 | 1500.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 1494.40 | 1497.93 | 1499.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 1495.00 | 1498.33 | 1499.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:00:00 | 1495.20 | 1497.17 | 1499.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 12:15:00 | 1504.40 | 1500.68 | 1500.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1504.40 | 1500.68 | 1500.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1515.00 | 1505.02 | 1502.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1507.85 | 1508.45 | 1505.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1510.25 | 1513.71 | 1509.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 1510.00 | 1513.71 | 1509.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1509.90 | 1512.94 | 1509.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 1511.40 | 1512.94 | 1509.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1510.05 | 1512.37 | 1509.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:15:00 | 1504.90 | 1512.37 | 1509.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1504.90 | 1510.87 | 1509.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1508.75 | 1510.87 | 1509.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1506.45 | 1509.99 | 1508.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 1506.45 | 1509.99 | 1508.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1515.00 | 1510.99 | 1509.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 1519.90 | 1510.79 | 1509.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 1520.00 | 1517.98 | 1513.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 1520.20 | 1518.86 | 1514.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 1522.90 | 1519.42 | 1515.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1500.00 | 1516.66 | 1515.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1500.00 | 1516.66 | 1515.56 | SL hit (close<static) qty=1.00 sl=1505.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1494.95 | 1511.42 | 1513.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 1480.50 | 1505.23 | 1510.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:15:00 | 1493.45 | 1496.29 | 1502.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1418.78 | 1452.10 | 1463.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 1431.45 | 1423.19 | 1438.00 | SL hit (close>ema200) qty=0.50 sl=1423.19 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1454.85 | 1439.81 | 1438.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1463.55 | 1447.90 | 1442.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 1442.40 | 1455.83 | 1451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1455.95 | 1455.85 | 1451.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:00:00 | 1465.20 | 1457.03 | 1452.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:45:00 | 1463.85 | 1461.03 | 1455.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 1460.40 | 1462.39 | 1457.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 1464.50 | 1485.61 | 1487.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1464.50 | 1485.61 | 1487.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1446.85 | 1464.26 | 1471.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 10:15:00 | 1412.70 | 1409.10 | 1421.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 13:15:00 | 1418.70 | 1410.82 | 1419.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1418.70 | 1410.82 | 1419.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 1417.55 | 1410.82 | 1419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1422.10 | 1413.08 | 1419.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 1429.70 | 1413.08 | 1419.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1419.95 | 1414.45 | 1419.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 1404.10 | 1414.45 | 1419.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 1397.10 | 1381.98 | 1380.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1397.10 | 1381.98 | 1380.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1400.60 | 1385.70 | 1382.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 1391.00 | 1393.85 | 1388.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 1391.00 | 1393.85 | 1388.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1383.00 | 1391.68 | 1387.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1383.00 | 1391.68 | 1387.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1385.50 | 1390.45 | 1387.53 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 1377.00 | 1384.86 | 1385.47 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 1394.00 | 1386.56 | 1386.13 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1362.95 | 1381.84 | 1384.02 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 1399.80 | 1387.03 | 1385.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 14:15:00 | 1411.00 | 1391.82 | 1387.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 12:15:00 | 1392.05 | 1395.22 | 1391.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:00:00 | 1392.05 | 1395.22 | 1391.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1395.60 | 1395.29 | 1391.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1419.95 | 1395.32 | 1392.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:15:00 | 1423.85 | 1398.11 | 1394.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:30:00 | 1406.90 | 1424.24 | 1422.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 1410.50 | 1420.48 | 1421.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1410.50 | 1420.48 | 1421.20 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1426.00 | 1419.76 | 1419.59 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 1415.00 | 1419.77 | 1419.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 1404.00 | 1416.62 | 1418.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 1415.50 | 1412.94 | 1416.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1419.40 | 1414.23 | 1416.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:00:00 | 1419.40 | 1414.23 | 1416.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 1419.90 | 1415.37 | 1416.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:15:00 | 1419.20 | 1415.37 | 1416.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 1420.05 | 1416.30 | 1417.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:30:00 | 1411.00 | 1416.58 | 1417.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 15:15:00 | 1424.45 | 1418.16 | 1417.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 15:15:00 | 1424.45 | 1418.16 | 1417.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 1450.00 | 1424.52 | 1420.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 12:15:00 | 1464.05 | 1465.85 | 1450.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 1464.05 | 1465.85 | 1450.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1449.40 | 1462.56 | 1450.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:30:00 | 1450.75 | 1462.56 | 1450.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1449.55 | 1459.96 | 1450.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 1439.95 | 1459.96 | 1450.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1451.00 | 1458.17 | 1450.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 1442.15 | 1458.17 | 1450.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1429.90 | 1452.51 | 1448.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1429.90 | 1452.51 | 1448.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 10:15:00 | 1412.70 | 1444.55 | 1445.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 11:15:00 | 1407.20 | 1437.08 | 1441.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 1433.35 | 1421.04 | 1430.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1467.90 | 1430.41 | 1433.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 1467.90 | 1430.41 | 1433.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 1433.55 | 1431.04 | 1433.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1424.40 | 1431.04 | 1433.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 15:00:00 | 1410.30 | 1400.85 | 1412.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:30:00 | 1425.65 | 1409.92 | 1411.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 1417.30 | 1412.88 | 1412.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 13:15:00 | 1417.30 | 1412.88 | 1412.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 1422.00 | 1414.51 | 1413.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 1412.80 | 1414.17 | 1413.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1415.45 | 1414.42 | 1413.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:30:00 | 1411.55 | 1414.42 | 1413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1414.60 | 1414.46 | 1413.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 1414.60 | 1414.46 | 1413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1424.45 | 1426.35 | 1422.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1424.45 | 1426.35 | 1422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1422.65 | 1425.61 | 1422.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1422.65 | 1425.61 | 1422.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1417.50 | 1423.99 | 1421.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1427.70 | 1425.11 | 1422.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 1421.40 | 1434.48 | 1434.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 1421.40 | 1434.48 | 1434.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 13:15:00 | 1413.45 | 1430.27 | 1432.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 12:15:00 | 1375.55 | 1369.54 | 1380.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:45:00 | 1373.80 | 1369.54 | 1380.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1368.05 | 1368.23 | 1376.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 1360.00 | 1368.23 | 1376.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 1365.05 | 1364.70 | 1372.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 1354.65 | 1364.70 | 1372.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1378.45 | 1359.21 | 1366.58 | SL hit (close>static) qty=1.00 sl=1376.30 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 1398.00 | 1372.28 | 1371.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 1407.60 | 1379.34 | 1374.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1371.00 | 1379.32 | 1376.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1370.05 | 1377.47 | 1375.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 1369.00 | 1377.47 | 1375.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 1363.20 | 1374.61 | 1374.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 1360.10 | 1367.94 | 1371.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1310.90 | 1321.86 | 1327.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:45:00 | 1313.05 | 1321.00 | 1327.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 1355.00 | 1328.38 | 1327.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 1355.00 | 1328.38 | 1327.47 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1309.90 | 1324.95 | 1326.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1299.75 | 1319.91 | 1323.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1295.70 | 1288.21 | 1300.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1293.30 | 1287.05 | 1294.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 1293.30 | 1287.05 | 1294.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1289.00 | 1287.44 | 1294.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1276.50 | 1284.61 | 1292.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 1279.75 | 1272.75 | 1280.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1279.10 | 1281.39 | 1281.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 1278.40 | 1270.50 | 1275.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1280.60 | 1272.52 | 1275.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 11:15:00 | 1275.60 | 1272.52 | 1275.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 1275.55 | 1273.62 | 1275.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 1275.00 | 1273.62 | 1275.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:30:00 | 1275.35 | 1275.06 | 1275.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 1285.05 | 1277.06 | 1276.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 1285.05 | 1277.06 | 1276.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 1298.55 | 1281.36 | 1278.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 1282.40 | 1291.09 | 1286.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1280.00 | 1288.87 | 1285.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:45:00 | 1279.05 | 1288.87 | 1285.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1300.45 | 1304.04 | 1298.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1300.45 | 1304.04 | 1298.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1301.65 | 1303.56 | 1298.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1290.00 | 1303.56 | 1298.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1292.80 | 1301.41 | 1298.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 1292.55 | 1301.41 | 1298.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1268.05 | 1294.74 | 1295.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1265.55 | 1288.90 | 1292.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1274.00 | 1268.61 | 1279.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1274.00 | 1268.61 | 1279.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1276.65 | 1270.22 | 1279.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1276.65 | 1270.22 | 1279.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1277.90 | 1271.76 | 1279.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1278.75 | 1271.76 | 1279.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1283.00 | 1274.01 | 1279.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 1282.50 | 1274.01 | 1279.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1275.40 | 1274.28 | 1279.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 1275.00 | 1274.28 | 1279.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1268.00 | 1255.23 | 1263.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1268.00 | 1255.23 | 1263.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1269.40 | 1258.07 | 1263.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 1262.00 | 1262.59 | 1264.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 1256.35 | 1237.53 | 1237.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1256.35 | 1237.53 | 1237.39 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 1229.10 | 1237.07 | 1237.58 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1244.25 | 1238.37 | 1238.07 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 1227.60 | 1238.11 | 1239.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1221.60 | 1234.81 | 1237.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 1232.35 | 1226.31 | 1231.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1231.50 | 1227.35 | 1231.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 1233.30 | 1227.35 | 1231.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1236.40 | 1229.16 | 1232.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:45:00 | 1234.55 | 1229.16 | 1232.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1235.20 | 1230.37 | 1232.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 1236.05 | 1230.37 | 1232.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1234.95 | 1231.28 | 1232.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 1221.20 | 1231.28 | 1232.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:45:00 | 1225.45 | 1227.23 | 1230.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1160.14 | 1182.92 | 1194.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1164.18 | 1182.92 | 1194.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 1146.10 | 1136.30 | 1153.70 | SL hit (close>ema200) qty=0.50 sl=1136.30 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1170.00 | 1156.76 | 1155.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1176.75 | 1168.08 | 1163.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 13:15:00 | 1178.55 | 1178.56 | 1173.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1186.40 | 1180.13 | 1174.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1169.70 | 1178.17 | 1174.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1169.70 | 1178.17 | 1174.61 | SL hit (close<ema400) qty=1.00 sl=1174.61 alert=retest1 |

### Cycle 54 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1158.10 | 1169.93 | 1171.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 1143.60 | 1155.77 | 1163.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 12:15:00 | 1110.05 | 1109.85 | 1121.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:45:00 | 1110.00 | 1109.85 | 1121.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1110.00 | 1109.91 | 1119.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1110.00 | 1109.91 | 1119.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1095.90 | 1107.44 | 1116.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1086.30 | 1106.11 | 1111.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1031.98 | 1059.31 | 1080.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 1032.00 | 1020.71 | 1033.30 | SL hit (close>ema200) qty=0.50 sl=1020.71 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 891.00 | 846.37 | 843.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 898.70 | 856.84 | 848.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 920.80 | 921.17 | 911.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 920.80 | 921.17 | 911.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 918.50 | 921.25 | 912.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 919.20 | 921.25 | 912.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 960.20 | 968.72 | 960.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 954.20 | 968.72 | 960.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 965.15 | 968.00 | 960.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 976.65 | 965.72 | 962.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 12:45:00 | 972.25 | 967.18 | 963.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 972.70 | 968.06 | 964.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 972.25 | 968.06 | 964.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 977.70 | 981.46 | 977.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 977.70 | 981.46 | 977.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 971.55 | 979.48 | 976.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 971.55 | 979.48 | 976.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 969.85 | 977.55 | 975.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 14:30:00 | 976.80 | 977.24 | 975.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 977.05 | 975.41 | 975.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 1074.32 | 1023.49 | 1008.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 996.40 | 1006.77 | 1006.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 985.25 | 1000.58 | 1003.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1024.85 | 983.85 | 987.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1024.10 | 991.90 | 991.04 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 990.30 | 998.25 | 998.38 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1010.65 | 999.73 | 998.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1014.55 | 1004.74 | 1001.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 996.45 | 1004.89 | 1004.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 941.10 | 987.28 | 996.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 964.65 | 961.41 | 976.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 959.15 | 960.96 | 974.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:45:00 | 952.50 | 958.38 | 972.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:15:00 | 949.90 | 952.12 | 956.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:45:00 | 949.85 | 951.46 | 955.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 954.50 | 952.02 | 954.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 962.40 | 954.09 | 954.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 979.30 | 959.13 | 957.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 979.30 | 959.13 | 957.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 985.60 | 964.43 | 959.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1012.10 | 1013.62 | 1001.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 1012.10 | 1013.62 | 1001.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1054.00 | 1054.81 | 1042.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1045.90 | 1054.81 | 1042.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1057.10 | 1063.63 | 1052.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 1054.70 | 1063.63 | 1052.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1054.40 | 1061.79 | 1052.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 1054.80 | 1061.79 | 1052.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1048.90 | 1058.12 | 1052.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 1048.90 | 1058.12 | 1052.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1049.90 | 1056.48 | 1052.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:15:00 | 1049.10 | 1056.48 | 1052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1010.60 | 1045.10 | 1048.12 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1038.80 | 1033.62 | 1032.94 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1019.50 | 1031.84 | 1033.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1009.70 | 1023.85 | 1028.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1021.00 | 1015.95 | 1022.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1021.50 | 1017.06 | 1022.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1021.50 | 1017.06 | 1022.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1022.20 | 1018.09 | 1022.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1022.20 | 1018.09 | 1022.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1022.20 | 1018.91 | 1022.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:30:00 | 1008.80 | 1016.27 | 1020.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 12:15:00 | 958.36 | 977.27 | 993.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1008.00 | 976.92 | 987.07 | SL hit (close>ema200) qty=0.50 sl=976.92 alert=retest2 |

### Cycle 65 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 990.80 | 979.12 | 977.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 996.00 | 984.72 | 980.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 965.10 | 979.09 | 980.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 11:15:00 | 960.50 | 975.37 | 979.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 978.10 | 969.79 | 974.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 988.70 | 973.57 | 975.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 988.70 | 973.57 | 975.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 981.10 | 975.08 | 975.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 991.00 | 975.08 | 975.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 987.30 | 977.52 | 977.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 989.40 | 981.33 | 979.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 982.00 | 982.41 | 980.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:30:00 | 981.20 | 982.41 | 980.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 991.50 | 984.31 | 981.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 987.30 | 984.31 | 981.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 995.90 | 995.56 | 990.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 997.00 | 995.56 | 990.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 995.20 | 995.49 | 990.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 992.60 | 995.49 | 990.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 994.10 | 1000.09 | 996.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 994.10 | 1000.09 | 996.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 998.50 | 999.77 | 996.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1005.40 | 999.77 | 996.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1001.20 | 1003.46 | 999.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 1003.20 | 1000.31 | 999.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 1002.00 | 1000.49 | 999.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1000.00 | 1000.71 | 1000.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1000.10 | 1000.71 | 1000.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 999.50 | 1000.47 | 1000.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 994.50 | 998.56 | 999.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 1002.30 | 999.00 | 999.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 999.50 | 999.10 | 999.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1000.00 | 999.10 | 999.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 999.00 | 999.08 | 999.27 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1001.20 | 999.50 | 999.45 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 997.70 | 999.24 | 999.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 993.10 | 997.99 | 998.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 1000.20 | 997.85 | 998.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 995.50 | 997.38 | 998.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 999.30 | 997.38 | 998.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 994.20 | 995.18 | 996.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:15:00 | 988.70 | 995.18 | 996.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 991.30 | 988.97 | 992.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 989.00 | 988.16 | 989.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 990.60 | 983.40 | 985.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 986.40 | 984.00 | 985.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 981.60 | 983.39 | 985.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.26 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.73 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.55 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.07 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 932.52 | 943.16 | 948.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 11:15:00 | 943.40 | 938.63 | 943.51 | SL hit (close>ema200) qty=0.50 sl=938.63 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 950.55 | 943.04 | 942.72 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 937.15 | 942.36 | 942.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 930.00 | 938.85 | 940.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 13:15:00 | 936.25 | 935.42 | 938.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 936.25 | 935.42 | 938.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 938.05 | 935.94 | 938.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 936.25 | 935.94 | 938.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 931.35 | 934.85 | 937.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 928.60 | 933.30 | 936.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 924.95 | 930.08 | 934.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 939.35 | 929.63 | 931.07 | SL hit (close>static) qty=1.00 sl=938.35 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 942.10 | 932.01 | 930.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 954.35 | 936.48 | 933.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 967.05 | 968.35 | 957.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 967.05 | 968.35 | 957.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 960.70 | 964.92 | 959.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 960.85 | 964.92 | 959.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 964.50 | 964.84 | 960.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 976.70 | 964.97 | 960.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 973.00 | 965.14 | 962.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 968.40 | 963.77 | 962.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:15:00 | 969.10 | 964.57 | 963.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 964.50 | 964.56 | 963.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 964.50 | 964.56 | 963.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 974.40 | 966.53 | 964.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 982.35 | 966.53 | 964.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 980.45 | 988.02 | 989.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 985.00 | 984.91 | 987.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 995.00 | 984.91 | 987.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 987.00 | 985.33 | 987.27 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 993.00 | 987.76 | 987.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 999.20 | 990.05 | 988.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 992.15 | 992.41 | 990.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:15:00 | 991.25 | 992.41 | 990.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 999.80 | 993.89 | 991.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 992.00 | 993.89 | 991.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 997.40 | 995.06 | 992.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 996.60 | 995.06 | 992.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 991.75 | 995.13 | 993.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 991.75 | 995.13 | 993.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 983.00 | 992.70 | 992.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 979.00 | 986.80 | 989.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 983.80 | 981.95 | 986.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 983.80 | 981.95 | 986.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 982.85 | 982.13 | 986.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 976.90 | 981.51 | 985.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 978.00 | 980.81 | 984.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 995.80 | 984.44 | 985.83 | SL hit (close>static) qty=1.00 sl=986.45 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 993.05 | 987.13 | 986.87 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 989.85 | 993.49 | 993.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 985.60 | 991.33 | 992.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 995.10 | 991.99 | 992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 988.35 | 991.26 | 992.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 986.40 | 989.86 | 991.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 983.75 | 988.64 | 990.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 937.08 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 934.56 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 940.00 | 933.66 | 942.91 | SL hit (close>ema200) qty=0.50 sl=933.66 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 947.55 | 944.81 | 944.53 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 930.40 | 943.94 | 944.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 926.25 | 936.47 | 939.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 920.85 | 920.53 | 927.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:00:00 | 910.55 | 918.54 | 925.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 865.02 | 883.60 | 898.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 851.60 | 846.99 | 860.14 | SL hit (close>ema200) qty=0.50 sl=846.99 alert=retest1 |

### Cycle 81 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 869.90 | 853.22 | 852.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 880.00 | 858.57 | 854.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 11:15:00 | 871.20 | 871.49 | 864.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:30:00 | 871.95 | 871.49 | 864.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 874.95 | 872.18 | 865.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 875.50 | 871.87 | 866.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 875.55 | 871.89 | 867.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-20 11:15:00 | 963.05 | 905.72 | 885.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 913.50 | 927.76 | 928.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 904.10 | 914.94 | 921.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 905.00 | 904.07 | 910.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 905.75 | 904.07 | 910.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 911.20 | 905.50 | 910.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 911.20 | 905.50 | 910.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 919.15 | 908.23 | 911.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 919.15 | 908.23 | 911.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 918.60 | 910.30 | 912.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 915.00 | 910.30 | 912.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 922.00 | 914.85 | 913.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 948.70 | 949.92 | 940.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 948.70 | 949.92 | 940.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 945.00 | 948.94 | 941.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 947.50 | 948.94 | 941.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 945.80 | 948.31 | 941.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 957.45 | 948.48 | 944.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 953.05 | 949.48 | 946.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 956.10 | 949.48 | 946.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 962.30 | 954.86 | 949.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 970.60 | 966.15 | 959.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 959.50 | 966.15 | 959.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 988.35 | 992.71 | 987.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 990.00 | 992.71 | 987.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 985.80 | 991.33 | 987.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 10:15:00 | 973.95 | 980.04 | 982.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 980.00 | 978.91 | 981.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 980.00 | 978.91 | 981.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 85 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 999.90 | 983.11 | 982.96 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 982.10 | 989.21 | 989.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 977.80 | 984.36 | 986.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 15:15:00 | 957.00 | 955.50 | 965.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 962.40 | 955.50 | 965.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 955.75 | 955.55 | 964.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 962.45 | 955.55 | 964.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 970.00 | 959.39 | 965.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 968.50 | 959.39 | 965.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 971.00 | 961.71 | 965.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 971.00 | 961.71 | 965.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 974.40 | 968.37 | 967.96 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 963.60 | 967.20 | 967.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 960.40 | 965.84 | 966.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 924.90 | 919.75 | 928.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 925.75 | 920.99 | 926.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 925.75 | 920.99 | 926.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 921.65 | 921.12 | 925.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 920.60 | 921.12 | 925.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 918.90 | 920.37 | 924.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 919.30 | 916.95 | 921.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 918.65 | 917.94 | 920.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 920.95 | 918.54 | 920.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 920.95 | 918.54 | 920.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | SL hit (close>static) qty=1.00 sl=926.35 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 930.25 | 922.63 | 922.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 938.60 | 927.19 | 924.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 931.10 | 931.30 | 927.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 931.10 | 931.30 | 927.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 928.00 | 930.64 | 927.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 927.25 | 930.64 | 927.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 921.20 | 928.75 | 926.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 921.20 | 928.75 | 926.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 921.20 | 927.24 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 922.00 | 927.24 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 919.90 | 925.78 | 925.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 914.90 | 921.59 | 923.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 929.85 | 922.17 | 923.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 919.25 | 921.59 | 922.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 918.25 | 921.59 | 922.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 918.70 | 920.24 | 921.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 915.45 | 914.01 | 915.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 915.80 | 914.01 | 915.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 921.50 | 915.51 | 916.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 920.25 | 915.51 | 916.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 916.40 | 915.68 | 916.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 939.30 | 922.39 | 919.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 913.40 | 921.61 | 922.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 910.00 | 916.24 | 918.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 915.65 | 914.45 | 916.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 915.65 | 914.45 | 916.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 916.55 | 914.87 | 916.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 916.50 | 914.87 | 916.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 917.25 | 915.35 | 916.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 917.40 | 915.35 | 916.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 912.95 | 914.87 | 916.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 909.00 | 912.38 | 914.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 908.00 | 903.15 | 906.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 914.90 | 907.37 | 905.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 909.55 | 909.68 | 907.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 909.55 | 909.68 | 907.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 909.50 | 909.49 | 907.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 903.35 | 909.49 | 907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 910.25 | 909.64 | 907.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 907.20 | 909.64 | 907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 910.80 | 909.87 | 908.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:00:00 | 914.65 | 910.83 | 908.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 914.70 | 912.17 | 909.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 897.65 | 910.79 | 913.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 907.95 | 905.94 | 909.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:45:00 | 908.50 | 905.94 | 909.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 907.95 | 906.34 | 909.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 908.35 | 906.34 | 909.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 910.40 | 907.15 | 909.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 910.40 | 907.15 | 909.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 909.90 | 907.70 | 909.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 915.40 | 907.70 | 909.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 913.95 | 908.95 | 910.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 916.55 | 908.95 | 910.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 922.10 | 911.58 | 911.15 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 907.85 | 910.84 | 910.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 901.50 | 908.97 | 910.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 895.00 | 890.13 | 895.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 904.50 | 893.01 | 896.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 904.50 | 893.01 | 896.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 905.50 | 895.51 | 897.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 905.50 | 895.51 | 897.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 916.10 | 899.62 | 898.81 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 893.45 | 903.09 | 904.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 15:15:00 | 890.10 | 897.81 | 901.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 885.00 | 892.42 | 894.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 888.00 | 889.74 | 892.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 885.05 | 886.75 | 889.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 843.60 | 856.72 | 867.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.75 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.80 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 830.90 | 826.71 | 834.01 | SL hit (close>ema200) qty=0.50 sl=826.71 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 835.15 | 833.19 | 833.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 874.50 | 841.45 | 836.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 874.00 | 875.55 | 859.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 873.85 | 875.55 | 859.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 868.70 | 871.60 | 861.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 870.60 | 871.40 | 862.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 870.00 | 870.77 | 863.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 869.90 | 870.02 | 863.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 855.00 | 867.00 | 863.26 | SL hit (close<static) qty=1.00 sl=861.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 851.50 | 859.93 | 860.63 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 871.00 | 861.30 | 860.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 874.90 | 865.92 | 863.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 872.85 | 875.09 | 869.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 872.85 | 875.09 | 869.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 871.60 | 874.39 | 869.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 870.95 | 874.39 | 869.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 885.10 | 876.53 | 871.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:00:00 | 888.15 | 881.79 | 875.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 891.20 | 883.10 | 876.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 888.45 | 885.70 | 879.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 894.25 | 886.06 | 880.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 880.10 | 885.55 | 882.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 880.95 | 885.55 | 882.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 878.25 | 884.09 | 881.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:15:00 | 876.15 | 884.09 | 881.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 878.90 | 883.05 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 877.20 | 883.05 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 887.00 | 883.84 | 882.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 878.40 | 883.84 | 882.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 875.10 | 882.09 | 881.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 873.00 | 882.09 | 881.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 883.40 | 882.35 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 882.90 | 882.35 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 873.20 | 879.35 | 880.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 850.55 | 848.23 | 857.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 850.55 | 848.23 | 857.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 874.90 | 853.86 | 857.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 874.90 | 853.86 | 857.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 877.00 | 858.49 | 859.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 863.55 | 858.49 | 859.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 873.90 | 865.78 | 862.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 869.80 | 871.42 | 867.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 869.80 | 871.42 | 867.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 859.10 | 869.04 | 867.48 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 862.20 | 866.20 | 866.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 858.10 | 862.63 | 864.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 849.55 | 846.75 | 852.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 849.55 | 846.75 | 852.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 852.00 | 847.80 | 852.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 848.00 | 849.18 | 852.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 859.80 | 852.22 | 852.91 | SL hit (close>static) qty=1.00 sl=856.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 853.45 | 850.83 | 850.54 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 846.00 | 849.84 | 850.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 842.40 | 847.92 | 849.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 850.45 | 848.42 | 849.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 853.85 | 849.51 | 849.73 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 851.75 | 849.96 | 849.91 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 847.75 | 849.52 | 849.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 837.40 | 847.06 | 848.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 845.60 | 834.67 | 837.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 853.00 | 838.33 | 839.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 853.00 | 838.33 | 839.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 852.10 | 841.09 | 840.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 853.20 | 843.51 | 841.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:15:00 | 845.10 | 847.80 | 845.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 845.00 | 847.24 | 845.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 848.95 | 849.58 | 846.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 850.30 | 858.25 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 837.35 | 846.52 | 851.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 807.20 | 804.36 | 812.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 805.85 | 804.36 | 812.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 808.25 | 803.81 | 809.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 814.05 | 803.81 | 809.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 809.50 | 804.95 | 809.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 804.80 | 808.19 | 809.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 820.75 | 813.54 | 811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 803.35 | 814.25 | 812.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 805.70 | 812.54 | 811.77 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 803.10 | 809.70 | 810.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 795.75 | 806.50 | 808.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 812.35 | 798.15 | 802.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 805.80 | 799.68 | 803.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 810.60 | 799.68 | 803.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 806.10 | 802.87 | 804.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:15:00 | 802.50 | 802.87 | 804.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 801.40 | 803.67 | 803.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 830.15 | 820.23 | 816.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 821.00 | 827.88 | 822.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 816.65 | 825.63 | 822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 816.10 | 825.63 | 822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 815.65 | 821.85 | 821.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 815.65 | 821.85 | 821.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 804.30 | 818.34 | 819.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 799.00 | 812.31 | 816.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 798.15 | 794.62 | 802.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:00:00 | 798.15 | 794.62 | 802.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 777.15 | 765.80 | 777.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 782.00 | 765.80 | 777.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 787.10 | 770.06 | 778.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 787.10 | 770.06 | 778.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 799.00 | 775.85 | 780.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 799.00 | 775.85 | 780.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 805.75 | 785.78 | 784.05 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 765.15 | 787.38 | 790.18 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 821.95 | 788.56 | 786.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 827.25 | 796.30 | 790.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 823.00 | 823.84 | 814.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 820.25 | 823.84 | 814.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 821.10 | 830.26 | 823.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 823.65 | 830.26 | 823.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 831.60 | 830.53 | 824.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 837.85 | 830.53 | 824.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 833.50 | 834.10 | 829.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 829.00 | 826.42 | 826.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 831.20 | 828.49 | 827.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 832.25 | 841.09 | 837.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 841.00 | 841.08 | 837.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 857.00 | 843.27 | 838.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 850.00 | 848.44 | 844.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 849.35 | 848.01 | 844.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 831.55 | 842.09 | 843.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 842.95 | 842.00 | 843.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 842.95 | 842.00 | 843.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 841.15 | 841.89 | 843.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 837.20 | 841.27 | 842.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:30:00 | 837.30 | 838.72 | 841.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:45:00 | 838.00 | 835.92 | 837.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.34 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.43 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 796.10 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 813.00 | 806.62 | 815.78 | SL hit (close>ema200) qty=0.50 sl=806.62 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 813.00 | 808.01 | 807.75 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 805.40 | 807.49 | 807.54 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 808.15 | 807.62 | 807.60 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 805.45 | 807.17 | 807.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 804.25 | 806.60 | 807.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 797.40 | 794.55 | 799.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 799.95 | 795.63 | 799.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 798.90 | 795.63 | 799.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 800.00 | 796.50 | 799.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 801.30 | 796.50 | 799.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 806.40 | 798.48 | 800.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 807.75 | 798.48 | 800.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 804.00 | 799.58 | 800.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 800.85 | 799.58 | 800.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 798.55 | 800.71 | 800.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 796.95 | 799.95 | 800.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 790.00 | 786.28 | 791.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 786.85 | 786.39 | 790.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 773.45 | 786.63 | 790.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 754.05 | 759.85 | 760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 748.45 | 756.30 | 758.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 749.80 | 754.19 | 756.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 747.80 | 757.21 | 757.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 12:15:00 | 748.65 | 754.33 | 755.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 766.50 | 757.74 | 756.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 781.50 | 759.38 | 757.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 768.40 | 770.82 | 768.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 766.15 | 769.65 | 768.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 11:15:00 | 771.15 | 768.50 | 768.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 777.15 | 771.68 | 769.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 13:15:00 | 832.05 | 833.68 | 820.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 14:00:00 | 832.05 | 833.68 | 820.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 833.75 | 831.69 | 822.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 835.35 | 831.69 | 822.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:15:00 | 838.55 | 832.30 | 823.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 14:15:00 | 918.89 | 873.26 | 864.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 929.15 | 939.69 | 940.19 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 946.25 | 939.33 | 939.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 952.70 | 945.64 | 942.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 952.35 | 968.48 | 962.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 946.25 | 964.03 | 960.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 945.95 | 964.03 | 960.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 948.90 | 958.20 | 958.56 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 972.00 | 959.10 | 958.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 987.95 | 973.94 | 969.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-30 13:00:00 | 1599.15 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2024-05-31 11:00:00 | 1598.85 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2024-05-31 14:30:00 | 1600.20 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-05-31 15:00:00 | 1600.20 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-06-06 13:00:00 | 1572.85 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-06 13:45:00 | 1573.00 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-06-07 09:15:00 | 1571.00 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-06-07 11:45:00 | 1571.55 | 2024-06-07 13:15:00 | 1588.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2024-06-12 09:15:00 | 1708.95 | 2024-06-18 09:15:00 | 1794.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-12 09:15:00 | 1708.95 | 2024-06-18 10:15:00 | 1750.05 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-07-11 14:00:00 | 1682.95 | 2024-07-12 10:15:00 | 1702.20 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-07-11 15:00:00 | 1685.15 | 2024-07-12 10:15:00 | 1702.20 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-18 11:15:00 | 1735.50 | 2024-07-19 12:15:00 | 1720.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-07-19 12:00:00 | 1720.00 | 2024-07-19 12:15:00 | 1720.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-07-26 11:00:00 | 1725.10 | 2024-07-30 15:15:00 | 1719.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-08-07 10:30:00 | 1603.35 | 2024-08-09 09:15:00 | 1528.36 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-08-07 13:30:00 | 1608.80 | 2024-08-09 14:15:00 | 1523.18 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-08-07 14:45:00 | 1604.25 | 2024-08-09 14:15:00 | 1524.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 10:30:00 | 1603.35 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2024-08-07 13:30:00 | 1608.80 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-08-07 14:45:00 | 1604.25 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2024-08-28 11:00:00 | 1529.90 | 2024-09-05 11:15:00 | 1532.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-08-28 13:30:00 | 1524.00 | 2024-09-05 11:15:00 | 1532.30 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-09-10 12:15:00 | 1496.15 | 2024-09-10 14:15:00 | 1525.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-09-12 11:15:00 | 1529.00 | 2024-09-13 10:15:00 | 1505.85 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-12 13:15:00 | 1526.80 | 2024-09-13 10:15:00 | 1505.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-20 13:30:00 | 1494.40 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-09-20 15:15:00 | 1495.00 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-23 10:00:00 | 1495.20 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-09-26 12:15:00 | 1519.90 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-26 15:00:00 | 1520.00 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-09-27 10:15:00 | 1520.20 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-27 11:30:00 | 1522.90 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1493.45 | 2024-10-07 09:15:00 | 1418.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1493.45 | 2024-10-08 11:15:00 | 1431.45 | STOP_HIT | 0.50 | 4.15% |
| BUY | retest2 | 2024-10-11 14:00:00 | 1465.20 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-10-14 09:45:00 | 1463.85 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-10-14 10:45:00 | 1460.40 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-10-25 09:15:00 | 1404.10 | 2024-10-30 11:15:00 | 1397.10 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1419.95 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-06 11:15:00 | 1423.85 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-11-08 11:30:00 | 1406.90 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-11-13 14:30:00 | 1411.00 | 2024-11-13 15:15:00 | 1424.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-11-21 12:15:00 | 1424.40 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-11-22 15:00:00 | 1410.30 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-11-26 09:30:00 | 1425.65 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-11-29 09:30:00 | 1427.70 | 2024-12-03 12:15:00 | 1421.40 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-12-10 10:15:00 | 1360.00 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-10 12:30:00 | 1365.05 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-10 13:00:00 | 1354.65 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1310.90 | 2024-12-20 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-12-19 09:45:00 | 1313.05 | 2024-12-20 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-12-26 09:30:00 | 1276.50 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-27 10:15:00 | 1279.75 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-30 09:30:00 | 1279.10 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-12-31 10:00:00 | 1278.40 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-31 11:15:00 | 1275.60 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-31 12:30:00 | 1275.55 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-31 13:15:00 | 1275.00 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-31 14:30:00 | 1275.35 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-09 15:15:00 | 1262.00 | 2025-01-15 10:15:00 | 1256.35 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1221.20 | 2025-01-27 09:15:00 | 1160.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:45:00 | 1225.45 | 2025-01-27 09:15:00 | 1164.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1221.20 | 2025-01-28 14:15:00 | 1146.10 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2025-01-21 09:45:00 | 1225.45 | 2025-01-28 14:15:00 | 1146.10 | STOP_HIT | 0.50 | 6.48% |
| BUY | retest1 | 2025-02-01 15:00:00 | 1186.40 | 2025-02-03 09:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1086.30 | 2025-02-11 09:15:00 | 1031.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1086.30 | 2025-02-13 11:15:00 | 1032.00 | STOP_HIT | 0.50 | 5.00% |
| BUY | retest2 | 2025-03-17 11:30:00 | 976.65 | 2025-03-25 09:15:00 | 1074.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 12:45:00 | 972.25 | 2025-03-25 09:15:00 | 1069.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:30:00 | 972.70 | 2025-03-25 09:15:00 | 1069.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 14:15:00 | 972.25 | 2025-03-25 09:15:00 | 1069.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-19 14:30:00 | 976.80 | 2025-03-25 09:15:00 | 1074.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 10:15:00 | 977.05 | 2025-03-25 09:15:00 | 1074.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:45:00 | 952.50 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-04-11 11:15:00 | 949.90 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-04-11 11:45:00 | 949.85 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-04-15 09:15:00 | 954.50 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-05 13:30:00 | 1008.80 | 2025-05-07 12:15:00 | 958.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 13:30:00 | 1008.80 | 2025-05-08 09:15:00 | 1008.00 | STOP_HIT | 0.50 | 0.08% |
| BUY | retest2 | 2025-05-21 13:15:00 | 1005.40 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1001.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-05-23 12:00:00 | 1003.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-23 12:45:00 | 1002.00 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-12 11:15:00 | 939.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-12 11:15:00 | 941.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-12 11:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-12 11:15:00 | 941.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-12 13:15:00 | 932.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-06-19 10:30:00 | 928.60 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-19 12:45:00 | 924.95 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-20 15:15:00 | 923.90 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-06-23 09:30:00 | 926.35 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-27 10:15:00 | 976.70 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-06-30 09:15:00 | 973.00 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-30 12:30:00 | 968.40 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-06-30 14:15:00 | 969.10 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-07-01 09:15:00 | 982.35 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-14 12:15:00 | 976.90 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-07-14 13:00:00 | 978.00 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-28 13:15:00 | 937.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-28 13:15:00 | 934.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-07 10:15:00 | 865.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-11 11:15:00 | 851.60 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2025-08-12 11:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-12 12:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-13 13:30:00 | 850.90 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-14 10:30:00 | 852.85 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-19 14:30:00 | 875.50 | 2025-08-20 11:15:00 | 963.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-20 09:15:00 | 875.55 | 2025-08-20 11:15:00 | 963.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 12:15:00 | 915.00 | 2025-09-01 10:15:00 | 916.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-05 09:45:00 | 957.45 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-09-05 13:30:00 | 953.05 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2025-09-05 14:15:00 | 956.10 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2025-09-08 09:30:00 | 962.30 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2025-10-01 09:15:00 | 920.60 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-01 09:45:00 | 918.90 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-01 14:45:00 | 919.30 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-03 10:00:00 | 918.65 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-08 11:15:00 | 918.25 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-08 12:45:00 | 918.70 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-10 09:45:00 | 915.45 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-10 10:15:00 | 915.80 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-17 11:30:00 | 909.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-10-21 14:15:00 | 908.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-29 12:00:00 | 914.65 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-29 13:30:00 | 914.70 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-19 10:15:00 | 843.60 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-19 12:15:00 | 840.75 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-19 12:15:00 | 840.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.12% |
| BUY | retest2 | 2025-11-27 13:00:00 | 870.60 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-27 14:15:00 | 870.00 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-27 15:15:00 | 869.90 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-12-03 10:00:00 | 888.15 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-03 10:30:00 | 891.20 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-03 14:00:00 | 888.45 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-03 15:15:00 | 894.25 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-12-10 09:15:00 | 863.55 | 2025-12-11 14:15:00 | 869.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-19 09:30:00 | 848.00 | 2025-12-19 13:15:00 | 859.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-22 10:00:00 | 848.00 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-23 09:15:00 | 844.80 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-23 14:45:00 | 847.70 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-01-01 12:30:00 | 848.95 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-06 09:45:00 | 850.30 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-14 14:30:00 | 804.80 | 2026-01-16 09:15:00 | 816.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-21 13:15:00 | 802.50 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-22 09:30:00 | 801.40 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-02-12 12:15:00 | 837.85 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-13 12:00:00 | 833.50 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-02-20 09:45:00 | 857.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-23 09:15:00 | 850.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-23 10:45:00 | 849.35 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 09:15:00 | 795.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 09:15:00 | 795.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 09:15:00 | 796.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-10 12:15:00 | 800.85 | 2026-03-11 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-13 09:15:00 | 773.45 | 2026-03-18 13:15:00 | 770.20 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 10:45:00 | 749.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-03-23 09:15:00 | 747.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-23 12:15:00 | 748.65 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-03-25 09:15:00 | 781.50 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-27 13:15:00 | 768.40 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-03-27 14:15:00 | 766.15 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-04-07 10:15:00 | 835.35 | 2026-04-10 14:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:15:00 | 838.55 | 2026-04-17 10:15:00 | 922.40 | TARGET_HIT | 1.00 | 10.00% |
