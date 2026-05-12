# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1389.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 42 |
| ALERT2 | 42 |
| ALERT2_SKIP | 24 |
| ALERT3 | 108 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 63 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 32
- **Target hits / Stop hits / Partials:** 10 / 54 / 8
- **Avg / median % per leg:** 1.65% / 0.22%
- **Sum % (uncompounded):** 118.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 13 | 46.4% | 8 | 20 | 0 | 2.24% | 62.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 13 | 46.4% | 8 | 20 | 0 | 2.24% | 62.8% |
| SELL (all) | 44 | 27 | 61.4% | 2 | 34 | 8 | 1.27% | 55.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.08% | -3.1% |
| SELL @ 3rd Alert (retest2) | 43 | 27 | 62.8% | 2 | 33 | 8 | 1.37% | 59.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.08% | -3.1% |
| retest2 (combined) | 71 | 40 | 56.3% | 10 | 53 | 8 | 1.71% | 121.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1456.90 | 1447.20 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 1477.80 | 1453.32 | 1449.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1454.60 | 1466.24 | 1458.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1476.80 | 1468.35 | 1460.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 1460.70 | 1468.35 | 1460.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1470.30 | 1476.42 | 1468.66 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1454.00 | 1465.02 | 1466.14 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1475.70 | 1461.86 | 1461.78 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1457.60 | 1461.01 | 1461.40 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1469.10 | 1462.63 | 1462.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1525.00 | 1475.90 | 1468.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 1474.90 | 1485.33 | 1475.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1504.20 | 1489.10 | 1478.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 1563.90 | 1512.36 | 1499.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1534.30 | 1510.13 | 1505.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1520.40 | 1516.42 | 1509.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1558.90 | 1586.25 | 1588.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1558.90 | 1586.25 | 1588.42 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 1608.90 | 1584.23 | 1582.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1621.00 | 1606.34 | 1598.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1629.20 | 1636.72 | 1622.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 1629.20 | 1636.72 | 1622.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1625.50 | 1633.03 | 1623.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 1628.80 | 1633.03 | 1623.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1632.80 | 1632.99 | 1624.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 1622.50 | 1632.99 | 1624.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1628.80 | 1633.22 | 1626.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 1622.20 | 1633.22 | 1626.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1632.00 | 1632.97 | 1627.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1634.30 | 1632.97 | 1627.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:45:00 | 1634.90 | 1633.69 | 1628.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 1633.90 | 1633.53 | 1628.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-12 10:15:00 | 1797.73 | 1738.80 | 1701.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1778.50 | 1788.75 | 1790.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1767.90 | 1784.58 | 1788.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1613.80 | 1610.64 | 1637.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:15:00 | 1623.00 | 1610.64 | 1637.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1625.70 | 1618.49 | 1634.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1634.00 | 1618.49 | 1634.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1629.00 | 1623.26 | 1632.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1626.60 | 1623.26 | 1632.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1617.60 | 1622.13 | 1631.50 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 1676.40 | 1638.24 | 1633.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 15:15:00 | 1695.00 | 1667.02 | 1651.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 1663.00 | 1668.12 | 1654.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1690.60 | 1672.61 | 1657.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 1660.10 | 1672.61 | 1657.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1663.20 | 1673.55 | 1665.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1663.20 | 1673.55 | 1665.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1653.00 | 1669.44 | 1664.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 1653.00 | 1669.44 | 1664.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1685.00 | 1666.91 | 1664.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1657.40 | 1666.91 | 1664.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 1640.60 | 1661.64 | 1662.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 1637.20 | 1656.76 | 1659.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 1656.80 | 1642.39 | 1649.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1667.20 | 1647.35 | 1650.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:15:00 | 1670.00 | 1647.35 | 1650.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 1702.40 | 1658.36 | 1655.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1715.00 | 1688.94 | 1672.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1720.00 | 1727.14 | 1712.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1713.30 | 1724.37 | 1712.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 1727.40 | 1716.26 | 1712.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 1724.80 | 1726.72 | 1721.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 1721.80 | 1726.66 | 1725.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 1714.20 | 1724.17 | 1724.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1714.20 | 1724.17 | 1724.87 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1741.60 | 1726.28 | 1725.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 1760.90 | 1735.99 | 1729.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 1770.10 | 1770.23 | 1758.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:15:00 | 1774.40 | 1770.23 | 1758.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1764.90 | 1768.95 | 1760.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1764.90 | 1768.95 | 1760.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1776.30 | 1771.73 | 1763.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 1776.30 | 1771.73 | 1763.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1770.20 | 1774.12 | 1767.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1773.30 | 1774.12 | 1767.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1771.10 | 1773.51 | 1767.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1769.80 | 1773.51 | 1767.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1757.40 | 1770.29 | 1767.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 1757.40 | 1770.29 | 1767.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1754.50 | 1767.13 | 1765.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:15:00 | 1758.70 | 1767.13 | 1765.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1754.00 | 1764.49 | 1764.89 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1769.80 | 1765.42 | 1765.24 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 1763.10 | 1765.05 | 1765.11 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 1768.70 | 1765.39 | 1765.23 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 1763.50 | 1765.01 | 1765.07 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1766.50 | 1765.31 | 1765.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 1791.90 | 1773.84 | 1769.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 1769.60 | 1780.06 | 1774.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 1770.00 | 1778.04 | 1773.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 1770.00 | 1778.04 | 1773.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1790.00 | 1776.10 | 1773.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1814.90 | 1794.76 | 1787.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 1763.70 | 1790.21 | 1788.45 | SL hit (close<static) qty=1.00 sl=1772.60 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 1778.20 | 1785.96 | 1786.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 1762.50 | 1777.14 | 1782.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1791.60 | 1774.87 | 1778.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1795.00 | 1778.90 | 1780.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1813.80 | 1778.90 | 1780.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1816.50 | 1786.42 | 1783.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1827.80 | 1794.70 | 1787.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1811.20 | 1813.73 | 1800.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1811.20 | 1813.73 | 1800.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1798.90 | 1810.84 | 1801.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1823.70 | 1808.56 | 1802.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:00:00 | 1831.40 | 1811.10 | 1805.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 1827.50 | 1815.46 | 1808.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1771.00 | 1804.56 | 1805.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 1771.00 | 1804.56 | 1805.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1697.50 | 1775.64 | 1786.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 1719.90 | 1701.99 | 1723.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 1719.90 | 1701.99 | 1723.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1722.10 | 1706.01 | 1723.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1714.00 | 1707.61 | 1722.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 1732.90 | 1715.00 | 1723.32 | SL hit (close>static) qty=1.00 sl=1729.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1785.80 | 1701.12 | 1692.86 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 11:15:00 | 1762.90 | 1786.77 | 1789.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1739.40 | 1761.72 | 1767.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1699.20 | 1708.00 | 1716.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1725.10 | 1712.13 | 1716.46 | SL hit (close>static) qty=1.00 sl=1722.60 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 1665.00 | 1650.23 | 1650.09 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1640.50 | 1649.28 | 1650.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1634.90 | 1641.53 | 1644.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1647.10 | 1634.45 | 1637.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1637.70 | 1635.10 | 1637.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1630.10 | 1634.28 | 1636.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 1630.00 | 1633.44 | 1636.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1606.80 | 1595.47 | 1594.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1606.80 | 1595.47 | 1594.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1623.90 | 1603.73 | 1598.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 1605.20 | 1608.23 | 1601.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1603.10 | 1607.20 | 1602.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 1607.80 | 1607.32 | 1602.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 1592.20 | 1604.91 | 1602.69 | SL hit (close<static) qty=1.00 sl=1593.80 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1590.80 | 1599.67 | 1600.73 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1607.20 | 1601.35 | 1601.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1617.50 | 1604.58 | 1602.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1591.70 | 1606.35 | 1604.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1599.70 | 1605.02 | 1604.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 1592.30 | 1605.02 | 1604.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1589.90 | 1602.00 | 1602.82 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1611.40 | 1599.10 | 1598.76 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1597.40 | 1598.76 | 1598.77 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 1602.00 | 1599.41 | 1599.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 1606.50 | 1600.83 | 1599.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 1600.00 | 1601.39 | 1600.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1608.20 | 1602.75 | 1600.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 1600.00 | 1602.75 | 1600.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1594.90 | 1602.34 | 1601.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1592.40 | 1602.34 | 1601.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1593.10 | 1600.49 | 1600.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 1584.20 | 1600.49 | 1600.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1578.20 | 1596.03 | 1598.41 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1604.20 | 1598.80 | 1598.18 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1586.80 | 1596.76 | 1597.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 1577.90 | 1591.20 | 1594.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1588.20 | 1586.40 | 1590.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1587.30 | 1586.58 | 1590.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 1587.60 | 1586.58 | 1590.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1589.00 | 1587.06 | 1590.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 1579.80 | 1587.06 | 1590.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 1595.50 | 1589.51 | 1590.54 | SL hit (close>static) qty=1.00 sl=1592.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1599.00 | 1591.40 | 1591.31 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1571.00 | 1589.02 | 1590.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 1604.90 | 1593.47 | 1591.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 1606.00 | 1597.64 | 1594.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1581.60 | 1594.24 | 1592.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1586.90 | 1592.77 | 1592.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1588.30 | 1592.77 | 1592.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1588.00 | 1591.81 | 1592.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1588.00 | 1591.81 | 1592.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1583.80 | 1589.39 | 1590.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1590.50 | 1589.61 | 1590.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1584.60 | 1588.61 | 1589.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 15:15:00 | 1575.00 | 1587.71 | 1589.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1593.40 | 1586.81 | 1588.57 | SL hit (close>static) qty=1.00 sl=1591.90 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1602.50 | 1591.27 | 1590.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1627.00 | 1598.42 | 1593.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1605.00 | 1606.38 | 1599.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 1605.00 | 1606.38 | 1599.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1607.60 | 1606.63 | 1599.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1620.40 | 1610.26 | 1603.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 1621.90 | 1617.76 | 1614.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1618.00 | 1618.20 | 1615.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1596.80 | 1611.55 | 1612.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1596.80 | 1611.55 | 1612.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1590.90 | 1607.42 | 1610.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1610.60 | 1602.35 | 1606.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1620.80 | 1606.04 | 1608.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 1620.80 | 1606.04 | 1608.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1592.80 | 1605.49 | 1607.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 1591.70 | 1600.98 | 1604.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 1589.20 | 1597.97 | 1601.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 1592.20 | 1594.00 | 1598.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:30:00 | 1591.40 | 1588.87 | 1592.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1589.94 | 1592.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 1584.20 | 1588.62 | 1591.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1584.30 | 1588.38 | 1590.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1579.00 | 1585.10 | 1586.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 1584.20 | 1586.19 | 1586.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1574.00 | 1583.75 | 1585.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1561.60 | 1580.63 | 1583.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1512.12 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1509.74 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1512.59 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1511.83 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1540.00 | 1536.68 | 1550.75 | SL hit (close>ema200) qty=0.50 sl=1536.68 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1560.40 | 1552.75 | 1552.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1566.90 | 1555.58 | 1554.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1553.90 | 1556.75 | 1554.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1566.00 | 1558.60 | 1555.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1571.80 | 1561.50 | 1557.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 1586.30 | 1570.70 | 1565.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1616.30 | 1644.06 | 1644.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 1616.30 | 1644.06 | 1644.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 10:15:00 | 1612.30 | 1637.71 | 1641.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1611.50 | 1621.36 | 1630.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 1605.70 | 1615.84 | 1624.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1609.00 | 1614.47 | 1623.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1530.92 | 1544.69 | 1553.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1528.55 | 1544.69 | 1553.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1608.00 | 1552.66 | 1553.88 | SL hit (close>ema200) qty=0.50 sl=1552.66 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 1606.00 | 1563.33 | 1558.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 1640.00 | 1586.29 | 1570.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 1629.00 | 1634.44 | 1601.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 1624.80 | 1634.44 | 1601.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1593.00 | 1616.98 | 1606.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1582.00 | 1616.98 | 1606.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1593.10 | 1612.21 | 1605.20 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 1582.70 | 1599.12 | 1600.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 1572.00 | 1587.56 | 1593.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1576.10 | 1574.99 | 1582.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1578.60 | 1574.99 | 1582.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1581.70 | 1576.56 | 1581.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 1581.70 | 1576.56 | 1581.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1582.20 | 1577.68 | 1581.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:15:00 | 1592.00 | 1577.68 | 1581.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1592.00 | 1580.55 | 1582.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1579.40 | 1580.55 | 1582.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1584.00 | 1581.24 | 1582.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:45:00 | 1574.20 | 1578.99 | 1581.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1571.40 | 1564.41 | 1563.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1571.40 | 1564.41 | 1563.89 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 1552.10 | 1563.44 | 1564.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1550.80 | 1555.64 | 1559.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1505.00 | 1504.77 | 1516.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1503.00 | 1504.77 | 1516.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1508.10 | 1501.97 | 1511.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1508.10 | 1501.97 | 1511.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1506.10 | 1503.67 | 1510.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:15:00 | 1514.30 | 1503.67 | 1510.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1514.30 | 1505.79 | 1510.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1499.70 | 1505.79 | 1510.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1501.90 | 1505.01 | 1509.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 1492.80 | 1500.90 | 1507.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1520.00 | 1507.88 | 1507.93 | SL hit (close>static) qty=1.00 sl=1517.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1528.20 | 1511.95 | 1509.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1533.20 | 1520.08 | 1514.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 1550.00 | 1550.23 | 1542.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 1550.00 | 1550.23 | 1542.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1558.50 | 1553.40 | 1547.63 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1532.10 | 1543.17 | 1544.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1522.60 | 1534.99 | 1539.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1482.60 | 1481.99 | 1495.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1482.60 | 1481.99 | 1495.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1416.80 | 1393.09 | 1407.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1420.00 | 1393.09 | 1407.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1415.80 | 1397.63 | 1407.83 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1440.00 | 1417.58 | 1414.87 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 1382.60 | 1410.59 | 1411.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 1360.90 | 1385.45 | 1396.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 1351.20 | 1349.14 | 1365.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 1351.20 | 1349.14 | 1365.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1350.00 | 1350.45 | 1361.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1338.10 | 1350.45 | 1361.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 1382.10 | 1360.45 | 1358.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 1382.10 | 1360.45 | 1358.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 15:15:00 | 1385.00 | 1370.52 | 1363.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 1395.10 | 1376.07 | 1374.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 1398.10 | 1413.34 | 1414.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 1398.10 | 1413.34 | 1414.04 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1426.90 | 1414.74 | 1414.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1430.80 | 1417.95 | 1415.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 11:15:00 | 1451.80 | 1458.16 | 1444.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 12:00:00 | 1451.80 | 1458.16 | 1444.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1452.20 | 1460.14 | 1451.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 1484.20 | 1471.94 | 1461.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1440.60 | 1472.85 | 1474.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 1440.60 | 1472.85 | 1474.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 1414.10 | 1461.10 | 1469.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 1346.20 | 1343.21 | 1363.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 1346.20 | 1343.21 | 1363.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1347.90 | 1339.91 | 1347.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 1347.00 | 1339.91 | 1347.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1341.70 | 1340.27 | 1347.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1333.00 | 1342.04 | 1346.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 1335.50 | 1340.73 | 1345.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 1334.50 | 1340.15 | 1344.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1349.10 | 1342.66 | 1344.49 | SL hit (close>static) qty=1.00 sl=1348.20 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 1358.00 | 1345.73 | 1345.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 1359.50 | 1348.49 | 1346.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 1373.00 | 1373.42 | 1362.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:15:00 | 1371.30 | 1373.42 | 1362.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1384.90 | 1379.07 | 1371.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 1376.60 | 1379.07 | 1371.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1360.00 | 1375.36 | 1370.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 1363.10 | 1375.36 | 1370.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1358.00 | 1371.89 | 1369.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1360.00 | 1371.89 | 1369.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 1361.90 | 1367.73 | 1368.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1354.30 | 1362.69 | 1364.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 1364.70 | 1358.58 | 1361.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1367.20 | 1360.30 | 1362.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 1367.20 | 1360.30 | 1362.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1363.00 | 1360.84 | 1362.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1343.40 | 1360.84 | 1362.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1352.40 | 1359.05 | 1361.31 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 1365.00 | 1361.71 | 1361.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1371.10 | 1364.62 | 1362.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1361.70 | 1365.09 | 1363.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1363.10 | 1364.70 | 1363.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1364.60 | 1364.70 | 1363.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1366.70 | 1365.10 | 1363.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 1364.70 | 1365.10 | 1363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1365.00 | 1365.35 | 1364.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 1365.00 | 1365.35 | 1364.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1362.30 | 1364.74 | 1363.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1362.30 | 1364.74 | 1363.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1359.00 | 1363.59 | 1363.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 1320.80 | 1363.59 | 1363.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1319.80 | 1354.83 | 1359.51 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1358.00 | 1348.37 | 1348.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1358.40 | 1350.37 | 1349.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 1350.50 | 1353.45 | 1351.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1349.90 | 1352.74 | 1351.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1349.90 | 1352.74 | 1351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1349.70 | 1352.13 | 1351.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1342.70 | 1352.13 | 1351.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1339.40 | 1349.58 | 1349.99 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 1352.00 | 1350.01 | 1349.79 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1338.70 | 1347.75 | 1348.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1335.30 | 1345.26 | 1347.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1298.30 | 1295.51 | 1307.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 1298.30 | 1295.51 | 1307.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1310.90 | 1301.31 | 1305.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 1310.90 | 1301.31 | 1305.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1301.00 | 1301.25 | 1305.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 1296.10 | 1301.19 | 1304.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 1299.40 | 1301.19 | 1304.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1261.20 | 1288.02 | 1292.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1310.10 | 1285.43 | 1283.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1310.10 | 1285.43 | 1283.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1335.90 | 1312.74 | 1298.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1316.50 | 1322.55 | 1307.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 1304.20 | 1316.58 | 1308.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1304.20 | 1316.58 | 1308.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 1304.20 | 1316.58 | 1308.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1334.90 | 1320.24 | 1310.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1341.20 | 1324.43 | 1313.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 1289.90 | 1319.26 | 1314.04 | SL hit (close<static) qty=1.00 sl=1303.90 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 1289.00 | 1308.21 | 1309.61 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1320.40 | 1305.83 | 1305.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1334.80 | 1311.63 | 1307.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1298.20 | 1312.48 | 1309.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1310.40 | 1312.06 | 1309.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 1317.20 | 1312.91 | 1309.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1317.90 | 1313.63 | 1310.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1334.50 | 1313.63 | 1310.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1325.90 | 1317.14 | 1313.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1315.80 | 1318.32 | 1315.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 1315.80 | 1318.32 | 1315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1313.20 | 1317.29 | 1314.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 1313.20 | 1317.29 | 1314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1316.40 | 1317.11 | 1314.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:30:00 | 1310.00 | 1317.11 | 1314.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1316.00 | 1316.89 | 1315.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:45:00 | 1330.80 | 1320.13 | 1316.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 1448.92 | 1403.65 | 1382.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1428.70 | 1440.33 | 1441.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 1370.90 | 1421.51 | 1432.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 10:15:00 | 1368.70 | 1366.25 | 1385.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:30:00 | 1352.30 | 1363.30 | 1382.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1394.00 | 1370.79 | 1380.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 1394.00 | 1370.79 | 1380.92 | SL hit (close>ema400) qty=1.00 sl=1380.92 alert=retest1 |

### Cycle 69 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1352.60 | 1336.41 | 1335.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 1356.50 | 1347.19 | 1341.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1388.70 | 1389.39 | 1380.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 1385.60 | 1389.39 | 1380.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:30:00 | 1445.20 | 2025-05-14 10:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-13 11:00:00 | 1458.80 | 2025-05-14 10:15:00 | 1456.90 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1563.90 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1534.30 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-05-26 11:00:00 | 1520.40 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-06-09 11:15:00 | 1634.30 | 2025-06-12 10:15:00 | 1797.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 12:45:00 | 1634.90 | 2025-06-12 10:15:00 | 1798.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 14:15:00 | 1633.90 | 2025-06-12 10:15:00 | 1797.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:30:00 | 1727.40 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-10 10:30:00 | 1724.80 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-07-11 11:30:00 | 1721.80 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1814.90 | 2025-07-28 12:15:00 | 1763.70 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-07-31 13:30:00 | 1823.70 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-08-01 12:00:00 | 1831.40 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-08-01 12:45:00 | 1827.50 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1714.00 | 2025-08-08 13:15:00 | 1732.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-08 15:15:00 | 1710.00 | 2025-08-18 09:15:00 | 1785.80 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1683.30 | 2025-08-18 09:15:00 | 1785.80 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1699.20 | 2025-09-11 09:15:00 | 1725.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-11 13:00:00 | 1697.20 | 2025-09-18 11:15:00 | 1612.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 1698.30 | 2025-09-18 11:15:00 | 1613.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 13:00:00 | 1697.20 | 2025-09-18 14:15:00 | 1634.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-09-11 14:45:00 | 1698.30 | 2025-09-18 14:15:00 | 1634.30 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1630.10 | 2025-10-01 12:15:00 | 1606.80 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2025-09-25 13:45:00 | 1630.00 | 2025-10-01 12:15:00 | 1606.80 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-10-03 12:00:00 | 1607.80 | 2025-10-03 14:15:00 | 1592.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-17 09:15:00 | 1579.80 | 2025-10-17 12:15:00 | 1595.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-23 09:15:00 | 1588.30 | 2025-10-23 09:15:00 | 1588.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-10-24 15:15:00 | 1575.00 | 2025-10-27 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1620.40 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-30 12:45:00 | 1621.90 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1618.00 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-04 12:15:00 | 1591.70 | 2025-11-13 13:15:00 | 1512.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1589.20 | 2025-11-13 13:15:00 | 1509.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 13:00:00 | 1592.20 | 2025-11-13 13:15:00 | 1512.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 14:30:00 | 1591.40 | 2025-11-13 13:15:00 | 1511.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 1591.70 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1589.20 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2025-11-06 13:00:00 | 1592.20 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-11-07 14:30:00 | 1591.40 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-11-10 11:00:00 | 1584.20 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1584.30 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1579.00 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-11-12 12:00:00 | 1584.20 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1561.60 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-11-18 11:30:00 | 1571.80 | 2025-11-26 09:15:00 | 1616.30 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-11-19 12:45:00 | 1586.30 | 2025-11-26 09:15:00 | 1616.30 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1611.50 | 2025-12-08 12:15:00 | 1530.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1605.70 | 2025-12-08 12:15:00 | 1528.55 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1611.50 | 2025-12-09 09:15:00 | 1608.00 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1605.70 | 2025-12-09 09:15:00 | 1608.00 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2025-11-27 15:00:00 | 1609.00 | 2025-12-09 10:15:00 | 1606.00 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-12-16 10:45:00 | 1574.20 | 2025-12-19 14:15:00 | 1571.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-01-01 11:30:00 | 1492.80 | 2026-01-02 10:15:00 | 1520.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1338.10 | 2026-01-30 12:15:00 | 1382.10 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-02-02 15:15:00 | 1395.10 | 2026-02-06 09:15:00 | 1398.10 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-02-12 09:45:00 | 1484.20 | 2026-02-13 13:15:00 | 1440.60 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-02-23 09:15:00 | 1333.00 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-23 10:00:00 | 1335.50 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-02-23 11:45:00 | 1334.50 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-19 10:30:00 | 1296.10 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-19 11:00:00 | 1299.40 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1261.20 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2026-03-27 15:00:00 | 1341.20 | 2026-03-30 10:15:00 | 1289.90 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-04-06 11:30:00 | 1317.20 | 2026-04-15 10:15:00 | 1448.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 1317.90 | 2026-04-15 10:15:00 | 1449.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1334.50 | 2026-04-16 09:15:00 | 1467.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1325.90 | 2026-04-16 09:15:00 | 1458.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:45:00 | 1330.80 | 2026-04-16 09:15:00 | 1463.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-23 11:30:00 | 1352.30 | 2026-04-23 14:15:00 | 1394.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-04-27 14:00:00 | 1348.30 | 2026-04-28 09:15:00 | 1213.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 15:15:00 | 1359.00 | 2026-04-28 09:15:00 | 1223.10 | TARGET_HIT | 1.00 | 10.00% |
