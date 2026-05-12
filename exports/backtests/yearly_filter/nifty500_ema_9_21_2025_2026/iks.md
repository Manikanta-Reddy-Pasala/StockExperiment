# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 1745.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 23 |
| ALERT3 | 150 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 63 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 7 / 55
- **Target hits / Stop hits / Partials:** 0 / 59 / 3
- **Avg / median % per leg:** -1.10% / -1.63%
- **Sum % (uncompounded):** -68.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 0 | 15 | 0 | -1.43% | -21.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 0 | 15 | 0 | -1.43% | -21.5% |
| SELL (all) | 47 | 6 | 12.8% | 0 | 44 | 3 | -0.99% | -46.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.82% | -1.6% |
| SELL @ 3rd Alert (retest2) | 45 | 6 | 13.3% | 0 | 42 | 3 | -1.00% | -45.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.82% | -1.6% |
| retest2 (combined) | 60 | 7 | 11.7% | 0 | 57 | 3 | -1.11% | -66.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 1605.90 | 1636.25 | 1638.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 1596.50 | 1619.44 | 1629.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1589.00 | 1581.55 | 1601.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1589.00 | 1581.55 | 1601.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1589.00 | 1581.55 | 1601.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1577.40 | 1581.55 | 1601.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 1574.50 | 1574.96 | 1589.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:45:00 | 1594.20 | 1574.96 | 1589.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1584.30 | 1577.15 | 1588.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:15:00 | 1577.40 | 1577.15 | 1588.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1569.10 | 1575.54 | 1586.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 1565.70 | 1575.54 | 1586.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 1566.00 | 1572.49 | 1583.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 1566.90 | 1572.49 | 1583.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 1568.10 | 1571.85 | 1581.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1590.00 | 1572.87 | 1579.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 1593.10 | 1572.87 | 1579.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1570.30 | 1572.35 | 1578.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 1590.10 | 1572.35 | 1578.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1561.10 | 1565.65 | 1572.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1583.70 | 1565.65 | 1572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1592.40 | 1571.00 | 1574.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1592.40 | 1571.00 | 1574.43 | SL hit (close>static) qty=1.00 sl=1590.80 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1594.20 | 1577.57 | 1576.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 1571.80 | 1576.28 | 1576.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1546.00 | 1570.23 | 1573.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1532.30 | 1517.39 | 1531.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1532.30 | 1517.39 | 1531.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1532.30 | 1517.39 | 1531.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 1532.30 | 1517.39 | 1531.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1528.60 | 1519.63 | 1531.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:00:00 | 1501.10 | 1515.92 | 1528.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1543.70 | 1516.43 | 1523.03 | SL hit (close>static) qty=1.00 sl=1542.30 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1545.50 | 1524.12 | 1523.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1556.00 | 1530.49 | 1526.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1527.00 | 1536.31 | 1530.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 1527.00 | 1536.31 | 1530.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1527.00 | 1536.31 | 1530.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:45:00 | 1528.60 | 1536.31 | 1530.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1523.40 | 1533.73 | 1529.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 1524.90 | 1533.73 | 1529.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1522.20 | 1531.42 | 1529.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 1521.60 | 1531.42 | 1529.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1585.00 | 1592.68 | 1579.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1576.00 | 1592.68 | 1579.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1577.80 | 1589.70 | 1579.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1573.00 | 1589.70 | 1579.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1592.80 | 1590.32 | 1580.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 1574.10 | 1590.32 | 1580.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1715.00 | 1721.89 | 1704.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1732.00 | 1721.89 | 1704.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 1726.00 | 1767.49 | 1763.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 1713.00 | 1756.59 | 1758.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1713.00 | 1756.59 | 1758.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1702.10 | 1745.69 | 1753.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 14:15:00 | 1600.50 | 1600.40 | 1615.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1600.50 | 1600.40 | 1615.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1597.00 | 1602.06 | 1613.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 13:45:00 | 1586.00 | 1599.92 | 1604.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 15:00:00 | 1585.40 | 1597.01 | 1602.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1585.70 | 1595.97 | 1601.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 1626.00 | 1601.79 | 1601.87 | SL hit (close>static) qty=1.00 sl=1614.40 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 1623.00 | 1606.03 | 1603.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 1631.00 | 1620.21 | 1614.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1632.60 | 1636.89 | 1631.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1632.60 | 1636.89 | 1631.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1632.60 | 1636.89 | 1631.29 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1603.10 | 1624.09 | 1626.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 1600.00 | 1619.27 | 1624.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1568.00 | 1565.46 | 1582.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 13:00:00 | 1568.00 | 1565.46 | 1582.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1580.50 | 1568.47 | 1582.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 1580.50 | 1568.47 | 1582.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1605.40 | 1575.86 | 1584.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1605.40 | 1575.86 | 1584.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1604.00 | 1581.49 | 1586.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1620.60 | 1581.49 | 1586.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1625.00 | 1590.19 | 1589.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1638.90 | 1599.93 | 1594.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 1610.00 | 1610.62 | 1602.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 1604.00 | 1610.62 | 1602.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.80 | 1608.45 | 1602.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1599.80 | 1608.45 | 1602.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1594.00 | 1605.56 | 1601.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1594.20 | 1605.56 | 1601.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1594.60 | 1603.37 | 1601.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 1599.70 | 1603.37 | 1601.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 1584.00 | 1597.36 | 1598.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 1584.00 | 1597.36 | 1598.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 14:15:00 | 1583.90 | 1594.67 | 1597.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 1598.00 | 1588.01 | 1592.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 1598.00 | 1588.01 | 1592.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1598.00 | 1588.01 | 1592.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 1602.40 | 1588.01 | 1592.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1594.80 | 1589.37 | 1592.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 1599.50 | 1589.37 | 1592.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1590.30 | 1589.55 | 1592.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1586.50 | 1588.78 | 1591.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 1597.70 | 1591.07 | 1591.90 | SL hit (close>static) qty=1.00 sl=1595.70 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1620.00 | 1597.56 | 1594.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1600.20 | 1607.38 | 1607.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 1589.10 | 1603.72 | 1606.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1602.40 | 1598.73 | 1602.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 1602.40 | 1598.73 | 1602.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1602.40 | 1598.73 | 1602.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1602.40 | 1598.73 | 1602.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1602.70 | 1599.53 | 1602.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 1603.60 | 1599.53 | 1602.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1598.00 | 1599.22 | 1602.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1602.00 | 1599.22 | 1602.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1593.00 | 1595.60 | 1599.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 1582.80 | 1595.60 | 1599.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1585.40 | 1582.89 | 1589.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1599.30 | 1593.88 | 1593.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 1599.30 | 1593.88 | 1593.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1626.60 | 1603.99 | 1598.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1606.00 | 1610.23 | 1604.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 1606.00 | 1610.23 | 1604.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1609.60 | 1610.10 | 1604.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1615.10 | 1610.10 | 1604.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1601.80 | 1608.44 | 1604.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 1595.10 | 1608.44 | 1604.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1588.50 | 1604.45 | 1603.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 1586.90 | 1604.45 | 1603.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1584.10 | 1600.38 | 1601.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1583.80 | 1594.36 | 1597.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1588.90 | 1587.89 | 1592.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 1588.90 | 1587.89 | 1592.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1579.30 | 1586.17 | 1591.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:15:00 | 1577.10 | 1586.17 | 1591.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 1577.00 | 1580.80 | 1585.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1568.80 | 1578.93 | 1583.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 1575.00 | 1578.14 | 1582.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1576.40 | 1576.96 | 1581.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 1576.40 | 1576.96 | 1581.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1601.00 | 1581.77 | 1583.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1601.00 | 1581.77 | 1583.08 | SL hit (close>static) qty=1.00 sl=1591.60 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1601.10 | 1585.63 | 1584.72 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1552.00 | 1579.82 | 1582.33 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 13:15:00 | 1607.00 | 1587.68 | 1585.26 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 1572.40 | 1584.79 | 1584.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 1567.00 | 1577.85 | 1581.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 1588.90 | 1579.12 | 1581.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 1588.90 | 1579.12 | 1581.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1588.90 | 1579.12 | 1581.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 1565.80 | 1576.44 | 1579.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 1569.90 | 1575.10 | 1578.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:45:00 | 1567.60 | 1574.86 | 1578.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 1568.90 | 1574.23 | 1577.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1572.60 | 1573.55 | 1576.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:45:00 | 1560.90 | 1571.80 | 1575.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:15:00 | 1557.40 | 1571.80 | 1575.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 1564.10 | 1570.38 | 1574.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 1605.50 | 1574.86 | 1575.36 | SL hit (close>static) qty=1.00 sl=1593.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 1590.00 | 1577.89 | 1576.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 09:15:00 | 1622.90 | 1586.89 | 1580.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 1609.50 | 1610.75 | 1599.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 1609.50 | 1610.75 | 1599.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1609.50 | 1610.75 | 1599.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 1605.30 | 1610.75 | 1599.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1599.50 | 1608.43 | 1601.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1599.50 | 1608.43 | 1601.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1599.90 | 1606.72 | 1601.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 1609.70 | 1606.72 | 1601.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1606.90 | 1606.76 | 1602.21 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 13:15:00 | 1597.00 | 1599.41 | 1599.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 14:15:00 | 1589.90 | 1597.50 | 1598.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 14:15:00 | 1555.50 | 1552.85 | 1565.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:45:00 | 1558.00 | 1552.85 | 1565.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1564.50 | 1556.81 | 1565.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 1553.20 | 1562.76 | 1566.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1596.20 | 1567.92 | 1567.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1596.20 | 1567.92 | 1567.69 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 1567.90 | 1579.65 | 1580.60 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1595.00 | 1582.06 | 1581.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 1613.50 | 1590.86 | 1585.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1586.60 | 1592.90 | 1587.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 1586.60 | 1592.90 | 1587.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1586.60 | 1592.90 | 1587.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1586.60 | 1592.90 | 1587.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1582.00 | 1590.72 | 1587.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1573.20 | 1588.29 | 1586.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1572.20 | 1585.08 | 1585.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 1563.10 | 1578.41 | 1582.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 1559.10 | 1557.62 | 1568.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 1559.10 | 1557.62 | 1568.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1568.70 | 1559.84 | 1568.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 1577.40 | 1559.84 | 1568.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1568.70 | 1561.61 | 1568.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 1570.00 | 1561.61 | 1568.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1600.10 | 1569.31 | 1571.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1600.10 | 1569.31 | 1571.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 15:15:00 | 1600.00 | 1575.45 | 1573.89 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1559.40 | 1580.28 | 1581.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1515.00 | 1551.60 | 1564.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 1513.90 | 1513.89 | 1529.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:30:00 | 1512.10 | 1513.89 | 1529.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1525.30 | 1515.55 | 1527.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1511.00 | 1515.21 | 1525.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 1510.50 | 1508.61 | 1517.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 1540.00 | 1520.27 | 1520.97 | SL hit (close>static) qty=1.00 sl=1534.90 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 1533.70 | 1522.95 | 1522.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1570.90 | 1532.54 | 1526.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1562.90 | 1565.50 | 1554.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 10:45:00 | 1561.70 | 1565.50 | 1554.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1558.70 | 1562.72 | 1554.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 1558.50 | 1562.72 | 1554.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1571.90 | 1576.90 | 1568.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 1571.90 | 1576.90 | 1568.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1564.20 | 1574.06 | 1569.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:15:00 | 1575.00 | 1574.06 | 1569.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1573.00 | 1573.85 | 1569.57 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 1552.40 | 1568.34 | 1569.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1548.10 | 1561.87 | 1566.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 1545.00 | 1542.66 | 1550.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 1545.00 | 1542.66 | 1550.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1535.10 | 1541.15 | 1548.96 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 1563.50 | 1551.60 | 1550.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1575.60 | 1556.40 | 1552.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 1557.10 | 1562.76 | 1558.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 1557.10 | 1562.76 | 1558.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1557.10 | 1562.76 | 1558.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1557.10 | 1562.76 | 1558.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1561.80 | 1562.57 | 1558.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 1558.10 | 1562.57 | 1558.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1555.70 | 1561.12 | 1558.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 1555.70 | 1561.12 | 1558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1561.10 | 1561.12 | 1558.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:15:00 | 1555.00 | 1561.12 | 1558.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1555.00 | 1559.89 | 1558.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1561.30 | 1559.89 | 1558.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1554.60 | 1558.83 | 1558.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 1554.60 | 1558.83 | 1558.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1553.30 | 1557.73 | 1557.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 1553.30 | 1557.73 | 1557.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 11:15:00 | 1552.30 | 1556.64 | 1557.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 1546.80 | 1554.20 | 1555.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 12:15:00 | 1534.00 | 1531.96 | 1539.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 13:00:00 | 1534.00 | 1531.96 | 1539.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1554.70 | 1537.25 | 1540.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1554.70 | 1537.25 | 1540.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1540.00 | 1537.80 | 1540.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1514.70 | 1537.80 | 1540.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 1438.96 | 1473.78 | 1487.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 1460.30 | 1459.14 | 1474.95 | SL hit (close>ema200) qty=0.50 sl=1459.14 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 1510.50 | 1470.20 | 1470.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 1541.60 | 1509.01 | 1495.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1527.90 | 1535.96 | 1518.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 1527.90 | 1535.96 | 1518.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1529.20 | 1533.54 | 1525.89 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 1520.10 | 1524.15 | 1524.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 1519.20 | 1522.91 | 1523.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 13:15:00 | 1525.00 | 1522.85 | 1523.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 1525.00 | 1522.85 | 1523.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1525.00 | 1522.85 | 1523.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:45:00 | 1530.80 | 1522.85 | 1523.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1530.20 | 1524.32 | 1523.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 1537.90 | 1529.07 | 1526.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 1528.10 | 1528.87 | 1526.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1528.10 | 1528.87 | 1526.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1528.10 | 1528.87 | 1526.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:15:00 | 1530.50 | 1528.87 | 1526.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1531.60 | 1529.42 | 1527.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 1536.00 | 1529.42 | 1527.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1515.30 | 1527.65 | 1526.87 | SL hit (close<static) qty=1.00 sl=1524.80 alert=retest2 |

### Cycle 33 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1515.00 | 1525.12 | 1525.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1503.90 | 1520.87 | 1523.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 1505.50 | 1505.05 | 1512.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:30:00 | 1509.40 | 1505.05 | 1512.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1516.10 | 1507.73 | 1512.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1516.10 | 1507.73 | 1512.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1520.80 | 1510.34 | 1512.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 1523.70 | 1510.34 | 1512.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1523.60 | 1514.62 | 1514.54 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1500.40 | 1517.39 | 1517.92 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1530.70 | 1518.82 | 1518.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 1539.40 | 1524.87 | 1521.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 1533.30 | 1535.16 | 1528.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:45:00 | 1532.90 | 1535.16 | 1528.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1524.60 | 1533.05 | 1527.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 1526.40 | 1533.05 | 1527.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1530.30 | 1532.50 | 1528.08 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1513.20 | 1524.53 | 1525.77 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1529.90 | 1525.98 | 1525.94 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1518.10 | 1525.18 | 1525.69 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 1537.40 | 1526.58 | 1526.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 1550.10 | 1531.28 | 1528.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1552.30 | 1561.30 | 1551.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1552.30 | 1561.30 | 1551.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1552.30 | 1561.30 | 1551.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1552.30 | 1561.30 | 1551.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1571.60 | 1563.36 | 1553.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1563.20 | 1563.36 | 1553.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1614.00 | 1574.30 | 1562.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 1680.00 | 1635.55 | 1604.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 1681.50 | 1665.79 | 1656.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:45:00 | 1680.50 | 1668.30 | 1658.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 15:15:00 | 1680.00 | 1667.15 | 1660.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1680.00 | 1669.72 | 1662.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 1685.20 | 1672.58 | 1664.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1652.00 | 1664.26 | 1663.70 | SL hit (close<static) qty=1.00 sl=1652.50 alert=retest2 |

### Cycle 41 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 1652.70 | 1661.95 | 1662.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 12:15:00 | 1649.90 | 1655.19 | 1658.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1623.30 | 1616.13 | 1626.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1623.30 | 1616.13 | 1626.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1623.30 | 1616.13 | 1626.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:30:00 | 1604.10 | 1616.45 | 1623.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:15:00 | 1602.50 | 1614.72 | 1621.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 1634.00 | 1599.62 | 1596.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 1634.00 | 1599.62 | 1596.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 1638.00 | 1612.38 | 1603.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 12:15:00 | 1640.50 | 1647.17 | 1628.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:45:00 | 1641.10 | 1647.17 | 1628.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1632.90 | 1642.02 | 1629.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 1625.00 | 1642.02 | 1629.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1625.00 | 1638.61 | 1629.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1669.00 | 1638.61 | 1629.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1675.70 | 1686.34 | 1687.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1675.70 | 1686.34 | 1687.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1666.50 | 1682.37 | 1685.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1682.20 | 1674.33 | 1679.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1682.20 | 1674.33 | 1679.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1682.20 | 1674.33 | 1679.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1682.20 | 1674.33 | 1679.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1688.60 | 1677.18 | 1679.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1688.60 | 1677.18 | 1679.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1676.50 | 1677.05 | 1679.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 1676.10 | 1677.66 | 1679.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 1592.29 | 1617.51 | 1638.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1605.70 | 1590.98 | 1608.03 | SL hit (close>ema200) qty=0.50 sl=1590.98 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1627.90 | 1591.97 | 1591.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1641.00 | 1610.62 | 1600.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 1677.30 | 1685.35 | 1671.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 1677.30 | 1685.35 | 1671.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1680.00 | 1684.28 | 1672.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1666.50 | 1684.28 | 1672.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1649.80 | 1677.39 | 1670.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 1649.80 | 1677.39 | 1670.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1651.40 | 1672.19 | 1668.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:15:00 | 1651.00 | 1672.19 | 1668.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 1651.20 | 1665.24 | 1666.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 1649.40 | 1662.07 | 1664.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1663.30 | 1659.50 | 1662.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1663.30 | 1659.50 | 1662.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1663.30 | 1659.50 | 1662.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1663.30 | 1659.50 | 1662.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1654.90 | 1658.58 | 1661.77 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1676.10 | 1664.11 | 1663.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1685.70 | 1668.43 | 1665.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1750.00 | 1754.16 | 1735.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1727.70 | 1748.25 | 1737.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1727.70 | 1748.25 | 1737.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1727.70 | 1748.25 | 1737.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1732.00 | 1745.00 | 1736.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 1727.50 | 1745.00 | 1736.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1722.10 | 1739.20 | 1736.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1722.10 | 1739.20 | 1736.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1738.00 | 1738.96 | 1736.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1709.00 | 1738.96 | 1736.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1705.10 | 1732.19 | 1733.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1689.90 | 1723.73 | 1729.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1675.00 | 1672.67 | 1688.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1675.20 | 1672.67 | 1688.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1670.40 | 1672.86 | 1685.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:30:00 | 1666.00 | 1671.83 | 1684.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 1666.10 | 1671.83 | 1684.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:00:00 | 1664.30 | 1670.32 | 1682.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 1666.60 | 1667.02 | 1676.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1679.30 | 1670.99 | 1675.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:30:00 | 1681.30 | 1670.99 | 1675.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1690.10 | 1674.81 | 1676.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 1690.10 | 1674.81 | 1676.41 | SL hit (close>static) qty=1.00 sl=1688.50 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1682.90 | 1678.06 | 1677.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 1687.50 | 1680.52 | 1678.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1694.80 | 1697.18 | 1691.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1694.80 | 1697.18 | 1691.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1694.80 | 1697.18 | 1691.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1694.00 | 1697.18 | 1691.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1688.30 | 1695.40 | 1690.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1688.30 | 1695.40 | 1690.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1698.30 | 1695.98 | 1691.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 1686.80 | 1695.98 | 1691.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1689.30 | 1694.65 | 1691.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 1691.90 | 1694.65 | 1691.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1668.70 | 1689.46 | 1689.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1668.70 | 1689.46 | 1689.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1674.80 | 1686.53 | 1688.02 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 1695.00 | 1688.87 | 1688.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 1699.00 | 1692.05 | 1690.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1685.00 | 1692.92 | 1691.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1685.00 | 1692.92 | 1691.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1685.00 | 1692.92 | 1691.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1689.20 | 1692.92 | 1691.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1670.00 | 1688.34 | 1689.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1664.90 | 1683.65 | 1687.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 13:15:00 | 1659.10 | 1657.21 | 1668.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:45:00 | 1656.90 | 1657.21 | 1668.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1661.30 | 1650.48 | 1659.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 1661.30 | 1650.48 | 1659.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1662.10 | 1652.80 | 1659.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 1661.20 | 1652.80 | 1659.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1663.90 | 1655.02 | 1660.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1669.70 | 1655.02 | 1660.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1675.90 | 1659.20 | 1661.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1678.50 | 1659.20 | 1661.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1674.10 | 1662.18 | 1662.66 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1684.40 | 1666.62 | 1664.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 1697.10 | 1676.86 | 1670.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1696.40 | 1697.31 | 1687.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1696.40 | 1697.31 | 1687.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1696.40 | 1697.31 | 1687.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 1683.70 | 1697.31 | 1687.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1687.90 | 1696.67 | 1692.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 1685.00 | 1696.67 | 1692.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1681.20 | 1693.57 | 1691.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 1680.60 | 1693.57 | 1691.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1676.60 | 1690.18 | 1689.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 1676.60 | 1690.18 | 1689.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 1682.00 | 1688.54 | 1689.16 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 15:15:00 | 1694.70 | 1689.82 | 1689.56 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1678.60 | 1687.58 | 1688.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1645.40 | 1678.68 | 1683.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1666.10 | 1659.95 | 1669.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 1666.10 | 1659.95 | 1669.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1666.10 | 1659.95 | 1669.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1676.80 | 1659.95 | 1669.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1694.80 | 1666.92 | 1671.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1694.80 | 1666.92 | 1671.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1681.70 | 1675.56 | 1674.82 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 1658.30 | 1674.24 | 1674.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 1640.50 | 1663.61 | 1669.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1631.60 | 1596.25 | 1622.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1631.60 | 1596.25 | 1622.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1631.60 | 1596.25 | 1622.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1631.60 | 1596.25 | 1622.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1633.90 | 1603.78 | 1623.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 1611.30 | 1603.78 | 1623.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:15:00 | 1530.73 | 1570.79 | 1593.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1547.10 | 1546.40 | 1568.77 | SL hit (close>ema200) qty=0.50 sl=1546.40 alert=retest2 |

### Cycle 58 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 1573.90 | 1564.28 | 1564.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 12:15:00 | 1587.70 | 1570.51 | 1567.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 1623.00 | 1629.86 | 1609.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 1606.00 | 1629.86 | 1609.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1616.90 | 1627.27 | 1610.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 1612.70 | 1627.27 | 1610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1590.00 | 1619.81 | 1608.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 1590.00 | 1619.81 | 1608.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1601.00 | 1616.05 | 1607.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 1591.30 | 1616.05 | 1607.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1604.90 | 1610.09 | 1606.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1604.90 | 1610.09 | 1606.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1629.00 | 1613.88 | 1608.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 1680.90 | 1613.88 | 1608.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1669.00 | 1715.09 | 1715.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1669.00 | 1715.09 | 1715.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 1666.50 | 1698.80 | 1707.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 15:15:00 | 1665.00 | 1653.33 | 1670.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1625.10 | 1653.33 | 1670.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 1629.10 | 1625.86 | 1641.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1640.50 | 1630.30 | 1640.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1640.50 | 1630.30 | 1640.05 | SL hit (close>ema400) qty=1.00 sl=1640.05 alert=retest1 |

### Cycle 60 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1659.40 | 1642.90 | 1642.35 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 1617.50 | 1641.05 | 1644.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 1592.40 | 1631.32 | 1639.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1437.40 | 1433.75 | 1479.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 1437.40 | 1433.75 | 1479.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1302.50 | 1307.31 | 1330.17 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1381.00 | 1329.67 | 1324.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 1402.30 | 1357.86 | 1339.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1378.00 | 1379.34 | 1362.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 1378.00 | 1379.34 | 1362.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1370.00 | 1376.30 | 1363.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1354.90 | 1376.30 | 1363.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1372.40 | 1378.39 | 1369.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 1372.40 | 1378.39 | 1369.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1370.00 | 1376.71 | 1369.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 1355.00 | 1376.71 | 1369.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1343.10 | 1369.99 | 1367.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 1343.10 | 1369.99 | 1367.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1340.70 | 1364.13 | 1365.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1335.50 | 1354.59 | 1360.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 1350.00 | 1346.52 | 1354.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 1362.00 | 1346.52 | 1354.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1354.90 | 1348.20 | 1354.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 1370.30 | 1348.20 | 1354.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1337.00 | 1345.96 | 1353.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 1350.00 | 1345.96 | 1353.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1353.60 | 1332.58 | 1336.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1355.00 | 1332.58 | 1336.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1344.50 | 1334.97 | 1337.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 1347.40 | 1334.97 | 1337.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1352.30 | 1340.36 | 1339.28 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 1330.00 | 1339.48 | 1340.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 15:15:00 | 1325.00 | 1331.82 | 1335.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 1318.40 | 1306.40 | 1318.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 1330.00 | 1306.40 | 1318.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1308.50 | 1306.82 | 1317.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1292.50 | 1304.34 | 1315.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 1342.00 | 1317.95 | 1319.62 | SL hit (close>static) qty=1.00 sl=1330.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1347.90 | 1323.94 | 1322.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1367.50 | 1336.39 | 1328.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 11:15:00 | 1338.10 | 1338.43 | 1330.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 11:15:00 | 1338.10 | 1338.43 | 1330.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 1338.10 | 1338.43 | 1330.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:45:00 | 1331.30 | 1338.43 | 1330.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 1328.00 | 1336.34 | 1330.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:45:00 | 1325.90 | 1336.34 | 1330.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 1319.00 | 1332.87 | 1329.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:00:00 | 1319.00 | 1332.87 | 1329.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1311.90 | 1327.26 | 1327.61 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1361.50 | 1330.02 | 1327.98 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1311.00 | 1331.71 | 1333.94 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1341.50 | 1336.17 | 1335.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1351.90 | 1340.07 | 1337.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 1460.00 | 1469.05 | 1448.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 10:00:00 | 1460.00 | 1469.05 | 1448.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1518.50 | 1512.06 | 1484.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1524.40 | 1513.86 | 1487.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1530.00 | 1516.89 | 1491.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1546.50 | 1518.76 | 1500.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:45:00 | 1523.60 | 1515.91 | 1502.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 1512.10 | 1520.34 | 1512.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 1512.10 | 1520.34 | 1512.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1516.00 | 1519.48 | 1512.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 1513.00 | 1519.48 | 1512.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1517.20 | 1519.02 | 1513.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1498.80 | 1510.18 | 1511.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 1498.80 | 1510.18 | 1511.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 15:15:00 | 1494.00 | 1505.95 | 1508.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 12:15:00 | 1446.30 | 1445.48 | 1464.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 13:00:00 | 1446.30 | 1445.48 | 1464.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 1436.60 | 1429.41 | 1437.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:00:00 | 1436.60 | 1429.41 | 1437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 1441.30 | 1431.79 | 1437.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:45:00 | 1443.40 | 1431.79 | 1437.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1433.70 | 1432.17 | 1437.44 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 1495.00 | 1445.25 | 1442.50 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 09:15:00 | 1440.60 | 1442.74 | 1442.76 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1453.50 | 1444.89 | 1443.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1468.60 | 1449.63 | 1446.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 15:15:00 | 1679.70 | 1684.36 | 1654.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:15:00 | 1673.70 | 1684.36 | 1654.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1665.00 | 1678.92 | 1663.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:00:00 | 1665.00 | 1678.92 | 1663.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1662.80 | 1675.69 | 1663.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 1664.20 | 1675.69 | 1663.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1668.00 | 1674.15 | 1663.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1685.40 | 1674.15 | 1663.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 1691.90 | 1677.70 | 1666.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1685.00 | 1679.38 | 1668.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 1685.90 | 1684.45 | 1671.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1686.00 | 1697.24 | 1689.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:15:00 | 1689.00 | 1697.24 | 1689.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 1712.80 | 1700.36 | 1692.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 10:30:00 | 1719.50 | 1702.96 | 1693.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 11:15:00 | 1722.50 | 1702.96 | 1693.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 11:15:00 | 1565.70 | 2025-05-26 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-05-22 12:45:00 | 1566.00 | 2025-05-26 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-05-22 13:15:00 | 1566.90 | 2025-05-26 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-05-22 14:15:00 | 1568.10 | 2025-05-26 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-05-30 12:00:00 | 1501.10 | 2025-06-02 09:15:00 | 1543.70 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-06-02 11:15:00 | 1521.20 | 2025-06-03 09:15:00 | 1545.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-02 15:00:00 | 1520.50 | 2025-06-03 09:15:00 | 1545.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1732.00 | 2025-06-18 12:15:00 | 1713.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-18 11:30:00 | 1726.00 | 2025-06-18 12:15:00 | 1713.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-26 13:45:00 | 1586.00 | 2025-06-27 14:15:00 | 1626.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-26 15:00:00 | 1585.40 | 2025-06-27 14:15:00 | 1626.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-06-27 09:15:00 | 1585.70 | 2025-06-27 14:15:00 | 1626.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-07-10 12:15:00 | 1599.70 | 2025-07-10 13:15:00 | 1584.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-11 14:45:00 | 1586.50 | 2025-07-14 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1588.10 | 2025-07-15 09:15:00 | 1620.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-14 14:00:00 | 1587.10 | 2025-07-15 09:15:00 | 1620.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-22 10:15:00 | 1582.80 | 2025-07-23 13:15:00 | 1599.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-23 09:30:00 | 1585.40 | 2025-07-23 13:15:00 | 1599.30 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-29 13:15:00 | 1577.10 | 2025-07-31 12:15:00 | 1601.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-07-30 12:30:00 | 1577.00 | 2025-07-31 12:15:00 | 1601.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1568.80 | 2025-07-31 12:15:00 | 1601.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-31 10:00:00 | 1575.00 | 2025-07-31 12:15:00 | 1601.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1565.80 | 2025-08-06 14:15:00 | 1605.50 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-08-05 12:45:00 | 1569.90 | 2025-08-06 14:15:00 | 1605.50 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-08-05 13:45:00 | 1567.60 | 2025-08-06 14:15:00 | 1605.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-05 14:30:00 | 1568.90 | 2025-08-06 15:15:00 | 1590.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-06 10:45:00 | 1560.90 | 2025-08-06 15:15:00 | 1590.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-06 11:15:00 | 1557.40 | 2025-08-06 15:15:00 | 1590.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-06 12:15:00 | 1564.10 | 2025-08-06 15:15:00 | 1590.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-14 15:15:00 | 1553.20 | 2025-08-18 09:15:00 | 1596.20 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1511.00 | 2025-09-03 12:15:00 | 1540.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-03 09:30:00 | 1510.50 | 2025-09-03 12:15:00 | 1540.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1514.70 | 2025-09-29 09:15:00 | 1438.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1514.70 | 2025-09-29 13:15:00 | 1460.30 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2025-10-13 15:15:00 | 1536.00 | 2025-10-14 09:15:00 | 1515.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-03 10:15:00 | 1680.00 | 2025-11-11 10:15:00 | 1652.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-07 11:15:00 | 1681.50 | 2025-11-11 11:15:00 | 1652.70 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-11-07 11:45:00 | 1680.50 | 2025-11-11 11:15:00 | 1652.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-07 15:15:00 | 1680.00 | 2025-11-11 11:15:00 | 1652.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-10 09:45:00 | 1685.20 | 2025-11-11 11:15:00 | 1652.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-18 09:30:00 | 1604.10 | 2025-11-24 11:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-18 11:15:00 | 1602.50 | 2025-11-24 11:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-26 09:15:00 | 1669.00 | 2025-12-02 15:15:00 | 1675.70 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-12-04 13:15:00 | 1676.10 | 2025-12-08 14:15:00 | 1592.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:15:00 | 1676.10 | 2025-12-10 09:15:00 | 1605.70 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-12-31 11:30:00 | 1666.00 | 2026-01-01 15:15:00 | 1690.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-12-31 12:15:00 | 1666.10 | 2026-01-01 15:15:00 | 1690.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1664.30 | 2026-01-01 15:15:00 | 1690.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-01 10:15:00 | 1666.60 | 2026-01-01 15:15:00 | 1690.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-02 10:00:00 | 1673.10 | 2026-01-02 13:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-28 09:15:00 | 1611.30 | 2026-01-29 10:15:00 | 1530.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 09:15:00 | 1611.30 | 2026-01-30 09:15:00 | 1547.10 | STOP_HIT | 0.50 | 3.98% |
| BUY | retest2 | 2026-02-05 09:15:00 | 1680.90 | 2026-02-12 10:15:00 | 1669.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2026-02-16 09:15:00 | 1625.10 | 2026-02-17 13:15:00 | 1640.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2026-02-17 10:30:00 | 1629.10 | 2026-02-17 13:15:00 | 1640.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-02-18 12:30:00 | 1635.00 | 2026-02-19 09:15:00 | 1659.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-02-18 15:00:00 | 1633.10 | 2026-02-19 09:15:00 | 1659.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-03-24 10:30:00 | 1292.50 | 2026-03-24 13:15:00 | 1342.00 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1524.40 | 2026-04-17 11:15:00 | 1498.80 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1530.00 | 2026-04-17 11:15:00 | 1498.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1546.50 | 2026-04-17 11:15:00 | 1498.80 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-04-15 10:45:00 | 1523.60 | 2026-04-17 11:15:00 | 1498.80 | STOP_HIT | 1.00 | -1.63% |
