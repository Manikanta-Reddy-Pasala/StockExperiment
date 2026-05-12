# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1671.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 23 |
| ALERT3 | 137 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 76 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 62
- **Target hits / Stop hits / Partials:** 2 / 77 / 5
- **Avg / median % per leg:** -0.18% / -0.81%
- **Sum % (uncompounded):** -15.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 9 | 22.5% | 0 | 40 | 0 | -0.49% | -19.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.37% | -2.7% |
| BUY @ 3rd Alert (retest2) | 38 | 9 | 23.7% | 0 | 38 | 0 | -0.44% | -16.7% |
| SELL (all) | 44 | 13 | 29.5% | 2 | 37 | 5 | 0.09% | 4.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.97% | -2.0% |
| SELL @ 3rd Alert (retest2) | 43 | 13 | 30.2% | 2 | 36 | 5 | 0.14% | 5.9% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.57% | -4.7% |
| retest2 (combined) | 81 | 22 | 27.2% | 2 | 74 | 5 | -0.13% | -10.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1576.00 | 1548.03 | 1544.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1577.00 | 1557.82 | 1549.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1573.90 | 1575.86 | 1565.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 1573.90 | 1575.86 | 1565.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1601.00 | 1581.66 | 1570.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1611.50 | 1600.30 | 1589.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 1607.80 | 1601.80 | 1591.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 1608.10 | 1616.99 | 1615.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1609.90 | 1614.23 | 1614.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1609.90 | 1614.23 | 1614.74 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1649.80 | 1621.92 | 1618.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 1651.60 | 1634.52 | 1625.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 11:15:00 | 1634.50 | 1640.80 | 1632.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 1634.50 | 1640.80 | 1632.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1632.80 | 1639.20 | 1632.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1632.80 | 1639.20 | 1632.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1620.80 | 1635.52 | 1631.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 1619.50 | 1635.52 | 1631.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1631.20 | 1634.65 | 1631.72 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1615.40 | 1629.80 | 1629.96 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1638.20 | 1628.18 | 1628.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 1643.40 | 1634.76 | 1631.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 1654.50 | 1654.57 | 1646.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:45:00 | 1650.80 | 1654.57 | 1646.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1645.90 | 1654.20 | 1649.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 1645.70 | 1654.20 | 1649.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1646.60 | 1652.68 | 1649.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 1634.20 | 1652.68 | 1649.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 1614.80 | 1645.10 | 1646.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 1607.50 | 1629.85 | 1638.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1579.00 | 1575.17 | 1593.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 1579.00 | 1575.17 | 1593.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1591.50 | 1580.26 | 1592.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 1575.60 | 1580.85 | 1591.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 13:15:00 | 1496.82 | 1532.25 | 1556.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1517.20 | 1516.16 | 1530.48 | SL hit (close>ema200) qty=0.50 sl=1516.16 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1614.00 | 1547.19 | 1542.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1644.50 | 1595.36 | 1571.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 1630.10 | 1631.75 | 1614.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 1613.80 | 1631.75 | 1614.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1630.20 | 1631.44 | 1615.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 1609.60 | 1631.44 | 1615.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1616.20 | 1627.47 | 1616.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:45:00 | 1615.70 | 1627.47 | 1616.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1611.30 | 1624.24 | 1616.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 1613.70 | 1624.24 | 1616.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1602.10 | 1619.81 | 1614.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1602.10 | 1619.81 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1609.00 | 1615.61 | 1613.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1624.70 | 1615.61 | 1613.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1600.50 | 1614.81 | 1613.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1600.50 | 1614.81 | 1613.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 1595.20 | 1610.89 | 1612.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1581.60 | 1602.75 | 1608.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1562.90 | 1560.09 | 1574.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1562.90 | 1560.09 | 1574.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1566.70 | 1564.96 | 1571.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:15:00 | 1569.30 | 1564.96 | 1571.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1560.30 | 1564.02 | 1570.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 1558.00 | 1564.02 | 1570.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 1586.60 | 1562.58 | 1565.76 | SL hit (close>static) qty=1.00 sl=1576.40 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 1582.90 | 1569.40 | 1568.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 13:15:00 | 1587.80 | 1574.90 | 1571.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1577.00 | 1578.58 | 1574.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1577.00 | 1578.58 | 1574.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1563.70 | 1575.61 | 1573.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 1566.90 | 1575.61 | 1573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1547.70 | 1570.02 | 1570.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1543.40 | 1564.70 | 1568.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 1551.50 | 1550.62 | 1557.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:30:00 | 1552.50 | 1550.62 | 1557.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1557.90 | 1552.07 | 1557.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1557.90 | 1552.07 | 1557.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1550.10 | 1551.68 | 1556.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1549.70 | 1551.68 | 1556.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1569.00 | 1556.70 | 1558.08 | SL hit (close>static) qty=1.00 sl=1565.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1572.40 | 1559.84 | 1559.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1587.00 | 1567.65 | 1563.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1600.40 | 1602.44 | 1594.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 1600.40 | 1602.44 | 1594.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1619.90 | 1605.93 | 1596.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:00:00 | 1622.40 | 1610.07 | 1600.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 1621.40 | 1613.86 | 1602.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:45:00 | 1639.20 | 1620.51 | 1608.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 1620.40 | 1631.69 | 1625.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1628.00 | 1630.39 | 1626.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1599.80 | 1630.39 | 1626.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1591.00 | 1622.51 | 1623.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1591.00 | 1622.51 | 1623.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 1585.10 | 1609.91 | 1616.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 1519.40 | 1518.39 | 1529.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:30:00 | 1510.30 | 1517.23 | 1527.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1524.00 | 1519.49 | 1524.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1524.90 | 1519.49 | 1524.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1540.00 | 1523.59 | 1526.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1540.00 | 1523.59 | 1526.18 | SL hit (close>ema400) qty=1.00 sl=1526.18 alert=retest1 |

### Cycle 13 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1545.60 | 1528.00 | 1527.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 1565.10 | 1542.25 | 1535.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1546.00 | 1549.99 | 1541.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 1546.00 | 1549.99 | 1541.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1548.40 | 1548.68 | 1543.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 1543.50 | 1548.68 | 1543.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1530.40 | 1544.60 | 1542.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 1534.80 | 1544.60 | 1542.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1531.90 | 1542.06 | 1541.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1531.90 | 1542.06 | 1541.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1532.30 | 1540.11 | 1540.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 1528.50 | 1534.91 | 1537.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 1533.30 | 1535.47 | 1537.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1549.60 | 1534.34 | 1535.03 | SL hit (close>static) qty=1.00 sl=1547.70 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1546.00 | 1536.67 | 1536.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1556.60 | 1544.09 | 1539.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 1542.50 | 1547.71 | 1543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1542.60 | 1546.69 | 1543.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 1552.70 | 1549.01 | 1544.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1541.50 | 1553.12 | 1553.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1541.50 | 1553.12 | 1553.34 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1565.70 | 1555.64 | 1554.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1572.70 | 1559.05 | 1556.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1559.40 | 1566.15 | 1561.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1559.70 | 1564.86 | 1561.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 1559.70 | 1564.86 | 1561.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1561.50 | 1564.19 | 1561.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 1565.50 | 1564.19 | 1561.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 1566.50 | 1565.61 | 1562.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1552.90 | 1562.86 | 1561.96 | SL hit (close<static) qty=1.00 sl=1556.10 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1553.50 | 1560.98 | 1561.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 1549.70 | 1553.26 | 1556.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1490.50 | 1489.73 | 1502.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:30:00 | 1491.90 | 1489.73 | 1502.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1487.00 | 1487.72 | 1497.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1483.50 | 1487.72 | 1497.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 1480.10 | 1486.68 | 1496.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 1409.33 | 1446.12 | 1461.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 1406.09 | 1446.12 | 1461.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1439.80 | 1434.33 | 1448.89 | SL hit (close>ema200) qty=0.50 sl=1434.33 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1472.80 | 1453.65 | 1453.57 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1455.00 | 1458.84 | 1459.33 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 1464.80 | 1458.72 | 1458.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 11:15:00 | 1472.00 | 1461.38 | 1459.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1470.80 | 1473.34 | 1467.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 1470.80 | 1473.34 | 1467.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1471.20 | 1472.91 | 1467.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1471.20 | 1472.91 | 1467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1469.30 | 1472.19 | 1467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1469.30 | 1472.19 | 1467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1467.20 | 1471.19 | 1467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 1467.20 | 1471.19 | 1467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1464.00 | 1469.75 | 1467.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 1464.00 | 1469.75 | 1467.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1450.30 | 1465.86 | 1465.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1450.30 | 1465.86 | 1465.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1450.00 | 1462.69 | 1464.26 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1479.50 | 1465.34 | 1464.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 1483.30 | 1468.93 | 1466.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 1472.40 | 1475.29 | 1471.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1460.50 | 1472.33 | 1470.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 1460.50 | 1472.33 | 1470.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1463.00 | 1470.46 | 1469.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 1459.40 | 1470.46 | 1469.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 1454.80 | 1466.30 | 1467.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 1450.60 | 1463.16 | 1466.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 1472.50 | 1462.26 | 1465.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1473.00 | 1464.41 | 1465.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 1473.00 | 1464.41 | 1465.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1460.10 | 1464.94 | 1465.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 1459.00 | 1462.76 | 1464.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 1458.70 | 1463.49 | 1464.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1527.40 | 1475.61 | 1469.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1527.40 | 1475.61 | 1469.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1530.10 | 1522.18 | 1509.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 1524.00 | 1526.26 | 1516.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 1524.00 | 1526.26 | 1516.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1519.10 | 1524.55 | 1517.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1520.00 | 1524.55 | 1517.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1520.10 | 1523.33 | 1518.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 1520.40 | 1523.33 | 1518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1514.60 | 1521.58 | 1518.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 1517.60 | 1521.58 | 1518.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1519.00 | 1521.07 | 1518.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1520.90 | 1520.45 | 1518.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:45:00 | 1520.20 | 1520.46 | 1519.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:15:00 | 1520.30 | 1520.46 | 1519.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 1509.00 | 1519.94 | 1519.91 | SL hit (close<static) qty=1.00 sl=1514.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1511.50 | 1518.26 | 1519.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1504.90 | 1515.58 | 1517.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1448.40 | 1448.19 | 1465.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:30:00 | 1449.00 | 1448.19 | 1465.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1451.30 | 1437.70 | 1448.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1451.30 | 1437.70 | 1448.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1452.60 | 1440.68 | 1448.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1451.70 | 1440.68 | 1448.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1450.30 | 1444.64 | 1449.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1438.00 | 1444.64 | 1449.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1447.80 | 1446.08 | 1449.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:15:00 | 1444.60 | 1446.08 | 1449.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1443.90 | 1445.64 | 1448.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 1444.50 | 1445.52 | 1448.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1440.50 | 1445.52 | 1448.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1435.70 | 1443.55 | 1447.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1426.60 | 1439.67 | 1444.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 1425.50 | 1435.30 | 1440.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 1427.60 | 1434.62 | 1440.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1468.80 | 1443.39 | 1442.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1468.80 | 1443.39 | 1442.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 1487.70 | 1469.87 | 1459.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 1491.10 | 1492.38 | 1479.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1497.20 | 1492.38 | 1479.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1510.00 | 1516.93 | 1507.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 1510.00 | 1516.93 | 1507.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1520.20 | 1517.58 | 1508.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1494.30 | 1511.59 | 1508.63 | SL hit (close<ema400) qty=1.00 sl=1508.63 alert=retest1 |

### Cycle 28 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 1597.10 | 1615.89 | 1617.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1589.90 | 1608.57 | 1613.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1586.50 | 1582.17 | 1593.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1586.70 | 1579.95 | 1588.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1586.70 | 1579.95 | 1588.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1590.70 | 1582.10 | 1588.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1587.50 | 1582.10 | 1588.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1594.70 | 1584.62 | 1589.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1589.70 | 1584.62 | 1589.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1585.30 | 1584.75 | 1588.99 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 1604.60 | 1590.60 | 1590.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 1611.90 | 1594.86 | 1592.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1585.20 | 1595.48 | 1593.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1597.60 | 1595.90 | 1593.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:00:00 | 1603.50 | 1597.98 | 1595.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 1602.90 | 1598.83 | 1595.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 1603.90 | 1599.84 | 1596.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1576.00 | 1595.58 | 1595.14 | SL hit (close<static) qty=1.00 sl=1582.10 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 1563.70 | 1589.20 | 1592.29 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 1602.30 | 1589.00 | 1587.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 1611.60 | 1593.52 | 1589.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1613.60 | 1619.04 | 1609.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1608.00 | 1616.83 | 1609.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1608.00 | 1616.83 | 1609.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1608.20 | 1615.10 | 1609.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 1607.00 | 1615.10 | 1609.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1607.50 | 1613.58 | 1609.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:45:00 | 1615.00 | 1610.69 | 1608.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 1654.20 | 1662.01 | 1662.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1654.20 | 1662.01 | 1662.09 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 1668.90 | 1662.74 | 1662.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1687.70 | 1673.26 | 1668.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 1732.80 | 1740.58 | 1721.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1732.80 | 1740.58 | 1721.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1720.90 | 1733.06 | 1723.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1720.90 | 1733.06 | 1723.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1723.40 | 1731.13 | 1723.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:15:00 | 1718.60 | 1731.13 | 1723.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1716.20 | 1728.14 | 1722.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1713.60 | 1728.14 | 1722.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1723.90 | 1727.29 | 1722.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1729.00 | 1727.29 | 1722.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1689.00 | 1719.91 | 1720.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1689.00 | 1719.91 | 1720.26 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1716.90 | 1711.84 | 1711.65 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1694.60 | 1708.96 | 1710.41 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1739.30 | 1713.76 | 1712.26 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1675.70 | 1717.56 | 1721.17 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1739.40 | 1709.08 | 1708.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1752.50 | 1725.56 | 1716.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 1722.20 | 1732.59 | 1722.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1712.40 | 1728.55 | 1721.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1712.40 | 1728.55 | 1721.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1724.80 | 1727.80 | 1722.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 1731.90 | 1729.40 | 1723.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 1729.90 | 1730.57 | 1728.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:30:00 | 1730.00 | 1728.12 | 1727.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1716.00 | 1725.69 | 1726.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1716.00 | 1725.69 | 1726.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 10:15:00 | 1710.60 | 1720.95 | 1723.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1722.00 | 1715.67 | 1718.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1713.20 | 1715.17 | 1718.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 1710.10 | 1714.06 | 1717.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1709.50 | 1713.84 | 1716.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 1705.10 | 1712.09 | 1715.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 1705.50 | 1692.68 | 1692.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1705.50 | 1692.68 | 1692.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1708.00 | 1695.75 | 1693.74 | Break + close above crossover candle high |

### Cycle 42 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 1673.80 | 1691.36 | 1691.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 1665.30 | 1686.15 | 1689.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 1689.50 | 1672.59 | 1679.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1679.50 | 1673.97 | 1679.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:30:00 | 1676.90 | 1673.28 | 1678.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 1676.80 | 1666.89 | 1672.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 1684.90 | 1676.18 | 1675.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1684.90 | 1676.18 | 1675.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1705.10 | 1683.82 | 1679.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 1725.50 | 1733.65 | 1724.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1721.20 | 1730.99 | 1724.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1721.20 | 1730.99 | 1724.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1722.50 | 1729.29 | 1724.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 1720.50 | 1729.29 | 1724.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1723.50 | 1728.74 | 1725.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1727.00 | 1728.74 | 1725.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1718.40 | 1726.67 | 1724.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:45:00 | 1739.00 | 1728.70 | 1725.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1709.50 | 1723.58 | 1724.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1709.50 | 1723.58 | 1724.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1684.90 | 1711.64 | 1717.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1677.30 | 1673.35 | 1686.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 1677.30 | 1673.35 | 1686.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1700.00 | 1679.58 | 1686.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1700.00 | 1679.58 | 1686.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1707.40 | 1685.15 | 1688.24 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1725.50 | 1697.02 | 1693.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 1731.30 | 1708.35 | 1699.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1722.20 | 1722.69 | 1711.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1722.20 | 1722.69 | 1711.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1716.10 | 1721.37 | 1711.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 15:00:00 | 1725.00 | 1722.10 | 1712.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1706.70 | 1719.71 | 1713.41 | SL hit (close<static) qty=1.00 sl=1708.90 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 1720.00 | 1728.60 | 1729.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 1708.00 | 1723.21 | 1726.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1727.50 | 1723.67 | 1726.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1728.70 | 1724.68 | 1726.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1728.70 | 1724.68 | 1726.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1736.40 | 1727.02 | 1727.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1736.40 | 1727.02 | 1727.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1734.00 | 1728.42 | 1728.05 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1712.80 | 1725.29 | 1726.67 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1731.90 | 1727.09 | 1726.82 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1718.30 | 1726.60 | 1726.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1707.00 | 1718.69 | 1722.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1685.30 | 1683.66 | 1697.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1685.30 | 1683.66 | 1697.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1677.80 | 1680.89 | 1690.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 1675.00 | 1678.99 | 1688.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 14:15:00 | 1591.25 | 1622.77 | 1646.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 1683.40 | 1628.85 | 1644.81 | SL hit (close>ema200) qty=0.50 sl=1628.85 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1694.30 | 1661.65 | 1657.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 1701.70 | 1681.03 | 1669.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1697.20 | 1698.86 | 1689.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:30:00 | 1697.10 | 1698.86 | 1689.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1690.30 | 1697.31 | 1690.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1691.10 | 1697.31 | 1690.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1695.30 | 1696.91 | 1690.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1690.50 | 1696.91 | 1690.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1693.60 | 1696.25 | 1691.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 1693.90 | 1696.25 | 1691.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1693.50 | 1695.70 | 1691.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 1694.60 | 1695.70 | 1691.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1693.80 | 1695.32 | 1691.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1693.80 | 1695.32 | 1691.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1698.00 | 1695.86 | 1692.20 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1681.40 | 1689.81 | 1690.53 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 1700.50 | 1692.04 | 1691.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 1704.30 | 1694.49 | 1692.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 1694.10 | 1696.51 | 1694.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1697.50 | 1696.71 | 1694.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 1705.00 | 1698.01 | 1695.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:30:00 | 1705.60 | 1698.81 | 1695.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1703.00 | 1699.44 | 1696.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1703.60 | 1700.28 | 1697.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1720.60 | 1704.34 | 1699.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 1726.00 | 1713.37 | 1705.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1744.60 | 1776.43 | 1779.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1744.60 | 1776.43 | 1779.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1742.20 | 1769.58 | 1775.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 1707.50 | 1706.85 | 1720.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 1707.50 | 1706.85 | 1720.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1697.10 | 1702.75 | 1714.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 1689.10 | 1699.51 | 1710.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1684.50 | 1697.49 | 1707.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1688.20 | 1695.63 | 1706.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1679.50 | 1695.43 | 1705.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1681.70 | 1692.68 | 1702.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 1719.40 | 1699.89 | 1700.35 | SL hit (close>static) qty=1.00 sl=1714.70 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 1705.10 | 1700.93 | 1700.78 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1678.40 | 1697.63 | 1699.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1660.80 | 1682.30 | 1691.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1655.50 | 1641.18 | 1656.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1643.60 | 1641.66 | 1655.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1640.50 | 1641.66 | 1655.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 1664.40 | 1650.06 | 1655.03 | SL hit (close>static) qty=1.00 sl=1659.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1651.60 | 1640.95 | 1640.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1661.10 | 1644.98 | 1642.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 1623.70 | 1643.08 | 1642.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 1627.80 | 1640.03 | 1641.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 1611.30 | 1630.89 | 1636.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1596.90 | 1584.12 | 1601.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1596.90 | 1584.12 | 1601.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1687.10 | 1606.74 | 1608.65 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1679.40 | 1621.27 | 1615.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 1698.10 | 1648.21 | 1629.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 1722.00 | 1723.70 | 1702.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:00:00 | 1722.00 | 1723.70 | 1702.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1711.00 | 1718.40 | 1706.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1708.80 | 1718.40 | 1706.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1733.10 | 1755.34 | 1740.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1733.10 | 1755.34 | 1740.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1747.00 | 1753.67 | 1741.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:45:00 | 1751.90 | 1753.70 | 1742.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 1751.00 | 1752.24 | 1742.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 1725.00 | 1744.99 | 1741.03 | SL hit (close<static) qty=1.00 sl=1728.30 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 1719.40 | 1736.58 | 1737.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1704.70 | 1722.70 | 1729.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 10:15:00 | 1726.50 | 1723.46 | 1729.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 1726.50 | 1723.46 | 1729.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1735.50 | 1725.87 | 1729.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1735.50 | 1725.87 | 1729.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1731.70 | 1727.03 | 1730.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 1738.20 | 1727.03 | 1730.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1729.40 | 1728.09 | 1730.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1729.40 | 1728.09 | 1730.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1738.00 | 1730.08 | 1730.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1718.90 | 1730.08 | 1730.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1723.80 | 1713.90 | 1713.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1723.80 | 1713.90 | 1713.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 1733.80 | 1717.88 | 1715.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1719.30 | 1720.30 | 1716.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1717.90 | 1719.82 | 1716.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 1717.90 | 1719.82 | 1716.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1718.60 | 1719.58 | 1717.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 1717.60 | 1719.58 | 1717.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1712.50 | 1718.16 | 1716.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:45:00 | 1713.20 | 1718.16 | 1716.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1719.00 | 1718.33 | 1716.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:30:00 | 1713.10 | 1718.33 | 1716.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1717.60 | 1718.18 | 1716.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 1717.20 | 1718.18 | 1716.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1716.00 | 1717.75 | 1716.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 1709.20 | 1717.75 | 1716.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1706.00 | 1715.40 | 1715.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 1699.00 | 1712.12 | 1714.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1701.80 | 1683.23 | 1691.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1702.60 | 1687.10 | 1692.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1702.60 | 1687.10 | 1692.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1687.00 | 1687.08 | 1691.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1676.70 | 1685.63 | 1689.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1711.00 | 1690.70 | 1691.31 | SL hit (close>static) qty=1.00 sl=1704.30 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 11:15:00 | 1714.10 | 1695.38 | 1693.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 1720.80 | 1700.47 | 1695.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 1697.80 | 1733.94 | 1723.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1704.90 | 1728.13 | 1722.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1730.80 | 1724.16 | 1721.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1709.40 | 1726.99 | 1728.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1709.40 | 1726.99 | 1728.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1691.70 | 1716.62 | 1722.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 1647.60 | 1660.60 | 1676.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1651.80 | 1657.60 | 1669.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 1482.84 | 1620.90 | 1644.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1474.20 | 1431.54 | 1428.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1476.30 | 1452.66 | 1439.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 1433.60 | 1454.01 | 1444.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1429.30 | 1449.07 | 1442.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 1421.40 | 1449.07 | 1442.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1418.20 | 1437.00 | 1438.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1376.00 | 1421.09 | 1430.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1382.40 | 1377.69 | 1398.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1313.28 | 1362.48 | 1381.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1349.10 | 1344.25 | 1365.38 | SL hit (close>ema200) qty=0.50 sl=1344.25 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1391.50 | 1369.76 | 1368.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1408.90 | 1381.14 | 1374.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1529.00 | 1529.38 | 1498.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1569.80 | 1529.38 | 1498.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1529.80 | 1560.76 | 1536.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1529.80 | 1560.76 | 1536.28 | SL hit (close<ema400) qty=1.00 sl=1536.28 alert=retest1 |

### Cycle 68 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1505.50 | 1524.59 | 1526.09 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1537.40 | 1527.15 | 1527.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1597.30 | 1555.23 | 1543.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1570.50 | 1570.81 | 1556.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 1570.50 | 1570.81 | 1556.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1567.30 | 1569.82 | 1558.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 1570.20 | 1569.82 | 1558.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:45:00 | 1572.90 | 1569.99 | 1559.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1569.00 | 1572.42 | 1565.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1584.90 | 1571.63 | 1568.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1570.50 | 1581.32 | 1576.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:15:00 | 1568.90 | 1581.32 | 1576.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1573.90 | 1579.84 | 1576.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 1564.80 | 1574.02 | 1574.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 1564.80 | 1574.02 | 1574.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1535.50 | 1564.87 | 1570.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1553.50 | 1544.03 | 1551.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1568.00 | 1548.82 | 1553.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 1569.30 | 1548.82 | 1553.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1570.00 | 1553.06 | 1554.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1545.00 | 1553.06 | 1554.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1560.00 | 1551.70 | 1553.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1560.00 | 1551.70 | 1553.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1554.00 | 1552.16 | 1553.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 1549.10 | 1552.31 | 1553.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 1558.80 | 1554.97 | 1554.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1558.80 | 1554.97 | 1554.65 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 1537.40 | 1553.00 | 1554.20 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1565.00 | 1552.19 | 1551.61 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1499.30 | 1541.54 | 1547.12 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 1570.50 | 1544.76 | 1544.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1634.70 | 1562.75 | 1552.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 1687.50 | 1691.82 | 1668.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:45:00 | 1685.00 | 1691.82 | 1668.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1676.30 | 1687.58 | 1675.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:00:00 | 1689.60 | 1685.01 | 1677.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 1671.00 | 1680.34 | 1676.48 | SL hit (close<static) qty=1.00 sl=1672.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 13:00:00 | 1611.50 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-05-15 14:00:00 | 1607.80 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-05-20 13:00:00 | 1608.10 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-06-03 11:15:00 | 1575.60 | 2025-06-04 13:15:00 | 1496.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 11:15:00 | 1575.60 | 2025-06-06 09:15:00 | 1517.20 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-06-17 11:15:00 | 1558.00 | 2025-06-18 09:15:00 | 1586.60 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1549.70 | 2025-06-23 10:15:00 | 1569.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-26 14:00:00 | 1622.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-06-26 14:30:00 | 1621.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-27 10:45:00 | 1639.20 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-06-30 13:45:00 | 1620.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest1 | 2025-07-08 10:30:00 | 1510.30 | 2025-07-09 09:15:00 | 1540.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-14 11:45:00 | 1533.30 | 2025-07-15 10:15:00 | 1549.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-16 13:30:00 | 1552.70 | 2025-07-21 09:15:00 | 1541.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-22 12:15:00 | 1565.50 | 2025-07-23 09:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-22 15:00:00 | 1566.50 | 2025-07-23 09:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1483.50 | 2025-08-01 11:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:45:00 | 1480.10 | 2025-08-01 11:15:00 | 1406.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1483.50 | 2025-08-04 09:15:00 | 1439.80 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2025-07-30 10:45:00 | 1480.10 | 2025-08-04 09:15:00 | 1439.80 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-08-14 12:30:00 | 1459.00 | 2025-08-18 09:15:00 | 1527.40 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2025-08-14 13:45:00 | 1458.70 | 2025-08-18 09:15:00 | 1527.40 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1520.90 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-08-22 13:45:00 | 1520.20 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-22 14:15:00 | 1520.30 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-02 11:15:00 | 1444.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1443.90 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1444.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1440.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1426.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-09-03 13:15:00 | 1425.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-09-03 13:45:00 | 1427.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2025-09-09 09:15:00 | 1497.20 | 2025-09-11 14:15:00 | 1494.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1532.00 | 2025-09-25 12:15:00 | 1597.10 | STOP_HIT | 1.00 | 4.25% |
| BUY | retest2 | 2025-10-01 13:00:00 | 1603.50 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-01 13:30:00 | 1602.90 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-01 15:00:00 | 1603.90 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-09 10:45:00 | 1615.00 | 2025-10-17 14:15:00 | 1654.20 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1729.00 | 2025-10-29 09:15:00 | 1689.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-11-11 12:30:00 | 1731.90 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-12 14:00:00 | 1729.90 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-12 14:30:00 | 1730.00 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-14 12:45:00 | 1710.10 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1709.50 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-11-17 10:00:00 | 1705.10 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-11-24 11:30:00 | 1676.90 | 2025-11-25 14:15:00 | 1684.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1676.80 | 2025-11-25 14:15:00 | 1684.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-02 10:45:00 | 1739.00 | 2025-12-02 12:15:00 | 1709.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-08 15:00:00 | 1725.00 | 2025-12-09 09:15:00 | 1706.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-09 13:00:00 | 1726.50 | 2025-12-11 15:15:00 | 1720.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1675.00 | 2025-12-22 14:15:00 | 1591.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1675.00 | 2025-12-23 09:15:00 | 1683.40 | STOP_HIT | 0.50 | -0.50% |
| BUY | retest2 | 2025-12-31 13:30:00 | 1705.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-12-31 14:30:00 | 1705.60 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2026-01-01 09:15:00 | 1703.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2026-01-01 10:00:00 | 1703.60 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2026-01-01 15:00:00 | 1726.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2026-01-14 12:00:00 | 1689.10 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1684.50 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1688.20 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1679.50 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1640.50 | 2026-01-22 14:15:00 | 1664.40 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-27 10:45:00 | 1640.10 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1635.00 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-27 13:00:00 | 1633.80 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-28 14:30:00 | 1632.30 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1614.70 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-01-29 11:30:00 | 1625.50 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-10 11:45:00 | 1751.90 | 2026-02-10 14:15:00 | 1725.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-02-10 12:45:00 | 1751.00 | 2026-02-10 14:15:00 | 1725.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1718.90 | 2026-02-17 13:15:00 | 1723.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-02-24 09:30:00 | 1676.70 | 2026-02-24 10:15:00 | 1711.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1730.80 | 2026-03-02 10:15:00 | 1709.40 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-05 12:45:00 | 1647.60 | 2026-03-09 09:15:00 | 1482.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1651.80 | 2026-03-09 09:15:00 | 1486.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1382.40 | 2026-04-02 09:15:00 | 1313.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1382.40 | 2026-04-02 13:15:00 | 1349.10 | STOP_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1569.80 | 2026-04-13 09:15:00 | 1529.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-04-17 10:15:00 | 1570.20 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-17 10:45:00 | 1572.90 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-04-20 10:00:00 | 1569.00 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1584.90 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1549.10 | 2026-04-27 14:15:00 | 1558.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-05-08 14:00:00 | 1689.60 | 2026-05-08 15:15:00 | 1671.00 | STOP_HIT | 1.00 | -1.10% |
