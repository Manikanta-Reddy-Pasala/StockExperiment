# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1659.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 102 |
| ALERT2 | 99 |
| ALERT2_SKIP | 61 |
| ALERT3 | 287 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 129 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 152 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 63 / 89
- **Target hits / Stop hits / Partials:** 7 / 126 / 19
- **Avg / median % per leg:** 1.09% / -0.58%
- **Sum % (uncompounded):** 165.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 16 | 25.8% | 3 | 59 | 0 | -0.02% | -0.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.71% | -1.7% |
| BUY @ 3rd Alert (retest2) | 61 | 16 | 26.2% | 3 | 58 | 0 | 0.01% | 0.8% |
| SELL (all) | 90 | 47 | 52.2% | 4 | 67 | 19 | 1.85% | 166.9% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.45% | 12.3% |
| SELL @ 3rd Alert (retest2) | 85 | 45 | 52.9% | 3 | 64 | 18 | 1.82% | 154.6% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.76% | 10.5% |
| retest2 (combined) | 146 | 61 | 41.8% | 6 | 122 | 18 | 1.06% | 155.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1525.90 | 1544.15 | 1544.41 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1572.25 | 1542.49 | 1539.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 1578.00 | 1549.59 | 1543.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1550.15 | 1556.33 | 1549.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 1550.15 | 1556.33 | 1549.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1550.15 | 1556.33 | 1549.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 1550.15 | 1556.33 | 1549.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1553.95 | 1555.86 | 1549.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 1541.25 | 1552.94 | 1548.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1529.65 | 1548.29 | 1547.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 1529.65 | 1548.29 | 1547.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 1532.25 | 1545.08 | 1545.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 14:15:00 | 1527.15 | 1538.07 | 1542.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 1547.35 | 1538.32 | 1541.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1547.35 | 1538.32 | 1541.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1547.35 | 1538.32 | 1541.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 1549.70 | 1538.32 | 1541.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1550.00 | 1540.65 | 1542.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 1555.60 | 1540.65 | 1542.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 1542.00 | 1542.33 | 1542.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 14:15:00 | 1534.05 | 1542.33 | 1542.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 1547.50 | 1543.57 | 1543.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 1547.50 | 1543.57 | 1543.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 1569.90 | 1548.83 | 1545.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 1540.15 | 1549.40 | 1546.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 12:15:00 | 1540.15 | 1549.40 | 1546.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 1540.15 | 1549.40 | 1546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 1553.00 | 1549.40 | 1546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1551.90 | 1549.90 | 1547.08 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 1537.40 | 1545.36 | 1545.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 1535.00 | 1543.29 | 1544.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 1588.50 | 1517.40 | 1523.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 1588.50 | 1517.40 | 1523.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1588.50 | 1517.40 | 1523.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 1588.50 | 1517.40 | 1523.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1542.00 | 1522.32 | 1525.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 15:15:00 | 1520.00 | 1522.32 | 1525.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 12:15:00 | 1444.00 | 1477.41 | 1495.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 1475.05 | 1471.29 | 1486.07 | SL hit (close>ema200) qty=0.50 sl=1471.29 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 1491.15 | 1487.18 | 1487.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 14:15:00 | 1496.80 | 1489.11 | 1487.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1489.80 | 1490.99 | 1489.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1489.80 | 1490.99 | 1489.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1489.80 | 1490.99 | 1489.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:45:00 | 1486.70 | 1490.99 | 1489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1491.00 | 1490.99 | 1489.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 11:15:00 | 1502.80 | 1490.99 | 1489.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 11:15:00 | 1498.30 | 1504.59 | 1499.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 12:15:00 | 1495.70 | 1502.58 | 1498.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:00:00 | 1495.25 | 1501.12 | 1498.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1486.45 | 1497.38 | 1497.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1486.45 | 1497.38 | 1497.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 1480.00 | 1493.90 | 1495.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 1480.00 | 1493.90 | 1495.69 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1510.60 | 1499.36 | 1497.99 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1442.50 | 1490.16 | 1495.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1365.40 | 1465.20 | 1483.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 1454.00 | 1416.02 | 1442.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 11:15:00 | 1454.00 | 1416.02 | 1442.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1454.00 | 1416.02 | 1442.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 1454.00 | 1416.02 | 1442.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1452.45 | 1423.30 | 1443.33 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1485.95 | 1453.44 | 1452.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1530.50 | 1468.86 | 1459.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 1681.00 | 1684.71 | 1653.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 11:00:00 | 1681.00 | 1684.71 | 1653.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1678.45 | 1682.06 | 1662.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:30:00 | 1660.80 | 1682.06 | 1662.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1672.50 | 1680.15 | 1663.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 1657.40 | 1675.94 | 1662.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1654.70 | 1671.69 | 1662.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:30:00 | 1656.65 | 1671.69 | 1662.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1655.00 | 1668.35 | 1661.44 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 09:15:00 | 1655.30 | 1658.34 | 1658.45 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 1679.50 | 1659.06 | 1658.31 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 1648.15 | 1663.10 | 1663.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 1637.50 | 1653.66 | 1659.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 1631.20 | 1628.81 | 1635.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 1631.20 | 1628.81 | 1635.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1631.20 | 1628.81 | 1635.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 1631.15 | 1628.81 | 1635.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 1632.25 | 1629.85 | 1635.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 1635.00 | 1629.85 | 1635.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1607.80 | 1625.44 | 1632.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:15:00 | 1600.00 | 1624.24 | 1628.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 1634.00 | 1619.73 | 1624.24 | SL hit (close>static) qty=1.00 sl=1633.70 alert=retest2 |

### Cycle 14 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 1631.55 | 1626.41 | 1625.88 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1595.60 | 1621.67 | 1624.00 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 1642.00 | 1623.22 | 1622.87 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 1620.90 | 1628.72 | 1628.87 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1640.75 | 1629.36 | 1628.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 11:15:00 | 1652.90 | 1634.48 | 1631.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 15:15:00 | 1740.00 | 1745.08 | 1716.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 09:15:00 | 1729.05 | 1745.08 | 1716.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1714.75 | 1739.02 | 1716.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1710.00 | 1739.02 | 1716.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1710.35 | 1733.28 | 1715.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1710.35 | 1733.28 | 1715.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1704.90 | 1727.61 | 1714.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1704.90 | 1727.61 | 1714.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1796.95 | 1809.80 | 1799.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 1798.85 | 1809.80 | 1799.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1799.00 | 1807.64 | 1799.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 1831.00 | 1807.64 | 1799.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 1788.00 | 1804.29 | 1800.80 | SL hit (close<static) qty=1.00 sl=1791.05 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 1788.95 | 1796.89 | 1797.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 1762.35 | 1789.98 | 1794.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 1785.35 | 1779.56 | 1787.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 13:15:00 | 1785.35 | 1779.56 | 1787.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1785.35 | 1779.56 | 1787.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:30:00 | 1788.25 | 1779.56 | 1787.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1801.55 | 1783.96 | 1788.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 1801.55 | 1783.96 | 1788.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1792.05 | 1785.58 | 1788.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1781.30 | 1785.58 | 1788.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1788.95 | 1786.17 | 1788.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:15:00 | 1793.60 | 1786.17 | 1788.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 1787.00 | 1786.34 | 1788.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:15:00 | 1783.00 | 1786.34 | 1788.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 14:15:00 | 1800.20 | 1783.66 | 1786.15 | SL hit (close>static) qty=1.00 sl=1798.75 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 1806.05 | 1790.75 | 1789.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 11:15:00 | 1823.15 | 1798.73 | 1793.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 13:15:00 | 1801.00 | 1801.66 | 1795.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:00:00 | 1801.00 | 1801.66 | 1795.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 21 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1734.00 | 1793.34 | 1793.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1725.60 | 1779.80 | 1787.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 1749.95 | 1746.63 | 1762.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 1749.95 | 1746.63 | 1762.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1749.95 | 1746.63 | 1762.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1749.55 | 1746.63 | 1762.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1736.55 | 1744.97 | 1759.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:30:00 | 1748.00 | 1744.97 | 1759.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1747.90 | 1713.93 | 1726.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1747.90 | 1713.93 | 1726.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1754.90 | 1722.12 | 1729.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1754.90 | 1722.12 | 1729.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1756.30 | 1735.82 | 1734.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 1785.10 | 1751.76 | 1743.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 1795.75 | 1798.15 | 1780.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 13:45:00 | 1805.95 | 1797.94 | 1782.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 1775.00 | 1793.35 | 1781.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-29 14:15:00 | 1775.00 | 1793.35 | 1781.53 | SL hit (close<ema400) qty=1.00 sl=1781.53 alert=retest1 |

### Cycle 23 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1843.95 | 1872.22 | 1873.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 1814.30 | 1860.64 | 1868.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1814.05 | 1786.94 | 1804.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 13:15:00 | 1814.05 | 1786.94 | 1804.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1814.05 | 1786.94 | 1804.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1837.35 | 1786.94 | 1804.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1788.05 | 1787.16 | 1802.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 11:00:00 | 1781.35 | 1788.47 | 1799.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:45:00 | 1785.95 | 1789.08 | 1798.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:15:00 | 1785.00 | 1789.36 | 1797.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1785.00 | 1790.81 | 1796.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1790.00 | 1790.65 | 1796.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:45:00 | 1755.90 | 1777.51 | 1785.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1692.28 | 1720.21 | 1742.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1696.65 | 1720.21 | 1742.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1695.75 | 1720.21 | 1742.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1695.75 | 1720.21 | 1742.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 11:15:00 | 1668.11 | 1703.54 | 1730.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 1693.95 | 1684.14 | 1697.41 | SL hit (close>ema200) qty=0.50 sl=1684.14 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 1723.00 | 1700.15 | 1698.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 1736.15 | 1721.99 | 1711.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 1740.00 | 1740.99 | 1727.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:15:00 | 1744.25 | 1740.99 | 1727.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1738.10 | 1740.41 | 1728.49 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1715.70 | 1727.70 | 1727.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 1701.75 | 1717.95 | 1722.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 1693.45 | 1690.09 | 1700.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 1693.45 | 1690.09 | 1700.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1693.45 | 1690.09 | 1700.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 1701.70 | 1690.09 | 1700.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1697.05 | 1692.27 | 1699.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:45:00 | 1684.55 | 1691.07 | 1698.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:30:00 | 1685.00 | 1688.26 | 1696.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:00:00 | 1680.05 | 1680.51 | 1688.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 15:00:00 | 1679.95 | 1680.26 | 1685.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1680.15 | 1680.20 | 1684.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-30 12:15:00 | 1703.60 | 1689.39 | 1688.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 1703.60 | 1689.39 | 1688.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1706.95 | 1694.79 | 1690.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 1696.45 | 1698.70 | 1694.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 12:15:00 | 1696.45 | 1698.70 | 1694.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1696.45 | 1698.70 | 1694.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 1696.45 | 1698.70 | 1694.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1690.35 | 1697.03 | 1694.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 1690.35 | 1697.03 | 1694.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1693.50 | 1696.32 | 1694.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:15:00 | 1693.40 | 1696.32 | 1694.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1693.40 | 1695.74 | 1694.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1721.35 | 1695.74 | 1694.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1699.50 | 1707.82 | 1704.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 11:30:00 | 1698.70 | 1705.73 | 1703.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:00:00 | 1697.40 | 1705.73 | 1703.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1697.15 | 1703.09 | 1702.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 1697.15 | 1703.09 | 1702.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 1700.00 | 1702.47 | 1702.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 1700.00 | 1702.47 | 1702.63 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 15:15:00 | 1705.00 | 1702.98 | 1702.84 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 1696.05 | 1701.59 | 1702.23 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1711.55 | 1703.28 | 1702.86 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 1695.60 | 1707.09 | 1707.90 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1781.10 | 1722.23 | 1714.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 1815.00 | 1784.15 | 1759.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 1783.05 | 1795.72 | 1773.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:00:00 | 1783.05 | 1795.72 | 1773.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1775.30 | 1789.54 | 1774.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 1773.90 | 1789.54 | 1774.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1772.00 | 1786.03 | 1774.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 1772.00 | 1786.03 | 1774.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 1775.10 | 1783.84 | 1774.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1780.05 | 1780.34 | 1774.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 1796.30 | 1836.57 | 1840.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1796.30 | 1836.57 | 1840.22 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 1861.05 | 1836.82 | 1836.16 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 1827.00 | 1839.13 | 1839.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 1821.65 | 1835.63 | 1838.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 15:15:00 | 1825.00 | 1823.34 | 1827.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 15:15:00 | 1825.00 | 1823.34 | 1827.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1825.00 | 1823.34 | 1827.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1822.80 | 1823.34 | 1827.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1805.10 | 1819.69 | 1825.43 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 1829.00 | 1819.56 | 1819.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 1841.05 | 1823.86 | 1821.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 1933.85 | 1938.05 | 1905.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 13:45:00 | 1931.15 | 1938.05 | 1905.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1932.15 | 1946.90 | 1931.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 1903.90 | 1935.32 | 1927.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1857.30 | 1919.72 | 1921.26 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 1943.40 | 1913.50 | 1910.51 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1919.00 | 1932.22 | 1932.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 1914.85 | 1926.79 | 1930.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1901.20 | 1885.55 | 1896.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1901.20 | 1885.55 | 1896.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1901.20 | 1885.55 | 1896.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 1906.30 | 1885.55 | 1896.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1946.45 | 1897.73 | 1900.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1946.45 | 1897.73 | 1900.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1941.05 | 1906.40 | 1904.36 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 1897.00 | 1910.76 | 1912.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1865.70 | 1894.24 | 1903.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1819.05 | 1800.89 | 1819.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1819.05 | 1800.89 | 1819.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1819.05 | 1800.89 | 1819.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:15:00 | 1830.45 | 1800.89 | 1819.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1838.25 | 1808.36 | 1821.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1838.25 | 1808.36 | 1821.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1861.65 | 1819.02 | 1825.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1852.60 | 1819.02 | 1825.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 13:15:00 | 1853.40 | 1833.05 | 1830.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 15:15:00 | 1871.70 | 1845.29 | 1837.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 09:15:00 | 1818.05 | 1839.85 | 1835.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 1818.05 | 1839.85 | 1835.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1818.05 | 1839.85 | 1835.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 1818.05 | 1839.85 | 1835.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1823.95 | 1836.67 | 1834.32 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 1816.70 | 1832.67 | 1832.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 1801.90 | 1826.52 | 1829.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 1794.20 | 1789.14 | 1801.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 11:00:00 | 1794.20 | 1789.14 | 1801.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1820.40 | 1795.39 | 1802.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:00:00 | 1820.40 | 1795.39 | 1802.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 1834.50 | 1803.21 | 1805.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 1831.20 | 1803.21 | 1805.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 1833.95 | 1809.36 | 1808.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 1846.50 | 1816.79 | 1811.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 1902.00 | 1908.28 | 1890.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 1902.00 | 1908.28 | 1890.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1887.10 | 1904.04 | 1890.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1887.10 | 1904.04 | 1890.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1952.60 | 1913.75 | 1895.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 1979.55 | 1913.75 | 1895.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 1970.60 | 1935.08 | 1909.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 1963.00 | 1946.78 | 1919.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 11:15:00 | 2177.51 | 2077.14 | 2014.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 2121.00 | 2159.37 | 2163.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 2104.00 | 2148.30 | 2158.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2111.95 | 2080.25 | 2105.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2111.95 | 2080.25 | 2105.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2111.95 | 2080.25 | 2105.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:15:00 | 2157.70 | 2080.25 | 2105.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 2149.15 | 2094.03 | 2109.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 2136.05 | 2094.03 | 2109.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1985.70 | 2084.76 | 2102.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 1947.45 | 2051.85 | 2086.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-18 09:15:00 | 1752.71 | 1964.79 | 2039.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 13:15:00 | 1775.85 | 1731.47 | 1731.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 1788.55 | 1742.89 | 1736.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 15:15:00 | 1775.10 | 1776.64 | 1761.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:15:00 | 1777.05 | 1776.64 | 1761.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1757.25 | 1772.76 | 1761.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 1757.25 | 1772.76 | 1761.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1719.90 | 1762.19 | 1757.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 1719.90 | 1762.19 | 1757.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1738.70 | 1757.49 | 1755.81 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1720.50 | 1750.09 | 1752.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 13:15:00 | 1714.00 | 1730.80 | 1740.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1721.65 | 1712.24 | 1723.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 1721.65 | 1712.24 | 1723.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1721.65 | 1712.24 | 1723.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1721.65 | 1712.24 | 1723.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1728.05 | 1715.40 | 1723.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1709.45 | 1715.40 | 1723.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 12:15:00 | 1623.98 | 1641.43 | 1662.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 1616.05 | 1610.06 | 1630.94 | SL hit (close>ema200) qty=0.50 sl=1610.06 alert=retest2 |

### Cycle 48 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1631.80 | 1612.07 | 1610.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 09:15:00 | 1662.05 | 1626.32 | 1617.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 1647.85 | 1651.17 | 1636.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 1647.85 | 1651.17 | 1636.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1647.85 | 1651.17 | 1636.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 13:00:00 | 1672.05 | 1658.09 | 1648.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 1693.65 | 1679.95 | 1668.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 1672.50 | 1685.34 | 1679.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 1660.55 | 1674.15 | 1675.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 1660.55 | 1674.15 | 1675.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 12:15:00 | 1653.05 | 1666.93 | 1671.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1603.05 | 1592.81 | 1613.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1603.05 | 1592.81 | 1613.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1603.05 | 1592.81 | 1613.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1603.05 | 1592.81 | 1613.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1594.80 | 1592.11 | 1606.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:30:00 | 1585.45 | 1592.01 | 1604.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 1584.00 | 1592.01 | 1604.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1580.25 | 1587.52 | 1600.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1561.50 | 1557.42 | 1556.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1561.50 | 1557.42 | 1556.93 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 1549.75 | 1556.34 | 1556.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 13:15:00 | 1546.40 | 1551.33 | 1553.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1551.40 | 1547.97 | 1551.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 1551.40 | 1547.97 | 1551.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1551.40 | 1547.97 | 1551.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 1551.40 | 1547.97 | 1551.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1547.05 | 1547.79 | 1550.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:15:00 | 1542.80 | 1547.79 | 1550.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 1557.95 | 1548.19 | 1548.34 | SL hit (close>static) qty=1.00 sl=1552.25 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 13:15:00 | 1554.35 | 1549.42 | 1548.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 15:15:00 | 1565.00 | 1552.92 | 1550.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 1566.50 | 1567.52 | 1560.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 1566.50 | 1567.52 | 1560.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1565.25 | 1567.07 | 1561.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1570.30 | 1567.07 | 1561.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1572.45 | 1568.14 | 1562.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 1580.15 | 1570.66 | 1564.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:45:00 | 1581.00 | 1575.88 | 1568.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 11:30:00 | 1585.20 | 1582.25 | 1574.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 15:15:00 | 1550.90 | 1572.76 | 1572.20 | SL hit (close<static) qty=1.00 sl=1552.35 alert=retest2 |

### Cycle 53 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1636.90 | 1640.99 | 1641.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1631.80 | 1638.56 | 1640.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1607.00 | 1605.07 | 1618.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1607.00 | 1605.07 | 1618.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1615.75 | 1605.44 | 1616.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1615.75 | 1605.44 | 1616.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1618.90 | 1608.13 | 1616.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1618.80 | 1608.13 | 1616.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1610.00 | 1608.50 | 1616.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 1607.10 | 1609.68 | 1615.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 1526.74 | 1563.11 | 1585.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 11:15:00 | 1446.39 | 1509.07 | 1550.33 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1504.15 | 1485.05 | 1483.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1580.80 | 1530.13 | 1518.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1574.15 | 1607.90 | 1573.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1574.15 | 1607.90 | 1573.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1574.15 | 1607.90 | 1573.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1569.45 | 1607.90 | 1573.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1574.45 | 1601.21 | 1573.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1574.45 | 1601.21 | 1573.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1587.15 | 1598.40 | 1574.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:30:00 | 1561.55 | 1598.40 | 1574.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1574.85 | 1593.69 | 1574.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 1574.35 | 1593.69 | 1574.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1576.10 | 1590.17 | 1575.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 1576.10 | 1590.17 | 1575.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1599.20 | 1591.98 | 1577.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 1611.00 | 1591.98 | 1577.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 1562.35 | 1589.10 | 1578.70 | SL hit (close<static) qty=1.00 sl=1573.85 alert=retest2 |

### Cycle 55 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 1552.20 | 1571.67 | 1572.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 1541.35 | 1565.61 | 1569.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 15:15:00 | 1523.30 | 1520.08 | 1537.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1491.30 | 1520.08 | 1537.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 1514.85 | 1503.81 | 1517.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 1483.75 | 1503.81 | 1517.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1477.40 | 1498.53 | 1514.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 1466.95 | 1489.91 | 1504.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 11:15:00 | 1416.73 | 1448.95 | 1476.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 1393.60 | 1417.56 | 1449.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 15:15:00 | 1342.17 | 1372.57 | 1409.94 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 56 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 1387.10 | 1359.26 | 1357.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 10:15:00 | 1395.50 | 1376.00 | 1367.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 1359.80 | 1383.96 | 1376.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 1359.80 | 1383.96 | 1376.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1359.80 | 1383.96 | 1376.83 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 1347.85 | 1369.52 | 1372.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 1338.55 | 1363.33 | 1368.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 1281.65 | 1278.34 | 1301.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 12:00:00 | 1281.65 | 1278.34 | 1301.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 1280.15 | 1278.70 | 1299.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 1280.15 | 1278.70 | 1299.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 1305.50 | 1284.06 | 1299.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:00:00 | 1305.50 | 1284.06 | 1299.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1323.45 | 1291.94 | 1301.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 1315.55 | 1291.94 | 1301.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1281.10 | 1276.47 | 1286.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 1288.75 | 1276.47 | 1286.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1280.80 | 1277.33 | 1285.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 1280.80 | 1277.33 | 1285.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 1269.75 | 1275.82 | 1284.18 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 1302.50 | 1286.86 | 1285.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1309.90 | 1293.91 | 1288.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1340.25 | 1350.19 | 1336.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1340.25 | 1350.19 | 1336.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1340.25 | 1350.19 | 1336.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1340.25 | 1350.19 | 1336.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1339.00 | 1347.15 | 1338.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 1339.00 | 1347.15 | 1338.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1336.05 | 1344.93 | 1338.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 1336.05 | 1344.93 | 1338.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1327.00 | 1341.34 | 1337.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 1333.40 | 1341.34 | 1337.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1310.00 | 1334.06 | 1334.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 1298.10 | 1315.92 | 1323.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 1312.00 | 1304.54 | 1313.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 15:15:00 | 1312.00 | 1304.54 | 1313.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1312.00 | 1304.54 | 1313.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1298.10 | 1304.54 | 1313.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 1301.95 | 1303.92 | 1311.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 1294.40 | 1289.82 | 1299.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 1331.15 | 1302.48 | 1300.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1331.15 | 1302.48 | 1300.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 1341.05 | 1315.23 | 1307.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 09:15:00 | 1459.40 | 1484.88 | 1461.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 1459.40 | 1484.88 | 1461.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1459.40 | 1484.88 | 1461.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 1459.45 | 1484.88 | 1461.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1462.85 | 1480.47 | 1461.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 1471.45 | 1462.84 | 1460.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:15:00 | 1474.45 | 1462.41 | 1460.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 1456.90 | 1461.30 | 1459.92 | SL hit (close<static) qty=1.00 sl=1459.40 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 1453.70 | 1458.87 | 1459.06 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1473.20 | 1461.73 | 1460.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 1482.90 | 1465.97 | 1462.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1468.90 | 1471.76 | 1466.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1468.90 | 1471.76 | 1466.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1468.90 | 1471.76 | 1466.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 1472.00 | 1464.74 | 1464.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 12:15:00 | 1454.95 | 1462.82 | 1463.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 12:15:00 | 1454.95 | 1462.82 | 1463.79 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1468.40 | 1464.04 | 1463.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 1469.75 | 1466.54 | 1465.00 | Break + close above crossover candle high |

### Cycle 65 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1444.00 | 1462.03 | 1463.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 1437.25 | 1457.07 | 1460.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1362.10 | 1351.82 | 1380.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1362.10 | 1351.82 | 1380.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1376.10 | 1357.65 | 1378.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 1376.10 | 1357.65 | 1378.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1379.20 | 1361.96 | 1378.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 1379.20 | 1361.96 | 1378.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1388.85 | 1367.34 | 1379.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 1388.25 | 1367.34 | 1379.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1360.85 | 1372.54 | 1379.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 1359.60 | 1372.54 | 1379.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 1360.75 | 1363.10 | 1372.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 15:15:00 | 1410.00 | 1377.70 | 1377.71 | SL hit (close>static) qty=1.00 sl=1399.85 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1440.00 | 1390.16 | 1383.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 1454.60 | 1418.04 | 1399.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1446.20 | 1456.94 | 1450.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 14:15:00 | 1446.20 | 1456.94 | 1450.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1446.20 | 1456.94 | 1450.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1446.20 | 1456.94 | 1450.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1440.00 | 1453.55 | 1449.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1432.40 | 1453.55 | 1449.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1441.40 | 1451.12 | 1448.61 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 12:15:00 | 1436.50 | 1445.16 | 1446.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 13:15:00 | 1425.20 | 1441.17 | 1444.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 1440.00 | 1406.21 | 1413.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 1440.00 | 1406.21 | 1413.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1440.00 | 1406.21 | 1413.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 1440.00 | 1406.21 | 1413.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1433.30 | 1411.63 | 1415.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 11:15:00 | 1432.80 | 1411.63 | 1415.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 1430.90 | 1418.66 | 1417.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 1430.90 | 1418.66 | 1417.97 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 1405.00 | 1415.64 | 1416.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1361.60 | 1404.84 | 1411.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 12:15:00 | 1406.10 | 1396.82 | 1405.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 12:15:00 | 1406.10 | 1396.82 | 1405.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 1406.10 | 1396.82 | 1405.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:00:00 | 1406.10 | 1396.82 | 1405.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 1399.60 | 1397.37 | 1404.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:45:00 | 1393.40 | 1397.36 | 1404.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 1382.30 | 1397.67 | 1403.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:00:00 | 1390.50 | 1398.10 | 1402.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:00:00 | 1395.00 | 1397.48 | 1401.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1394.10 | 1395.28 | 1399.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 1387.80 | 1393.64 | 1397.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 1387.90 | 1392.49 | 1396.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:00:00 | 1388.10 | 1391.53 | 1395.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 1323.73 | 1355.81 | 1372.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 1325.25 | 1355.81 | 1372.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:15:00 | 1313.18 | 1342.84 | 1363.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:15:00 | 1320.97 | 1342.84 | 1363.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:15:00 | 1318.41 | 1342.84 | 1363.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:15:00 | 1318.51 | 1342.84 | 1363.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:15:00 | 1318.69 | 1342.84 | 1363.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 1300.00 | 1298.96 | 1321.85 | SL hit (close>ema200) qty=0.50 sl=1298.96 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 1311.00 | 1304.27 | 1304.09 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1295.00 | 1303.21 | 1303.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 1280.80 | 1298.73 | 1301.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1348.90 | 1287.53 | 1288.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1348.90 | 1287.53 | 1288.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1348.90 | 1287.53 | 1288.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 1354.30 | 1287.53 | 1288.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1345.50 | 1299.12 | 1294.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1397.20 | 1337.49 | 1317.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 1458.70 | 1484.24 | 1450.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 1458.70 | 1484.24 | 1450.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1458.70 | 1484.24 | 1450.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 1439.90 | 1484.24 | 1450.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1452.80 | 1477.96 | 1450.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1452.80 | 1477.96 | 1450.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1450.40 | 1472.44 | 1450.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 1459.00 | 1472.44 | 1450.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 1462.60 | 1461.94 | 1451.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 1447.40 | 1458.73 | 1455.08 | SL hit (close<static) qty=1.00 sl=1448.90 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1456.60 | 1465.39 | 1466.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 1450.00 | 1462.32 | 1464.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 1470.40 | 1463.93 | 1465.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1470.40 | 1463.93 | 1465.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1470.40 | 1463.93 | 1465.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1470.40 | 1463.93 | 1465.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1504.00 | 1471.95 | 1468.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1515.10 | 1480.58 | 1472.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1620.60 | 1639.29 | 1620.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1620.60 | 1639.29 | 1620.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1620.60 | 1639.29 | 1620.41 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 1587.40 | 1608.78 | 1610.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 1585.10 | 1595.82 | 1599.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 1600.00 | 1596.66 | 1599.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 1600.00 | 1596.66 | 1599.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1600.00 | 1596.66 | 1599.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 1600.30 | 1596.66 | 1599.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1596.00 | 1596.53 | 1599.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 1594.00 | 1596.02 | 1598.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1604.70 | 1597.76 | 1599.27 | SL hit (close>static) qty=1.00 sl=1603.90 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 1625.60 | 1603.62 | 1601.69 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 1595.10 | 1603.06 | 1604.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 1590.50 | 1600.55 | 1602.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 1574.90 | 1574.26 | 1582.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:30:00 | 1573.40 | 1574.26 | 1582.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1571.50 | 1573.71 | 1581.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:30:00 | 1565.40 | 1572.46 | 1580.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1558.00 | 1572.85 | 1579.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 10:15:00 | 1487.13 | 1509.04 | 1529.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 10:15:00 | 1480.10 | 1509.04 | 1529.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1504.90 | 1503.86 | 1520.33 | SL hit (close>ema200) qty=0.50 sl=1503.86 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1502.30 | 1493.07 | 1492.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1510.00 | 1496.90 | 1494.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1482.20 | 1495.27 | 1494.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1482.20 | 1495.27 | 1494.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1482.20 | 1495.27 | 1494.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1480.50 | 1495.27 | 1494.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 1476.90 | 1491.60 | 1492.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 12:15:00 | 1472.00 | 1485.37 | 1489.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 1494.60 | 1484.10 | 1487.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1494.60 | 1484.10 | 1487.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1494.60 | 1484.10 | 1487.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 1487.20 | 1484.10 | 1487.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1487.40 | 1481.75 | 1484.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 1509.10 | 1481.99 | 1480.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1509.10 | 1481.99 | 1480.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1520.10 | 1489.61 | 1483.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1553.50 | 1559.30 | 1544.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 1553.50 | 1559.30 | 1544.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1557.80 | 1559.00 | 1545.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 1545.10 | 1559.00 | 1545.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1725.60 | 1741.23 | 1730.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1725.60 | 1741.23 | 1730.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1735.60 | 1740.10 | 1731.09 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1700.70 | 1722.00 | 1724.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 1693.00 | 1711.71 | 1718.94 | Break + close below crossover candle low |

### Cycle 82 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1792.00 | 1720.65 | 1720.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1838.00 | 1777.67 | 1754.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 1894.90 | 1906.17 | 1876.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 15:00:00 | 1894.90 | 1906.17 | 1876.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1955.70 | 1982.07 | 1966.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1955.70 | 1982.07 | 1966.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1958.40 | 1977.34 | 1965.75 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 1955.00 | 1959.68 | 1960.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1940.60 | 1955.86 | 1958.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1885.00 | 1881.14 | 1904.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1885.00 | 1881.14 | 1904.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1885.00 | 1881.14 | 1904.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 1887.80 | 1881.14 | 1904.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1869.00 | 1864.62 | 1878.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1869.00 | 1864.62 | 1878.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1886.00 | 1869.28 | 1878.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1886.00 | 1869.28 | 1878.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1880.10 | 1871.44 | 1878.42 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1895.00 | 1884.19 | 1883.06 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 1880.00 | 1894.34 | 1894.95 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 1913.50 | 1898.17 | 1896.63 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 1885.40 | 1895.99 | 1896.85 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 1908.30 | 1898.45 | 1897.89 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 1872.60 | 1893.85 | 1895.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1812.70 | 1872.36 | 1885.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1846.30 | 1836.20 | 1855.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1846.30 | 1836.20 | 1855.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1846.30 | 1836.20 | 1855.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1863.10 | 1836.20 | 1855.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1861.40 | 1837.02 | 1845.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 1855.40 | 1837.02 | 1845.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1869.10 | 1843.44 | 1847.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:15:00 | 1875.20 | 1843.44 | 1847.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 1886.90 | 1856.06 | 1852.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 13:15:00 | 1893.00 | 1863.45 | 1856.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 1857.30 | 1862.22 | 1856.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 1857.30 | 1862.22 | 1856.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1857.30 | 1862.22 | 1856.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1857.30 | 1862.22 | 1856.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1865.00 | 1862.77 | 1857.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 1855.00 | 1862.77 | 1857.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1824.20 | 1855.06 | 1854.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 1826.70 | 1855.06 | 1854.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 1819.30 | 1847.91 | 1851.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 1793.60 | 1837.05 | 1845.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 13:15:00 | 1814.60 | 1813.41 | 1824.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 1814.60 | 1813.41 | 1824.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1842.00 | 1818.56 | 1824.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 1843.90 | 1818.56 | 1824.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1826.40 | 1820.13 | 1824.31 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1868.50 | 1831.73 | 1828.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 1881.60 | 1861.18 | 1854.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 14:15:00 | 1847.00 | 1861.67 | 1855.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 14:15:00 | 1847.00 | 1861.67 | 1855.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1847.00 | 1861.67 | 1855.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1847.00 | 1861.67 | 1855.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1834.90 | 1856.32 | 1854.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 1825.00 | 1856.32 | 1854.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 1827.10 | 1848.66 | 1850.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 11:15:00 | 1817.50 | 1842.43 | 1847.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1837.20 | 1821.59 | 1832.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1837.20 | 1821.59 | 1832.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1837.20 | 1821.59 | 1832.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 1843.70 | 1821.59 | 1832.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1847.80 | 1826.83 | 1833.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1847.80 | 1826.83 | 1833.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1843.30 | 1832.45 | 1834.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1843.30 | 1832.45 | 1834.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1840.30 | 1834.02 | 1835.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1874.50 | 1834.02 | 1835.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1870.40 | 1841.30 | 1838.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1910.50 | 1878.75 | 1862.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1863.00 | 1882.71 | 1872.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1863.00 | 1882.71 | 1872.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1863.00 | 1882.71 | 1872.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1864.50 | 1882.71 | 1872.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1865.10 | 1879.19 | 1871.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1853.00 | 1879.19 | 1871.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1845.30 | 1864.06 | 1865.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1836.00 | 1855.34 | 1861.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1836.10 | 1816.04 | 1830.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1836.10 | 1816.04 | 1830.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1836.10 | 1816.04 | 1830.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1836.10 | 1816.04 | 1830.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1853.60 | 1823.55 | 1832.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1856.20 | 1823.55 | 1832.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1841.10 | 1835.64 | 1836.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1877.00 | 1835.64 | 1836.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1906.00 | 1849.72 | 1842.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1908.40 | 1861.45 | 1848.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 1996.00 | 1998.31 | 1963.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:30:00 | 1982.50 | 1998.31 | 1963.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1968.10 | 1984.58 | 1972.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 1968.10 | 1984.58 | 1972.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1969.50 | 1981.57 | 1972.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1969.50 | 1981.57 | 1972.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1972.00 | 1979.65 | 1972.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 2002.30 | 1979.65 | 1972.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 2055.50 | 2065.31 | 2065.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 2055.50 | 2065.31 | 2065.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 2034.10 | 2059.07 | 2062.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 2024.00 | 1990.50 | 2017.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 2024.00 | 1990.50 | 2017.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2024.00 | 1990.50 | 2017.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 2024.00 | 1990.50 | 2017.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2025.00 | 1997.40 | 2017.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 2029.20 | 1997.40 | 2017.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2059.00 | 2009.72 | 2021.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 2059.00 | 2009.72 | 2021.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 2062.60 | 2032.04 | 2029.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 2064.90 | 2042.61 | 2035.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 2047.90 | 2087.79 | 2074.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 2047.90 | 2087.79 | 2074.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2047.90 | 2087.79 | 2074.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 2050.00 | 2087.79 | 2074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 2047.00 | 2079.63 | 2072.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 2047.00 | 2079.63 | 2072.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 2034.00 | 2068.20 | 2068.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 2025.00 | 2059.56 | 2064.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 2052.00 | 2047.79 | 2056.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:30:00 | 2036.20 | 2047.79 | 2056.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1955.00 | 1927.39 | 1947.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1955.00 | 1927.39 | 1947.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1945.70 | 1931.05 | 1947.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 1943.30 | 1935.82 | 1947.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:30:00 | 1943.90 | 1937.04 | 1947.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 1941.90 | 1937.04 | 1947.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 1943.90 | 1938.75 | 1947.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1943.90 | 1939.78 | 1946.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1921.00 | 1939.78 | 1946.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1936.10 | 1939.04 | 1945.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 1955.80 | 1939.04 | 1945.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1961.00 | 1943.43 | 1947.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 1961.00 | 1943.43 | 1947.19 | SL hit (close>static) qty=1.00 sl=1960.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 13:15:00 | 1958.00 | 1949.79 | 1949.39 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 1937.00 | 1946.96 | 1948.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 1919.80 | 1941.53 | 1945.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1915.00 | 1895.81 | 1907.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 15:15:00 | 1915.00 | 1895.81 | 1907.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1915.00 | 1895.81 | 1907.05 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1923.40 | 1912.19 | 1911.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1980.00 | 1927.31 | 1918.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1959.00 | 1961.66 | 1944.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 1940.90 | 1957.51 | 1944.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1940.90 | 1957.51 | 1944.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 1940.90 | 1957.51 | 1944.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1938.10 | 1953.62 | 1943.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 1937.20 | 1953.62 | 1943.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1934.10 | 1948.76 | 1943.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 1934.10 | 1948.76 | 1943.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1944.50 | 1947.90 | 1943.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 1957.00 | 1947.90 | 1943.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:00:00 | 1951.40 | 1948.57 | 1945.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 1946.40 | 1946.30 | 1945.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:30:00 | 1948.90 | 1945.70 | 1945.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1941.00 | 1945.13 | 1944.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 1941.00 | 1945.13 | 1944.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1941.50 | 1944.40 | 1944.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1941.50 | 1944.40 | 1944.64 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1958.60 | 1946.56 | 1945.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1967.80 | 1950.80 | 1947.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 1947.70 | 1953.57 | 1949.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 1947.70 | 1953.57 | 1949.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1947.70 | 1953.57 | 1949.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 1946.60 | 1953.57 | 1949.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1950.00 | 1952.85 | 1949.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1954.40 | 1952.85 | 1949.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1950.10 | 1952.30 | 1949.93 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 1940.70 | 1948.27 | 1948.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1927.20 | 1942.71 | 1945.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 10:15:00 | 1935.00 | 1932.07 | 1937.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 1935.00 | 1932.07 | 1937.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1935.80 | 1933.56 | 1936.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 1937.70 | 1933.56 | 1936.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1935.00 | 1933.85 | 1936.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 1935.00 | 1933.85 | 1936.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1929.50 | 1932.98 | 1936.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1925.00 | 1932.98 | 1936.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1918.00 | 1929.98 | 1934.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1904.00 | 1923.48 | 1928.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1915.10 | 1901.64 | 1900.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 1915.10 | 1901.64 | 1900.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1932.60 | 1909.97 | 1904.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 1880.00 | 1917.47 | 1914.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1880.00 | 1917.47 | 1914.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1880.00 | 1917.47 | 1914.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 1880.00 | 1917.47 | 1914.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1864.90 | 1906.96 | 1910.26 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1904.90 | 1900.83 | 1900.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1917.00 | 1904.07 | 1902.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1898.00 | 1903.00 | 1902.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1898.00 | 1903.00 | 1902.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1898.00 | 1903.00 | 1902.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1895.50 | 1903.00 | 1902.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1915.50 | 1905.50 | 1903.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 1931.20 | 1918.87 | 1912.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 1906.00 | 1932.98 | 1934.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1906.00 | 1932.98 | 1934.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1901.50 | 1926.68 | 1931.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 1930.00 | 1927.28 | 1930.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 13:15:00 | 1930.00 | 1927.28 | 1930.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1930.00 | 1927.28 | 1930.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 1930.00 | 1927.28 | 1930.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1891.20 | 1920.07 | 1927.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1933.80 | 1920.07 | 1927.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1952.80 | 1914.34 | 1916.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1952.80 | 1914.34 | 1916.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1955.00 | 1922.47 | 1919.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1961.70 | 1934.99 | 1926.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1936.70 | 1938.48 | 1931.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1936.70 | 1938.48 | 1931.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1936.70 | 1938.48 | 1931.04 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 1920.00 | 1927.22 | 1927.51 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1935.00 | 1928.78 | 1928.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 1949.00 | 1932.82 | 1930.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1947.40 | 1948.81 | 1940.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:00:00 | 1947.40 | 1948.81 | 1940.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1954.00 | 1949.40 | 1942.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1945.00 | 1949.40 | 1942.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1947.70 | 1960.78 | 1953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 1947.70 | 1960.78 | 1953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1946.90 | 1958.00 | 1953.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1947.60 | 1958.00 | 1953.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1947.10 | 1951.62 | 1951.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 1947.10 | 1951.62 | 1951.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1948.10 | 1950.91 | 1950.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1948.50 | 1950.91 | 1950.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1947.50 | 1950.23 | 1950.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 14:15:00 | 1939.00 | 1947.16 | 1949.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 15:15:00 | 1932.50 | 1930.79 | 1937.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:45:00 | 1915.10 | 1926.97 | 1935.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 12:15:00 | 1915.70 | 1924.24 | 1932.57 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 15:00:00 | 1910.20 | 1917.80 | 1927.26 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 1931.20 | 1917.54 | 1923.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 1931.20 | 1917.54 | 1923.11 | SL hit (close>ema400) qty=1.00 sl=1923.11 alert=retest1 |

### Cycle 114 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1925.00 | 1893.89 | 1891.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 1925.60 | 1904.78 | 1896.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1916.00 | 1917.26 | 1907.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 1916.00 | 1917.26 | 1907.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1931.90 | 1935.76 | 1928.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1931.90 | 1935.76 | 1928.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1929.00 | 1934.08 | 1928.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 1928.50 | 1934.08 | 1928.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1933.00 | 1934.53 | 1930.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 1921.20 | 1934.53 | 1930.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1931.40 | 1933.90 | 1930.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1938.50 | 1933.90 | 1930.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1933.90 | 1933.61 | 1930.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 1931.10 | 1933.61 | 1930.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1933.00 | 1933.49 | 1931.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 1931.90 | 1933.49 | 1931.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1931.80 | 1933.15 | 1931.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 1932.30 | 1933.15 | 1931.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1937.90 | 1934.10 | 1931.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1932.20 | 1934.10 | 1931.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1931.10 | 1933.50 | 1931.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 1928.60 | 1933.50 | 1931.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1930.20 | 1932.84 | 1931.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:15:00 | 1930.00 | 1932.84 | 1931.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1933.70 | 1933.01 | 1931.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:15:00 | 1928.20 | 1933.01 | 1931.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1930.30 | 1932.47 | 1931.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 1935.00 | 1931.47 | 1931.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1923.80 | 1930.70 | 1931.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 1923.80 | 1930.70 | 1931.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 1905.10 | 1925.58 | 1928.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 12:15:00 | 1909.10 | 1904.22 | 1913.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:00:00 | 1909.10 | 1904.22 | 1913.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1908.50 | 1905.08 | 1913.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 1913.40 | 1905.08 | 1913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1901.00 | 1885.55 | 1890.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 1905.40 | 1885.55 | 1890.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1887.00 | 1885.84 | 1890.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1881.00 | 1885.90 | 1889.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 1882.50 | 1885.66 | 1889.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 1883.10 | 1880.95 | 1884.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:45:00 | 1883.70 | 1881.96 | 1884.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1893.60 | 1884.29 | 1885.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 1893.60 | 1884.29 | 1885.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1885.10 | 1884.45 | 1885.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:15:00 | 1895.00 | 1884.45 | 1885.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 15:15:00 | 1895.00 | 1886.56 | 1886.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 1895.00 | 1886.56 | 1886.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1897.50 | 1888.75 | 1887.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1908.90 | 1909.18 | 1900.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:30:00 | 1903.20 | 1909.18 | 1900.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1905.00 | 1908.34 | 1900.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1898.00 | 1908.34 | 1900.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1906.30 | 1907.93 | 1901.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 1902.50 | 1907.93 | 1901.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1907.90 | 1912.23 | 1906.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1926.00 | 1914.25 | 1907.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 1920.00 | 1914.80 | 1908.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:15:00 | 1919.70 | 1915.20 | 1909.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 1918.30 | 1915.12 | 1910.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1918.30 | 1915.76 | 1911.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1914.20 | 1915.76 | 1911.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1911.50 | 1914.91 | 1911.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1892.10 | 1908.33 | 1909.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1892.10 | 1908.33 | 1909.65 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 1919.20 | 1909.38 | 1909.25 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1908.00 | 1909.10 | 1909.14 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 1910.00 | 1909.28 | 1909.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1925.80 | 1912.58 | 1910.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1981.80 | 1984.01 | 1964.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 1981.80 | 1984.01 | 1964.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1973.00 | 1985.11 | 1979.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1973.00 | 1985.11 | 1979.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1967.00 | 1981.49 | 1978.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1969.00 | 1981.49 | 1978.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1948.50 | 1972.57 | 1974.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1932.00 | 1955.00 | 1964.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 1936.80 | 1933.44 | 1943.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:45:00 | 1938.40 | 1933.44 | 1943.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1939.70 | 1935.26 | 1942.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1939.70 | 1935.26 | 1942.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1939.20 | 1936.05 | 1942.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1939.20 | 1936.05 | 1942.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1942.20 | 1937.28 | 1942.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1946.60 | 1937.28 | 1942.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1946.90 | 1939.20 | 1942.70 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1953.00 | 1945.44 | 1944.77 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1936.90 | 1943.78 | 1944.55 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 1963.00 | 1948.30 | 1946.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1972.20 | 1953.08 | 1948.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 1956.40 | 1957.85 | 1952.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 1956.40 | 1957.85 | 1952.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1956.40 | 1957.85 | 1952.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 1957.00 | 1957.85 | 1952.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1958.40 | 1957.96 | 1953.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:30:00 | 1955.80 | 1957.96 | 1953.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1949.70 | 1956.31 | 1952.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 1949.70 | 1956.31 | 1952.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1945.00 | 1954.05 | 1952.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1932.60 | 1954.05 | 1952.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1932.60 | 1949.76 | 1950.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 1928.40 | 1945.49 | 1948.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1915.70 | 1905.51 | 1916.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 1915.70 | 1905.51 | 1916.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1915.70 | 1905.51 | 1916.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1915.70 | 1905.51 | 1916.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1913.00 | 1907.01 | 1916.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 1893.40 | 1911.64 | 1915.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 1893.40 | 1908.26 | 1913.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 1884.10 | 1880.99 | 1880.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1884.10 | 1880.99 | 1880.72 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 1877.00 | 1880.19 | 1880.38 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 1888.10 | 1881.78 | 1881.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1892.40 | 1883.90 | 1882.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 1882.10 | 1883.54 | 1882.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 1882.10 | 1883.54 | 1882.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1882.10 | 1883.54 | 1882.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1886.30 | 1883.54 | 1882.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1882.00 | 1883.23 | 1882.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:30:00 | 1883.00 | 1883.23 | 1882.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1877.30 | 1882.05 | 1881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 1878.40 | 1882.05 | 1881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 1874.80 | 1880.60 | 1881.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1863.00 | 1876.98 | 1879.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1749.90 | 1746.61 | 1769.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:00:00 | 1749.90 | 1746.61 | 1769.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1744.70 | 1747.74 | 1759.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 1744.70 | 1747.74 | 1759.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1724.50 | 1721.45 | 1736.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1708.40 | 1719.76 | 1732.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 1706.00 | 1710.67 | 1723.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1767.00 | 1699.32 | 1694.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1767.00 | 1699.32 | 1694.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 1835.60 | 1780.81 | 1752.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 1805.90 | 1810.81 | 1787.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 1766.20 | 1810.81 | 1787.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1780.00 | 1804.65 | 1787.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1765.70 | 1804.65 | 1787.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1779.20 | 1799.56 | 1786.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:45:00 | 1782.20 | 1795.57 | 1785.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1782.90 | 1790.40 | 1784.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 1788.20 | 1803.32 | 1800.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 1782.10 | 1795.97 | 1797.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 1782.10 | 1795.97 | 1797.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1774.50 | 1789.08 | 1793.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1650.40 | 1630.31 | 1666.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 1655.90 | 1630.31 | 1666.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1677.90 | 1639.61 | 1645.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 1677.90 | 1639.61 | 1645.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1682.00 | 1648.09 | 1648.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 1687.00 | 1648.09 | 1648.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 1655.00 | 1650.00 | 1649.41 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1628.00 | 1645.34 | 1647.46 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1675.90 | 1650.55 | 1649.10 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 1640.30 | 1647.39 | 1647.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 1628.00 | 1643.51 | 1646.04 | Break + close below crossover candle low |

### Cycle 136 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1687.20 | 1650.41 | 1648.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1706.00 | 1661.53 | 1653.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 12:15:00 | 1837.30 | 1840.44 | 1816.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 12:30:00 | 1835.10 | 1840.44 | 1816.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1812.80 | 1834.45 | 1817.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1812.80 | 1834.45 | 1817.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1815.00 | 1830.56 | 1817.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1783.20 | 1830.56 | 1817.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1779.50 | 1820.35 | 1813.86 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1764.00 | 1801.60 | 1806.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 1751.00 | 1782.44 | 1795.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1718.80 | 1709.20 | 1741.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 1715.40 | 1709.20 | 1741.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1726.50 | 1717.54 | 1739.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:00:00 | 1710.30 | 1718.50 | 1736.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1763.90 | 1730.59 | 1737.81 | SL hit (close>static) qty=1.00 sl=1744.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1776.80 | 1748.19 | 1744.69 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1673.80 | 1735.19 | 1739.77 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1750.20 | 1725.77 | 1724.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1754.20 | 1735.18 | 1729.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1752.10 | 1752.89 | 1741.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 1752.10 | 1752.89 | 1741.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1750.90 | 1752.49 | 1742.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1734.10 | 1752.49 | 1742.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1749.90 | 1751.97 | 1743.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1761.30 | 1751.97 | 1743.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 1761.40 | 1777.00 | 1768.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 1764.80 | 1774.40 | 1768.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 1740.10 | 1762.21 | 1763.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 1740.10 | 1762.21 | 1763.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 1729.50 | 1750.38 | 1757.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 1725.00 | 1723.57 | 1736.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 1725.00 | 1723.57 | 1736.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1729.90 | 1724.74 | 1734.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1750.40 | 1724.74 | 1734.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1687.20 | 1671.41 | 1690.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 1689.10 | 1671.41 | 1690.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1674.70 | 1672.07 | 1688.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1684.80 | 1672.07 | 1688.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1738.00 | 1678.22 | 1685.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1738.00 | 1678.22 | 1685.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 1739.00 | 1690.38 | 1690.30 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1592.10 | 1670.72 | 1681.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1567.50 | 1650.08 | 1671.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1599.00 | 1598.35 | 1623.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1599.00 | 1598.35 | 1623.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1630.30 | 1606.22 | 1619.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1628.50 | 1606.22 | 1619.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1667.00 | 1618.38 | 1623.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 1667.00 | 1618.38 | 1623.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1667.60 | 1628.22 | 1627.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1670.90 | 1642.61 | 1634.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 1653.50 | 1654.04 | 1643.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:45:00 | 1651.70 | 1654.04 | 1643.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1644.00 | 1652.61 | 1645.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 1644.00 | 1652.61 | 1645.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1725.00 | 1667.09 | 1652.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:30:00 | 1636.10 | 1667.09 | 1652.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1672.70 | 1675.91 | 1659.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 1709.10 | 1692.11 | 1682.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 1705.70 | 1703.13 | 1691.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1724.70 | 1710.56 | 1700.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:30:00 | 1708.80 | 1707.43 | 1701.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1699.00 | 1706.24 | 1701.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 1699.00 | 1706.24 | 1701.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1683.50 | 1701.69 | 1699.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 1683.50 | 1701.69 | 1699.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1692.00 | 1699.75 | 1699.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 1689.00 | 1697.60 | 1698.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 1689.00 | 1697.60 | 1698.33 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1729.00 | 1703.88 | 1701.12 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1689.70 | 1712.47 | 1714.74 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1727.80 | 1711.66 | 1709.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1744.70 | 1722.09 | 1715.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 13:15:00 | 1735.30 | 1745.01 | 1736.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 13:15:00 | 1735.30 | 1745.01 | 1736.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 1735.30 | 1745.01 | 1736.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:00:00 | 1735.30 | 1745.01 | 1736.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1739.90 | 1743.99 | 1736.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:15:00 | 1738.30 | 1743.99 | 1736.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1738.30 | 1742.85 | 1736.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 1726.20 | 1742.85 | 1736.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1728.90 | 1740.06 | 1736.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 1713.80 | 1740.06 | 1736.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1724.20 | 1736.89 | 1734.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:15:00 | 1733.50 | 1736.89 | 1734.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1726.80 | 1733.54 | 1733.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 1715.90 | 1730.02 | 1732.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1733.50 | 1725.44 | 1729.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1733.50 | 1725.44 | 1729.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1733.50 | 1725.44 | 1729.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 1733.50 | 1725.44 | 1729.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1736.60 | 1727.67 | 1729.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 1736.60 | 1727.67 | 1729.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1727.00 | 1729.33 | 1730.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:00:00 | 1714.70 | 1726.40 | 1728.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1717.40 | 1725.79 | 1727.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 13:00:00 | 1720.00 | 1723.56 | 1726.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1733.60 | 1727.71 | 1727.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1733.60 | 1727.71 | 1727.65 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1722.50 | 1726.75 | 1727.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1698.80 | 1720.11 | 1724.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1727.80 | 1699.32 | 1707.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1727.80 | 1699.32 | 1707.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1727.80 | 1699.32 | 1707.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1727.80 | 1699.32 | 1707.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1720.10 | 1703.48 | 1708.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1712.10 | 1703.48 | 1708.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:15:00 | 1715.90 | 1706.42 | 1709.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 1687.70 | 1673.79 | 1673.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1687.70 | 1673.79 | 1673.78 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 1666.10 | 1672.25 | 1673.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 1660.60 | 1669.92 | 1671.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 1670.70 | 1631.61 | 1639.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1670.70 | 1631.61 | 1639.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1670.70 | 1631.61 | 1639.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1670.70 | 1631.61 | 1639.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1660.00 | 1637.28 | 1641.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:30:00 | 1645.00 | 1638.03 | 1641.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 1643.30 | 1639.86 | 1641.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 1643.70 | 1642.29 | 1642.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 15:15:00 | 1650.00 | 1643.83 | 1643.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 15:15:00 | 1650.00 | 1643.83 | 1643.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 10:15:00 | 1662.90 | 1647.98 | 1645.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1647.80 | 1647.94 | 1645.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 1647.80 | 1647.94 | 1645.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1647.80 | 1647.94 | 1645.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 1646.80 | 1647.94 | 1645.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1645.90 | 1647.53 | 1645.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 1648.20 | 1647.53 | 1645.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1663.00 | 1650.63 | 1646.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1671.50 | 1652.76 | 1648.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 14:15:00 | 1534.05 | 2024-05-17 15:15:00 | 1547.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-05-23 15:15:00 | 1520.00 | 2024-05-27 12:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 15:15:00 | 1520.00 | 2024-05-28 09:15:00 | 1475.05 | STOP_HIT | 0.50 | 2.96% |
| BUY | retest2 | 2024-05-30 11:15:00 | 1502.80 | 2024-05-31 15:15:00 | 1480.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-05-31 11:15:00 | 1498.30 | 2024-05-31 15:15:00 | 1480.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-05-31 12:15:00 | 1495.70 | 2024-05-31 15:15:00 | 1480.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-05-31 13:00:00 | 1495.25 | 2024-05-31 15:15:00 | 1480.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-06-25 14:15:00 | 1600.00 | 2024-06-26 10:15:00 | 1634.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-12 09:15:00 | 1831.00 | 2024-07-12 12:15:00 | 1788.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-07-16 12:15:00 | 1783.00 | 2024-07-16 14:15:00 | 1800.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-18 09:15:00 | 1784.45 | 2024-07-18 09:15:00 | 1806.05 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2024-07-29 13:45:00 | 1805.95 | 2024-07-29 14:15:00 | 1775.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-07-30 10:15:00 | 1801.80 | 2024-08-05 11:15:00 | 1843.95 | STOP_HIT | 1.00 | 2.34% |
| SELL | retest2 | 2024-08-08 11:00:00 | 1781.35 | 2024-08-14 09:15:00 | 1692.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 12:45:00 | 1785.95 | 2024-08-14 09:15:00 | 1696.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 14:15:00 | 1785.00 | 2024-08-14 09:15:00 | 1695.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 09:15:00 | 1785.00 | 2024-08-14 09:15:00 | 1695.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-12 11:45:00 | 1755.90 | 2024-08-14 11:15:00 | 1668.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 11:00:00 | 1781.35 | 2024-08-19 09:15:00 | 1693.95 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2024-08-08 12:45:00 | 1785.95 | 2024-08-19 09:15:00 | 1693.95 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2024-08-08 14:15:00 | 1785.00 | 2024-08-19 09:15:00 | 1693.95 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2024-08-09 09:15:00 | 1785.00 | 2024-08-19 09:15:00 | 1693.95 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2024-08-12 11:45:00 | 1755.90 | 2024-08-19 09:15:00 | 1693.95 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2024-08-28 10:45:00 | 1684.55 | 2024-08-30 12:15:00 | 1703.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-28 11:30:00 | 1685.00 | 2024-08-30 12:15:00 | 1703.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-29 10:00:00 | 1680.05 | 2024-08-30 12:15:00 | 1703.60 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-29 15:00:00 | 1679.95 | 2024-08-30 12:15:00 | 1703.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1721.35 | 2024-09-04 14:15:00 | 1700.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-04 10:30:00 | 1699.50 | 2024-09-04 14:15:00 | 1700.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-09-04 11:30:00 | 1698.70 | 2024-09-04 14:15:00 | 1700.00 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-09-04 12:00:00 | 1697.40 | 2024-09-04 14:15:00 | 1700.00 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-09-13 09:15:00 | 1780.05 | 2024-09-19 11:15:00 | 1796.30 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1979.55 | 2024-11-06 11:15:00 | 2177.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-04 12:45:00 | 1970.60 | 2024-11-06 11:15:00 | 2167.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-04 14:30:00 | 1963.00 | 2024-11-06 11:15:00 | 2159.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-14 14:45:00 | 1947.45 | 2024-11-18 09:15:00 | 1752.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-03 09:15:00 | 1709.45 | 2024-12-06 12:15:00 | 1623.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 09:15:00 | 1709.45 | 2024-12-09 13:15:00 | 1616.05 | STOP_HIT | 0.50 | 5.46% |
| BUY | retest2 | 2024-12-16 13:00:00 | 1672.05 | 2024-12-19 10:15:00 | 1660.55 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-12-17 14:15:00 | 1693.65 | 2024-12-19 10:15:00 | 1660.55 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-12-18 15:15:00 | 1672.50 | 2024-12-19 10:15:00 | 1660.55 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-24 14:30:00 | 1585.45 | 2025-01-01 11:15:00 | 1561.50 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-12-24 15:15:00 | 1584.00 | 2025-01-01 11:15:00 | 1561.50 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2024-12-26 09:30:00 | 1580.25 | 2025-01-01 11:15:00 | 1561.50 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-01-03 12:15:00 | 1542.80 | 2025-01-06 12:15:00 | 1557.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-01-08 13:15:00 | 1580.15 | 2025-01-09 15:15:00 | 1550.90 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-01-08 14:45:00 | 1581.00 | 2025-01-09 15:15:00 | 1550.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-01-09 11:30:00 | 1585.20 | 2025-01-09 15:15:00 | 1550.90 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-01-10 09:15:00 | 1579.90 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-01-10 13:15:00 | 1587.95 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2025-01-10 14:15:00 | 1599.00 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-01-13 14:45:00 | 1590.80 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2025-01-13 15:15:00 | 1600.00 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2025-01-14 13:00:00 | 1627.65 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-01-15 10:15:00 | 1630.75 | 2025-01-21 12:15:00 | 1636.90 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-01-23 13:15:00 | 1607.10 | 2025-01-24 14:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:15:00 | 1607.10 | 2025-01-27 11:15:00 | 1446.39 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-06 15:15:00 | 1611.00 | 2025-02-07 09:15:00 | 1562.35 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest1 | 2025-02-11 09:15:00 | 1491.30 | 2025-02-13 11:15:00 | 1416.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 14:15:00 | 1466.95 | 2025-02-14 09:15:00 | 1393.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-11 09:15:00 | 1491.30 | 2025-02-14 15:15:00 | 1342.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-12 14:15:00 | 1466.95 | 2025-02-17 09:15:00 | 1320.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1298.10 | 2025-03-18 10:15:00 | 1331.15 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-03-13 11:00:00 | 1301.95 | 2025-03-18 10:15:00 | 1331.15 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-03-17 11:00:00 | 1294.40 | 2025-03-18 10:15:00 | 1331.15 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-03-27 12:00:00 | 1471.45 | 2025-03-27 13:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-03-27 13:15:00 | 1474.45 | 2025-03-27 13:15:00 | 1456.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-28 09:15:00 | 1482.20 | 2025-03-28 10:15:00 | 1453.70 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-04-01 15:15:00 | 1472.00 | 2025-04-02 12:15:00 | 1454.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-09 10:15:00 | 1359.60 | 2025-04-09 15:15:00 | 1410.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-04-09 14:00:00 | 1360.75 | 2025-04-09 15:15:00 | 1410.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-04-24 11:15:00 | 1432.80 | 2025-04-24 12:15:00 | 1430.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-04-25 14:45:00 | 1393.40 | 2025-05-02 09:15:00 | 1323.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1382.30 | 2025-05-02 09:15:00 | 1325.25 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2025-04-28 13:00:00 | 1390.50 | 2025-05-02 11:15:00 | 1313.18 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-04-28 14:00:00 | 1395.00 | 2025-05-02 11:15:00 | 1320.97 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1387.80 | 2025-05-02 11:15:00 | 1318.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 13:00:00 | 1387.90 | 2025-05-02 11:15:00 | 1318.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 15:00:00 | 1388.10 | 2025-05-02 11:15:00 | 1318.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 14:45:00 | 1393.40 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1382.30 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2025-04-28 13:00:00 | 1390.50 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.51% |
| SELL | retest2 | 2025-04-28 14:00:00 | 1395.00 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.81% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1387.80 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-04-29 13:00:00 | 1387.90 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-04-29 15:00:00 | 1388.10 | 2025-05-05 14:15:00 | 1300.00 | STOP_HIT | 0.50 | 6.35% |
| BUY | retest2 | 2025-05-16 12:15:00 | 1459.00 | 2025-05-19 14:15:00 | 1447.40 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-19 09:15:00 | 1462.60 | 2025-05-19 14:15:00 | 1447.40 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-20 09:45:00 | 1460.00 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-05-20 10:30:00 | 1459.60 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-05-21 15:15:00 | 1465.00 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1465.80 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1473.90 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-05-26 10:15:00 | 1465.10 | 2025-05-26 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-05 12:00:00 | 1594.00 | 2025-06-05 12:15:00 | 1604.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-05 12:45:00 | 1594.50 | 2025-06-05 14:15:00 | 1625.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1565.40 | 2025-06-16 10:15:00 | 1487.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 09:15:00 | 1558.00 | 2025-06-16 10:15:00 | 1480.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1565.40 | 2025-06-16 14:15:00 | 1504.90 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-06-12 09:15:00 | 1558.00 | 2025-06-16 14:15:00 | 1504.90 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-06-24 10:15:00 | 1487.20 | 2025-06-27 09:15:00 | 1509.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1487.40 | 2025-06-27 09:15:00 | 1509.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-09-05 09:15:00 | 2002.30 | 2025-09-11 11:15:00 | 2055.50 | STOP_HIT | 1.00 | 2.66% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1943.30 | 2025-09-26 10:15:00 | 1961.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-25 13:30:00 | 1943.90 | 2025-09-26 10:15:00 | 1961.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-25 14:00:00 | 1941.90 | 2025-09-26 10:15:00 | 1961.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-25 15:15:00 | 1943.90 | 2025-09-26 10:15:00 | 1961.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-06 15:15:00 | 1957.00 | 2025-10-08 14:15:00 | 1941.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-07 12:00:00 | 1951.40 | 2025-10-08 14:15:00 | 1941.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-08 09:30:00 | 1946.40 | 2025-10-08 14:15:00 | 1941.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-08 11:30:00 | 1948.90 | 2025-10-08 14:15:00 | 1941.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-16 10:15:00 | 1904.00 | 2025-10-20 14:15:00 | 1915.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-31 09:30:00 | 1931.20 | 2025-11-06 10:15:00 | 1906.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2025-11-19 09:45:00 | 1915.10 | 2025-11-20 12:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest1 | 2025-11-19 12:15:00 | 1915.70 | 2025-11-20 12:15:00 | 1931.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest1 | 2025-11-19 15:00:00 | 1910.20 | 2025-11-20 12:15:00 | 1931.20 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-21 10:15:00 | 1903.40 | 2025-11-26 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-21 13:15:00 | 1902.20 | 2025-11-26 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-21 15:00:00 | 1888.00 | 2025-11-26 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-11-26 09:45:00 | 1902.10 | 2025-11-26 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-03 15:15:00 | 1935.00 | 2025-12-04 11:15:00 | 1923.80 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1881.00 | 2025-12-11 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-10 13:45:00 | 1882.50 | 2025-12-11 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-11 12:15:00 | 1883.10 | 2025-12-11 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-12-11 12:45:00 | 1883.70 | 2025-12-11 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1926.00 | 2025-12-18 09:15:00 | 1892.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-12-16 12:15:00 | 1920.00 | 2025-12-18 09:15:00 | 1892.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-16 14:15:00 | 1919.70 | 2025-12-18 09:15:00 | 1892.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-16 15:15:00 | 1918.30 | 2025-12-18 09:15:00 | 1892.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1893.40 | 2026-01-14 13:15:00 | 1884.10 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-08 12:30:00 | 1893.40 | 2026-01-14 13:15:00 | 1884.10 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-28 13:15:00 | 1708.40 | 2026-02-03 09:15:00 | 1767.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-01-29 09:30:00 | 1706.00 | 2026-02-03 09:15:00 | 1767.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2026-02-06 11:45:00 | 1782.20 | 2026-02-11 11:15:00 | 1782.10 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1782.90 | 2026-02-11 11:15:00 | 1782.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-11 10:15:00 | 1788.20 | 2026-02-11 11:15:00 | 1782.10 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-03-05 14:00:00 | 1710.30 | 2026-03-06 09:15:00 | 1763.90 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1761.30 | 2026-03-16 10:15:00 | 1740.10 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-13 14:00:00 | 1761.40 | 2026-03-16 10:15:00 | 1740.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-03-13 14:30:00 | 1764.80 | 2026-03-16 10:15:00 | 1740.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-02 13:45:00 | 1709.10 | 2026-04-07 15:15:00 | 1689.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-04-06 10:15:00 | 1705.70 | 2026-04-07 15:15:00 | 1689.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-04-07 09:15:00 | 1724.70 | 2026-04-07 15:15:00 | 1689.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-07 10:30:00 | 1708.80 | 2026-04-07 15:15:00 | 1689.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-21 15:00:00 | 1714.70 | 2026-04-23 09:15:00 | 1733.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-04-22 11:15:00 | 1717.40 | 2026-04-23 09:15:00 | 1733.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-22 13:00:00 | 1720.00 | 2026-04-23 09:15:00 | 1733.60 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1712.10 | 2026-05-04 10:15:00 | 1687.70 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2026-04-27 12:15:00 | 1715.90 | 2026-05-04 10:15:00 | 1687.70 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2026-05-07 09:30:00 | 1645.00 | 2026-05-07 15:15:00 | 1650.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-05-07 12:30:00 | 1643.30 | 2026-05-07 15:15:00 | 1650.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-05-07 14:30:00 | 1643.70 | 2026-05-07 15:15:00 | 1650.00 | STOP_HIT | 1.00 | -0.38% |
