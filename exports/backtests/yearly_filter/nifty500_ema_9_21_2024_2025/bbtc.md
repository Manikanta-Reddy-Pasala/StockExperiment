# Bombay Burmah Trading Corporation Ltd. (BBTC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1563.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 102 |
| ALERT2 | 102 |
| ALERT2_SKIP | 58 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 124 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 55 / 87
- **Target hits / Stop hits / Partials:** 7 / 115 / 20
- **Avg / median % per leg:** 0.58% / -0.58%
- **Sum % (uncompounded):** 82.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 15 | 23.8% | 4 | 59 | 0 | 0.05% | 3.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 63 | 15 | 23.8% | 4 | 59 | 0 | 0.05% | 3.1% |
| SELL (all) | 79 | 40 | 50.6% | 3 | 56 | 20 | 1.00% | 79.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 79 | 40 | 50.6% | 3 | 56 | 20 | 1.00% | 79.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 142 | 55 | 38.7% | 7 | 115 | 20 | 0.58% | 82.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1550.00 | 1527.81 | 1526.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 12:15:00 | 1576.80 | 1549.57 | 1540.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1550.95 | 1556.09 | 1545.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 1550.95 | 1556.09 | 1545.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1550.95 | 1556.09 | 1545.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 1550.95 | 1556.09 | 1545.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1549.95 | 1554.86 | 1546.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 1557.65 | 1554.86 | 1546.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:30:00 | 1554.50 | 1553.09 | 1546.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 11:45:00 | 1555.00 | 1552.55 | 1547.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:30:00 | 1552.25 | 1552.83 | 1547.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1536.60 | 1549.59 | 1546.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 1536.60 | 1549.59 | 1546.70 | SL hit (close<static) qty=1.00 sl=1544.95 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 1542.40 | 1547.43 | 1547.49 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 1556.00 | 1549.15 | 1548.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 1568.80 | 1553.83 | 1550.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1561.10 | 1561.20 | 1555.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 1561.10 | 1561.20 | 1555.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1561.10 | 1561.20 | 1555.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:15:00 | 1550.00 | 1561.20 | 1555.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1550.00 | 1558.96 | 1555.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1556.00 | 1558.96 | 1555.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1560.45 | 1559.26 | 1555.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1560.00 | 1559.26 | 1555.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1571.15 | 1561.63 | 1557.24 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 1544.25 | 1553.68 | 1554.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 1538.90 | 1548.05 | 1551.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 1546.50 | 1546.29 | 1549.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1546.50 | 1546.29 | 1549.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1546.50 | 1546.29 | 1549.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1545.60 | 1546.29 | 1549.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1560.00 | 1549.04 | 1550.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 1568.00 | 1549.04 | 1550.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1556.00 | 1550.43 | 1551.33 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 1566.65 | 1553.67 | 1552.72 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 1535.80 | 1549.43 | 1550.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 15:15:00 | 1530.00 | 1538.81 | 1543.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 1542.05 | 1523.64 | 1530.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1542.05 | 1523.64 | 1530.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1542.05 | 1523.64 | 1530.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1542.05 | 1523.64 | 1530.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1522.00 | 1523.32 | 1529.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 1511.50 | 1519.00 | 1526.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:30:00 | 1511.00 | 1517.19 | 1524.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1435.92 | 1482.89 | 1488.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1435.45 | 1482.89 | 1488.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 1360.35 | 1440.35 | 1466.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1510.85 | 1461.54 | 1457.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 1548.00 | 1486.43 | 1469.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1583.00 | 1584.66 | 1560.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 1583.00 | 1584.66 | 1560.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1644.00 | 1651.37 | 1643.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:00:00 | 1644.00 | 1651.37 | 1643.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1647.00 | 1650.49 | 1644.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:45:00 | 1647.00 | 1650.49 | 1644.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1646.95 | 1649.20 | 1645.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 1646.95 | 1649.20 | 1645.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1648.70 | 1649.10 | 1645.41 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 1635.00 | 1643.43 | 1643.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 1633.75 | 1639.84 | 1641.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1637.80 | 1628.40 | 1633.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1637.80 | 1628.40 | 1633.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1637.80 | 1628.40 | 1633.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 1630.00 | 1628.40 | 1633.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1631.95 | 1629.11 | 1633.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 1630.00 | 1632.77 | 1634.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1697.80 | 1645.37 | 1639.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1697.80 | 1645.37 | 1639.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 09:15:00 | 1806.90 | 1700.21 | 1671.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 1925.30 | 1927.40 | 1847.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 13:00:00 | 1925.30 | 1927.40 | 1847.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1897.50 | 1930.66 | 1912.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 1897.50 | 1930.66 | 1912.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1875.45 | 1919.62 | 1909.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 1875.45 | 1919.62 | 1909.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1882.95 | 1912.29 | 1906.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1876.15 | 1912.29 | 1906.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 2040.05 | 2061.31 | 2035.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:30:00 | 2041.25 | 2061.31 | 2035.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 2038.40 | 2056.73 | 2035.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 2038.40 | 2056.73 | 2035.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 2083.00 | 2061.98 | 2039.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:00:00 | 2084.00 | 2066.39 | 2043.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 2087.50 | 2068.66 | 2048.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:30:00 | 2095.90 | 2088.55 | 2071.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 2090.80 | 2087.42 | 2072.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 2073.40 | 2084.61 | 2073.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:45:00 | 2070.00 | 2084.61 | 2073.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 2070.30 | 2081.75 | 2073.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 2068.15 | 2081.75 | 2073.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 2058.50 | 2077.10 | 2072.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 2058.50 | 2077.10 | 2072.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 2055.00 | 2072.68 | 2070.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 2057.15 | 2072.68 | 2070.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 2049.15 | 2067.97 | 2068.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 2049.15 | 2067.97 | 2068.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 2024.95 | 2048.44 | 2057.30 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 2215.60 | 2063.34 | 2057.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 2295.00 | 2109.67 | 2078.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 13:15:00 | 2256.05 | 2262.96 | 2207.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:00:00 | 2256.05 | 2262.96 | 2207.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 2241.75 | 2253.77 | 2220.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 2258.05 | 2245.03 | 2219.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:45:00 | 2312.90 | 2260.57 | 2230.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:00:00 | 2305.00 | 2307.51 | 2283.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 14:15:00 | 2272.00 | 2287.15 | 2288.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 2272.00 | 2287.15 | 2288.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 2258.00 | 2281.32 | 2285.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 2203.50 | 2171.48 | 2201.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 2203.50 | 2171.48 | 2201.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2203.50 | 2171.48 | 2201.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 2203.50 | 2171.48 | 2201.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2171.30 | 2171.44 | 2198.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 2239.30 | 2171.44 | 2198.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 2153.70 | 2135.98 | 2157.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 2153.75 | 2135.98 | 2157.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 2146.00 | 2137.99 | 2156.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 2210.85 | 2137.99 | 2156.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2250.00 | 2160.39 | 2165.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2250.00 | 2160.39 | 2165.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 2280.00 | 2184.31 | 2175.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 2312.00 | 2260.58 | 2233.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 2284.10 | 2289.95 | 2271.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 2284.10 | 2289.95 | 2271.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2319.50 | 2296.37 | 2279.09 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 2268.80 | 2288.92 | 2290.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 2253.00 | 2281.74 | 2286.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2211.75 | 2163.35 | 2191.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2211.75 | 2163.35 | 2191.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2211.75 | 2163.35 | 2191.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 2211.75 | 2163.35 | 2191.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 2193.00 | 2169.28 | 2191.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:15:00 | 2158.00 | 2173.88 | 2189.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 2242.80 | 2158.73 | 2159.72 | SL hit (close>static) qty=1.00 sl=2231.55 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 2258.60 | 2178.70 | 2168.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 2321.90 | 2241.08 | 2208.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 11:15:00 | 2316.85 | 2325.67 | 2286.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 12:00:00 | 2316.85 | 2325.67 | 2286.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 2342.90 | 2321.01 | 2297.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 2348.05 | 2321.01 | 2297.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 2284.90 | 2311.96 | 2299.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 2284.90 | 2311.96 | 2299.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 2270.00 | 2303.57 | 2296.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 2270.00 | 2303.57 | 2296.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 2244.45 | 2291.74 | 2291.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 2238.50 | 2273.45 | 2283.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 2271.90 | 2267.14 | 2278.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:30:00 | 2262.00 | 2267.14 | 2278.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 2299.20 | 2273.55 | 2280.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 2295.75 | 2273.55 | 2280.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 2286.00 | 2276.04 | 2280.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:45:00 | 2268.10 | 2275.37 | 2279.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 2270.00 | 2275.37 | 2279.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:45:00 | 2270.00 | 2272.66 | 2277.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 2268.75 | 2271.88 | 2277.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 2252.15 | 2266.43 | 2273.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:45:00 | 2245.10 | 2266.43 | 2273.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 2287.85 | 2268.01 | 2272.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 2284.10 | 2268.01 | 2272.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 2287.00 | 2271.81 | 2274.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 2445.30 | 2271.81 | 2274.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 2445.20 | 2306.49 | 2289.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 2445.20 | 2306.49 | 2289.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 2519.95 | 2419.13 | 2375.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2531.50 | 2546.12 | 2508.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 2531.50 | 2546.12 | 2508.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 2538.25 | 2546.83 | 2533.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:30:00 | 2543.40 | 2546.83 | 2533.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 2533.95 | 2544.25 | 2533.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:30:00 | 2549.45 | 2549.79 | 2537.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 2548.35 | 2548.83 | 2540.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 2585.55 | 2544.91 | 2540.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 2534.75 | 2550.68 | 2551.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 2534.75 | 2550.68 | 2551.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 2495.70 | 2534.87 | 2543.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 2595.50 | 2496.53 | 2502.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 2595.50 | 2496.53 | 2502.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 2595.50 | 2496.53 | 2502.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 2595.50 | 2496.53 | 2502.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 2682.50 | 2533.73 | 2519.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 2732.95 | 2664.96 | 2613.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 2702.00 | 2723.43 | 2680.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 15:00:00 | 2702.00 | 2723.43 | 2680.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2649.45 | 2702.64 | 2678.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2649.45 | 2702.64 | 2678.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2648.00 | 2691.71 | 2675.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:15:00 | 2643.30 | 2691.71 | 2675.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 2638.95 | 2666.57 | 2666.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 2625.05 | 2658.27 | 2662.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 2629.50 | 2624.80 | 2641.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 2629.50 | 2624.80 | 2641.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 2629.50 | 2624.80 | 2641.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 2635.00 | 2624.80 | 2641.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 2662.80 | 2632.40 | 2642.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 2660.05 | 2632.40 | 2642.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2641.55 | 2634.23 | 2642.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 2704.20 | 2634.23 | 2642.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2686.35 | 2644.65 | 2646.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 2739.90 | 2644.65 | 2646.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 2669.40 | 2649.60 | 2648.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 2716.90 | 2671.75 | 2660.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 2663.50 | 2676.97 | 2667.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 11:15:00 | 2663.50 | 2676.97 | 2667.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 2663.50 | 2676.97 | 2667.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:00:00 | 2663.50 | 2676.97 | 2667.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 2672.70 | 2676.11 | 2667.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 14:30:00 | 2690.70 | 2673.17 | 2667.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 2674.00 | 2673.17 | 2667.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:45:00 | 2677.25 | 2683.33 | 2676.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 2699.00 | 2680.73 | 2676.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 2681.55 | 2680.90 | 2676.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:45:00 | 2745.95 | 2691.33 | 2682.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 2818.00 | 2691.99 | 2685.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 2706.50 | 2715.26 | 2706.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:15:00 | 2707.00 | 2711.49 | 2705.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 2706.15 | 2710.42 | 2705.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 2720.00 | 2715.07 | 2708.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 2720.75 | 2723.40 | 2716.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:00:00 | 2714.00 | 2721.52 | 2716.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 2696.95 | 2710.00 | 2711.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 2696.95 | 2710.00 | 2711.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 15:15:00 | 2687.00 | 2705.40 | 2709.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 2617.90 | 2608.25 | 2644.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 2617.90 | 2608.25 | 2644.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2617.90 | 2608.25 | 2644.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 2590.80 | 2614.48 | 2636.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 2696.50 | 2629.66 | 2629.96 | SL hit (close>static) qty=1.00 sl=2655.75 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 2675.85 | 2638.90 | 2634.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 2710.00 | 2677.78 | 2661.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 13:15:00 | 2682.55 | 2691.33 | 2677.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 14:00:00 | 2682.55 | 2691.33 | 2677.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2849.70 | 2882.01 | 2850.74 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 2780.05 | 2831.91 | 2836.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 2750.70 | 2794.08 | 2812.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2582.90 | 2568.02 | 2656.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2582.90 | 2568.02 | 2656.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 2634.50 | 2587.77 | 2637.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:30:00 | 2625.70 | 2587.77 | 2637.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 2652.80 | 2600.78 | 2639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 2652.80 | 2600.78 | 2639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 2640.00 | 2608.62 | 2639.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 2669.10 | 2608.62 | 2639.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2678.95 | 2622.69 | 2642.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 2678.95 | 2622.69 | 2642.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2681.30 | 2634.41 | 2646.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 2682.60 | 2634.41 | 2646.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 2684.95 | 2658.15 | 2655.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 2725.60 | 2675.36 | 2664.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 2687.00 | 2691.99 | 2679.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 2675.10 | 2691.99 | 2679.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2685.65 | 2690.72 | 2679.97 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 2650.00 | 2672.86 | 2673.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 2640.55 | 2666.40 | 2670.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 2635.15 | 2622.38 | 2641.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 2635.15 | 2622.38 | 2641.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 2635.15 | 2622.38 | 2641.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:45:00 | 2636.80 | 2622.38 | 2641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 2654.20 | 2628.74 | 2642.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:45:00 | 2643.70 | 2628.74 | 2642.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 2660.00 | 2634.99 | 2644.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 2680.05 | 2634.99 | 2644.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 2669.25 | 2650.80 | 2649.77 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 2639.00 | 2651.58 | 2652.00 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 12:15:00 | 2663.30 | 2653.92 | 2653.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 13:15:00 | 2668.90 | 2656.92 | 2654.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 13:15:00 | 2641.30 | 2675.74 | 2668.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 13:15:00 | 2641.30 | 2675.74 | 2668.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 2641.30 | 2675.74 | 2668.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 2641.30 | 2675.74 | 2668.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 2648.55 | 2670.30 | 2666.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:15:00 | 2644.05 | 2670.30 | 2666.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 2644.05 | 2665.05 | 2664.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 2611.80 | 2665.05 | 2664.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 2635.95 | 2659.23 | 2662.19 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 2676.80 | 2665.04 | 2664.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 2686.05 | 2674.42 | 2669.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 2724.70 | 2764.08 | 2728.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 2724.70 | 2764.08 | 2728.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2724.70 | 2764.08 | 2728.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 2724.70 | 2764.08 | 2728.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 2710.95 | 2753.45 | 2726.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:15:00 | 2692.55 | 2753.45 | 2726.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 2678.30 | 2738.42 | 2722.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:45:00 | 2691.10 | 2738.42 | 2722.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 2655.40 | 2701.75 | 2707.94 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 2792.95 | 2716.50 | 2709.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 10:15:00 | 2839.00 | 2741.00 | 2721.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 15:15:00 | 2792.55 | 2795.71 | 2761.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 09:15:00 | 2715.10 | 2795.71 | 2761.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 2669.80 | 2770.53 | 2753.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 2669.80 | 2770.53 | 2753.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 2624.70 | 2741.37 | 2741.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 2583.70 | 2676.53 | 2708.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 2673.55 | 2667.85 | 2696.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 2673.55 | 2667.85 | 2696.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 2673.55 | 2667.85 | 2696.31 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 2728.85 | 2703.90 | 2703.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 2744.25 | 2717.86 | 2710.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 2720.05 | 2734.69 | 2724.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 2720.05 | 2734.69 | 2724.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 2720.05 | 2734.69 | 2724.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 2720.05 | 2734.69 | 2724.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 2723.35 | 2732.42 | 2724.60 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 2708.40 | 2719.14 | 2719.81 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 2742.90 | 2723.89 | 2721.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 2773.00 | 2737.58 | 2729.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 2747.50 | 2748.74 | 2737.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 09:15:00 | 2705.65 | 2748.74 | 2737.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2657.95 | 2730.58 | 2729.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2657.95 | 2730.58 | 2729.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2640.90 | 2712.65 | 2721.84 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 2737.90 | 2709.43 | 2707.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 2756.30 | 2718.81 | 2712.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 2771.90 | 2781.56 | 2755.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 2771.90 | 2781.56 | 2755.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 2771.00 | 2780.16 | 2763.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 2762.15 | 2780.16 | 2763.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2753.75 | 2774.88 | 2762.28 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 2723.70 | 2749.05 | 2752.34 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 2815.75 | 2756.99 | 2753.08 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 2737.15 | 2754.68 | 2754.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 2715.00 | 2739.88 | 2747.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2621.85 | 2613.63 | 2659.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:30:00 | 2603.35 | 2613.63 | 2659.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 2583.95 | 2583.46 | 2616.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:00:00 | 2583.95 | 2583.46 | 2616.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 2616.10 | 2589.99 | 2616.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 2573.30 | 2588.19 | 2613.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 2581.25 | 2575.69 | 2595.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:45:00 | 2578.05 | 2575.36 | 2593.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 2583.65 | 2584.07 | 2594.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:15:00 | 2444.64 | 2499.31 | 2534.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:15:00 | 2452.19 | 2499.31 | 2534.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:15:00 | 2449.15 | 2499.31 | 2534.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:15:00 | 2454.47 | 2499.31 | 2534.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 2516.95 | 2486.56 | 2510.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 2516.95 | 2486.56 | 2510.50 | SL hit (close>ema200) qty=0.50 sl=2486.56 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 2536.95 | 2519.32 | 2519.28 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 2510.75 | 2518.94 | 2519.26 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 2536.75 | 2521.70 | 2520.36 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 2502.50 | 2517.03 | 2518.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 2486.65 | 2508.53 | 2513.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 2431.65 | 2406.31 | 2429.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 10:15:00 | 2431.65 | 2406.31 | 2429.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 2431.65 | 2406.31 | 2429.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 2431.65 | 2406.31 | 2429.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 2414.70 | 2407.99 | 2427.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 2409.15 | 2412.35 | 2425.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:30:00 | 2406.00 | 2407.97 | 2421.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 15:00:00 | 2405.75 | 2409.69 | 2417.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 2410.55 | 2381.49 | 2381.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 2410.55 | 2381.49 | 2381.29 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 2372.95 | 2381.82 | 2381.82 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 2402.60 | 2378.83 | 2378.83 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 2368.10 | 2380.30 | 2381.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 2341.90 | 2372.62 | 2377.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 2363.00 | 2339.75 | 2347.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 2363.00 | 2339.75 | 2347.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2363.00 | 2339.75 | 2347.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 2382.45 | 2339.75 | 2347.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2350.90 | 2341.98 | 2348.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 2356.55 | 2341.98 | 2348.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2370.05 | 2349.29 | 2350.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:15:00 | 2386.00 | 2349.29 | 2350.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 2384.20 | 2356.27 | 2353.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 2406.65 | 2366.35 | 2358.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 2375.00 | 2379.56 | 2369.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:00:00 | 2375.00 | 2379.56 | 2369.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 2365.60 | 2376.77 | 2369.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 2365.60 | 2376.77 | 2369.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 2360.00 | 2373.41 | 2368.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 2360.00 | 2373.41 | 2368.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 2346.25 | 2367.98 | 2366.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 2341.65 | 2367.98 | 2366.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 2325.00 | 2359.39 | 2362.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 2317.95 | 2341.92 | 2352.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 2164.35 | 2152.67 | 2191.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 11:00:00 | 2164.35 | 2152.67 | 2191.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 2118.55 | 2122.64 | 2136.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 2085.15 | 2122.94 | 2135.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 2115.00 | 2083.38 | 2097.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 2227.50 | 2128.15 | 2115.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 2227.50 | 2128.15 | 2115.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 2250.95 | 2195.21 | 2155.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 2223.05 | 2225.92 | 2201.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 12:15:00 | 2204.80 | 2219.18 | 2204.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 2204.80 | 2219.18 | 2204.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:45:00 | 2202.10 | 2219.18 | 2204.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 2203.95 | 2216.13 | 2204.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 2202.75 | 2216.13 | 2204.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2201.75 | 2213.26 | 2204.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2201.75 | 2213.26 | 2204.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2205.00 | 2211.61 | 2204.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2165.55 | 2211.61 | 2204.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2159.10 | 2201.10 | 2200.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 2159.10 | 2201.10 | 2200.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2120.30 | 2184.94 | 2192.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 2110.25 | 2159.29 | 2179.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 10:15:00 | 2075.95 | 2075.22 | 2104.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 10:15:00 | 2075.95 | 2075.22 | 2104.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 2075.95 | 2075.22 | 2104.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 2075.95 | 2075.22 | 2104.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 2134.00 | 2082.92 | 2096.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 2049.85 | 2085.00 | 2092.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 1947.36 | 1985.19 | 2021.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 1990.10 | 1979.91 | 2009.23 | SL hit (close>ema200) qty=0.50 sl=1979.91 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 2006.10 | 2000.47 | 1999.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 2022.90 | 2004.71 | 2001.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 2000.50 | 2003.87 | 2001.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 2000.50 | 2003.87 | 2001.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2000.50 | 2003.87 | 2001.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2000.50 | 2003.87 | 2001.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2004.00 | 2003.89 | 2001.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2001.25 | 2003.89 | 2001.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 2000.55 | 2003.22 | 2001.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:45:00 | 2006.70 | 2002.59 | 2001.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:45:00 | 2005.95 | 2001.50 | 2001.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 1999.50 | 2001.10 | 2001.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 1999.50 | 2001.10 | 2001.22 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 2014.20 | 2003.72 | 2002.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 2036.45 | 2012.15 | 2006.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 2003.05 | 2015.77 | 2010.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 2003.05 | 2015.77 | 2010.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2003.05 | 2015.77 | 2010.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 2003.05 | 2015.77 | 2010.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2003.80 | 2013.37 | 2010.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 2004.50 | 2013.37 | 2010.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 2002.50 | 2011.20 | 2009.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 2001.35 | 2011.20 | 2009.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 1991.30 | 2007.22 | 2007.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1981.15 | 2002.01 | 2005.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 11:15:00 | 2013.40 | 1997.50 | 2001.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 11:15:00 | 2013.40 | 1997.50 | 2001.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 2013.40 | 1997.50 | 2001.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:45:00 | 2010.55 | 1997.50 | 2001.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 12:15:00 | 2058.00 | 2009.60 | 2006.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 13:15:00 | 2077.00 | 2023.08 | 2012.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2161.90 | 2196.56 | 2143.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 2161.90 | 2196.56 | 2143.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 2155.90 | 2176.48 | 2149.98 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 2074.05 | 2129.64 | 2135.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 2026.35 | 2098.41 | 2117.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 2076.40 | 2075.95 | 2100.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 2076.40 | 2075.95 | 2100.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2127.50 | 2077.63 | 2092.36 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 2128.45 | 2104.27 | 2101.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 2167.75 | 2125.99 | 2118.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 2129.20 | 2138.04 | 2128.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 11:15:00 | 2129.20 | 2138.04 | 2128.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 2129.20 | 2138.04 | 2128.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 2120.50 | 2138.04 | 2128.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 2127.95 | 2136.02 | 2128.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 2113.05 | 2136.02 | 2128.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 2122.55 | 2133.33 | 2127.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 2115.25 | 2133.33 | 2127.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 2128.50 | 2132.36 | 2128.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:45:00 | 2115.10 | 2132.36 | 2128.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 2106.00 | 2127.09 | 2126.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 2138.10 | 2128.55 | 2126.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 2112.35 | 2125.31 | 2125.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 2112.35 | 2125.31 | 2125.45 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 2142.15 | 2127.55 | 2125.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 2166.70 | 2140.92 | 2134.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 2142.90 | 2144.77 | 2137.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 2142.90 | 2144.77 | 2137.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2134.65 | 2143.93 | 2139.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 2131.30 | 2143.93 | 2139.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2138.80 | 2142.91 | 2139.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 2142.00 | 2142.91 | 2139.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 2111.15 | 2133.65 | 2135.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 2111.15 | 2133.65 | 2135.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 2094.75 | 2121.47 | 2129.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1980.25 | 1965.05 | 2000.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 1980.25 | 1965.05 | 2000.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1888.30 | 1873.07 | 1899.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1888.30 | 1873.07 | 1899.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1842.85 | 1868.95 | 1893.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 1828.70 | 1858.30 | 1883.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:15:00 | 1824.00 | 1858.30 | 1883.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 10:15:00 | 1917.80 | 1867.11 | 1875.05 | SL hit (close>static) qty=1.00 sl=1905.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1896.95 | 1881.09 | 1880.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1902.00 | 1885.27 | 1882.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 1893.05 | 1893.07 | 1887.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:45:00 | 1896.00 | 1893.07 | 1887.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1897.90 | 1893.99 | 1888.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:15:00 | 1901.00 | 1893.99 | 1888.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 1839.60 | 1883.08 | 1885.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1839.60 | 1883.08 | 1885.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 1837.90 | 1858.57 | 1871.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 1852.60 | 1835.19 | 1853.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 1852.60 | 1835.19 | 1853.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1852.60 | 1835.19 | 1853.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 1865.00 | 1835.19 | 1853.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1834.70 | 1835.09 | 1851.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 13:15:00 | 1827.60 | 1835.09 | 1851.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 1736.22 | 1772.93 | 1802.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 09:15:00 | 1644.84 | 1721.52 | 1763.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1723.65 | 1699.61 | 1699.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1731.55 | 1705.99 | 1702.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 1766.80 | 1767.02 | 1743.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 1766.80 | 1767.02 | 1743.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1780.90 | 1781.26 | 1765.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1766.95 | 1781.26 | 1765.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1753.45 | 1775.70 | 1764.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1753.45 | 1775.70 | 1764.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1740.80 | 1768.72 | 1762.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 1740.80 | 1768.72 | 1762.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 1725.00 | 1753.90 | 1756.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1712.80 | 1745.68 | 1752.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1704.50 | 1700.47 | 1717.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 10:45:00 | 1703.80 | 1700.47 | 1717.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1703.65 | 1699.02 | 1710.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 1706.50 | 1699.02 | 1710.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1679.20 | 1694.09 | 1706.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 1657.95 | 1683.08 | 1698.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 1660.45 | 1675.46 | 1692.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 1686.55 | 1674.09 | 1672.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 1686.55 | 1674.09 | 1672.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1692.50 | 1677.77 | 1674.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 1732.00 | 1735.39 | 1719.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 1765.80 | 1735.39 | 1719.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1781.85 | 1811.90 | 1788.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1781.85 | 1811.90 | 1788.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1776.65 | 1804.85 | 1786.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 1776.65 | 1804.85 | 1786.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1776.05 | 1799.09 | 1785.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1776.40 | 1799.09 | 1785.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1765.00 | 1782.71 | 1781.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1778.70 | 1782.71 | 1781.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 1762.15 | 1778.60 | 1779.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1762.15 | 1778.60 | 1779.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 1752.10 | 1770.70 | 1775.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 1758.20 | 1740.10 | 1752.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 1758.20 | 1740.10 | 1752.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1758.20 | 1740.10 | 1752.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1758.20 | 1740.10 | 1752.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1759.40 | 1743.96 | 1753.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 1772.05 | 1743.96 | 1753.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1800.55 | 1763.02 | 1760.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 1811.10 | 1772.64 | 1765.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 1771.65 | 1778.83 | 1770.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 13:15:00 | 1771.65 | 1778.83 | 1770.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1771.65 | 1778.83 | 1770.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 1771.65 | 1778.83 | 1770.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1767.90 | 1776.64 | 1770.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 1767.90 | 1776.64 | 1770.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1759.90 | 1773.29 | 1769.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 1779.35 | 1775.47 | 1770.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:30:00 | 1779.00 | 1772.96 | 1770.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 1779.90 | 1773.42 | 1771.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1771.85 | 1776.51 | 1773.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1789.05 | 1779.02 | 1774.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:30:00 | 1786.40 | 1779.02 | 1774.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1795.15 | 1849.97 | 1834.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1795.15 | 1849.97 | 1834.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1794.05 | 1838.79 | 1830.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1791.85 | 1838.79 | 1830.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 1773.00 | 1816.46 | 1821.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 1773.00 | 1816.46 | 1821.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1758.70 | 1804.91 | 1815.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1737.10 | 1712.34 | 1744.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1737.10 | 1712.34 | 1744.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1737.10 | 1712.34 | 1744.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1722.00 | 1730.02 | 1740.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:30:00 | 1722.45 | 1728.98 | 1738.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 1782.20 | 1739.63 | 1742.83 | SL hit (close>static) qty=1.00 sl=1765.95 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 1771.95 | 1746.09 | 1745.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1818.00 | 1772.63 | 1759.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1895.90 | 1897.61 | 1868.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1895.90 | 1897.61 | 1868.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1924.20 | 1967.67 | 1959.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1924.20 | 1967.67 | 1959.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1907.90 | 1955.71 | 1954.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1907.90 | 1955.71 | 1954.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1912.30 | 1947.03 | 1950.89 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 1947.00 | 1934.16 | 1933.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 15:15:00 | 1960.00 | 1941.40 | 1937.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1933.10 | 1939.74 | 1937.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1933.10 | 1939.74 | 1937.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1933.10 | 1939.74 | 1937.00 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1903.90 | 1932.57 | 1933.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1894.80 | 1925.02 | 1930.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1888.80 | 1867.29 | 1884.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1888.80 | 1867.29 | 1884.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1888.80 | 1867.29 | 1884.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1888.80 | 1867.29 | 1884.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1896.20 | 1873.07 | 1885.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 1896.30 | 1873.07 | 1885.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1897.10 | 1877.88 | 1886.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1897.10 | 1877.88 | 1886.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1884.60 | 1884.79 | 1887.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1887.40 | 1884.79 | 1887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1906.10 | 1889.05 | 1889.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1906.10 | 1889.05 | 1889.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1889.80 | 1889.20 | 1889.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 1876.10 | 1889.20 | 1889.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1782.29 | 1842.99 | 1865.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1779.00 | 1734.83 | 1755.95 | SL hit (close>ema200) qty=0.50 sl=1734.83 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1800.00 | 1771.55 | 1768.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1808.40 | 1783.58 | 1775.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1777.60 | 1787.29 | 1779.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1777.60 | 1787.29 | 1779.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1780.00 | 1785.83 | 1779.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1780.00 | 1785.83 | 1779.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1796.30 | 1787.92 | 1781.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1819.00 | 1786.90 | 1781.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 09:15:00 | 2000.90 | 1922.58 | 1876.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 2014.00 | 2028.74 | 2029.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 2010.00 | 2024.99 | 2027.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 1985.00 | 1983.73 | 1992.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 09:15:00 | 1991.40 | 1983.73 | 1992.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2022.20 | 1991.42 | 1994.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 2022.20 | 1991.42 | 1994.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 2014.90 | 1996.12 | 1996.63 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 2017.90 | 2000.47 | 1998.56 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 1989.20 | 1999.98 | 2000.20 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2020.70 | 2003.21 | 2000.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 2072.60 | 2021.79 | 2011.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2009.90 | 2023.94 | 2016.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2009.90 | 2023.94 | 2016.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1993.60 | 2017.87 | 2014.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1993.60 | 2017.87 | 2014.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1965.00 | 2007.30 | 2009.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1927.90 | 1969.75 | 1985.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1955.50 | 1953.95 | 1971.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:30:00 | 1953.00 | 1953.95 | 1971.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1961.00 | 1957.11 | 1970.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1960.80 | 1957.11 | 1970.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1978.40 | 1961.37 | 1971.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1977.60 | 1961.37 | 1971.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1974.80 | 1964.06 | 1971.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1982.40 | 1964.06 | 1971.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1960.30 | 1963.31 | 1970.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 1941.90 | 1963.31 | 1970.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1955.70 | 1956.63 | 1965.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 1948.70 | 1955.27 | 1962.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1844.81 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1857.91 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1851.26 | 1879.45 | 1902.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 1880.30 | 1864.86 | 1880.88 | SL hit (close>ema200) qty=0.50 sl=1864.86 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1936.00 | 1895.00 | 1889.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1956.20 | 1929.28 | 1917.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 1979.30 | 1994.65 | 1976.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 1979.30 | 1994.65 | 1976.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1978.80 | 1994.40 | 1984.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1978.80 | 1994.40 | 1984.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2007.00 | 1996.92 | 1986.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 2007.50 | 1999.94 | 1988.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:30:00 | 2015.90 | 2001.35 | 1990.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 1980.00 | 1988.43 | 1989.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 1980.00 | 1988.43 | 1989.58 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 1991.00 | 1988.80 | 1988.75 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1978.20 | 1986.68 | 1987.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1960.50 | 1979.44 | 1983.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1971.30 | 1963.42 | 1971.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1971.30 | 1963.42 | 1971.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1960.80 | 1962.90 | 1970.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1955.90 | 1961.70 | 1969.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 1956.20 | 1961.94 | 1968.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 1955.10 | 1955.19 | 1963.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1974.30 | 1960.84 | 1962.95 | SL hit (close>static) qty=1.00 sl=1971.50 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1967.50 | 1964.82 | 1964.56 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1951.10 | 1962.07 | 1963.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1942.30 | 1956.46 | 1960.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1957.80 | 1956.73 | 1960.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 1957.80 | 1956.73 | 1960.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1959.00 | 1957.18 | 1960.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 1959.00 | 1957.18 | 1960.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1940.00 | 1953.75 | 1958.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1931.10 | 1945.05 | 1951.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1968.00 | 1948.10 | 1950.87 | SL hit (close>static) qty=1.00 sl=1961.70 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1972.70 | 1955.60 | 1953.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2004.40 | 1965.36 | 1958.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2013.10 | 2023.27 | 2007.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 2013.10 | 2023.27 | 2007.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2016.30 | 2021.88 | 2008.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:30:00 | 2019.80 | 2019.26 | 2009.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 2004.00 | 2014.53 | 2008.89 | SL hit (close<static) qty=1.00 sl=2005.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1978.20 | 2005.42 | 2005.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 1967.90 | 1992.27 | 1999.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1990.00 | 1962.86 | 1972.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 2002.00 | 1962.86 | 1972.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2017.90 | 1973.87 | 1976.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 2017.90 | 1973.87 | 1976.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 2012.60 | 1981.62 | 1979.99 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 1958.90 | 1980.34 | 1982.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1945.70 | 1966.02 | 1973.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1931.00 | 1927.19 | 1943.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1969.00 | 1927.19 | 1943.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1952.70 | 1932.29 | 1944.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 1970.00 | 1932.29 | 1944.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1941.90 | 1934.21 | 1944.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 1941.00 | 1934.21 | 1944.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1940.40 | 1935.45 | 1943.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 1942.10 | 1935.45 | 1943.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1944.00 | 1937.16 | 1943.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 1944.00 | 1937.16 | 1943.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1940.00 | 1937.73 | 1943.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 1935.10 | 1939.08 | 1943.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1952.40 | 1941.11 | 1943.69 | SL hit (close>static) qty=1.00 sl=1944.30 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 1863.00 | 1846.98 | 1846.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 15:15:00 | 1878.00 | 1853.18 | 1849.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1855.10 | 1856.65 | 1853.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 1855.10 | 1856.65 | 1853.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1860.00 | 1857.32 | 1853.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1847.00 | 1857.32 | 1853.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1846.00 | 1855.06 | 1853.15 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1833.20 | 1850.69 | 1851.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1831.00 | 1838.86 | 1844.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1851.10 | 1841.31 | 1845.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 1832.80 | 1840.69 | 1844.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 14:15:00 | 1852.40 | 1845.94 | 1845.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 1852.40 | 1845.94 | 1845.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 1859.90 | 1848.73 | 1847.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1847.70 | 1848.53 | 1847.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1869.30 | 1859.30 | 1853.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 1840.00 | 1854.33 | 1855.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1840.00 | 1854.33 | 1855.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1835.10 | 1848.19 | 1852.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1818.80 | 1791.48 | 1803.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 1818.80 | 1791.48 | 1803.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1831.20 | 1799.42 | 1806.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 1840.20 | 1799.42 | 1806.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1806.10 | 1802.79 | 1806.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:45:00 | 1809.60 | 1802.79 | 1806.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1810.00 | 1804.23 | 1806.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 1810.00 | 1804.23 | 1806.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1800.10 | 1803.40 | 1806.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 1800.10 | 1803.40 | 1806.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 1804.50 | 1803.62 | 1806.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 1824.30 | 1803.62 | 1806.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1790.00 | 1800.90 | 1804.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1783.40 | 1797.87 | 1802.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1787.10 | 1780.91 | 1790.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1782.00 | 1784.79 | 1788.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 1822.80 | 1797.26 | 1793.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1822.80 | 1797.26 | 1793.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1837.30 | 1826.91 | 1815.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1867.50 | 1867.78 | 1849.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 1848.20 | 1860.95 | 1849.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1848.20 | 1860.95 | 1849.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1848.20 | 1860.95 | 1849.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1846.00 | 1857.96 | 1848.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 1842.90 | 1857.96 | 1848.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1852.40 | 1856.85 | 1849.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1862.80 | 1855.32 | 1851.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 1858.70 | 1857.60 | 1854.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1860.30 | 1856.77 | 1854.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1867.00 | 1858.47 | 1855.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1865.37 | 1859.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1911.00 | 1883.83 | 1880.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1937.90 | 1947.60 | 1948.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1937.90 | 1947.60 | 1948.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 1931.10 | 1939.69 | 1943.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1890.00 | 1885.69 | 1900.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 1897.10 | 1885.69 | 1900.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1769.70 | 1767.05 | 1784.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1784.60 | 1767.05 | 1784.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1782.40 | 1773.37 | 1783.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 1782.40 | 1773.37 | 1783.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1789.30 | 1776.56 | 1783.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1789.30 | 1776.56 | 1783.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1808.30 | 1782.91 | 1785.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1808.30 | 1782.91 | 1785.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1809.00 | 1788.13 | 1788.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1817.00 | 1793.90 | 1790.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1887.90 | 1891.02 | 1863.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 1887.90 | 1891.02 | 1863.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1869.50 | 1880.97 | 1869.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 1884.60 | 1880.97 | 1869.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1844.20 | 1862.85 | 1864.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1844.20 | 1862.85 | 1864.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1840.30 | 1855.34 | 1860.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1860.90 | 1849.04 | 1853.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 1860.90 | 1849.04 | 1853.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1852.40 | 1849.71 | 1853.11 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1865.80 | 1855.67 | 1855.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1875.10 | 1859.55 | 1857.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1864.70 | 1865.29 | 1860.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 1861.00 | 1865.29 | 1860.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1857.40 | 1863.72 | 1860.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1851.40 | 1863.72 | 1860.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1873.50 | 1865.67 | 1861.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 1880.80 | 1865.67 | 1861.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:30:00 | 1882.00 | 1871.51 | 1864.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:00:00 | 1907.80 | 1888.52 | 1884.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-17 12:15:00 | 2068.88 | 1951.52 | 1921.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 2039.40 | 2054.40 | 2056.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 2031.60 | 2046.50 | 2052.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 2015.30 | 2011.28 | 2024.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 2015.30 | 2011.28 | 2024.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 2020.30 | 2012.99 | 2022.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:30:00 | 2025.00 | 2012.99 | 2022.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1997.00 | 2011.43 | 2019.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 1988.30 | 2003.99 | 2013.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1984.50 | 2000.22 | 2009.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 1977.90 | 1986.92 | 1995.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1888.88 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1885.27 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1879.01 | 1915.21 | 1939.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1905.40 | 1901.28 | 1925.92 | SL hit (close>ema200) qty=0.50 sl=1901.28 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 1939.00 | 1926.78 | 1925.87 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1875.40 | 1916.50 | 1921.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1853.30 | 1869.76 | 1880.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1855.40 | 1854.96 | 1869.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:15:00 | 1853.80 | 1854.96 | 1869.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 105 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 1954.70 | 1867.76 | 1867.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1982.80 | 1890.77 | 1877.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1964.80 | 1971.26 | 1936.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:00:00 | 1964.80 | 1971.26 | 1936.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1933.60 | 1963.77 | 1948.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1933.60 | 1963.77 | 1948.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1935.00 | 1958.02 | 1946.92 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 1920.00 | 1939.91 | 1940.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1896.20 | 1924.64 | 1932.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1863.70 | 1863.43 | 1884.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 1863.70 | 1863.43 | 1884.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1844.60 | 1843.58 | 1851.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1834.90 | 1844.25 | 1848.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1835.10 | 1843.38 | 1847.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1834.80 | 1843.38 | 1847.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 1833.00 | 1840.24 | 1845.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1850.00 | 1839.09 | 1842.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 1851.30 | 1839.09 | 1842.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1846.90 | 1840.65 | 1842.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1842.00 | 1841.52 | 1843.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1837.20 | 1841.51 | 1842.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 1885.70 | 1835.77 | 1835.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1885.70 | 1835.77 | 1835.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 12:15:00 | 1914.30 | 1851.48 | 1842.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1863.00 | 1867.59 | 1858.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1863.00 | 1867.59 | 1858.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1860.00 | 1866.07 | 1858.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1860.00 | 1866.07 | 1858.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1854.00 | 1863.65 | 1858.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1840.10 | 1863.65 | 1858.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1834.70 | 1857.86 | 1856.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1834.70 | 1857.86 | 1856.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 1830.00 | 1852.29 | 1853.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 1825.80 | 1846.99 | 1851.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1852.00 | 1844.55 | 1848.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1852.00 | 1844.55 | 1848.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1843.80 | 1844.40 | 1848.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1836.00 | 1844.40 | 1848.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1858.70 | 1835.73 | 1837.34 | SL hit (close>static) qty=1.00 sl=1857.30 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1858.00 | 1840.18 | 1839.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 1870.20 | 1846.18 | 1842.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1864.80 | 1877.73 | 1865.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1864.80 | 1877.73 | 1865.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1854.00 | 1872.98 | 1864.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1872.50 | 1872.98 | 1864.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 1872.00 | 1872.79 | 1864.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1863.70 | 1870.56 | 1871.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1863.70 | 1870.56 | 1871.34 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1877.80 | 1871.77 | 1871.65 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 1867.00 | 1871.52 | 1871.60 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1872.50 | 1871.72 | 1871.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 1918.10 | 1880.99 | 1875.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1858.50 | 1881.41 | 1878.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1858.50 | 1881.41 | 1878.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1855.00 | 1876.12 | 1876.76 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 1890.70 | 1879.04 | 1878.03 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 1877.60 | 1880.95 | 1881.00 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 1899.00 | 1884.56 | 1882.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1902.20 | 1888.96 | 1885.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 1907.90 | 1912.96 | 1903.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:00:00 | 1907.90 | 1912.96 | 1903.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1902.90 | 1910.95 | 1903.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:30:00 | 1901.20 | 1910.95 | 1903.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1909.00 | 1910.56 | 1903.76 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 1886.00 | 1900.79 | 1902.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 1882.90 | 1897.22 | 1900.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1899.00 | 1847.03 | 1852.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 1883.60 | 1847.03 | 1852.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 1905.20 | 1858.66 | 1857.15 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1862.30 | 1871.58 | 1872.63 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1878.10 | 1873.44 | 1873.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 1883.60 | 1876.59 | 1874.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1874.70 | 1876.21 | 1874.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 1874.70 | 1876.21 | 1874.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1884.00 | 1877.77 | 1875.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 1894.50 | 1880.41 | 1876.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 1889.80 | 1881.49 | 1877.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1872.80 | 1878.70 | 1877.10 | SL hit (close<static) qty=1.00 sl=1873.50 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 1870.80 | 1875.58 | 1875.86 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1886.10 | 1876.59 | 1876.20 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1873.10 | 1875.89 | 1875.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 1870.20 | 1874.75 | 1875.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1875.60 | 1871.84 | 1873.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1875.60 | 1871.84 | 1873.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1879.00 | 1873.27 | 1874.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 1878.20 | 1873.27 | 1874.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1887.50 | 1876.12 | 1875.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 1910.00 | 1882.90 | 1878.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 15:15:00 | 1896.10 | 1899.35 | 1889.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:15:00 | 1890.20 | 1899.35 | 1889.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1884.60 | 1896.40 | 1889.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1884.60 | 1896.40 | 1889.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1872.60 | 1891.64 | 1887.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1872.60 | 1891.64 | 1887.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1867.40 | 1882.10 | 1883.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1832.20 | 1869.58 | 1877.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1808.00 | 1803.88 | 1822.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1825.40 | 1803.88 | 1822.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1828.80 | 1808.86 | 1823.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1811.00 | 1814.24 | 1821.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1841.90 | 1825.00 | 1824.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1841.90 | 1825.00 | 1824.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 1848.60 | 1837.78 | 1831.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1835.60 | 1837.34 | 1832.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 1829.40 | 1837.34 | 1832.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1822.50 | 1834.37 | 1831.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 1822.50 | 1834.37 | 1831.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1826.40 | 1832.78 | 1830.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 1823.20 | 1832.78 | 1830.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1826.90 | 1831.60 | 1830.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 1823.10 | 1831.60 | 1830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 1821.00 | 1829.48 | 1829.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1812.80 | 1824.63 | 1827.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1758.00 | 1753.29 | 1777.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 1758.00 | 1753.29 | 1777.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1768.80 | 1747.55 | 1762.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1771.30 | 1747.55 | 1762.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1769.00 | 1751.84 | 1762.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 1768.70 | 1751.84 | 1762.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1761.10 | 1755.01 | 1762.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1765.00 | 1755.01 | 1762.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1776.00 | 1759.21 | 1763.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1776.00 | 1759.21 | 1763.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1777.20 | 1762.81 | 1764.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:15:00 | 1790.00 | 1762.81 | 1764.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1790.00 | 1768.25 | 1767.18 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1735.10 | 1762.25 | 1765.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1729.80 | 1755.76 | 1762.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1719.60 | 1715.27 | 1731.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1728.00 | 1715.27 | 1731.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1724.20 | 1716.32 | 1725.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 1727.20 | 1716.32 | 1725.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1728.80 | 1718.82 | 1725.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 1726.20 | 1718.82 | 1725.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1715.00 | 1718.06 | 1724.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 1708.20 | 1718.06 | 1724.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1710.50 | 1715.86 | 1720.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1692.00 | 1716.05 | 1720.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 1732.70 | 1723.24 | 1722.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1732.70 | 1723.24 | 1722.14 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1679.00 | 1713.01 | 1717.59 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1742.60 | 1704.77 | 1703.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1786.30 | 1776.62 | 1766.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1793.30 | 1793.75 | 1783.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1793.30 | 1793.75 | 1783.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1827.10 | 1836.10 | 1818.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 1822.60 | 1836.10 | 1818.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1816.80 | 1826.86 | 1820.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1816.80 | 1826.86 | 1820.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1808.00 | 1823.09 | 1819.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 1785.90 | 1823.09 | 1819.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1776.80 | 1813.83 | 1815.32 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1819.30 | 1807.76 | 1807.24 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1798.20 | 1810.42 | 1811.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1793.80 | 1807.10 | 1809.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 1782.60 | 1778.27 | 1790.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 1782.60 | 1778.27 | 1790.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1796.90 | 1783.02 | 1790.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 1800.50 | 1783.02 | 1790.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1799.50 | 1786.32 | 1791.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1799.50 | 1786.32 | 1791.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1766.40 | 1766.75 | 1774.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 1748.30 | 1762.87 | 1770.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1751.90 | 1760.05 | 1767.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1753.00 | 1758.90 | 1765.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1660.88 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1664.31 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1665.35 | 1697.87 | 1719.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 1668.00 | 1667.54 | 1692.35 | SL hit (close>ema200) qty=0.50 sl=1667.54 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 1592.80 | 1575.80 | 1574.07 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1527.60 | 1565.38 | 1570.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1515.10 | 1544.72 | 1558.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1506.00 | 1504.01 | 1521.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 14:15:00 | 1508.70 | 1506.84 | 1516.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1508.70 | 1506.84 | 1516.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1516.80 | 1506.84 | 1516.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1521.00 | 1509.67 | 1516.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1534.80 | 1509.67 | 1516.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1533.60 | 1514.46 | 1518.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1537.60 | 1514.46 | 1518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1541.10 | 1519.79 | 1520.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1541.10 | 1519.79 | 1520.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1543.90 | 1524.61 | 1522.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 1561.70 | 1532.03 | 1526.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1518.00 | 1534.35 | 1529.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1518.00 | 1534.35 | 1529.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1522.90 | 1532.06 | 1529.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 1531.50 | 1529.48 | 1528.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 1517.10 | 1527.01 | 1527.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1517.10 | 1527.01 | 1527.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1513.30 | 1524.27 | 1526.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1532.00 | 1524.43 | 1525.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1532.00 | 1524.43 | 1525.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1522.40 | 1524.02 | 1525.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1518.80 | 1524.02 | 1525.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1442.86 | 1487.25 | 1506.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1440.50 | 1425.43 | 1444.94 | SL hit (close>ema200) qty=0.50 sl=1425.43 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1405.80 | 1387.58 | 1385.24 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1364.00 | 1381.57 | 1383.17 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1399.00 | 1384.88 | 1384.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1413.30 | 1390.57 | 1387.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1422.70 | 1423.47 | 1411.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1422.70 | 1423.47 | 1411.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1448.80 | 1456.58 | 1447.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 1448.00 | 1456.58 | 1447.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1445.30 | 1454.32 | 1447.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1443.80 | 1454.32 | 1447.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1445.90 | 1452.64 | 1447.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 1466.20 | 1452.64 | 1447.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1485.10 | 1459.13 | 1450.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:45:00 | 1495.60 | 1469.46 | 1457.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:00:00 | 1490.10 | 1481.61 | 1466.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1458.30 | 1464.92 | 1464.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1458.30 | 1464.92 | 1464.97 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1483.80 | 1468.70 | 1466.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1507.20 | 1479.34 | 1472.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1477.20 | 1486.79 | 1479.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1477.20 | 1486.79 | 1479.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1500.50 | 1489.53 | 1481.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 1505.00 | 1489.53 | 1481.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1529.70 | 1495.32 | 1487.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 1513.30 | 1518.20 | 1506.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 1508.20 | 1532.31 | 1533.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1508.20 | 1532.31 | 1533.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1503.00 | 1526.45 | 1530.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1512.00 | 1506.77 | 1517.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1516.90 | 1506.77 | 1517.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1512.00 | 1507.82 | 1517.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1516.70 | 1507.82 | 1517.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1522.90 | 1511.33 | 1517.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1522.90 | 1511.33 | 1517.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1522.10 | 1513.48 | 1517.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 1517.20 | 1513.87 | 1517.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1528.50 | 1505.39 | 1505.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1528.50 | 1505.39 | 1505.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1547.30 | 1513.77 | 1509.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1519.80 | 1523.37 | 1517.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:15:00 | 1513.00 | 1523.37 | 1517.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1515.50 | 1521.80 | 1517.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 1511.60 | 1521.80 | 1517.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1511.20 | 1519.68 | 1516.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 1505.50 | 1519.68 | 1516.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1516.60 | 1519.06 | 1516.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:45:00 | 1517.00 | 1516.91 | 1515.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1525.80 | 1516.19 | 1515.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 1557.65 | 2024-05-16 13:15:00 | 1536.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-05-16 10:30:00 | 1554.50 | 2024-05-16 13:15:00 | 1536.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-05-16 11:45:00 | 1555.00 | 2024-05-16 13:15:00 | 1536.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-16 12:30:00 | 1552.25 | 2024-05-16 13:15:00 | 1536.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-05-16 15:15:00 | 1553.35 | 2024-05-17 15:15:00 | 1542.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-05-17 09:30:00 | 1552.20 | 2024-05-17 15:15:00 | 1542.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-05-17 10:45:00 | 1554.70 | 2024-05-17 15:15:00 | 1542.40 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-17 14:45:00 | 1554.45 | 2024-05-17 15:15:00 | 1542.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-29 13:45:00 | 1511.50 | 2024-06-04 09:15:00 | 1435.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 14:30:00 | 1511.00 | 2024-06-04 09:15:00 | 1435.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 13:45:00 | 1511.50 | 2024-06-04 12:15:00 | 1360.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-29 14:30:00 | 1511.00 | 2024-06-04 12:15:00 | 1359.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-20 14:15:00 | 1630.00 | 2024-06-21 09:15:00 | 1697.80 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-07-02 15:00:00 | 2084.00 | 2024-07-05 09:15:00 | 2049.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-07-03 10:15:00 | 2087.50 | 2024-07-05 09:15:00 | 2049.15 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-04 09:30:00 | 2095.90 | 2024-07-05 09:15:00 | 2049.15 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-07-04 11:15:00 | 2090.80 | 2024-07-05 09:15:00 | 2049.15 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-07-11 11:45:00 | 2258.05 | 2024-07-16 14:15:00 | 2272.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2024-07-11 13:45:00 | 2312.90 | 2024-07-16 14:15:00 | 2272.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-07-15 10:00:00 | 2305.00 | 2024-07-16 14:15:00 | 2272.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-08-06 13:15:00 | 2158.00 | 2024-08-08 09:15:00 | 2242.80 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2024-08-14 14:45:00 | 2268.10 | 2024-08-19 09:15:00 | 2445.20 | STOP_HIT | 1.00 | -7.81% |
| SELL | retest2 | 2024-08-14 15:15:00 | 2270.00 | 2024-08-19 09:15:00 | 2445.20 | STOP_HIT | 1.00 | -7.72% |
| SELL | retest2 | 2024-08-16 09:45:00 | 2270.00 | 2024-08-19 09:15:00 | 2445.20 | STOP_HIT | 1.00 | -7.72% |
| SELL | retest2 | 2024-08-16 11:00:00 | 2268.75 | 2024-08-19 09:15:00 | 2445.20 | STOP_HIT | 1.00 | -7.78% |
| BUY | retest2 | 2024-08-27 09:30:00 | 2549.45 | 2024-08-29 12:15:00 | 2534.75 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-08-27 14:15:00 | 2548.35 | 2024-08-29 12:15:00 | 2534.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-08-28 09:15:00 | 2585.55 | 2024-08-29 12:15:00 | 2534.75 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-09-11 14:30:00 | 2690.70 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-09-11 15:15:00 | 2674.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-09-12 14:45:00 | 2677.25 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-09-13 09:15:00 | 2699.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-09-13 11:45:00 | 2745.95 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-09-16 09:15:00 | 2818.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2024-09-17 09:30:00 | 2706.50 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-17 11:15:00 | 2707.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-09-17 12:30:00 | 2720.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-18 10:30:00 | 2720.75 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-09-18 12:00:00 | 2714.00 | 2024-09-18 14:15:00 | 2696.95 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-20 14:15:00 | 2590.80 | 2024-09-24 09:15:00 | 2696.50 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2024-11-18 12:45:00 | 2573.30 | 2024-11-22 10:15:00 | 2444.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 10:30:00 | 2581.25 | 2024-11-22 10:15:00 | 2452.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 11:45:00 | 2578.05 | 2024-11-22 10:15:00 | 2449.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 14:15:00 | 2583.65 | 2024-11-22 10:15:00 | 2454.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-18 12:45:00 | 2573.30 | 2024-11-25 09:15:00 | 2516.95 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2024-11-19 10:30:00 | 2581.25 | 2024-11-25 09:15:00 | 2516.95 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2024-11-19 11:45:00 | 2578.05 | 2024-11-25 09:15:00 | 2516.95 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-11-19 14:15:00 | 2583.65 | 2024-11-25 09:15:00 | 2516.95 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2024-12-03 15:15:00 | 2409.15 | 2024-12-09 10:15:00 | 2410.55 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-12-04 09:30:00 | 2406.00 | 2024-12-09 10:15:00 | 2410.55 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-12-04 15:00:00 | 2405.75 | 2024-12-09 10:15:00 | 2410.55 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-12-30 09:15:00 | 2085.15 | 2025-01-01 09:15:00 | 2227.50 | STOP_HIT | 1.00 | -6.83% |
| SELL | retest2 | 2024-12-31 13:15:00 | 2115.00 | 2025-01-01 09:15:00 | 2227.50 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2025-01-10 09:15:00 | 2049.85 | 2025-01-13 14:15:00 | 1947.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 2049.85 | 2025-01-14 10:15:00 | 1990.10 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2025-01-17 14:45:00 | 2006.70 | 2025-01-20 10:15:00 | 1999.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-01-20 09:45:00 | 2005.95 | 2025-01-20 10:15:00 | 1999.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-02-04 09:30:00 | 2138.10 | 2025-02-04 10:15:00 | 2112.35 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-02-07 11:15:00 | 2142.00 | 2025-02-07 13:15:00 | 2111.15 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-02-18 11:45:00 | 1828.70 | 2025-02-19 10:15:00 | 1917.80 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2025-02-18 12:15:00 | 1824.00 | 2025-02-19 10:15:00 | 1917.80 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2025-02-20 14:15:00 | 1901.00 | 2025-02-21 09:15:00 | 1839.60 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-02-24 13:15:00 | 1827.60 | 2025-02-27 11:15:00 | 1736.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 13:15:00 | 1827.60 | 2025-02-28 09:15:00 | 1644.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-13 11:30:00 | 1657.95 | 2025-03-18 13:15:00 | 1686.55 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-03-13 14:00:00 | 1660.45 | 2025-03-18 13:15:00 | 1686.55 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1778.70 | 2025-03-26 09:15:00 | 1762.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-04-01 09:30:00 | 1779.35 | 2025-04-04 12:15:00 | 1773.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-04-01 12:30:00 | 1779.00 | 2025-04-04 12:15:00 | 1773.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-04-01 15:15:00 | 1779.90 | 2025-04-04 12:15:00 | 1773.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-04-02 09:30:00 | 1771.85 | 2025-04-04 12:15:00 | 1773.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-04-09 10:00:00 | 1722.00 | 2025-04-09 11:15:00 | 1782.20 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-04-09 10:30:00 | 1722.45 | 2025-04-09 11:15:00 | 1782.20 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-05-06 11:15:00 | 1876.10 | 2025-05-07 09:15:00 | 1782.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 1876.10 | 2025-05-12 09:15:00 | 1779.00 | STOP_HIT | 0.50 | 5.18% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1819.00 | 2025-05-19 09:15:00 | 2000.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1941.90 | 2025-06-20 14:15:00 | 1844.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1955.70 | 2025-06-20 14:15:00 | 1857.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 10:45:00 | 1948.70 | 2025-06-20 14:15:00 | 1851.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1941.90 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1955.70 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-06-18 10:45:00 | 1948.70 | 2025-06-23 14:15:00 | 1880.30 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-07-02 11:45:00 | 2007.50 | 2025-07-03 15:15:00 | 1980.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-02 12:30:00 | 2015.90 | 2025-07-03 15:15:00 | 1980.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-09 12:45:00 | 1955.90 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-09 14:15:00 | 1956.20 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-10 09:30:00 | 1955.10 | 2025-07-10 14:15:00 | 1974.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1931.10 | 2025-07-15 10:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-18 12:30:00 | 2019.80 | 2025-07-18 14:15:00 | 2004.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-29 15:15:00 | 1935.10 | 2025-07-30 09:15:00 | 1952.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1933.20 | 2025-08-06 15:15:00 | 1836.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1933.20 | 2025-08-07 12:15:00 | 1855.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-08-18 12:00:00 | 1832.80 | 2025-08-18 14:15:00 | 1852.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-20 10:30:00 | 1869.30 | 2025-08-21 14:15:00 | 1840.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-08-29 12:15:00 | 1783.40 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-01 10:45:00 | 1787.10 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1782.00 | 2025-09-02 10:15:00 | 1822.80 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-09-08 15:00:00 | 1862.80 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-09-09 11:00:00 | 1858.70 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.26% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1860.30 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 4.17% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1867.00 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 3.80% |
| BUY | retest2 | 2025-09-15 10:15:00 | 1911.00 | 2025-09-19 14:15:00 | 1937.90 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-10-08 10:15:00 | 1884.60 | 2025-10-08 14:15:00 | 1844.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-13 12:15:00 | 1880.80 | 2025-10-17 12:15:00 | 2068.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-13 13:30:00 | 1882.00 | 2025-10-17 12:15:00 | 2070.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 15:00:00 | 1907.80 | 2025-10-23 09:15:00 | 2098.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1988.30 | 2025-11-07 09:15:00 | 1888.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1984.50 | 2025-11-07 09:15:00 | 1885.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1977.90 | 2025-11-07 09:15:00 | 1879.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1988.30 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1984.50 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1977.90 | 2025-11-07 12:15:00 | 1905.40 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1834.90 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-28 09:45:00 | 1835.10 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1834.80 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-28 12:00:00 | 1833.00 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1842.00 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-12-01 14:15:00 | 1837.20 | 2025-12-03 11:15:00 | 1885.70 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1836.00 | 2025-12-09 11:15:00 | 1858.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1872.50 | 2025-12-15 09:15:00 | 1863.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-11 10:00:00 | 1872.00 | 2025-12-15 09:15:00 | 1863.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-01-05 09:30:00 | 1894.50 | 2026-01-05 12:15:00 | 1872.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-05 10:45:00 | 1889.80 | 2026-01-05 12:15:00 | 1872.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1811.00 | 2026-01-14 10:15:00 | 1841.90 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-29 10:15:00 | 1708.20 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1710.50 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1692.00 | 2026-02-01 10:15:00 | 1732.70 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1748.30 | 2026-03-02 09:15:00 | 1660.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1751.90 | 2026-03-02 09:15:00 | 1664.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:30:00 | 1753.00 | 2026-03-02 09:15:00 | 1665.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1748.30 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1751.90 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2026-02-26 10:30:00 | 1753.00 | 2026-03-02 15:15:00 | 1668.00 | STOP_HIT | 0.50 | 4.85% |
| BUY | retest2 | 2026-03-19 12:45:00 | 1531.50 | 2026-03-19 13:15:00 | 1517.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1518.80 | 2026-03-23 10:15:00 | 1442.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1518.80 | 2026-03-25 09:15:00 | 1440.50 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2026-04-10 11:45:00 | 1495.60 | 2026-04-13 15:15:00 | 1458.30 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-10 15:00:00 | 1490.10 | 2026-04-13 15:15:00 | 1458.30 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-16 11:15:00 | 1505.00 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1529.70 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-20 09:45:00 | 1513.30 | 2026-04-24 10:15:00 | 1508.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-04-27 14:30:00 | 1517.20 | 2026-05-04 09:15:00 | 1528.50 | STOP_HIT | 1.00 | -0.74% |
