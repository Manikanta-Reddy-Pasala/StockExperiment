# Piramal Finance Ltd. (PIRAMALFIN)

## Backtest Summary

- **Window:** 2025-11-07 09:15:00 → 2026-05-08 15:15:00 (861 bars)
- **Last close:** 2015.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 42 |
| ALERT1 | 22 |
| ALERT2 | 21 |
| ALERT2_SKIP | 14 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 28
- **Target hits / Stop hits / Partials:** 0 / 40 / 8
- **Avg / median % per leg:** 0.44% / -0.55%
- **Sum % (uncompounded):** 21.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 3 | 14.3% | 0 | 21 | 0 | -1.51% | -31.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 3 | 14.3% | 0 | 21 | 0 | -1.51% | -31.6% |
| SELL (all) | 27 | 17 | 63.0% | 0 | 19 | 8 | 1.95% | 52.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 17 | 63.0% | 0 | 19 | 8 | 1.95% | 52.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 20 | 41.7% | 0 | 40 | 8 | 0.44% | 21.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 1594.00 | 1645.63 | 1645.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1592.90 | 1635.08 | 1641.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 1629.80 | 1609.55 | 1620.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 1629.80 | 1609.55 | 1620.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1629.80 | 1609.55 | 1620.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 1628.20 | 1609.55 | 1620.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1598.30 | 1607.30 | 1618.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1591.20 | 1607.30 | 1618.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1579.00 | 1601.64 | 1614.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:00:00 | 1568.30 | 1585.72 | 1597.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1570.00 | 1582.86 | 1595.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1568.00 | 1574.91 | 1589.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 1570.10 | 1576.94 | 1585.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1586.90 | 1569.37 | 1576.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 1582.70 | 1569.37 | 1576.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1575.60 | 1570.62 | 1576.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1561.00 | 1570.62 | 1576.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1489.88 | 1517.44 | 1538.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1491.50 | 1517.44 | 1538.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1489.60 | 1517.44 | 1538.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1491.59 | 1517.44 | 1538.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1482.95 | 1517.44 | 1538.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 1518.50 | 1502.69 | 1523.21 | SL hit (close>ema200) qty=0.50 sl=1502.69 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1516.90 | 1474.11 | 1473.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 1534.20 | 1504.58 | 1490.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 1519.30 | 1525.60 | 1508.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 1519.30 | 1525.60 | 1508.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1532.00 | 1539.83 | 1525.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1520.10 | 1539.83 | 1525.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1506.80 | 1533.22 | 1523.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1506.80 | 1533.22 | 1523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1507.50 | 1528.08 | 1522.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 1507.00 | 1528.08 | 1522.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1520.10 | 1523.68 | 1521.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 1523.80 | 1523.68 | 1521.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1520.00 | 1522.94 | 1521.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 1503.70 | 1522.94 | 1521.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1524.60 | 1523.27 | 1521.62 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 1515.10 | 1520.33 | 1520.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1493.00 | 1514.87 | 1518.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 1522.10 | 1505.39 | 1510.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 1522.10 | 1505.39 | 1510.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1522.10 | 1505.39 | 1510.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1522.10 | 1505.39 | 1510.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1505.00 | 1505.31 | 1510.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1551.20 | 1505.31 | 1510.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1531.60 | 1510.57 | 1512.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 1522.90 | 1510.57 | 1512.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1532.00 | 1514.85 | 1513.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 1532.00 | 1514.85 | 1513.99 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 1503.10 | 1513.64 | 1514.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1495.80 | 1510.07 | 1512.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1513.50 | 1510.76 | 1512.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 1513.50 | 1510.76 | 1512.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1513.50 | 1510.76 | 1512.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1516.80 | 1510.76 | 1512.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1518.00 | 1512.20 | 1512.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 1517.90 | 1512.20 | 1512.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1515.30 | 1512.82 | 1513.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 1519.90 | 1512.82 | 1513.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 1542.00 | 1519.05 | 1515.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 1564.50 | 1530.88 | 1522.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1616.90 | 1621.40 | 1586.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 1616.90 | 1621.40 | 1586.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1578.80 | 1606.12 | 1587.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 1576.90 | 1606.12 | 1587.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1577.80 | 1600.46 | 1586.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1571.40 | 1600.46 | 1586.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1616.70 | 1609.67 | 1596.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 1640.40 | 1620.02 | 1607.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 1637.00 | 1622.26 | 1610.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 1635.50 | 1630.68 | 1622.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 1594.70 | 1615.35 | 1617.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 1594.70 | 1615.35 | 1617.23 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 1652.10 | 1613.33 | 1612.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1668.40 | 1642.50 | 1632.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 1860.10 | 1865.12 | 1839.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 15:15:00 | 1860.10 | 1865.12 | 1839.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1860.10 | 1865.12 | 1839.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 1861.10 | 1859.14 | 1839.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1842.70 | 1855.85 | 1839.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 12:00:00 | 1854.90 | 1855.66 | 1840.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 1851.50 | 1854.53 | 1841.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 1850.30 | 1853.68 | 1842.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 10:00:00 | 1856.80 | 1858.80 | 1848.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1849.10 | 1855.89 | 1848.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 1872.30 | 1847.60 | 1846.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:00:00 | 1870.10 | 1852.10 | 1848.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 13:15:00 | 1837.00 | 1858.53 | 1853.88 | SL hit (close<static) qty=1.00 sl=1843.80 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 1830.60 | 1848.22 | 1850.29 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1894.80 | 1855.40 | 1850.48 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1840.00 | 1859.63 | 1859.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 1829.80 | 1853.66 | 1857.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1834.70 | 1821.87 | 1834.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1834.70 | 1821.87 | 1834.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1834.70 | 1821.87 | 1834.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:30:00 | 1832.50 | 1821.87 | 1834.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 1810.00 | 1819.49 | 1831.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 11:30:00 | 1785.80 | 1816.39 | 1829.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 1781.00 | 1805.26 | 1813.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 1782.00 | 1800.61 | 1810.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1757.10 | 1788.41 | 1800.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1762.00 | 1783.13 | 1796.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:30:00 | 1722.90 | 1772.98 | 1790.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 1727.20 | 1745.03 | 1755.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 13:15:00 | 1696.51 | 1722.93 | 1741.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 13:15:00 | 1691.95 | 1722.93 | 1741.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 13:15:00 | 1692.90 | 1722.93 | 1741.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1749.80 | 1728.31 | 1741.96 | SL hit (close>ema200) qty=0.50 sl=1728.31 alert=retest2 |

### Cycle 12 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1736.20 | 1709.67 | 1706.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1752.30 | 1718.20 | 1710.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 14:15:00 | 1722.90 | 1725.51 | 1716.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 15:00:00 | 1722.90 | 1725.51 | 1716.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1760.90 | 1732.35 | 1721.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:15:00 | 1782.60 | 1732.35 | 1721.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1775.40 | 1741.48 | 1726.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:30:00 | 1766.10 | 1750.44 | 1733.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:00:00 | 1763.40 | 1753.67 | 1742.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1784.50 | 1764.91 | 1752.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 1797.10 | 1771.35 | 1756.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1796.80 | 1784.04 | 1776.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 1798.00 | 1785.95 | 1777.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 1771.00 | 1777.98 | 1778.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 13:15:00 | 1771.00 | 1777.98 | 1778.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 1762.20 | 1774.83 | 1777.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1756.00 | 1752.77 | 1757.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1756.00 | 1752.77 | 1757.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1756.00 | 1752.77 | 1757.66 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 1770.00 | 1757.90 | 1757.78 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1731.30 | 1752.58 | 1755.38 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1778.90 | 1759.39 | 1757.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1781.90 | 1763.89 | 1759.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 15:15:00 | 1765.50 | 1769.84 | 1764.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 15:15:00 | 1765.50 | 1769.84 | 1764.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1765.50 | 1769.84 | 1764.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1789.90 | 1769.84 | 1764.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 1762.00 | 1767.76 | 1766.42 | SL hit (close<static) qty=1.00 sl=1763.00 alert=retest2 |

### Cycle 17 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1742.40 | 1762.69 | 1764.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1731.70 | 1744.96 | 1752.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1800.60 | 1749.94 | 1751.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1800.60 | 1749.94 | 1751.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1800.60 | 1749.94 | 1751.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 1800.60 | 1749.94 | 1751.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 15:15:00 | 1799.00 | 1759.75 | 1755.58 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 1733.70 | 1749.73 | 1751.45 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1793.00 | 1753.89 | 1752.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1808.00 | 1776.16 | 1766.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 1772.50 | 1778.12 | 1768.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 1772.50 | 1778.12 | 1768.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1772.50 | 1778.12 | 1768.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1772.50 | 1778.12 | 1768.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1752.40 | 1772.98 | 1767.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1756.00 | 1772.98 | 1767.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1739.70 | 1766.32 | 1764.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1739.70 | 1766.32 | 1764.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 1751.50 | 1763.36 | 1763.64 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 1780.00 | 1763.93 | 1763.68 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1695.70 | 1751.89 | 1758.35 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 1784.70 | 1745.17 | 1744.02 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 1724.50 | 1745.47 | 1745.66 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 14:15:00 | 1780.80 | 1749.40 | 1747.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 14:15:00 | 1793.80 | 1761.81 | 1755.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1748.30 | 1762.82 | 1757.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1748.30 | 1762.82 | 1757.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1748.30 | 1762.82 | 1757.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 1748.30 | 1762.82 | 1757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1745.30 | 1759.32 | 1756.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 1748.10 | 1759.32 | 1756.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 1742.70 | 1753.88 | 1754.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1721.00 | 1743.02 | 1748.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1788.30 | 1738.95 | 1742.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1788.30 | 1738.95 | 1742.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1788.30 | 1738.95 | 1742.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1788.30 | 1738.95 | 1742.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 1782.00 | 1747.56 | 1745.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 1841.00 | 1793.38 | 1776.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1790.00 | 1793.33 | 1779.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 10:30:00 | 1797.30 | 1793.33 | 1779.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1780.30 | 1788.99 | 1780.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 1780.30 | 1788.99 | 1780.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1809.20 | 1793.03 | 1783.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:45:00 | 1815.90 | 1799.02 | 1787.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 10:15:00 | 1748.60 | 1804.39 | 1801.62 | SL hit (close<static) qty=1.00 sl=1770.10 alert=retest2 |

### Cycle 29 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 1769.00 | 1797.32 | 1798.65 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 1827.50 | 1798.69 | 1798.40 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1785.50 | 1797.61 | 1798.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 11:15:00 | 1782.30 | 1794.55 | 1796.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 1794.90 | 1792.95 | 1795.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 14:15:00 | 1794.90 | 1792.95 | 1795.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1794.90 | 1792.95 | 1795.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 1820.00 | 1792.95 | 1795.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1787.90 | 1791.94 | 1794.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1786.30 | 1791.94 | 1794.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1791.00 | 1791.75 | 1794.26 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1811.00 | 1797.90 | 1796.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 1831.40 | 1807.45 | 1801.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 1790.00 | 1805.17 | 1801.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 11:15:00 | 1790.00 | 1805.17 | 1801.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1790.00 | 1805.17 | 1801.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1790.00 | 1805.17 | 1801.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1804.90 | 1805.11 | 1801.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 1810.50 | 1803.19 | 1801.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1869.20 | 1816.39 | 1807.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 1783.00 | 1814.35 | 1812.22 | SL hit (close<static) qty=1.00 sl=1786.60 alert=retest2 |

### Cycle 33 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1784.10 | 1808.30 | 1809.66 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 1835.00 | 1813.64 | 1811.97 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1802.00 | 1812.16 | 1812.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 15:15:00 | 1773.00 | 1804.33 | 1809.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 1757.00 | 1755.54 | 1772.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 12:00:00 | 1757.00 | 1755.54 | 1772.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1723.70 | 1718.70 | 1736.02 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 10:15:00 | 1756.50 | 1739.14 | 1736.93 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 1736.80 | 1738.26 | 1738.45 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 1755.00 | 1741.60 | 1739.96 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 1732.40 | 1741.18 | 1741.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 1708.00 | 1733.97 | 1738.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 14:15:00 | 1653.00 | 1650.85 | 1673.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-20 15:00:00 | 1653.00 | 1650.85 | 1673.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1687.30 | 1656.37 | 1668.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1687.30 | 1656.37 | 1668.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1710.50 | 1667.20 | 1671.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 1698.20 | 1667.20 | 1671.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 1699.10 | 1678.07 | 1676.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1730.10 | 1690.39 | 1682.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 11:15:00 | 1818.60 | 1837.65 | 1806.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 1818.60 | 1837.65 | 1806.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1850.00 | 1836.54 | 1811.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:30:00 | 1814.30 | 1836.54 | 1811.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 1837.90 | 1849.90 | 1835.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1924.40 | 1849.90 | 1835.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 1939.30 | 1964.88 | 1965.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 1939.30 | 1964.88 | 1965.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 1928.30 | 1957.56 | 1962.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1939.60 | 1918.46 | 1929.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1939.60 | 1918.46 | 1929.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1939.60 | 1918.46 | 1929.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 1940.00 | 1918.46 | 1929.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1942.70 | 1923.31 | 1931.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:45:00 | 1945.60 | 1923.31 | 1931.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1928.80 | 1924.41 | 1930.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1926.70 | 1925.32 | 1930.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 1925.60 | 1925.32 | 1930.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 1946.90 | 1929.68 | 1931.24 | SL hit (close>static) qty=1.00 sl=1943.90 alert=retest2 |

### Cycle 42 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1987.00 | 1933.28 | 1929.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 10:15:00 | 2026.50 | 1951.92 | 1938.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-27 12:00:00 | 1568.30 | 2025-12-03 09:15:00 | 1489.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1570.00 | 2025-12-03 09:15:00 | 1491.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 15:15:00 | 1568.00 | 2025-12-03 09:15:00 | 1489.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 12:00:00 | 1570.10 | 2025-12-03 09:15:00 | 1491.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1561.00 | 2025-12-03 09:15:00 | 1482.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:00:00 | 1568.30 | 2025-12-03 13:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1570.00 | 2025-12-03 13:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-11-27 15:15:00 | 1568.00 | 2025-12-03 13:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-11-28 12:00:00 | 1570.10 | 2025-12-03 13:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1561.00 | 2025-12-03 13:15:00 | 1518.50 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-12-17 10:15:00 | 1522.90 | 2025-12-17 10:15:00 | 1532.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-24 14:30:00 | 1640.40 | 2025-12-29 13:15:00 | 1594.70 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-12-26 09:45:00 | 1637.00 | 2025-12-29 13:15:00 | 1594.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-12-29 10:30:00 | 1635.50 | 2025-12-29 13:15:00 | 1594.70 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-01-09 12:00:00 | 1854.90 | 2026-01-13 13:15:00 | 1837.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-09 12:45:00 | 1851.50 | 2026-01-13 13:15:00 | 1837.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-01-09 14:00:00 | 1850.30 | 2026-01-14 10:15:00 | 1830.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-12 10:00:00 | 1856.80 | 2026-01-14 10:15:00 | 1830.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-13 09:15:00 | 1872.30 | 2026-01-14 10:15:00 | 1830.60 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-01-13 10:00:00 | 1870.10 | 2026-01-14 10:15:00 | 1830.60 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-01-21 11:30:00 | 1785.80 | 2026-02-01 13:15:00 | 1696.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1781.00 | 2026-02-01 13:15:00 | 1691.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 12:00:00 | 1782.00 | 2026-02-01 13:15:00 | 1692.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 11:30:00 | 1785.80 | 2026-02-01 14:15:00 | 1749.80 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1781.00 | 2026-02-01 14:15:00 | 1749.80 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2026-01-23 12:00:00 | 1782.00 | 2026-02-01 14:15:00 | 1749.80 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1757.10 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-01-27 10:30:00 | 1722.90 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-02-01 10:00:00 | 1727.20 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-02-02 09:15:00 | 1727.30 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-02-02 10:00:00 | 1726.70 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-02-03 10:30:00 | 1726.60 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-09 10:15:00 | 1723.00 | 2026-02-09 10:15:00 | 1736.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-02-10 10:15:00 | 1782.60 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-10 10:45:00 | 1775.40 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-10 12:30:00 | 1766.10 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2026-02-11 12:00:00 | 1763.40 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2026-02-12 11:00:00 | 1797.10 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-02-17 10:45:00 | 1796.80 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-17 12:15:00 | 1798.00 | 2026-02-18 13:15:00 | 1771.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1789.90 | 2026-02-26 15:15:00 | 1762.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-03-20 09:45:00 | 1815.90 | 2026-03-23 10:15:00 | 1748.60 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-03-27 13:30:00 | 1810.50 | 2026-03-30 12:15:00 | 1783.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-03-27 15:00:00 | 1869.20 | 2026-03-30 12:15:00 | 1783.00 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-04-28 09:15:00 | 1924.40 | 2026-05-04 12:15:00 | 1939.30 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2026-05-06 12:30:00 | 1926.70 | 2026-05-06 15:15:00 | 1946.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-05-06 13:15:00 | 1925.60 | 2026-05-06 15:15:00 | 1946.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-05-07 09:30:00 | 1925.50 | 2026-05-08 09:15:00 | 1987.00 | STOP_HIT | 1.00 | -3.19% |
