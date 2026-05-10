# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 25 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 66 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 45
- **Target hits / Stop hits / Partials:** 2 / 66 / 0
- **Avg / median % per leg:** 0.23% / -0.69%
- **Sum % (uncompounded):** 15.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 15 | 36.6% | 2 | 39 | 0 | 1.15% | 47.1% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.77% | -2.8% |
| BUY @ 3rd Alert (retest2) | 40 | 15 | 37.5% | 2 | 38 | 0 | 1.25% | 49.8% |
| SELL (all) | 27 | 8 | 29.6% | 0 | 27 | 0 | -1.15% | -31.1% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.47% | 0.5% |
| SELL @ 3rd Alert (retest2) | 26 | 7 | 26.9% | 0 | 26 | 0 | -1.21% | -31.6% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.15% | -2.3% |
| retest2 (combined) | 66 | 22 | 33.3% | 2 | 64 | 0 | 0.28% | 18.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1292.30 | 1276.04 | 1275.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 15:15:00 | 1299.90 | 1288.45 | 1284.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 1355.50 | 1357.96 | 1343.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:30:00 | 1357.30 | 1357.96 | 1343.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1380.80 | 1380.67 | 1372.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 1375.60 | 1380.67 | 1372.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1385.10 | 1382.31 | 1374.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 1391.70 | 1384.16 | 1377.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-02 12:15:00 | 1530.87 | 1516.06 | 1504.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 1503.00 | 1510.81 | 1511.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 1495.90 | 1507.82 | 1510.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1509.30 | 1507.67 | 1509.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1504.30 | 1506.99 | 1509.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 1495.50 | 1505.34 | 1508.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1514.80 | 1507.16 | 1508.06 | SL hit (close>static) qty=1.00 sl=1512.80 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1511.50 | 1508.68 | 1508.52 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1499.60 | 1506.86 | 1507.71 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 1514.20 | 1508.62 | 1508.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 1520.10 | 1510.92 | 1509.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 1513.90 | 1515.70 | 1512.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1516.20 | 1515.80 | 1513.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 1516.10 | 1515.80 | 1513.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1530.70 | 1531.05 | 1525.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 1526.20 | 1531.05 | 1525.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1520.10 | 1528.86 | 1525.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1520.10 | 1528.86 | 1525.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1520.20 | 1527.13 | 1524.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1519.60 | 1527.13 | 1524.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1513.90 | 1522.96 | 1523.07 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 1531.20 | 1523.75 | 1522.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 1551.00 | 1529.20 | 1525.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1589.70 | 1592.16 | 1578.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:00:00 | 1589.70 | 1592.16 | 1578.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1575.90 | 1588.25 | 1581.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1575.90 | 1588.25 | 1581.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1573.20 | 1585.24 | 1581.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 1569.40 | 1585.24 | 1581.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1576.50 | 1580.70 | 1579.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 1592.20 | 1583.06 | 1580.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 1591.90 | 1586.32 | 1583.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:30:00 | 1591.00 | 1587.97 | 1584.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 1598.70 | 1593.82 | 1592.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1621.40 | 1625.47 | 1618.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 1614.30 | 1625.47 | 1618.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1634.00 | 1631.62 | 1625.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1619.80 | 1631.62 | 1625.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1634.90 | 1632.28 | 1626.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 1651.70 | 1637.85 | 1631.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 1650.90 | 1643.04 | 1635.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 1653.60 | 1650.56 | 1641.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 1649.70 | 1651.72 | 1643.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1653.00 | 1651.97 | 1644.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 1645.90 | 1651.97 | 1644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1642.90 | 1649.96 | 1645.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 1642.90 | 1649.96 | 1645.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1643.10 | 1648.59 | 1645.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1638.90 | 1648.59 | 1645.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1631.80 | 1645.17 | 1644.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1631.80 | 1645.17 | 1644.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 1626.00 | 1639.45 | 1641.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1572.10 | 1565.51 | 1573.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1573.60 | 1567.13 | 1573.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1573.60 | 1567.13 | 1573.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1576.40 | 1568.98 | 1573.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1576.40 | 1568.98 | 1573.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1576.70 | 1570.52 | 1573.72 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1584.30 | 1576.24 | 1575.95 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1564.50 | 1575.09 | 1575.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1562.00 | 1572.47 | 1574.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1579.10 | 1570.59 | 1572.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1585.50 | 1573.57 | 1573.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1588.90 | 1573.57 | 1573.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1577.80 | 1574.42 | 1574.09 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 10:15:00 | 1570.50 | 1573.97 | 1574.33 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1585.00 | 1575.52 | 1574.79 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 1569.80 | 1574.26 | 1574.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1542.20 | 1566.97 | 1571.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 1539.30 | 1538.84 | 1550.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1544.50 | 1540.66 | 1545.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1544.50 | 1540.66 | 1545.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 1544.50 | 1540.66 | 1545.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1545.70 | 1541.67 | 1545.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1542.60 | 1541.67 | 1545.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1552.90 | 1543.92 | 1546.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1552.90 | 1543.92 | 1546.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1550.00 | 1545.13 | 1546.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1547.60 | 1545.05 | 1546.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 1556.60 | 1548.20 | 1547.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 1556.60 | 1548.20 | 1547.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 1564.80 | 1552.77 | 1549.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1556.30 | 1557.64 | 1553.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1551.40 | 1556.39 | 1553.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 1550.60 | 1556.39 | 1553.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1547.70 | 1554.65 | 1552.87 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 1550.00 | 1551.82 | 1551.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1536.20 | 1548.57 | 1550.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 1531.60 | 1528.62 | 1533.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 1532.30 | 1528.62 | 1533.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1552.20 | 1533.34 | 1535.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 1552.20 | 1533.34 | 1535.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1546.70 | 1536.01 | 1536.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 1540.10 | 1536.83 | 1536.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 1537.90 | 1537.04 | 1536.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 12:15:00 | 1537.90 | 1537.04 | 1536.97 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 1530.50 | 1535.73 | 1536.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 15:15:00 | 1524.00 | 1532.34 | 1534.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 1509.50 | 1500.73 | 1510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1504.60 | 1501.50 | 1509.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 1498.30 | 1499.20 | 1508.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 1501.70 | 1498.73 | 1505.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1502.30 | 1500.22 | 1505.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 12:15:00 | 1495.80 | 1488.47 | 1484.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1533.40 | 1533.91 | 1517.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:45:00 | 1535.40 | 1533.91 | 1517.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1616.50 | 1629.69 | 1614.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1609.70 | 1629.69 | 1614.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1620.90 | 1627.93 | 1614.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 1615.50 | 1627.93 | 1614.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1635.30 | 1632.46 | 1622.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:45:00 | 1643.90 | 1637.68 | 1627.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 1644.40 | 1639.31 | 1630.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:15:00 | 1645.40 | 1639.31 | 1630.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1644.60 | 1652.89 | 1644.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1646.10 | 1651.53 | 1644.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 1648.40 | 1651.53 | 1644.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1650.70 | 1651.37 | 1645.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1654.90 | 1645.97 | 1644.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1636.10 | 1650.03 | 1648.62 | SL hit (close<static) qty=1.00 sl=1644.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1629.10 | 1640.94 | 1644.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 1644.20 | 1636.42 | 1640.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1626.10 | 1634.36 | 1639.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:00:00 | 1623.90 | 1632.27 | 1637.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1620.80 | 1618.55 | 1619.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1640.00 | 1622.84 | 1621.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1640.00 | 1622.84 | 1621.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1640.00 | 1622.84 | 1621.51 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 1611.50 | 1622.64 | 1622.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 1603.50 | 1618.81 | 1621.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1615.30 | 1611.93 | 1616.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1620.00 | 1613.55 | 1616.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1590.60 | 1613.55 | 1616.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1603.10 | 1611.46 | 1615.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:45:00 | 1569.30 | 1599.04 | 1609.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 1570.00 | 1593.39 | 1605.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 1566.80 | 1581.66 | 1586.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1609.70 | 1591.95 | 1589.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1605.50 | 1602.76 | 1597.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1603.00 | 1602.81 | 1597.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1604.70 | 1602.81 | 1597.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 1607.00 | 1602.73 | 1598.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 1604.90 | 1602.73 | 1598.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 1604.40 | 1603.10 | 1598.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | SL hit (close<static) qty=1.00 sl=1596.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | SL hit (close<static) qty=1.00 sl=1596.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | SL hit (close<static) qty=1.00 sl=1596.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | SL hit (close<static) qty=1.00 sl=1596.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 1596.00 | 1602.22 | 1599.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1595.10 | 1600.79 | 1599.53 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1594.10 | 1598.58 | 1598.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 1588.30 | 1596.52 | 1597.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1577.80 | 1581.33 | 1586.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1572.80 | 1579.63 | 1585.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1566.60 | 1577.22 | 1583.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 1563.70 | 1574.78 | 1582.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 1567.00 | 1573.92 | 1581.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 1566.70 | 1572.28 | 1579.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1558.00 | 1552.95 | 1558.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1557.20 | 1552.95 | 1558.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1570.50 | 1556.46 | 1559.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1570.50 | 1556.46 | 1559.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1568.20 | 1558.80 | 1560.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 1557.80 | 1559.55 | 1560.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1563.00 | 1560.24 | 1560.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1555.70 | 1560.35 | 1560.67 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1565.50 | 1561.38 | 1560.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 11:15:00 | 1575.80 | 1564.27 | 1562.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 1577.90 | 1578.29 | 1571.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:00:00 | 1577.90 | 1578.29 | 1571.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1577.80 | 1581.67 | 1577.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1577.80 | 1581.67 | 1577.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1566.00 | 1578.54 | 1576.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1566.00 | 1578.54 | 1576.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1563.50 | 1575.53 | 1575.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 1560.30 | 1570.37 | 1573.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 1553.30 | 1550.72 | 1557.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1558.00 | 1552.18 | 1557.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1558.00 | 1552.18 | 1557.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1564.00 | 1554.54 | 1558.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1583.10 | 1554.54 | 1558.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1574.80 | 1558.60 | 1559.84 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 1580.70 | 1564.65 | 1562.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 1585.40 | 1575.25 | 1569.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 1603.20 | 1612.78 | 1599.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 1604.70 | 1612.78 | 1599.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 1604.50 | 1609.88 | 1600.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 1601.50 | 1609.88 | 1600.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1624.30 | 1612.77 | 1602.30 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1585.80 | 1599.76 | 1601.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1583.40 | 1592.34 | 1597.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1581.50 | 1576.92 | 1584.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 1581.50 | 1576.92 | 1584.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1587.00 | 1578.94 | 1584.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1587.00 | 1578.94 | 1584.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1592.00 | 1581.55 | 1585.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 1592.00 | 1581.55 | 1585.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 1596.00 | 1589.32 | 1588.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1598.40 | 1591.13 | 1589.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 1581.90 | 1590.46 | 1589.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 1580.20 | 1588.41 | 1588.88 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 12:15:00 | 1596.00 | 1590.34 | 1589.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 1607.30 | 1593.73 | 1591.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 1604.90 | 1608.40 | 1602.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1630.30 | 1608.40 | 1602.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1606.80 | 1611.99 | 1606.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 1602.80 | 1611.99 | 1606.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1610.00 | 1611.59 | 1606.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1585.10 | 1605.58 | 1605.28 | SL hit (close<ema400) qty=1.00 sl=1605.28 alert=retest1 |

### Cycle 34 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 1585.20 | 1601.50 | 1603.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 11:15:00 | 1569.10 | 1595.02 | 1600.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 1558.10 | 1557.45 | 1572.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 1558.10 | 1557.45 | 1572.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1520.00 | 1522.30 | 1531.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 1507.20 | 1517.71 | 1527.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1530.40 | 1524.81 | 1524.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1530.40 | 1524.81 | 1524.28 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1519.80 | 1523.81 | 1523.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 12:15:00 | 1513.00 | 1521.65 | 1522.88 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1536.40 | 1523.15 | 1522.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1560.30 | 1530.58 | 1526.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1545.70 | 1552.64 | 1545.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1545.80 | 1551.27 | 1545.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1547.30 | 1551.27 | 1545.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1557.00 | 1552.42 | 1546.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:45:00 | 1566.00 | 1554.77 | 1548.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1576.40 | 1557.36 | 1551.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-12 14:15:00 | 1722.60 | 1684.68 | 1654.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 1679.50 | 1683.52 | 1684.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 1679.50 | 1683.52 | 1684.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1674.40 | 1681.70 | 1683.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1687.90 | 1673.09 | 1676.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1688.10 | 1676.09 | 1677.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1696.50 | 1676.09 | 1677.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1688.20 | 1678.52 | 1678.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 1696.10 | 1684.01 | 1681.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 13:15:00 | 1673.90 | 1681.28 | 1681.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1663.30 | 1677.68 | 1679.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:15:00 | 1673.20 | 1660.35 | 1668.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1682.00 | 1664.68 | 1669.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1682.00 | 1664.68 | 1669.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1675.50 | 1666.85 | 1670.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1683.00 | 1666.85 | 1670.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1679.10 | 1671.39 | 1671.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 1677.50 | 1671.39 | 1671.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 1679.70 | 1673.05 | 1672.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 1688.20 | 1676.08 | 1673.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1721.50 | 1726.67 | 1716.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 1721.50 | 1726.67 | 1716.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1709.30 | 1723.19 | 1715.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 1709.30 | 1723.19 | 1715.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 1705.20 | 1719.60 | 1715.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 1699.30 | 1719.60 | 1715.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1698.40 | 1712.06 | 1712.18 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 1717.00 | 1710.28 | 1710.06 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1671.80 | 1702.58 | 1706.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 1667.40 | 1679.78 | 1689.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1676.90 | 1673.35 | 1683.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1676.90 | 1673.35 | 1683.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1677.80 | 1674.24 | 1682.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1679.50 | 1674.24 | 1682.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1678.80 | 1675.82 | 1681.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 1679.00 | 1675.82 | 1681.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1687.50 | 1678.15 | 1681.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 1686.50 | 1678.15 | 1681.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1685.00 | 1679.52 | 1682.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1666.70 | 1679.52 | 1682.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 12:15:00 | 1692.50 | 1682.89 | 1682.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1692.50 | 1682.89 | 1682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 1703.80 | 1690.08 | 1686.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:45:00 | 1681.60 | 1690.43 | 1687.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1693.10 | 1690.96 | 1688.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 14:45:00 | 1694.90 | 1691.91 | 1688.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1674.60 | 1689.41 | 1688.33 | SL hit (close<static) qty=1.00 sl=1680.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 12:15:00 | 1695.00 | 1689.32 | 1688.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 13:00:00 | 1694.90 | 1690.43 | 1689.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1694.70 | 1690.97 | 1689.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1696.80 | 1692.14 | 1690.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 11:30:00 | 1707.50 | 1698.01 | 1693.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 1710.70 | 1698.59 | 1694.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 1706.20 | 1699.91 | 1696.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:00:00 | 1709.50 | 1701.82 | 1697.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1704.20 | 1704.41 | 1700.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 1703.60 | 1704.41 | 1700.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1710.80 | 1713.39 | 1708.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1675.50 | 1690.29 | 1696.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1665.30 | 1664.33 | 1676.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 1665.30 | 1664.33 | 1676.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1672.70 | 1666.35 | 1674.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1673.00 | 1666.35 | 1674.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1686.00 | 1670.28 | 1675.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 1688.00 | 1670.28 | 1675.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1682.30 | 1672.68 | 1676.05 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 1685.20 | 1678.10 | 1677.90 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 09:15:00 | 1670.10 | 1676.50 | 1677.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 1661.00 | 1673.40 | 1675.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1679.10 | 1672.59 | 1674.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1690.80 | 1676.23 | 1676.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1697.80 | 1687.79 | 1683.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 1683.60 | 1687.85 | 1683.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1683.70 | 1687.02 | 1683.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1677.70 | 1687.02 | 1683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1693.40 | 1688.30 | 1684.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 1687.90 | 1688.30 | 1684.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1686.00 | 1690.61 | 1686.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1704.60 | 1690.61 | 1686.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1701.60 | 1692.55 | 1688.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 1701.90 | 1695.88 | 1690.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1681.80 | 1693.06 | 1690.12 | SL hit (close<static) qty=1.00 sl=1686.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1681.80 | 1693.06 | 1690.12 | SL hit (close<static) qty=1.00 sl=1686.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1681.80 | 1693.06 | 1690.12 | SL hit (close<static) qty=1.00 sl=1686.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1674.40 | 1686.86 | 1687.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1666.10 | 1682.71 | 1685.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 1675.00 | 1673.62 | 1678.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 1675.00 | 1673.62 | 1678.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1670.20 | 1673.16 | 1677.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:30:00 | 1674.20 | 1673.16 | 1677.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1642.80 | 1641.71 | 1653.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1642.40 | 1641.71 | 1653.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1648.40 | 1643.05 | 1653.38 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1673.00 | 1660.27 | 1658.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1685.50 | 1668.91 | 1663.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 1664.60 | 1674.93 | 1671.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1666.90 | 1673.33 | 1670.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1662.20 | 1673.33 | 1670.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1677.00 | 1673.62 | 1671.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1695.00 | 1673.62 | 1671.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1692.90 | 1713.66 | 1715.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1692.90 | 1713.66 | 1715.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 1681.30 | 1691.94 | 1701.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 1648.10 | 1647.52 | 1660.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:45:00 | 1658.30 | 1647.52 | 1660.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1647.40 | 1647.85 | 1655.20 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 1665.20 | 1655.86 | 1655.47 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1650.30 | 1654.49 | 1654.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1639.90 | 1650.42 | 1652.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1620.40 | 1617.15 | 1629.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1620.40 | 1617.15 | 1629.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1641.70 | 1622.36 | 1629.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1641.70 | 1622.36 | 1629.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1616.40 | 1621.17 | 1628.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1615.30 | 1621.17 | 1628.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 1616.30 | 1623.20 | 1626.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1611.90 | 1621.17 | 1624.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 1599.50 | 1608.20 | 1608.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 1584.20 | 1601.18 | 1605.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1597.00 | 1596.97 | 1602.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 1597.00 | 1596.97 | 1602.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1596.90 | 1596.96 | 1601.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 1608.90 | 1596.96 | 1601.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1618.80 | 1601.33 | 1603.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 1618.80 | 1601.33 | 1603.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1613.60 | 1603.78 | 1604.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 1615.90 | 1603.78 | 1604.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1619.00 | 1606.82 | 1605.63 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1596.80 | 1604.10 | 1604.82 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 1631.10 | 1609.50 | 1607.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 09:15:00 | 1636.10 | 1619.75 | 1613.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 1699.60 | 1700.91 | 1686.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 1699.60 | 1700.91 | 1686.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1697.60 | 1700.15 | 1688.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1722.70 | 1700.12 | 1693.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:00:00 | 1721.00 | 1704.29 | 1695.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1726.70 | 1709.31 | 1698.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 1727.00 | 1714.21 | 1702.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1750.00 | 1743.32 | 1733.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1780.50 | 1744.31 | 1737.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 1768.20 | 1755.02 | 1743.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1862.70 | 1853.02 | 1852.29 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1848.70 | 1851.58 | 1851.90 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1857.40 | 1852.74 | 1852.40 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1840.90 | 1850.67 | 1851.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 1838.20 | 1848.18 | 1850.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1747.80 | 1741.63 | 1763.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1747.80 | 1741.63 | 1763.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1705.00 | 1695.00 | 1711.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1726.30 | 1695.00 | 1711.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1720.00 | 1700.00 | 1711.85 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 1732.80 | 1718.14 | 1717.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 1742.80 | 1728.45 | 1722.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1729.20 | 1729.73 | 1724.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1729.20 | 1729.73 | 1724.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1731.10 | 1730.00 | 1724.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1739.40 | 1730.00 | 1724.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1722.40 | 1728.48 | 1724.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1722.40 | 1728.48 | 1724.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1722.10 | 1727.20 | 1724.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1685.60 | 1727.20 | 1724.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1686.00 | 1718.96 | 1720.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1674.50 | 1696.89 | 1706.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1645.60 | 1636.98 | 1657.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 1652.70 | 1636.98 | 1657.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1654.40 | 1640.63 | 1651.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 1654.40 | 1640.63 | 1651.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1642.90 | 1641.08 | 1650.32 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1663.00 | 1655.81 | 1655.10 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1643.40 | 1654.82 | 1656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1627.90 | 1643.25 | 1649.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1641.00 | 1638.99 | 1645.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1635.00 | 1637.80 | 1642.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1591.70 | 1637.80 | 1642.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 1610.20 | 1599.81 | 1599.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1610.20 | 1599.81 | 1599.09 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 1585.50 | 1596.79 | 1597.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 15:15:00 | 1581.10 | 1593.65 | 1596.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1537.00 | 1509.97 | 1534.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1498.90 | 1507.76 | 1531.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 1487.40 | 1505.79 | 1528.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:30:00 | 1495.00 | 1503.13 | 1525.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1469.20 | 1484.21 | 1485.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 1615.20 | 1524.71 | 1503.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1642.40 | 1637.61 | 1615.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 1643.40 | 1637.61 | 1615.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 1668.00 | 1677.67 | 1678.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 1668.00 | 1677.67 | 1678.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1668.00 | 1677.67 | 1678.58 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1690.20 | 1680.16 | 1679.55 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 1662.00 | 1675.82 | 1677.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 1657.00 | 1672.06 | 1675.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 1639.50 | 1634.57 | 1648.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1609.30 | 1634.57 | 1648.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1589.30 | 1587.27 | 1602.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 1601.70 | 1594.51 | 1600.38 | SL hit (close>ema400) qty=1.00 sl=1600.38 alert=retest1 |

### Cycle 75 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 1613.70 | 1604.97 | 1603.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1626.00 | 1610.94 | 1606.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1607.10 | 1612.72 | 1609.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1608.40 | 1611.85 | 1609.26 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1575.30 | 1604.55 | 1606.39 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 1608.70 | 1601.37 | 1600.40 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 1588.10 | 1598.72 | 1599.29 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1635.80 | 1603.34 | 1600.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1652.40 | 1629.23 | 1615.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1690.80 | 1691.67 | 1670.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 1690.80 | 1691.67 | 1670.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 14:15:00 | 1391.70 | 2025-06-02 12:15:00 | 1530.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-05 13:30:00 | 1495.50 | 2025-06-06 10:15:00 | 1514.80 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-20 10:45:00 | 1592.20 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-06-20 14:00:00 | 1591.90 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-06-20 14:30:00 | 1591.00 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-06-24 12:15:00 | 1598.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2025-07-01 09:15:00 | 1651.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-01 11:00:00 | 1650.90 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-01 15:00:00 | 1653.60 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 09:30:00 | 1649.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1547.60 | 2025-07-22 10:15:00 | 1556.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-28 12:00:00 | 1540.10 | 2025-07-28 12:15:00 | 1537.90 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-07-31 09:45:00 | 1498.30 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-31 13:00:00 | 1501.70 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1502.30 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-08-20 12:45:00 | 1643.90 | 2025-08-26 09:15:00 | 1636.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-20 14:30:00 | 1644.40 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-20 15:15:00 | 1645.40 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-08-22 09:30:00 | 1644.60 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-25 11:15:00 | 1654.90 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-28 13:00:00 | 1623.90 | 2025-09-02 09:15:00 | 1640.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1620.80 | 2025-09-02 09:15:00 | 1640.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-04 11:45:00 | 1569.30 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-04 12:45:00 | 1570.00 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-09 10:30:00 | 1566.80 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-09-11 11:15:00 | 1604.70 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1607.00 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-11 12:15:00 | 1604.90 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-11 12:45:00 | 1604.40 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-16 11:15:00 | 1566.60 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-09-16 11:45:00 | 1563.70 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-09-16 13:15:00 | 1567.00 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-09-16 13:45:00 | 1566.70 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-09-19 12:30:00 | 1557.80 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1563.00 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2025-10-15 09:15:00 | 1630.30 | 2025-10-16 09:15:00 | 1585.10 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-10-27 11:45:00 | 1507.20 | 2025-10-29 10:15:00 | 1530.40 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-03 12:45:00 | 1566.00 | 2025-11-12 14:15:00 | 1722.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1576.40 | 2025-11-18 15:15:00 | 1679.50 | STOP_HIT | 1.00 | 6.54% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1666.70 | 2025-12-05 12:15:00 | 1692.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-08 14:45:00 | 1694.90 | 2025-12-09 09:15:00 | 1674.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-09 12:15:00 | 1695.00 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-09 13:00:00 | 1694.90 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1694.70 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-10 11:30:00 | 1707.50 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-10 13:15:00 | 1710.70 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1706.20 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-11 12:00:00 | 1709.50 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1704.60 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1701.60 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 13:15:00 | 1701.90 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-05 09:15:00 | 1695.00 | 2026-01-08 11:15:00 | 1692.90 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1615.30 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-01-23 09:15:00 | 1616.30 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-01-23 13:15:00 | 1611.90 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1722.70 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.87% |
| BUY | retest2 | 2026-02-09 10:00:00 | 1721.00 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.97% |
| BUY | retest2 | 2026-02-09 10:30:00 | 1726.70 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2026-02-09 13:00:00 | 1727.00 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.60% |
| BUY | retest2 | 2026-02-12 11:15:00 | 1780.50 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 3.40% |
| BUY | retest2 | 2026-02-12 13:15:00 | 1768.20 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 4.12% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1591.70 | 2026-03-25 12:15:00 | 1610.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-01 11:45:00 | 1487.40 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1495.00 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1469.20 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1642.40 | 2026-04-20 12:15:00 | 1668.00 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2026-04-13 12:15:00 | 1643.40 | 2026-04-20 12:15:00 | 1668.00 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest1 | 2026-04-23 09:15:00 | 1609.30 | 2026-04-27 14:15:00 | 1601.70 | STOP_HIT | 1.00 | 0.47% |
