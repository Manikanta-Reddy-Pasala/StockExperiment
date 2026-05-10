# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1268.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 59 |
| ALERT1 | 41 |
| ALERT2 | 41 |
| ALERT2_SKIP | 25 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 39 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 17 / 34
- **Target hits / Stop hits / Partials:** 3 / 42 / 6
- **Avg / median % per leg:** 0.56% / -1.37%
- **Sum % (uncompounded):** 28.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 5 | 18.5% | 2 | 25 | 0 | -0.18% | -4.9% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.72% | -12.1% |
| BUY @ 3rd Alert (retest2) | 20 | 5 | 25.0% | 2 | 18 | 0 | 0.36% | 7.2% |
| SELL (all) | 24 | 12 | 50.0% | 1 | 17 | 6 | 1.40% | 33.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.18% | 8.4% |
| SELL @ 3rd Alert (retest2) | 22 | 10 | 45.5% | 1 | 16 | 5 | 1.15% | 25.3% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.41% | -3.7% |
| retest2 (combined) | 42 | 15 | 35.7% | 3 | 34 | 5 | 0.77% | 32.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1230.00 | 1243.84 | 1244.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 1228.00 | 1236.93 | 1239.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 1274.80 | 1231.80 | 1232.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1270.40 | 1239.52 | 1235.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 1293.90 | 1256.07 | 1244.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:00:00 | 1245.20 | 1266.91 | 1256.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1249.50 | 1263.43 | 1255.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1236.00 | 1263.43 | 1255.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1245.60 | 1256.21 | 1253.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1359.00 | 1257.19 | 1254.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 09:15:00 | 1494.90 | 1427.11 | 1387.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1516.10 | 1519.34 | 1519.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1477.20 | 1510.91 | 1515.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 1479.40 | 1461.53 | 1475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1471.30 | 1463.48 | 1475.48 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1527.90 | 1484.57 | 1480.89 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1483.10 | 1496.17 | 1496.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1461.50 | 1489.24 | 1493.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1472.10 | 1478.21 | 1485.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1486.50 | 1478.00 | 1482.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1486.50 | 1478.00 | 1482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1485.00 | 1479.40 | 1482.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1475.10 | 1479.40 | 1482.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1490.90 | 1482.59 | 1483.58 | SL hit (close>static) qty=1.00 sl=1488.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1492.40 | 1484.55 | 1484.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1515.50 | 1490.74 | 1487.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 1523.00 | 1523.25 | 1510.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1536.70 | 1523.25 | 1510.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:30:00 | 1535.00 | 1526.32 | 1515.65 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 12:15:00 | 1531.70 | 1526.32 | 1515.65 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | SL hit (close<ema400) qty=1.00 sl=1514.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | SL hit (close<ema400) qty=1.00 sl=1514.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | SL hit (close<ema400) qty=1.00 sl=1514.59 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 1504.00 | 1521.85 | 1514.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1499.00 | 1517.28 | 1513.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 1499.00 | 1517.28 | 1513.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 1482.80 | 1506.93 | 1509.15 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 1533.00 | 1512.90 | 1510.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1606.20 | 1531.56 | 1518.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1588.00 | 1590.05 | 1566.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1588.00 | 1590.05 | 1566.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1578.50 | 1589.34 | 1575.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1568.10 | 1589.34 | 1575.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1586.80 | 1588.83 | 1576.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 1596.80 | 1589.34 | 1578.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1602.00 | 1589.34 | 1578.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1575.00 | 1591.51 | 1591.39 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1575.00 | 1591.51 | 1591.39 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1578.30 | 1588.87 | 1590.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 1569.70 | 1582.70 | 1587.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1609.50 | 1581.57 | 1578.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1627.90 | 1608.65 | 1595.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 1619.00 | 1620.34 | 1609.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 1619.00 | 1620.34 | 1609.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1608.40 | 1617.92 | 1610.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1608.40 | 1617.92 | 1610.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1608.90 | 1616.11 | 1610.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1607.50 | 1616.11 | 1610.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1612.50 | 1615.39 | 1610.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 1607.60 | 1615.39 | 1610.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1605.00 | 1613.31 | 1609.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1600.60 | 1613.31 | 1609.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1598.00 | 1610.25 | 1608.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 1601.70 | 1610.25 | 1608.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1592.20 | 1606.64 | 1607.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1541.80 | 1587.69 | 1597.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1553.20 | 1552.67 | 1569.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1553.20 | 1552.67 | 1569.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1565.30 | 1554.85 | 1567.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1565.30 | 1554.85 | 1567.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1562.50 | 1556.38 | 1567.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:15:00 | 1572.50 | 1556.38 | 1567.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1574.50 | 1560.00 | 1568.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1574.50 | 1560.00 | 1568.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1571.00 | 1562.20 | 1568.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1561.80 | 1562.20 | 1568.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 13:15:00 | 1483.71 | 1503.01 | 1518.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-28 13:15:00 | 1405.62 | 1432.95 | 1453.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1457.10 | 1446.15 | 1445.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1477.00 | 1458.09 | 1452.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1461.10 | 1461.32 | 1454.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1451.00 | 1459.26 | 1454.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1439.80 | 1459.26 | 1454.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1432.60 | 1453.93 | 1452.60 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1437.20 | 1450.58 | 1451.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1426.60 | 1441.73 | 1446.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1438.90 | 1430.40 | 1436.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1425.00 | 1429.32 | 1435.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1453.00 | 1429.32 | 1435.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1435.50 | 1430.56 | 1435.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1409.00 | 1425.86 | 1430.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1445.70 | 1391.34 | 1390.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1445.70 | 1391.34 | 1390.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 1489.10 | 1451.08 | 1427.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 10:15:00 | 1473.10 | 1476.66 | 1453.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 10:45:00 | 1471.40 | 1476.66 | 1453.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1447.30 | 1468.49 | 1453.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 1447.30 | 1468.49 | 1453.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 1442.90 | 1463.37 | 1452.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 1450.40 | 1453.49 | 1450.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 1451.60 | 1450.49 | 1449.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:00:00 | 1453.10 | 1451.02 | 1450.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 11:15:00 | 1501.90 | 1514.91 | 1522.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1517.30 | 1513.44 | 1520.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1516.80 | 1514.11 | 1519.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 1516.60 | 1514.11 | 1519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1547.00 | 1520.83 | 1521.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1535.00 | 1520.83 | 1521.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1550.10 | 1526.69 | 1524.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1560.90 | 1537.48 | 1531.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 1535.40 | 1546.92 | 1541.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1528.50 | 1543.23 | 1540.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 1529.80 | 1543.23 | 1540.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 1518.40 | 1538.27 | 1538.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1511.10 | 1532.83 | 1535.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1542.10 | 1522.00 | 1525.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1536.30 | 1524.86 | 1526.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 1526.90 | 1524.86 | 1526.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:15:00 | 1450.56 | 1462.80 | 1477.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1451.80 | 1447.44 | 1460.58 | SL hit (close>ema200) qty=0.50 sl=1447.44 alert=retest2 |

### Cycle 18 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1351.00 | 1341.10 | 1340.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1371.50 | 1355.03 | 1347.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1357.10 | 1359.12 | 1351.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 1357.10 | 1359.12 | 1351.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1351.00 | 1357.50 | 1351.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 1353.80 | 1357.50 | 1351.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1356.30 | 1357.26 | 1351.49 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1334.50 | 1349.35 | 1350.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1333.30 | 1342.19 | 1346.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1343.90 | 1341.71 | 1345.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1340.00 | 1341.37 | 1344.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 1339.90 | 1341.37 | 1344.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1344.80 | 1342.06 | 1344.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 1350.30 | 1342.06 | 1344.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1338.80 | 1341.40 | 1344.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 1335.90 | 1340.82 | 1343.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 1333.80 | 1340.82 | 1343.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 1349.00 | 1329.77 | 1333.72 | SL hit (close>static) qty=1.00 sl=1346.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 1349.00 | 1329.77 | 1333.72 | SL hit (close>static) qty=1.00 sl=1346.10 alert=retest2 |

### Cycle 20 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1372.20 | 1342.18 | 1338.92 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 1340.30 | 1349.08 | 1349.66 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1360.00 | 1349.86 | 1349.59 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 1341.40 | 1349.20 | 1350.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1319.70 | 1339.70 | 1344.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 1321.50 | 1320.20 | 1330.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:30:00 | 1322.40 | 1320.20 | 1330.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1329.50 | 1322.83 | 1329.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 1330.00 | 1322.83 | 1329.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1334.00 | 1325.06 | 1330.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1350.50 | 1331.25 | 1332.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1354.00 | 1335.80 | 1334.61 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 1317.00 | 1334.47 | 1336.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 1311.70 | 1321.06 | 1328.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1320.70 | 1312.41 | 1319.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1322.90 | 1314.51 | 1320.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1322.90 | 1314.51 | 1320.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1324.70 | 1316.55 | 1320.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1328.80 | 1316.55 | 1320.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1328.90 | 1319.02 | 1321.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1329.70 | 1319.02 | 1321.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1333.00 | 1321.81 | 1322.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1333.50 | 1321.81 | 1322.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1331.20 | 1323.69 | 1323.14 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 1314.00 | 1321.77 | 1322.37 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1330.00 | 1323.57 | 1323.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1342.20 | 1326.80 | 1324.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1336.80 | 1338.88 | 1332.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1336.80 | 1338.88 | 1332.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1333.80 | 1337.86 | 1332.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 1333.80 | 1337.86 | 1332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1335.00 | 1337.29 | 1333.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 1348.70 | 1337.46 | 1334.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1341.10 | 1339.39 | 1335.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 1341.90 | 1340.56 | 1336.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1343.90 | 1336.00 | 1335.57 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1329.00 | 1336.44 | 1336.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1316.50 | 1332.45 | 1334.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1277.90 | 1270.99 | 1287.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1283.40 | 1275.54 | 1286.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1290.50 | 1275.54 | 1286.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1289.70 | 1278.37 | 1286.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 1282.50 | 1280.82 | 1286.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1272.90 | 1280.38 | 1284.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1218.38 | 1267.38 | 1273.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1209.26 | 1267.38 | 1273.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1241.50 | 1240.58 | 1250.65 | SL hit (close>ema200) qty=0.50 sl=1240.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1241.50 | 1240.58 | 1250.65 | SL hit (close>ema200) qty=0.50 sl=1240.58 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1185.80 | 1179.07 | 1178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1194.00 | 1184.15 | 1181.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1194.40 | 1195.91 | 1189.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1210.20 | 1195.91 | 1189.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1209.50 | 1198.15 | 1191.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:45:00 | 1211.50 | 1200.32 | 1192.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 15:00:00 | 1208.80 | 1206.06 | 1197.78 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1200.90 | 1204.22 | 1198.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1191.50 | 1201.68 | 1197.71 | SL hit (close<ema400) qty=1.00 sl=1197.71 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1191.50 | 1201.68 | 1197.71 | SL hit (close<ema400) qty=1.00 sl=1197.71 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1191.50 | 1201.68 | 1197.71 | SL hit (close<ema400) qty=1.00 sl=1197.71 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1191.50 | 1201.68 | 1197.71 | SL hit (close<ema400) qty=1.00 sl=1197.71 alert=retest1 |

### Cycle 33 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1174.20 | 1192.01 | 1194.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 1166.60 | 1186.93 | 1191.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 1147.00 | 1146.50 | 1160.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1129.00 | 1146.50 | 1160.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1072.55 | 1087.44 | 1105.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1091.00 | 1077.87 | 1092.86 | SL hit (close>ema200) qty=0.50 sl=1077.87 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1094.00 | 1081.10 | 1092.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:15:00 | 1095.20 | 1081.10 | 1092.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1087.30 | 1082.34 | 1092.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 1082.30 | 1083.03 | 1091.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 1098.90 | 1087.64 | 1092.50 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1082.50 | 1088.81 | 1091.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 1080.00 | 1087.30 | 1090.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1080.00 | 1084.51 | 1088.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1085.40 | 1084.69 | 1088.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1092.20 | 1084.69 | 1088.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1092.20 | 1086.19 | 1088.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 1093.50 | 1086.19 | 1088.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1093.80 | 1087.71 | 1089.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 1093.80 | 1087.71 | 1089.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1100.00 | 1090.17 | 1090.22 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1100.00 | 1090.17 | 1090.22 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1100.00 | 1090.17 | 1090.22 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1102.70 | 1092.68 | 1091.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1106.20 | 1097.32 | 1093.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1104.80 | 1102.42 | 1098.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1088.60 | 1099.95 | 1100.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1088.60 | 1099.95 | 1100.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1084.40 | 1096.47 | 1098.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 1072.20 | 1064.63 | 1073.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1075.00 | 1066.70 | 1073.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 1075.00 | 1066.70 | 1073.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1075.20 | 1068.40 | 1073.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1074.90 | 1068.40 | 1073.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1070.80 | 1068.88 | 1073.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 1066.80 | 1068.88 | 1073.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1090.00 | 1073.11 | 1074.86 | SL hit (close>static) qty=1.00 sl=1075.20 alert=retest2 |

### Cycle 36 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1096.00 | 1077.68 | 1076.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1104.00 | 1086.58 | 1082.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1116.80 | 1117.15 | 1104.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1116.80 | 1117.15 | 1104.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1105.80 | 1113.25 | 1106.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1105.80 | 1113.25 | 1106.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1107.20 | 1112.04 | 1106.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1107.20 | 1112.04 | 1106.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1102.80 | 1110.20 | 1106.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1102.00 | 1110.20 | 1106.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1104.90 | 1109.14 | 1106.34 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1096.60 | 1103.90 | 1104.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1092.30 | 1101.58 | 1103.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1064.00 | 1061.20 | 1072.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1074.70 | 1063.90 | 1072.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 1078.60 | 1063.90 | 1072.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1080.00 | 1067.12 | 1073.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1080.00 | 1067.12 | 1073.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1078.10 | 1069.32 | 1073.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1082.00 | 1069.32 | 1073.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1074.80 | 1075.06 | 1075.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 1075.00 | 1075.06 | 1075.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1068.10 | 1073.67 | 1074.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1065.10 | 1073.67 | 1074.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 1062.10 | 1071.35 | 1073.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1097.10 | 1074.50 | 1073.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1097.10 | 1074.50 | 1073.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1097.10 | 1074.50 | 1073.81 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1075.90 | 1090.48 | 1090.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1073.00 | 1085.50 | 1088.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1005.00 | 1004.37 | 1018.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1006.10 | 1004.37 | 1018.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 991.40 | 994.37 | 999.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 982.80 | 992.18 | 997.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 933.66 | 950.73 | 965.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 919.80 | 917.93 | 933.22 | SL hit (close>ema200) qty=0.50 sl=917.93 alert=retest2 |

### Cycle 40 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 939.70 | 914.97 | 911.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 986.00 | 946.12 | 933.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 982.25 | 995.43 | 976.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1005.00 | 997.34 | 979.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 1022.70 | 1000.68 | 982.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1029.20 | 993.62 | 985.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:45:00 | 1027.15 | 1000.17 | 989.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1039.55 | 1028.85 | 1028.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1074.80 | 1038.35 | 1033.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 1108.40 | 1109.63 | 1093.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:30:00 | 1107.95 | 1109.63 | 1093.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1078.15 | 1105.12 | 1096.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 1078.15 | 1105.12 | 1096.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1084.70 | 1101.04 | 1095.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 1071.70 | 1101.04 | 1095.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1081.95 | 1091.12 | 1092.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1042.80 | 1079.68 | 1086.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1050.40 | 1041.32 | 1054.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1055.00 | 1044.06 | 1054.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1051.20 | 1044.06 | 1054.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1077.50 | 1050.75 | 1056.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1079.50 | 1050.75 | 1056.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1071.85 | 1054.97 | 1057.84 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1068.45 | 1060.77 | 1060.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1091.55 | 1072.36 | 1066.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1131.90 | 1135.51 | 1118.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 1121.00 | 1135.51 | 1118.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1189.30 | 1162.07 | 1147.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1155.75 | 1162.07 | 1147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1160.05 | 1168.93 | 1161.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1160.05 | 1168.93 | 1161.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1153.35 | 1165.82 | 1160.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1180.00 | 1163.38 | 1160.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1147.30 | 1170.87 | 1171.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1147.30 | 1170.87 | 1171.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1063.00 | 1113.95 | 1125.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 1083.30 | 1081.39 | 1102.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 1083.30 | 1081.39 | 1102.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1100.20 | 1085.96 | 1100.87 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1126.60 | 1107.48 | 1104.95 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1083.70 | 1104.45 | 1105.53 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 1115.10 | 1106.68 | 1106.13 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1078.40 | 1101.49 | 1103.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1075.90 | 1086.13 | 1093.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1086.20 | 1076.35 | 1084.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1131.20 | 1087.27 | 1088.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1133.60 | 1087.27 | 1088.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1127.80 | 1095.37 | 1091.96 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 1095.90 | 1109.25 | 1109.99 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1127.30 | 1112.86 | 1111.56 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 1092.80 | 1108.50 | 1109.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1033.70 | 1086.96 | 1098.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1056.80 | 1040.26 | 1057.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1050.70 | 1042.35 | 1056.80 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1087.40 | 1066.06 | 1063.61 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1045.60 | 1063.68 | 1064.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1032.50 | 1050.76 | 1057.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1044.70 | 1011.05 | 1026.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1022.55 | 1013.35 | 1025.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 999.65 | 1025.87 | 1028.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1055.00 | 1031.20 | 1029.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1055.00 | 1031.20 | 1029.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1059.75 | 1042.97 | 1036.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 1039.70 | 1051.20 | 1043.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1036.10 | 1048.18 | 1043.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 1036.10 | 1048.18 | 1043.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1048.50 | 1048.25 | 1043.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1076.30 | 1045.74 | 1043.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 13:15:00 | 1183.93 | 1158.41 | 1130.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1221.45 | 1246.54 | 1249.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1220.00 | 1237.83 | 1244.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1276.50 | 1237.96 | 1241.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1277.00 | 1245.77 | 1245.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1297.20 | 1274.90 | 1268.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1274.20 | 1288.23 | 1279.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1265.75 | 1283.74 | 1278.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1265.75 | 1283.74 | 1278.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1281.75 | 1281.59 | 1278.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 1289.15 | 1281.82 | 1278.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1325.20 | 1281.86 | 1279.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1289.30 | 1299.31 | 1293.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 1289.00 | 1297.23 | 1293.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 1263.60 | 1277.40 | 1283.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1286.90 | 1277.75 | 1282.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1292.00 | 1280.60 | 1283.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1291.60 | 1280.60 | 1283.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1289.00 | 1282.65 | 1283.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1289.00 | 1282.65 | 1283.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1288.10 | 1283.74 | 1284.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 1286.00 | 1283.74 | 1284.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1279.90 | 1282.97 | 1283.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:30:00 | 1289.90 | 1282.97 | 1283.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1280.70 | 1282.52 | 1283.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 1283.80 | 1282.52 | 1283.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1278.90 | 1281.79 | 1283.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 1283.00 | 1281.79 | 1283.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1288.50 | 1283.13 | 1283.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 1274.70 | 1283.13 | 1283.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1260.50 | 1278.61 | 1281.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 1251.90 | 1275.09 | 1279.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 15:00:00 | 1253.90 | 1266.05 | 1273.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 11:15:00 | 1259.20 | 2025-05-20 13:15:00 | 1241.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1359.00 | 2025-05-30 09:15:00 | 1494.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1475.10 | 2025-06-23 11:15:00 | 1490.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2025-06-25 09:15:00 | 1536.70 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2025-06-25 11:30:00 | 1535.00 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-06-25 12:15:00 | 1531.70 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-01 12:45:00 | 1596.80 | 2025-07-03 09:15:00 | 1575.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-01 13:15:00 | 1602.00 | 2025-07-03 09:15:00 | 1575.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1561.80 | 2025-07-23 13:15:00 | 1483.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1561.80 | 2025-07-28 13:15:00 | 1405.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1409.00 | 2025-08-13 09:15:00 | 1445.70 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-19 10:15:00 | 1450.40 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 4.19% |
| BUY | retest2 | 2025-08-19 13:45:00 | 1451.60 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-08-19 15:00:00 | 1453.10 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 3.99% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1526.90 | 2025-09-12 09:15:00 | 1450.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1526.90 | 2025-09-15 09:15:00 | 1451.80 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-10-09 09:30:00 | 1335.90 | 2025-10-10 10:15:00 | 1349.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-09 10:00:00 | 1333.80 | 2025-10-10 10:15:00 | 1349.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-30 14:30:00 | 1348.70 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-10-31 11:15:00 | 1341.10 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-10-31 11:45:00 | 1341.90 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1282.50 | 2025-11-13 09:15:00 | 1218.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1272.90 | 2025-11-13 09:15:00 | 1209.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1282.50 | 2025-11-14 12:15:00 | 1241.50 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1272.90 | 2025-11-14 12:15:00 | 1241.50 | STOP_HIT | 0.50 | 2.47% |
| BUY | retest1 | 2025-11-28 10:15:00 | 1210.20 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest1 | 2025-11-28 11:15:00 | 1209.50 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2025-11-28 11:45:00 | 1211.50 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2025-11-28 15:00:00 | 1208.80 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest1 | 2025-12-04 09:15:00 | 1129.00 | 2025-12-08 12:15:00 | 1072.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-04 09:15:00 | 1129.00 | 2025-12-09 10:15:00 | 1091.00 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-12-09 14:15:00 | 1082.30 | 2025-12-09 15:15:00 | 1098.90 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1082.50 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-12-10 13:45:00 | 1080.00 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1080.00 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-12-15 10:15:00 | 1104.80 | 2025-12-16 11:15:00 | 1088.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-19 14:15:00 | 1066.80 | 2025-12-19 14:15:00 | 1090.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1065.10 | 2026-01-02 10:15:00 | 1097.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-01-01 13:00:00 | 1062.10 | 2026-01-02 10:15:00 | 1097.10 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-01-16 12:15:00 | 982.80 | 2026-01-20 11:15:00 | 933.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 982.80 | 2026-01-22 09:15:00 | 919.80 | STOP_HIT | 0.50 | 6.41% |
| BUY | retest2 | 2026-02-02 09:45:00 | 1022.70 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2026-02-03 09:15:00 | 1029.20 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-03 09:45:00 | 1027.15 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1180.00 | 2026-03-02 09:15:00 | 1147.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-02 09:15:00 | 999.65 | 2026-04-02 13:15:00 | 1055.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1076.30 | 2026-04-15 13:15:00 | 1183.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 14:30:00 | 1289.15 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1325.20 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2026-05-05 11:00:00 | 1289.30 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-05-05 12:15:00 | 1289.00 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.34% |
