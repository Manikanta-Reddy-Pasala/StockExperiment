# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 250 |
| ALERT1 | 145 |
| ALERT2 | 142 |
| ALERT2_SKIP | 83 |
| ALERT3 | 358 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 191 |
| PARTIAL | 18 |
| TARGET_HIT | 11 |
| STOP_HIT | 189 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 214 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 145
- **Target hits / Stop hits / Partials:** 11 / 185 / 18
- **Avg / median % per leg:** 0.11% / -1.07%
- **Sum % (uncompounded):** 24.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 93 | 22 | 23.7% | 10 | 83 | 0 | 0.11% | 10.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.25% | -3.8% |
| BUY @ 3rd Alert (retest2) | 90 | 22 | 24.4% | 10 | 80 | 0 | 0.16% | 14.4% |
| SELL (all) | 121 | 47 | 38.8% | 1 | 102 | 18 | 0.12% | 14.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.72% | -5.4% |
| SELL @ 3rd Alert (retest2) | 119 | 47 | 39.5% | 1 | 100 | 18 | 0.16% | 19.4% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.84% | -9.2% |
| retest2 (combined) | 209 | 69 | 33.0% | 11 | 180 | 18 | 0.16% | 33.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 09:15:00 | 1389.00 | 1407.47 | 1408.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 15:15:00 | 1381.00 | 1391.23 | 1398.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 10:15:00 | 1415.90 | 1393.61 | 1398.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 10:15:00 | 1415.90 | 1393.61 | 1398.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 1415.90 | 1393.61 | 1398.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:45:00 | 1417.15 | 1393.61 | 1398.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 1411.80 | 1397.25 | 1399.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 11:30:00 | 1417.75 | 1397.25 | 1399.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 12:15:00 | 1447.45 | 1407.29 | 1403.68 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 1383.15 | 1408.05 | 1410.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 10:15:00 | 1375.50 | 1391.38 | 1399.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 10:15:00 | 1375.90 | 1367.85 | 1375.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 10:15:00 | 1375.90 | 1367.85 | 1375.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 1375.90 | 1367.85 | 1375.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 11:00:00 | 1375.90 | 1367.85 | 1375.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 1378.70 | 1370.02 | 1375.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 11:30:00 | 1378.95 | 1370.02 | 1375.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 1377.35 | 1371.49 | 1376.04 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 10:15:00 | 1386.00 | 1378.69 | 1378.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 11:15:00 | 1404.30 | 1383.81 | 1380.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 12:15:00 | 1416.05 | 1419.15 | 1409.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 13:00:00 | 1416.05 | 1419.15 | 1409.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 1404.20 | 1414.76 | 1409.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:45:00 | 1418.15 | 1416.52 | 1410.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:00:00 | 1419.20 | 1417.06 | 1411.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:45:00 | 1423.30 | 1416.64 | 1411.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:15:00 | 1420.95 | 1416.64 | 1411.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 1412.15 | 1415.74 | 1411.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 15:00:00 | 1427.85 | 1416.44 | 1412.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-14 09:15:00 | 1559.97 | 1531.98 | 1524.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 1550.90 | 1567.67 | 1568.44 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 09:15:00 | 1590.50 | 1571.28 | 1569.61 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 15:15:00 | 1565.00 | 1575.97 | 1577.04 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 1596.30 | 1580.04 | 1578.79 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 14:15:00 | 1565.45 | 1576.55 | 1577.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 15:15:00 | 1555.30 | 1572.30 | 1575.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 09:15:00 | 1568.55 | 1560.78 | 1566.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 1568.55 | 1560.78 | 1566.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1568.55 | 1560.78 | 1566.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:30:00 | 1567.95 | 1560.78 | 1566.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 1580.25 | 1564.67 | 1567.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 11:00:00 | 1580.25 | 1564.67 | 1567.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 1586.05 | 1568.95 | 1569.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:00:00 | 1586.05 | 1568.95 | 1569.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 12:15:00 | 1588.80 | 1572.92 | 1571.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 13:15:00 | 1598.10 | 1577.95 | 1573.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 1573.35 | 1577.03 | 1573.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 1573.35 | 1577.03 | 1573.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 1573.35 | 1577.03 | 1573.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 1573.35 | 1577.03 | 1573.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 1563.00 | 1574.23 | 1572.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 1591.80 | 1574.23 | 1572.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 09:15:00 | 1598.80 | 1584.10 | 1580.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 15:15:00 | 1623.05 | 1647.67 | 1648.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 1623.05 | 1647.67 | 1648.15 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 1684.00 | 1654.94 | 1651.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 15:15:00 | 1707.00 | 1689.33 | 1676.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 11:15:00 | 1689.10 | 1691.40 | 1680.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 12:00:00 | 1689.10 | 1691.40 | 1680.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 1679.45 | 1689.01 | 1680.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:45:00 | 1678.10 | 1689.01 | 1680.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 1668.20 | 1684.85 | 1679.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:00:00 | 1668.20 | 1684.85 | 1679.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 1695.90 | 1687.06 | 1680.96 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 1658.45 | 1679.03 | 1679.69 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 1684.95 | 1679.61 | 1679.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 1708.00 | 1685.29 | 1681.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 11:15:00 | 1705.15 | 1705.69 | 1695.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 11:30:00 | 1703.30 | 1705.69 | 1695.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 1698.00 | 1702.66 | 1696.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 1698.00 | 1702.66 | 1696.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 1700.00 | 1702.13 | 1697.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 1705.45 | 1702.13 | 1697.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:45:00 | 1706.80 | 1705.48 | 1699.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 11:15:00 | 1680.95 | 1702.53 | 1698.94 | SL hit (close<static) qty=1.00 sl=1697.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 1695.00 | 1696.46 | 1696.64 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 1700.00 | 1697.17 | 1696.95 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 10:15:00 | 1686.00 | 1694.93 | 1695.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 11:15:00 | 1685.00 | 1692.95 | 1694.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 1690.00 | 1687.40 | 1691.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 1690.00 | 1687.40 | 1691.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 1690.00 | 1687.40 | 1691.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:30:00 | 1684.60 | 1687.40 | 1691.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 1703.15 | 1690.55 | 1692.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:00:00 | 1703.15 | 1690.55 | 1692.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 1689.80 | 1690.40 | 1691.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 13:30:00 | 1684.95 | 1689.50 | 1691.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 14:00:00 | 1686.80 | 1689.50 | 1691.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 14:45:00 | 1685.15 | 1689.86 | 1691.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 09:30:00 | 1685.95 | 1690.27 | 1691.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 1685.20 | 1689.26 | 1690.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:30:00 | 1688.85 | 1689.26 | 1690.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 1687.85 | 1688.98 | 1690.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:30:00 | 1683.95 | 1688.98 | 1690.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 1687.10 | 1688.60 | 1690.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 12:45:00 | 1691.05 | 1688.60 | 1690.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 1691.70 | 1689.22 | 1690.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 13:45:00 | 1693.60 | 1689.22 | 1690.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1689.40 | 1689.26 | 1690.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 1689.40 | 1689.26 | 1690.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1693.00 | 1690.01 | 1690.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 1700.05 | 1690.01 | 1690.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-24 09:15:00 | 1698.35 | 1691.67 | 1691.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 1698.35 | 1691.67 | 1691.17 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 1687.00 | 1690.71 | 1690.82 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 15:15:00 | 1700.00 | 1692.34 | 1691.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 1722.35 | 1698.34 | 1694.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 13:15:00 | 1705.40 | 1705.65 | 1699.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 13:45:00 | 1714.05 | 1705.65 | 1699.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 1694.55 | 1703.43 | 1699.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 15:00:00 | 1694.55 | 1703.43 | 1699.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 1692.90 | 1701.32 | 1698.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 1707.05 | 1701.32 | 1698.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 09:15:00 | 1686.55 | 1698.37 | 1697.49 | SL hit (close<static) qty=1.00 sl=1690.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 10:15:00 | 1683.90 | 1695.47 | 1696.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 14:15:00 | 1663.65 | 1681.76 | 1688.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 1670.00 | 1650.92 | 1658.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 1670.00 | 1650.92 | 1658.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1670.00 | 1650.92 | 1658.36 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 12:15:00 | 1682.45 | 1664.08 | 1663.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 1693.40 | 1672.85 | 1667.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 1692.40 | 1702.87 | 1691.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 11:15:00 | 1692.40 | 1702.87 | 1691.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 1692.40 | 1702.87 | 1691.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 1695.00 | 1702.87 | 1691.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 1695.00 | 1701.30 | 1691.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:45:00 | 1680.55 | 1701.30 | 1691.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 1695.00 | 1700.04 | 1692.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 1695.00 | 1700.04 | 1692.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 1695.00 | 1699.03 | 1692.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 15:15:00 | 1680.00 | 1699.03 | 1692.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 1680.00 | 1695.22 | 1691.37 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 11:15:00 | 1681.60 | 1687.54 | 1688.35 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 1719.10 | 1693.96 | 1691.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 15:15:00 | 1737.50 | 1702.67 | 1695.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 09:15:00 | 1695.15 | 1701.16 | 1695.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 1695.15 | 1701.16 | 1695.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1695.15 | 1701.16 | 1695.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:45:00 | 1695.00 | 1701.16 | 1695.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 1685.50 | 1698.03 | 1694.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:45:00 | 1683.00 | 1698.03 | 1694.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 1690.00 | 1696.43 | 1693.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 12:30:00 | 1696.00 | 1694.23 | 1693.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 13:15:00 | 1695.95 | 1694.23 | 1693.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 13:45:00 | 1697.95 | 1695.73 | 1693.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 15:15:00 | 1679.00 | 1691.07 | 1692.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 15:15:00 | 1679.00 | 1691.07 | 1692.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 1672.00 | 1678.48 | 1683.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 09:15:00 | 1692.90 | 1667.53 | 1671.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 09:15:00 | 1692.90 | 1667.53 | 1671.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 1692.90 | 1667.53 | 1671.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 09:45:00 | 1682.20 | 1667.53 | 1671.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 1694.30 | 1672.89 | 1673.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 10:45:00 | 1703.35 | 1672.89 | 1673.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 11:15:00 | 1698.60 | 1678.03 | 1675.70 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 1652.95 | 1675.25 | 1677.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 15:15:00 | 1636.15 | 1659.67 | 1668.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 10:15:00 | 1605.70 | 1603.84 | 1619.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-18 10:30:00 | 1603.10 | 1603.84 | 1619.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1612.50 | 1608.15 | 1614.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:45:00 | 1603.85 | 1607.88 | 1614.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 15:15:00 | 1627.30 | 1616.84 | 1616.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 1627.30 | 1616.84 | 1616.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1633.40 | 1620.15 | 1618.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 15:15:00 | 1632.00 | 1633.39 | 1626.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:00:00 | 1649.80 | 1636.67 | 1628.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 12:15:00 | 1647.20 | 1638.07 | 1630.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 13:00:00 | 1645.85 | 1639.63 | 1632.33 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 1636.10 | 1640.17 | 1633.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 1636.10 | 1640.17 | 1633.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 1627.00 | 1637.54 | 1633.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-23 15:15:00 | 1627.00 | 1637.54 | 1633.29 | SL hit (close<ema400) qty=1.00 sl=1633.29 alert=retest1 |

### Cycle 29 — SELL (started 2023-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 14:15:00 | 1612.05 | 1640.08 | 1640.80 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 1639.00 | 1632.90 | 1632.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 13:15:00 | 1643.70 | 1635.06 | 1633.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 10:15:00 | 1658.85 | 1661.77 | 1653.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 1658.85 | 1661.77 | 1653.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 1658.85 | 1661.77 | 1653.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:45:00 | 1658.25 | 1661.77 | 1653.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 1654.05 | 1660.23 | 1653.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:00:00 | 1654.05 | 1660.23 | 1653.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 1653.45 | 1658.87 | 1653.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:30:00 | 1652.55 | 1658.87 | 1653.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 1660.00 | 1659.10 | 1653.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:30:00 | 1651.55 | 1659.10 | 1653.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 1656.30 | 1658.60 | 1654.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 09:15:00 | 1667.60 | 1658.60 | 1654.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 10:15:00 | 1629.00 | 1652.71 | 1652.67 | SL hit (close<static) qty=1.00 sl=1647.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 1629.65 | 1648.10 | 1650.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 11:15:00 | 1616.80 | 1633.78 | 1637.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 1632.75 | 1625.17 | 1631.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 1632.75 | 1625.17 | 1631.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 1632.75 | 1625.17 | 1631.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 13:15:00 | 1617.00 | 1626.65 | 1630.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 14:15:00 | 1616.00 | 1625.24 | 1629.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 10:00:00 | 1616.50 | 1626.61 | 1628.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 10:15:00 | 1606.55 | 1607.54 | 1615.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 1600.20 | 1606.07 | 1614.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 11:15:00 | 1593.05 | 1606.07 | 1614.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 10:00:00 | 1585.40 | 1584.02 | 1597.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 12:15:00 | 1607.20 | 1594.83 | 1593.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 12:15:00 | 1607.20 | 1594.83 | 1593.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 1621.00 | 1601.94 | 1596.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 11:15:00 | 1613.50 | 1635.86 | 1625.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 11:15:00 | 1613.50 | 1635.86 | 1625.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 1613.50 | 1635.86 | 1625.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:30:00 | 1618.40 | 1635.86 | 1625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 1612.90 | 1631.26 | 1624.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:30:00 | 1611.95 | 1631.26 | 1624.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 1613.80 | 1625.70 | 1623.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:00:00 | 1613.80 | 1625.70 | 1623.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 1615.80 | 1623.72 | 1622.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 1665.55 | 1623.72 | 1622.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 15:15:00 | 1607.00 | 1622.07 | 1623.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 1607.00 | 1622.07 | 1623.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 13:15:00 | 1601.55 | 1611.36 | 1616.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 10:15:00 | 1594.85 | 1594.45 | 1601.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 11:00:00 | 1594.85 | 1594.45 | 1601.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 1592.20 | 1594.00 | 1600.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:45:00 | 1590.35 | 1593.20 | 1599.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 15:15:00 | 1590.10 | 1592.31 | 1598.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 10:45:00 | 1587.80 | 1594.33 | 1596.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 1591.00 | 1588.96 | 1592.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 1588.10 | 1589.29 | 1591.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 1602.00 | 1592.96 | 1592.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 1602.00 | 1592.96 | 1592.57 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 1582.30 | 1590.93 | 1591.93 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 09:15:00 | 1602.85 | 1592.84 | 1592.59 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 1576.15 | 1590.30 | 1591.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 14:15:00 | 1566.05 | 1585.45 | 1589.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 15:15:00 | 1590.10 | 1586.38 | 1589.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 15:15:00 | 1590.10 | 1586.38 | 1589.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 1590.10 | 1586.38 | 1589.33 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 1599.25 | 1590.26 | 1590.19 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 15:15:00 | 1589.00 | 1591.77 | 1592.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1554.20 | 1584.26 | 1588.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1565.10 | 1561.42 | 1571.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1565.10 | 1561.42 | 1571.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1565.10 | 1561.42 | 1571.67 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1604.80 | 1574.50 | 1573.32 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 15:15:00 | 1582.90 | 1590.26 | 1591.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 14:15:00 | 1576.75 | 1583.67 | 1587.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 1586.00 | 1583.55 | 1586.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 1586.00 | 1583.55 | 1586.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 1586.00 | 1583.55 | 1586.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:00:00 | 1586.00 | 1583.55 | 1586.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 1584.90 | 1583.82 | 1586.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:30:00 | 1587.10 | 1583.82 | 1586.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 1581.40 | 1583.34 | 1586.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 13:00:00 | 1579.25 | 1582.52 | 1585.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 13:45:00 | 1579.80 | 1582.02 | 1584.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 1615.65 | 1588.74 | 1587.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 14:15:00 | 1615.65 | 1588.74 | 1587.70 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 1580.25 | 1604.92 | 1607.91 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 1607.20 | 1600.92 | 1600.13 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 14:15:00 | 1590.35 | 1599.51 | 1599.75 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 1611.05 | 1600.77 | 1600.21 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 1594.45 | 1600.74 | 1600.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 1582.45 | 1597.08 | 1599.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 14:15:00 | 1596.75 | 1593.65 | 1596.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 1596.75 | 1593.65 | 1596.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 1596.75 | 1593.65 | 1596.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 14:30:00 | 1601.00 | 1593.65 | 1596.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 1602.00 | 1595.32 | 1597.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:15:00 | 1596.35 | 1595.32 | 1597.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 1596.00 | 1595.46 | 1597.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 10:15:00 | 1582.25 | 1595.46 | 1597.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 11:15:00 | 1604.85 | 1598.36 | 1598.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 11:15:00 | 1604.85 | 1598.36 | 1598.16 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 1589.50 | 1596.59 | 1597.38 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 12:15:00 | 1603.25 | 1597.97 | 1597.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 14:15:00 | 1666.55 | 1628.71 | 1614.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 1700.35 | 1704.20 | 1673.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 09:30:00 | 1700.05 | 1704.20 | 1673.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1710.00 | 1712.12 | 1695.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:45:00 | 1700.10 | 1712.12 | 1695.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 1668.05 | 1709.30 | 1701.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 1666.05 | 1709.30 | 1701.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 1674.20 | 1702.28 | 1699.26 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 11:15:00 | 1664.00 | 1694.62 | 1696.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 12:15:00 | 1652.00 | 1686.10 | 1692.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 1635.00 | 1632.75 | 1647.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-13 10:00:00 | 1635.00 | 1632.75 | 1647.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 1614.95 | 1601.56 | 1613.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 1614.95 | 1601.56 | 1613.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 1600.30 | 1601.31 | 1612.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 11:15:00 | 1594.85 | 1601.31 | 1612.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 14:45:00 | 1595.10 | 1601.02 | 1608.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 15:15:00 | 1594.00 | 1601.02 | 1608.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 1623.00 | 1604.29 | 1608.86 | SL hit (close>static) qty=1.00 sl=1614.85 alert=retest2 |

### Cycle 52 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 1631.20 | 1614.06 | 1612.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 1641.55 | 1627.12 | 1620.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 15:15:00 | 1630.00 | 1630.29 | 1625.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 15:15:00 | 1630.00 | 1630.29 | 1625.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 1630.00 | 1630.29 | 1625.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:15:00 | 1624.10 | 1630.29 | 1625.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 1625.10 | 1629.25 | 1625.27 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 13:15:00 | 1608.95 | 1621.95 | 1622.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 15:15:00 | 1605.05 | 1617.30 | 1620.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1615.15 | 1597.98 | 1602.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1615.15 | 1597.98 | 1602.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1615.15 | 1597.98 | 1602.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 1615.15 | 1597.98 | 1602.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1613.45 | 1601.08 | 1603.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:30:00 | 1618.95 | 1601.08 | 1603.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 12:15:00 | 1637.90 | 1610.82 | 1607.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 12:15:00 | 1647.00 | 1628.49 | 1619.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 14:15:00 | 1629.55 | 1630.38 | 1621.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 14:45:00 | 1629.75 | 1630.38 | 1621.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 1644.70 | 1633.18 | 1624.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 12:45:00 | 1650.05 | 1643.99 | 1635.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 13:15:00 | 1694.55 | 1704.33 | 1704.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 13:15:00 | 1694.55 | 1704.33 | 1704.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 14:15:00 | 1681.30 | 1699.73 | 1702.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 14:15:00 | 1662.25 | 1660.19 | 1673.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 15:00:00 | 1662.25 | 1660.19 | 1673.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1692.00 | 1668.12 | 1674.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:45:00 | 1682.30 | 1668.12 | 1674.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 1695.90 | 1673.68 | 1676.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:45:00 | 1698.65 | 1673.68 | 1676.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 1711.35 | 1681.21 | 1679.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 12:15:00 | 1723.90 | 1689.75 | 1683.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 14:15:00 | 1694.40 | 1708.92 | 1700.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 14:15:00 | 1694.40 | 1708.92 | 1700.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1694.40 | 1708.92 | 1700.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 1694.40 | 1708.92 | 1700.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 1698.00 | 1706.74 | 1700.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 09:15:00 | 1708.00 | 1706.74 | 1700.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 13:15:00 | 1690.00 | 1699.73 | 1699.23 | SL hit (close<static) qty=1.00 sl=1692.30 alert=retest2 |

### Cycle 57 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 1688.80 | 1697.54 | 1698.28 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 10:15:00 | 1717.30 | 1701.14 | 1699.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 11:15:00 | 1721.30 | 1705.18 | 1701.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 1713.00 | 1718.29 | 1710.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 1713.00 | 1718.29 | 1710.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1713.00 | 1718.29 | 1710.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 13:15:00 | 1804.90 | 1726.62 | 1716.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 14:45:00 | 1758.95 | 1772.39 | 1767.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 15:15:00 | 1767.00 | 1772.39 | 1767.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 11:15:00 | 1780.00 | 1803.69 | 1805.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 11:15:00 | 1780.00 | 1803.69 | 1805.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 12:15:00 | 1776.50 | 1798.25 | 1803.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 15:15:00 | 1818.00 | 1797.50 | 1801.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 15:15:00 | 1818.00 | 1797.50 | 1801.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 1818.00 | 1797.50 | 1801.34 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 1823.00 | 1805.66 | 1804.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 1870.95 | 1821.62 | 1811.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 15:15:00 | 1918.00 | 1919.94 | 1894.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 09:15:00 | 1909.70 | 1919.94 | 1894.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1888.95 | 1913.74 | 1894.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 1888.95 | 1913.74 | 1894.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1875.05 | 1906.00 | 1892.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 1879.80 | 1906.00 | 1892.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 1883.45 | 1896.57 | 1890.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 15:00:00 | 1891.00 | 1893.15 | 1889.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 09:15:00 | 1873.25 | 1887.55 | 1887.49 | SL hit (close<static) qty=1.00 sl=1876.95 alert=retest2 |

### Cycle 61 — SELL (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 10:15:00 | 1871.30 | 1884.30 | 1886.02 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 1906.75 | 1888.79 | 1887.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 12:15:00 | 1922.65 | 1901.40 | 1895.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 2090.20 | 2118.15 | 2060.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 15:00:00 | 2090.20 | 2118.15 | 2060.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 2107.65 | 2116.05 | 2065.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 2218.00 | 2116.05 | 2065.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 13:30:00 | 2116.40 | 2121.59 | 2088.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 09:15:00 | 2040.55 | 2087.41 | 2079.54 | SL hit (close<static) qty=1.00 sl=2047.20 alert=retest2 |

### Cycle 63 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 2012.25 | 2065.16 | 2070.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 15:15:00 | 1990.00 | 2020.84 | 2036.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 1980.05 | 1967.69 | 1985.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 1980.05 | 1967.69 | 1985.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 1980.05 | 1967.69 | 1985.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 1974.45 | 1967.69 | 1985.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 2004.60 | 1975.07 | 1987.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:00:00 | 2004.60 | 1975.07 | 1987.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 1992.05 | 1978.47 | 1987.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:00:00 | 1992.05 | 1978.47 | 1987.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 1968.95 | 1976.56 | 1986.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 13:30:00 | 1955.35 | 1974.25 | 1984.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 14:00:00 | 1965.00 | 1974.25 | 1984.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 2042.65 | 1976.40 | 1975.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 14:15:00 | 2042.65 | 1976.40 | 1975.01 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 2007.80 | 2044.71 | 2045.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 10:15:00 | 1989.80 | 2033.73 | 2040.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 2003.25 | 2001.00 | 2017.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 09:30:00 | 1999.00 | 2001.00 | 2017.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1999.00 | 1994.19 | 2006.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 2036.00 | 1994.19 | 2006.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 2048.60 | 2005.07 | 2010.68 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 2037.60 | 2017.49 | 2015.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 2052.65 | 2027.77 | 2022.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 10:15:00 | 2134.05 | 2139.46 | 2117.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-02 11:00:00 | 2134.05 | 2139.46 | 2117.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 2120.05 | 2132.64 | 2118.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:45:00 | 2107.45 | 2132.64 | 2118.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 2144.60 | 2135.03 | 2120.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:45:00 | 2117.85 | 2135.03 | 2120.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 2125.00 | 2132.54 | 2122.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 2138.75 | 2132.54 | 2122.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 2144.90 | 2135.01 | 2124.09 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 09:15:00 | 2072.30 | 2118.31 | 2124.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 10:15:00 | 2053.20 | 2105.28 | 2117.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 1819.00 | 1806.32 | 1854.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 11:00:00 | 1819.00 | 1806.32 | 1854.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 1829.75 | 1817.64 | 1832.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 1829.75 | 1817.64 | 1832.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 1813.25 | 1816.76 | 1830.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 10:45:00 | 1806.85 | 1814.22 | 1826.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 10:00:00 | 1803.40 | 1801.20 | 1813.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 09:30:00 | 1800.00 | 1794.29 | 1802.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 11:15:00 | 1833.25 | 1807.58 | 1807.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 1833.25 | 1807.58 | 1807.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 11:15:00 | 1858.00 | 1835.24 | 1825.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 09:15:00 | 1838.00 | 1843.14 | 1833.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 1838.00 | 1843.14 | 1833.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1838.00 | 1843.14 | 1833.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 11:30:00 | 1855.85 | 1843.79 | 1838.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 10:30:00 | 1851.25 | 1853.16 | 1846.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 1851.50 | 1852.05 | 1847.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 1824.25 | 1846.38 | 1847.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1824.25 | 1846.38 | 1847.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1813.40 | 1839.78 | 1844.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 1836.00 | 1825.20 | 1833.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 10:15:00 | 1836.00 | 1825.20 | 1833.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 1836.00 | 1825.20 | 1833.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:30:00 | 1832.95 | 1825.20 | 1833.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1865.00 | 1833.16 | 1836.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:00:00 | 1865.00 | 1833.16 | 1836.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1848.55 | 1836.24 | 1837.53 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 1854.90 | 1839.97 | 1839.11 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 12:15:00 | 1829.40 | 1839.10 | 1839.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 09:15:00 | 1820.00 | 1834.20 | 1836.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 11:15:00 | 1840.00 | 1834.52 | 1836.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 11:15:00 | 1840.00 | 1834.52 | 1836.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 1840.00 | 1834.52 | 1836.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:30:00 | 1840.00 | 1834.52 | 1836.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 1836.15 | 1834.85 | 1836.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 15:15:00 | 1830.00 | 1836.25 | 1836.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 09:30:00 | 1825.00 | 1831.74 | 1834.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 14:15:00 | 1823.80 | 1810.73 | 1810.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 1823.80 | 1810.73 | 1810.46 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 1793.60 | 1809.04 | 1810.01 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 13:15:00 | 1819.20 | 1809.62 | 1808.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 14:15:00 | 1825.25 | 1812.75 | 1810.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1795.00 | 1812.76 | 1810.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 1795.00 | 1812.76 | 1810.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1795.00 | 1812.76 | 1810.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 1795.00 | 1812.76 | 1810.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 1787.00 | 1807.61 | 1808.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1778.75 | 1801.83 | 1805.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 15:15:00 | 1775.00 | 1760.63 | 1773.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 15:15:00 | 1775.00 | 1760.63 | 1773.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 1775.00 | 1760.63 | 1773.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 1826.00 | 1760.63 | 1773.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1820.25 | 1772.56 | 1777.32 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 10:15:00 | 1825.10 | 1783.06 | 1781.67 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 1782.00 | 1792.60 | 1792.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 1775.00 | 1789.08 | 1791.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 15:15:00 | 1763.90 | 1762.37 | 1773.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-21 09:15:00 | 1770.35 | 1762.37 | 1773.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1781.05 | 1766.10 | 1773.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:45:00 | 1781.75 | 1766.10 | 1773.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1781.50 | 1769.18 | 1774.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:30:00 | 1775.90 | 1769.18 | 1774.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 1781.00 | 1771.55 | 1775.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 13:00:00 | 1776.45 | 1772.53 | 1775.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 14:15:00 | 1776.65 | 1773.62 | 1775.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 14:45:00 | 1777.10 | 1775.22 | 1776.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 09:15:00 | 1788.00 | 1778.38 | 1777.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 1788.00 | 1778.38 | 1777.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 1815.80 | 1788.23 | 1782.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 1788.35 | 1788.72 | 1783.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 1788.35 | 1788.72 | 1783.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 1786.00 | 1788.18 | 1784.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 09:15:00 | 1802.85 | 1788.18 | 1784.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 10:15:00 | 1814.95 | 1789.19 | 1784.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1776.70 | 1785.74 | 1784.08 | SL hit (close<static) qty=1.00 sl=1778.60 alert=retest2 |

### Cycle 79 — SELL (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 13:15:00 | 1774.60 | 1783.02 | 1783.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 1763.90 | 1779.20 | 1781.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 1795.00 | 1778.48 | 1780.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 10:15:00 | 1795.00 | 1778.48 | 1780.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 1795.00 | 1778.48 | 1780.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:00:00 | 1795.00 | 1778.48 | 1780.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 1795.00 | 1781.78 | 1781.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 13:15:00 | 1829.65 | 1794.59 | 1787.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 09:15:00 | 1791.85 | 1799.40 | 1791.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 1791.85 | 1799.40 | 1791.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1791.85 | 1799.40 | 1791.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:15:00 | 1782.90 | 1799.40 | 1791.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1795.80 | 1798.68 | 1792.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 1797.25 | 1798.68 | 1792.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 12:15:00 | 1807.80 | 1797.95 | 1792.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 11:15:00 | 1895.75 | 1906.06 | 1906.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 1895.75 | 1906.06 | 1906.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 1874.15 | 1896.57 | 1901.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1882.95 | 1863.97 | 1877.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1882.95 | 1863.97 | 1877.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1882.95 | 1863.97 | 1877.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 1882.95 | 1863.97 | 1877.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1900.85 | 1871.35 | 1879.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 1900.85 | 1871.35 | 1879.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1899.20 | 1876.92 | 1881.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 1901.50 | 1876.92 | 1881.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 1872.85 | 1877.72 | 1880.85 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 1889.70 | 1883.47 | 1882.78 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 1866.75 | 1879.72 | 1881.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 1855.85 | 1873.27 | 1877.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 13:15:00 | 1870.00 | 1850.01 | 1854.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 13:15:00 | 1870.00 | 1850.01 | 1854.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 1870.00 | 1850.01 | 1854.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:30:00 | 1870.90 | 1850.01 | 1854.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1867.40 | 1853.49 | 1855.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 1867.40 | 1853.49 | 1855.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 15:15:00 | 1879.70 | 1858.73 | 1857.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 1936.00 | 1874.19 | 1864.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 13:15:00 | 1967.30 | 1976.24 | 1959.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 14:00:00 | 1967.30 | 1976.24 | 1959.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1957.00 | 1970.99 | 1960.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 2005.80 | 1970.99 | 1960.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 15:15:00 | 1991.00 | 1981.85 | 1971.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 1954.60 | 1974.32 | 1970.64 | SL hit (close<static) qty=1.00 sl=1955.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 1957.35 | 1967.69 | 1968.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 1943.05 | 1959.11 | 1963.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 13:15:00 | 1945.90 | 1941.80 | 1948.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 13:15:00 | 1945.90 | 1941.80 | 1948.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 13:15:00 | 1945.90 | 1941.80 | 1948.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 14:00:00 | 1945.90 | 1941.80 | 1948.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 1969.45 | 1947.33 | 1950.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 15:00:00 | 1969.45 | 1947.33 | 1950.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 15:15:00 | 1975.00 | 1952.86 | 1952.63 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 1935.95 | 1949.48 | 1951.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 1922.00 | 1943.98 | 1948.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1922.00 | 1915.01 | 1928.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:30:00 | 1919.85 | 1915.01 | 1928.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1926.15 | 1917.24 | 1928.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 1926.15 | 1917.24 | 1928.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1942.85 | 1922.36 | 1929.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 1942.85 | 1922.36 | 1929.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1940.50 | 1925.99 | 1930.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 1929.90 | 1931.71 | 1932.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 09:15:00 | 1962.05 | 1937.49 | 1935.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 1962.05 | 1937.49 | 1935.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 09:15:00 | 2091.75 | 1981.30 | 1958.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 2051.15 | 2056.82 | 2016.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-13 10:00:00 | 2051.15 | 2056.82 | 2016.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 2034.75 | 2050.48 | 2032.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 2042.10 | 2050.48 | 2032.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 2030.50 | 2046.48 | 2032.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:15:00 | 2021.65 | 2046.48 | 2032.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 2021.70 | 2041.52 | 2031.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:45:00 | 2017.90 | 2041.52 | 2031.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 2022.55 | 2037.73 | 2030.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 2048.00 | 2030.13 | 2028.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 10:15:00 | 2028.05 | 2028.92 | 2028.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 11:30:00 | 2031.45 | 2028.71 | 2028.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 2025.20 | 2033.92 | 2034.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 2025.20 | 2033.92 | 2034.61 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 2248.65 | 2074.95 | 2053.02 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 2075.00 | 2114.55 | 2117.60 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 2150.00 | 2121.64 | 2120.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 13:15:00 | 2197.65 | 2185.44 | 2166.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 13:15:00 | 2186.60 | 2190.90 | 2179.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 2186.60 | 2190.90 | 2179.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 2186.60 | 2190.90 | 2179.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 2196.30 | 2190.90 | 2179.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 2189.40 | 2192.85 | 2185.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 2186.20 | 2192.85 | 2185.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 2173.45 | 2188.97 | 2184.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:45:00 | 2169.35 | 2188.97 | 2184.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 2161.50 | 2183.48 | 2182.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 2161.50 | 2183.48 | 2182.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 2171.00 | 2180.98 | 2181.36 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 2190.00 | 2182.30 | 2181.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 10:15:00 | 2200.95 | 2188.85 | 2184.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 2188.75 | 2195.06 | 2189.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 2188.75 | 2195.06 | 2189.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 2188.75 | 2195.06 | 2189.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:45:00 | 2191.75 | 2195.06 | 2189.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 2200.00 | 2196.05 | 2190.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 2202.90 | 2196.05 | 2190.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 2207.45 | 2197.38 | 2192.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:00:00 | 2206.05 | 2228.23 | 2219.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 2157.40 | 2214.07 | 2213.62 | SL hit (close<static) qty=1.00 sl=2182.80 alert=retest2 |

### Cycle 95 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2088.15 | 2188.88 | 2202.22 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 2262.45 | 2210.63 | 2208.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 2321.90 | 2232.89 | 2218.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 2419.35 | 2423.71 | 2387.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 14:45:00 | 2419.75 | 2423.71 | 2387.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2475.95 | 2470.56 | 2449.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 2449.85 | 2470.56 | 2449.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2677.05 | 2707.88 | 2665.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 2706.65 | 2707.88 | 2665.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 2649.15 | 2696.13 | 2664.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 2664.15 | 2696.13 | 2664.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 2655.00 | 2687.91 | 2663.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:30:00 | 2667.55 | 2687.91 | 2663.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 2636.90 | 2677.70 | 2661.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 2636.90 | 2677.70 | 2661.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 2583.85 | 2647.62 | 2649.53 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 2720.60 | 2654.92 | 2652.06 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 2650.00 | 2677.04 | 2679.78 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 2751.70 | 2691.65 | 2685.92 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 2616.05 | 2677.28 | 2684.79 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 2702.65 | 2674.40 | 2673.75 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 2659.85 | 2671.49 | 2672.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 2650.00 | 2667.19 | 2670.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 2684.90 | 2662.92 | 2667.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 2684.90 | 2662.92 | 2667.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 2684.90 | 2662.92 | 2667.08 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 2712.05 | 2672.75 | 2671.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 2725.95 | 2701.55 | 2690.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 2749.70 | 2750.23 | 2737.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 14:45:00 | 2750.00 | 2750.23 | 2737.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 2735.70 | 2753.81 | 2744.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 2735.70 | 2753.81 | 2744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 2724.80 | 2748.01 | 2742.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 2713.30 | 2748.01 | 2742.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2748.40 | 2742.24 | 2740.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 2748.35 | 2742.24 | 2740.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 2733.45 | 2740.48 | 2740.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 2733.45 | 2740.48 | 2740.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 2727.05 | 2737.80 | 2739.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 2697.60 | 2728.51 | 2734.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 2619.30 | 2603.14 | 2616.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 2619.30 | 2603.14 | 2616.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2619.30 | 2603.14 | 2616.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 2619.30 | 2603.14 | 2616.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 2600.80 | 2602.67 | 2615.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 2619.80 | 2602.67 | 2615.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2555.00 | 2580.28 | 2597.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:15:00 | 2553.25 | 2580.28 | 2597.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 2647.40 | 2591.33 | 2599.61 | SL hit (close>static) qty=1.00 sl=2601.55 alert=retest2 |

### Cycle 106 — BUY (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 12:15:00 | 2686.00 | 2610.26 | 2607.47 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 2600.90 | 2616.41 | 2616.55 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 2636.70 | 2617.14 | 2616.50 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 10:15:00 | 2601.10 | 2613.93 | 2615.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 11:15:00 | 2597.25 | 2610.59 | 2613.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 2606.15 | 2562.78 | 2577.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2606.15 | 2562.78 | 2577.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2606.15 | 2562.78 | 2577.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2606.15 | 2562.78 | 2577.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2582.35 | 2566.70 | 2577.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 2571.25 | 2566.70 | 2577.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:00:00 | 2565.15 | 2555.98 | 2566.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 2594.00 | 2572.17 | 2569.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 2594.00 | 2572.17 | 2569.61 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 2554.75 | 2567.08 | 2568.56 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 2584.65 | 2570.60 | 2570.02 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 2565.90 | 2569.56 | 2569.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 2558.45 | 2565.70 | 2567.65 | Break + close below crossover candle low |

### Cycle 114 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 2603.75 | 2573.31 | 2570.93 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 2540.25 | 2579.45 | 2584.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 10:15:00 | 2540.10 | 2562.63 | 2573.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 15:15:00 | 2552.25 | 2546.37 | 2559.55 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 09:15:00 | 2461.05 | 2546.37 | 2559.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2487.15 | 2471.11 | 2503.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-06 12:15:00 | 2522.00 | 2484.62 | 2502.04 | SL hit (close>ema400) qty=1.00 sl=2502.04 alert=retest1 |

### Cycle 116 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 2550.00 | 2504.25 | 2502.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 12:15:00 | 2567.85 | 2531.91 | 2517.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 2577.25 | 2578.69 | 2557.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:45:00 | 2566.35 | 2578.69 | 2557.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2547.95 | 2570.57 | 2557.63 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 2533.55 | 2551.57 | 2551.72 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 15:15:00 | 2561.00 | 2552.58 | 2552.10 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 2541.15 | 2550.30 | 2551.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 10:15:00 | 2539.15 | 2548.07 | 2550.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 12:15:00 | 2549.90 | 2546.50 | 2548.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 2549.90 | 2546.50 | 2548.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 2549.90 | 2546.50 | 2548.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 2544.10 | 2546.50 | 2548.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 2537.45 | 2544.69 | 2547.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:30:00 | 2499.90 | 2541.14 | 2545.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:45:00 | 2520.45 | 2509.09 | 2523.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 2519.20 | 2516.13 | 2523.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 13:30:00 | 2518.45 | 2504.38 | 2504.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 14:15:00 | 2511.10 | 2505.72 | 2505.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 2511.10 | 2505.72 | 2505.43 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 2484.10 | 2502.08 | 2503.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 10:15:00 | 2459.55 | 2493.58 | 2499.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 12:15:00 | 2492.90 | 2486.47 | 2495.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 12:15:00 | 2492.90 | 2486.47 | 2495.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 2492.90 | 2486.47 | 2495.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 2492.90 | 2486.47 | 2495.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 2508.90 | 2490.96 | 2496.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 2508.90 | 2490.96 | 2496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 2526.90 | 2498.14 | 2499.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 2526.90 | 2498.14 | 2499.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 2530.00 | 2504.52 | 2501.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 2549.85 | 2513.58 | 2506.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 2578.65 | 2588.31 | 2565.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 2578.65 | 2588.31 | 2565.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 2578.65 | 2588.31 | 2565.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 2582.05 | 2588.31 | 2565.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 2545.75 | 2580.40 | 2565.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 2545.75 | 2580.40 | 2565.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 2548.90 | 2574.10 | 2564.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 2548.90 | 2574.10 | 2564.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 2539.90 | 2567.26 | 2561.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:45:00 | 2526.65 | 2567.26 | 2561.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 2546.70 | 2558.04 | 2558.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 2531.80 | 2552.79 | 2555.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 12:15:00 | 2518.15 | 2516.44 | 2529.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:00:00 | 2518.15 | 2516.44 | 2529.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2494.75 | 2494.58 | 2504.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 2466.15 | 2491.73 | 2499.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 2467.15 | 2484.11 | 2495.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 12:15:00 | 2509.95 | 2486.94 | 2484.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 2509.95 | 2486.94 | 2484.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 2531.65 | 2502.41 | 2493.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 11:15:00 | 2512.60 | 2532.11 | 2519.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 11:15:00 | 2512.60 | 2532.11 | 2519.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 2512.60 | 2532.11 | 2519.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:45:00 | 2512.65 | 2532.11 | 2519.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 2510.95 | 2527.88 | 2519.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 2541.85 | 2518.28 | 2516.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 11:15:00 | 2539.05 | 2521.89 | 2521.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 11:15:00 | 2507.05 | 2518.92 | 2519.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 2507.05 | 2518.92 | 2519.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 2493.55 | 2513.85 | 2517.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 2519.95 | 2511.35 | 2515.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 15:15:00 | 2519.95 | 2511.35 | 2515.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2519.95 | 2511.35 | 2515.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 2530.65 | 2511.35 | 2515.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2502.00 | 2509.48 | 2513.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 2496.50 | 2509.48 | 2513.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:00:00 | 2497.65 | 2505.83 | 2511.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 2453.45 | 2440.14 | 2440.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 13:15:00 | 2453.45 | 2440.14 | 2440.06 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 2415.45 | 2440.83 | 2440.92 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 13:15:00 | 2466.95 | 2442.02 | 2440.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 2534.00 | 2460.42 | 2449.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 2442.35 | 2515.67 | 2491.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 2442.35 | 2515.67 | 2491.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 2442.35 | 2515.67 | 2491.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 2442.35 | 2515.67 | 2491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 2434.20 | 2499.37 | 2485.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 2414.85 | 2499.37 | 2485.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 2428.95 | 2470.83 | 2474.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 2393.45 | 2426.74 | 2442.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 2420.95 | 2376.32 | 2386.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 2420.95 | 2376.32 | 2386.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 2420.95 | 2376.32 | 2386.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 2420.00 | 2376.32 | 2386.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 2387.50 | 2378.56 | 2386.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:30:00 | 2375.10 | 2379.07 | 2385.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 2402.40 | 2391.33 | 2389.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 2402.40 | 2391.33 | 2389.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 2410.00 | 2395.07 | 2391.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 13:15:00 | 2396.75 | 2397.80 | 2393.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 2396.75 | 2397.80 | 2393.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 2396.75 | 2397.80 | 2393.71 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 2375.70 | 2390.61 | 2391.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 2332.95 | 2369.11 | 2380.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 2214.90 | 2211.53 | 2253.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 2218.45 | 2211.53 | 2253.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2277.40 | 2227.32 | 2242.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 2279.40 | 2227.32 | 2242.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2276.00 | 2237.06 | 2245.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 2278.80 | 2237.06 | 2245.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 2320.50 | 2253.75 | 2251.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 2329.10 | 2268.82 | 2258.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 2338.00 | 2340.33 | 2323.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:45:00 | 2343.10 | 2340.33 | 2323.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 2330.25 | 2336.07 | 2325.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 2339.30 | 2336.07 | 2325.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 2344.30 | 2337.72 | 2327.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 2351.65 | 2337.88 | 2330.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 12:15:00 | 2316.55 | 2333.87 | 2330.02 | SL hit (close<static) qty=1.00 sl=2326.40 alert=retest2 |

### Cycle 133 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 2315.40 | 2327.00 | 2327.37 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 2335.00 | 2328.60 | 2328.06 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 2317.40 | 2326.36 | 2327.09 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 2391.10 | 2339.31 | 2332.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 14:15:00 | 2441.60 | 2369.36 | 2349.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2336.85 | 2374.34 | 2356.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 2336.85 | 2374.34 | 2356.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2336.85 | 2374.34 | 2356.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 2336.85 | 2374.34 | 2356.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 2380.50 | 2375.57 | 2358.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:00:00 | 2386.70 | 2377.80 | 2361.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:45:00 | 2386.00 | 2385.14 | 2371.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:30:00 | 2388.00 | 2384.73 | 2374.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 2391.65 | 2383.95 | 2376.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2393.95 | 2385.95 | 2378.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 2363.70 | 2385.95 | 2378.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 2360.10 | 2380.74 | 2377.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:30:00 | 2360.05 | 2380.74 | 2377.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 2365.20 | 2377.63 | 2376.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 2365.20 | 2377.63 | 2376.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2378.45 | 2380.15 | 2377.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 2378.45 | 2380.15 | 2377.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 2376.00 | 2379.32 | 2377.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 2378.05 | 2379.32 | 2377.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2381.30 | 2379.72 | 2377.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-22 11:15:00 | 2356.65 | 2376.38 | 2376.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 2356.65 | 2376.38 | 2376.84 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 2373.70 | 2370.35 | 2370.02 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 2360.35 | 2369.89 | 2369.96 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 2385.00 | 2372.91 | 2371.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 14:15:00 | 2393.30 | 2378.54 | 2374.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 10:15:00 | 2378.95 | 2381.92 | 2377.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 10:15:00 | 2378.95 | 2381.92 | 2377.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 2378.95 | 2381.92 | 2377.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 2378.75 | 2381.92 | 2377.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 2373.95 | 2380.33 | 2376.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:45:00 | 2371.50 | 2380.33 | 2376.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 2374.95 | 2379.25 | 2376.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 2377.95 | 2378.99 | 2376.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 14:45:00 | 2378.00 | 2378.51 | 2376.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 2353.35 | 2372.92 | 2374.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 2353.35 | 2372.92 | 2374.54 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 2425.00 | 2363.51 | 2355.37 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 12:15:00 | 2364.50 | 2384.72 | 2385.91 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 2421.50 | 2387.41 | 2386.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 2577.25 | 2469.49 | 2437.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 09:15:00 | 2493.90 | 2503.05 | 2472.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 09:30:00 | 2493.85 | 2503.05 | 2472.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2468.00 | 2491.40 | 2474.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:00:00 | 2468.00 | 2491.40 | 2474.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 2476.30 | 2488.38 | 2474.39 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 2398.65 | 2463.72 | 2465.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 2391.35 | 2423.30 | 2442.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2437.80 | 2419.28 | 2437.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2437.80 | 2419.28 | 2437.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2437.80 | 2419.28 | 2437.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 2437.80 | 2419.28 | 2437.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 2409.35 | 2417.29 | 2434.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 2449.00 | 2417.29 | 2434.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2393.05 | 2374.58 | 2390.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 2391.75 | 2374.58 | 2390.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2397.20 | 2379.11 | 2391.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 2397.20 | 2379.11 | 2391.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 2387.80 | 2380.85 | 2390.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 2400.00 | 2380.85 | 2390.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 2379.65 | 2380.61 | 2389.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 2379.65 | 2380.61 | 2389.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2345.25 | 2365.43 | 2379.06 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2488.10 | 2385.85 | 2377.36 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 2361.75 | 2386.37 | 2389.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 2351.10 | 2379.31 | 2385.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 2350.05 | 2336.08 | 2352.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 2350.05 | 2336.08 | 2352.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 2350.00 | 2338.87 | 2352.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 2352.05 | 2338.87 | 2352.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2356.60 | 2342.41 | 2352.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 2363.15 | 2342.41 | 2352.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 2368.15 | 2347.56 | 2354.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 2368.15 | 2347.56 | 2354.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 2368.35 | 2351.72 | 2355.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 2367.90 | 2351.72 | 2355.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 2349.35 | 2350.57 | 2354.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 2349.35 | 2350.57 | 2354.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 2360.00 | 2351.65 | 2353.88 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 2360.00 | 2355.04 | 2354.97 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 2351.25 | 2354.97 | 2355.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 2338.45 | 2349.54 | 2352.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 2348.50 | 2348.40 | 2351.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:00:00 | 2348.50 | 2348.40 | 2351.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 2333.00 | 2344.85 | 2348.65 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 2354.55 | 2345.99 | 2345.28 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 2342.95 | 2345.19 | 2345.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 2334.35 | 2343.02 | 2344.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 15:15:00 | 2346.00 | 2343.62 | 2344.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 15:15:00 | 2346.00 | 2343.62 | 2344.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 2346.00 | 2343.62 | 2344.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 2345.65 | 2343.62 | 2344.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2343.05 | 2343.50 | 2344.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 2348.90 | 2343.50 | 2344.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 2337.30 | 2342.26 | 2343.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 2333.00 | 2339.95 | 2341.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 2216.35 | 2253.10 | 2268.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 2230.35 | 2223.18 | 2238.55 | SL hit (close>ema200) qty=0.50 sl=2223.18 alert=retest2 |

### Cycle 152 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 2249.55 | 2176.45 | 2166.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 2258.55 | 2192.87 | 2175.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 2184.75 | 2202.78 | 2183.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 2184.75 | 2202.78 | 2183.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 2184.75 | 2202.78 | 2183.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 2184.00 | 2202.78 | 2183.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 2182.95 | 2198.82 | 2183.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 2182.95 | 2198.82 | 2183.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 2182.15 | 2195.48 | 2183.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 2191.15 | 2192.54 | 2183.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:45:00 | 2189.95 | 2191.70 | 2184.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 2192.20 | 2191.70 | 2184.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 2159.90 | 2190.25 | 2189.89 | SL hit (close<static) qty=1.00 sl=2178.45 alert=retest2 |

### Cycle 153 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2116.80 | 2175.56 | 2183.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 2114.75 | 2163.40 | 2177.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 2123.20 | 2110.68 | 2131.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 2123.20 | 2110.68 | 2131.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 2123.20 | 2110.68 | 2131.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 2123.20 | 2110.68 | 2131.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 2125.00 | 2113.54 | 2130.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 2110.80 | 2113.54 | 2130.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 2115.80 | 2114.49 | 2129.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:45:00 | 2116.00 | 2116.29 | 2126.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 2113.15 | 2113.95 | 2120.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 2113.65 | 2113.89 | 2119.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:45:00 | 2097.85 | 2108.46 | 2116.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 2005.26 | 2057.45 | 2079.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 2010.01 | 2057.45 | 2079.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 2010.20 | 2057.45 | 2079.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 2007.49 | 2057.45 | 2079.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 1992.96 | 2047.60 | 2073.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 2038.95 | 2032.18 | 2051.95 | SL hit (close>ema200) qty=0.50 sl=2032.18 alert=retest2 |

### Cycle 154 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 2085.85 | 2059.51 | 2058.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 15:15:00 | 2122.95 | 2098.47 | 2086.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 2098.35 | 2098.45 | 2087.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:00:00 | 2098.35 | 2098.45 | 2087.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2104.90 | 2106.88 | 2098.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 11:30:00 | 2119.80 | 2105.75 | 2102.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:45:00 | 2120.00 | 2108.02 | 2104.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 2116.30 | 2109.08 | 2105.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 2079.70 | 2101.43 | 2103.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 2079.70 | 2101.43 | 2103.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 2027.70 | 2079.18 | 2090.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1988.90 | 1982.65 | 2010.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 1992.40 | 1982.65 | 2010.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1961.85 | 1980.17 | 2000.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:30:00 | 1980.45 | 1980.17 | 2000.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1995.80 | 1978.43 | 1990.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 1995.80 | 1978.43 | 1990.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1989.45 | 1980.64 | 1990.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 1981.65 | 1980.64 | 1990.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 1986.40 | 1982.57 | 1986.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1985.00 | 1983.77 | 1986.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 2006.85 | 1988.59 | 1988.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 2006.85 | 1988.59 | 1988.15 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 09:15:00 | 1984.35 | 1994.61 | 1995.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 1971.95 | 1990.08 | 1993.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 13:15:00 | 1964.80 | 1963.36 | 1973.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:00:00 | 1964.80 | 1963.36 | 1973.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2004.15 | 1967.33 | 1972.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 2004.15 | 1967.33 | 1972.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1991.10 | 1972.09 | 1973.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 2016.70 | 1972.09 | 1973.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1976.00 | 1973.21 | 1974.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1976.00 | 1973.21 | 1974.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 13:15:00 | 2001.45 | 1978.86 | 1976.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 14:15:00 | 2008.40 | 1984.76 | 1979.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 11:15:00 | 1984.45 | 1989.67 | 1983.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 11:15:00 | 1984.45 | 1989.67 | 1983.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1984.45 | 1989.67 | 1983.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 1984.45 | 1989.67 | 1983.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1977.85 | 1987.31 | 1983.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 1977.85 | 1987.31 | 1983.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1963.50 | 1982.55 | 1981.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 1963.50 | 1982.55 | 1981.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 1972.45 | 1980.53 | 1980.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1955.70 | 1972.44 | 1976.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 1874.85 | 1871.89 | 1898.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 1874.85 | 1871.89 | 1898.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1877.10 | 1872.93 | 1896.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 1884.05 | 1872.93 | 1896.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1922.60 | 1882.86 | 1898.76 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 1930.10 | 1910.82 | 1908.72 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 1867.00 | 1903.41 | 1905.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 1864.65 | 1882.69 | 1894.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 1835.00 | 1833.78 | 1855.45 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:15:00 | 1794.80 | 1833.78 | 1855.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1847.85 | 1826.72 | 1838.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 1847.85 | 1826.72 | 1838.40 | SL hit (close>ema400) qty=1.00 sl=1838.40 alert=retest1 |

### Cycle 162 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1858.60 | 1845.37 | 1845.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 15:15:00 | 1894.95 | 1874.34 | 1864.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1863.15 | 1872.10 | 1864.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1863.15 | 1872.10 | 1864.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1863.15 | 1872.10 | 1864.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1863.15 | 1872.10 | 1864.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1877.50 | 1873.18 | 1865.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1879.95 | 1873.66 | 1866.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 1881.15 | 1874.66 | 1868.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 12:15:00 | 1853.10 | 1864.72 | 1865.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 1853.10 | 1864.72 | 1865.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 1840.20 | 1857.99 | 1862.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 1784.35 | 1770.38 | 1790.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 1733.50 | 1770.38 | 1790.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1723.00 | 1760.91 | 1784.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:45:00 | 1715.60 | 1750.30 | 1777.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:00:00 | 1710.00 | 1727.31 | 1756.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:15:00 | 1710.00 | 1721.94 | 1748.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1784.65 | 1753.42 | 1752.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1784.65 | 1753.42 | 1752.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1847.85 | 1788.37 | 1771.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 1887.05 | 1887.66 | 1860.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 13:00:00 | 1887.05 | 1887.66 | 1860.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1880.25 | 1883.68 | 1863.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 1886.70 | 1866.20 | 1862.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 1890.50 | 1872.12 | 1866.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1961.35 | 1986.57 | 1986.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 1961.35 | 1986.57 | 1986.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 11:15:00 | 1947.50 | 1967.26 | 1975.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 2007.70 | 1973.86 | 1975.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 2007.70 | 1973.86 | 1975.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2007.70 | 1973.86 | 1975.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 2007.70 | 1973.86 | 1975.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1980.20 | 1975.13 | 1975.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 1962.85 | 1975.13 | 1975.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1864.71 | 1892.65 | 1915.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 1766.57 | 1828.71 | 1868.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 166 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 1871.10 | 1838.31 | 1835.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1911.75 | 1864.67 | 1850.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 1865.20 | 1879.88 | 1861.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 13:15:00 | 1865.20 | 1879.88 | 1861.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 1865.20 | 1879.88 | 1861.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:45:00 | 1866.25 | 1879.88 | 1861.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 1862.85 | 1876.47 | 1861.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:30:00 | 1859.10 | 1876.47 | 1861.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 1860.00 | 1873.18 | 1861.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1917.50 | 1873.18 | 1861.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 1949.00 | 1957.01 | 1957.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 1949.00 | 1957.01 | 1957.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1926.90 | 1949.86 | 1953.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 1919.90 | 1916.54 | 1926.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 1919.90 | 1916.54 | 1926.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1919.90 | 1916.54 | 1926.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:30:00 | 1900.10 | 1914.19 | 1924.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 1901.40 | 1911.59 | 1922.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:30:00 | 1897.70 | 1900.58 | 1910.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 1922.70 | 1881.10 | 1878.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 1922.70 | 1881.10 | 1878.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 1980.10 | 1900.90 | 1887.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 2025.30 | 2050.58 | 2006.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 09:15:00 | 2018.20 | 2050.58 | 2006.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2055.00 | 2051.47 | 2010.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:30:00 | 2101.60 | 2063.11 | 2023.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 2101.70 | 2065.20 | 2037.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-21 09:15:00 | 2311.76 | 2247.46 | 2228.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 2420.00 | 2424.63 | 2425.15 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 2436.00 | 2426.90 | 2426.14 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 2406.50 | 2424.60 | 2425.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 2399.80 | 2419.64 | 2423.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 2440.00 | 2424.09 | 2422.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 2465.00 | 2437.72 | 2429.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 2503.80 | 2508.25 | 2478.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:30:00 | 2520.90 | 2508.25 | 2478.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 2482.20 | 2502.18 | 2480.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 2482.20 | 2502.18 | 2480.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 2471.00 | 2495.95 | 2479.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 2471.00 | 2495.95 | 2479.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 2485.00 | 2493.76 | 2480.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 2473.80 | 2493.76 | 2480.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 2476.70 | 2490.35 | 2479.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 2476.70 | 2490.35 | 2479.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 2479.00 | 2488.08 | 2479.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 2512.70 | 2488.08 | 2479.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2485.80 | 2508.12 | 2508.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2485.80 | 2508.12 | 2508.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 2463.90 | 2499.28 | 2504.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 2508.60 | 2435.78 | 2454.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2453.00 | 2439.23 | 2454.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 2431.60 | 2449.05 | 2454.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 2438.70 | 2446.23 | 2452.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 2441.50 | 2443.37 | 2449.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 2437.20 | 2441.93 | 2446.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 2424.00 | 2438.34 | 2444.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 2421.20 | 2438.34 | 2444.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 2423.30 | 2433.05 | 2440.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 2419.30 | 2432.24 | 2439.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 2423.90 | 2432.26 | 2437.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2414.70 | 2428.75 | 2434.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 2408.50 | 2428.75 | 2434.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 2400.10 | 2422.68 | 2430.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2567.00 | 2447.93 | 2440.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 2567.00 | 2447.93 | 2440.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 2615.00 | 2481.34 | 2456.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 12:15:00 | 2546.70 | 2550.00 | 2517.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 13:00:00 | 2546.70 | 2550.00 | 2517.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2520.30 | 2541.37 | 2523.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 2514.80 | 2541.37 | 2523.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2504.10 | 2533.91 | 2521.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 2508.80 | 2533.91 | 2521.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2506.40 | 2528.41 | 2520.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 2528.00 | 2528.41 | 2520.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 2513.80 | 2521.71 | 2519.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 14:15:00 | 2780.80 | 2649.25 | 2592.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 2779.00 | 2810.38 | 2811.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 2730.70 | 2777.66 | 2790.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2650.70 | 2641.32 | 2679.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:00:00 | 2650.70 | 2641.32 | 2679.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2686.20 | 2650.85 | 2669.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 2685.00 | 2650.85 | 2669.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2681.60 | 2657.00 | 2670.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2649.60 | 2657.00 | 2670.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2645.60 | 2632.45 | 2642.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 2635.10 | 2632.45 | 2642.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2637.50 | 2633.46 | 2641.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 2643.90 | 2633.46 | 2641.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 2640.90 | 2634.95 | 2641.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 2640.90 | 2634.95 | 2641.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 2666.30 | 2641.22 | 2643.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 2666.10 | 2641.22 | 2643.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 2679.60 | 2648.89 | 2647.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 2708.70 | 2666.94 | 2656.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 2701.90 | 2702.70 | 2685.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 2693.30 | 2700.03 | 2690.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2693.30 | 2700.03 | 2690.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:15:00 | 2704.90 | 2700.03 | 2690.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2704.90 | 2701.00 | 2692.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2679.10 | 2701.00 | 2692.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2688.40 | 2698.48 | 2691.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 2690.40 | 2698.48 | 2691.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2702.20 | 2699.23 | 2692.69 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 2678.00 | 2691.24 | 2692.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 2670.40 | 2687.07 | 2690.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 2655.00 | 2631.82 | 2643.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2629.70 | 2631.40 | 2642.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:30:00 | 2615.00 | 2629.86 | 2640.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 2619.60 | 2629.86 | 2640.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 2601.70 | 2630.20 | 2634.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 2484.25 | 2524.66 | 2543.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 2488.62 | 2524.66 | 2543.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 2471.61 | 2506.29 | 2531.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2510.00 | 2507.03 | 2529.23 | SL hit (close>ema200) qty=0.50 sl=2507.03 alert=retest2 |

### Cycle 178 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 2645.00 | 2544.41 | 2541.48 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 2544.60 | 2567.48 | 2569.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2512.80 | 2547.16 | 2556.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 2575.60 | 2539.16 | 2535.97 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2505.00 | 2531.04 | 2533.16 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 2542.60 | 2535.41 | 2534.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 2587.80 | 2545.89 | 2539.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 2529.70 | 2553.59 | 2545.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 2530.00 | 2548.87 | 2544.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2598.00 | 2548.87 | 2544.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 10:15:00 | 2857.80 | 2682.24 | 2619.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 2819.30 | 2845.31 | 2845.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 2810.90 | 2833.63 | 2839.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 10:15:00 | 2830.50 | 2829.88 | 2836.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:00:00 | 2830.50 | 2829.88 | 2836.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 2869.90 | 2837.26 | 2839.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 2869.90 | 2837.26 | 2839.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 2853.00 | 2840.41 | 2840.33 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 2787.80 | 2829.89 | 2835.56 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 2905.60 | 2845.37 | 2840.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 2912.90 | 2874.94 | 2859.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 13:15:00 | 2879.00 | 2881.84 | 2867.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:45:00 | 2871.90 | 2881.84 | 2867.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 2874.60 | 2880.40 | 2868.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 2864.20 | 2880.40 | 2868.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 2855.90 | 2875.50 | 2866.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 2913.30 | 2875.50 | 2866.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 2828.10 | 2906.82 | 2895.54 | SL hit (close<static) qty=1.00 sl=2852.20 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 2877.70 | 2887.55 | 2888.47 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 2939.90 | 2898.02 | 2893.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 2952.00 | 2915.52 | 2902.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 2946.60 | 2917.99 | 2909.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 2927.10 | 2920.00 | 2911.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 2930.30 | 2920.00 | 2911.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 2944.20 | 2989.49 | 2995.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 2944.20 | 2989.49 | 2995.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 2923.20 | 2976.23 | 2988.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 2914.70 | 2867.24 | 2882.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2925.00 | 2878.79 | 2885.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:15:00 | 2924.00 | 2878.79 | 2885.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 2918.90 | 2894.05 | 2892.13 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 2875.40 | 2890.87 | 2891.41 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 2909.80 | 2893.41 | 2891.26 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 2880.00 | 2890.50 | 2891.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 2874.00 | 2886.61 | 2889.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 13:15:00 | 2877.10 | 2873.06 | 2880.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 2877.10 | 2873.06 | 2880.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2888.10 | 2871.84 | 2878.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 2915.50 | 2871.84 | 2878.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2898.30 | 2877.14 | 2879.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 2895.00 | 2877.14 | 2879.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 2871.80 | 2876.77 | 2879.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 2876.20 | 2876.77 | 2879.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 2873.10 | 2876.03 | 2878.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 2871.00 | 2876.03 | 2878.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 2856.10 | 2872.05 | 2876.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 2847.30 | 2868.40 | 2874.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 2849.20 | 2865.90 | 2872.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 2704.93 | 2760.99 | 2799.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 2706.74 | 2760.99 | 2799.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 2753.00 | 2743.48 | 2780.28 | SL hit (close>ema200) qty=0.50 sl=2743.48 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 2793.70 | 2754.90 | 2753.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 2820.50 | 2768.02 | 2759.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 12:15:00 | 2942.10 | 2943.60 | 2913.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 12:45:00 | 2941.20 | 2943.60 | 2913.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 2914.00 | 2937.28 | 2915.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 2914.00 | 2937.28 | 2915.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 2921.80 | 2934.18 | 2916.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 2915.00 | 2934.18 | 2916.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2931.00 | 2933.55 | 2917.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 2922.00 | 2933.55 | 2917.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2904.60 | 2927.76 | 2916.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 2901.80 | 2927.76 | 2916.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2920.80 | 2926.37 | 2916.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 2892.40 | 2926.37 | 2916.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2931.80 | 2927.45 | 2918.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 2921.40 | 2927.45 | 2918.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2899.80 | 2922.71 | 2919.04 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 2889.00 | 2915.97 | 2916.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 2877.00 | 2893.03 | 2902.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 2800.10 | 2799.69 | 2824.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 14:00:00 | 2800.10 | 2799.69 | 2824.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2825.20 | 2804.79 | 2824.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 2825.20 | 2804.79 | 2824.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2827.80 | 2809.39 | 2825.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2862.50 | 2809.39 | 2825.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2921.00 | 2831.72 | 2833.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 2921.00 | 2831.72 | 2833.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2909.80 | 2847.33 | 2840.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 2938.30 | 2886.71 | 2862.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 2918.50 | 2919.48 | 2892.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 2918.50 | 2919.48 | 2892.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2904.00 | 2916.55 | 2896.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 2904.00 | 2916.55 | 2896.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2887.10 | 2910.66 | 2895.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2939.70 | 2910.66 | 2895.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 2917.70 | 2943.15 | 2940.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 2910.00 | 2936.52 | 2937.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 2910.00 | 2936.52 | 2937.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 2904.00 | 2930.02 | 2934.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 2938.10 | 2920.74 | 2927.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2936.00 | 2923.79 | 2928.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 2938.10 | 2923.79 | 2928.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 2927.10 | 2924.22 | 2927.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 2929.50 | 2924.22 | 2927.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 2923.30 | 2924.04 | 2927.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 2876.60 | 2916.17 | 2922.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 2884.80 | 2867.64 | 2865.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 2884.80 | 2867.64 | 2865.88 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 2838.20 | 2861.75 | 2863.36 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 2875.40 | 2864.48 | 2864.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 2895.00 | 2870.59 | 2867.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 2855.10 | 2882.46 | 2876.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 2870.30 | 2880.02 | 2875.82 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 2854.50 | 2871.76 | 2872.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 2821.00 | 2858.40 | 2866.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 2850.20 | 2839.11 | 2852.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 2850.20 | 2839.11 | 2852.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 2875.00 | 2846.29 | 2854.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 2875.00 | 2846.29 | 2854.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2875.90 | 2852.21 | 2856.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 2862.10 | 2854.19 | 2857.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 2862.20 | 2832.70 | 2833.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 2876.40 | 2841.44 | 2837.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 2876.40 | 2841.44 | 2837.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 2901.50 | 2860.95 | 2847.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 2709.80 | 2845.54 | 2845.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 2700.70 | 2816.57 | 2832.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 2669.00 | 2766.95 | 2805.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 2696.90 | 2692.76 | 2729.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 09:15:00 | 2708.20 | 2692.76 | 2729.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2711.90 | 2649.31 | 2671.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 2709.80 | 2649.31 | 2671.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2715.60 | 2662.57 | 2675.23 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 2721.00 | 2684.20 | 2683.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 2728.50 | 2693.06 | 2687.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 2736.00 | 2709.00 | 2697.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 2727.20 | 2714.85 | 2702.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 2757.80 | 2732.58 | 2716.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 2693.30 | 2709.87 | 2711.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 2693.30 | 2709.87 | 2711.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2677.40 | 2703.38 | 2708.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2662.80 | 2686.19 | 2696.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 2663.00 | 2681.55 | 2693.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 2661.50 | 2681.10 | 2684.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 2661.70 | 2673.30 | 2680.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 2678.00 | 2672.52 | 2677.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 2678.00 | 2672.52 | 2677.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2670.00 | 2672.02 | 2676.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 2643.60 | 2672.02 | 2676.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2649.10 | 2667.43 | 2674.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 2636.40 | 2663.53 | 2671.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 2638.20 | 2656.04 | 2666.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 2635.10 | 2654.70 | 2664.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 2728.90 | 2666.41 | 2667.71 | SL hit (close>static) qty=1.00 sl=2709.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 2704.80 | 2674.09 | 2671.08 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 2650.00 | 2675.22 | 2675.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 2627.70 | 2661.49 | 2668.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 2669.90 | 2657.13 | 2662.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 2660.00 | 2657.70 | 2662.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 2643.00 | 2657.70 | 2662.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 2644.80 | 2655.12 | 2660.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 2632.90 | 2650.36 | 2655.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 2669.40 | 2657.58 | 2657.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2669.40 | 2657.58 | 2657.00 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 2641.20 | 2654.67 | 2655.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2625.00 | 2647.25 | 2651.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 2628.60 | 2627.58 | 2638.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 2628.60 | 2627.58 | 2638.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2628.50 | 2629.32 | 2637.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2618.30 | 2629.32 | 2637.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2607.50 | 2624.70 | 2633.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 2642.10 | 2594.26 | 2592.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2642.10 | 2594.26 | 2592.91 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 2567.40 | 2597.57 | 2599.83 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 14:15:00 | 2620.70 | 2600.95 | 2600.40 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 2579.00 | 2599.61 | 2600.07 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 2614.00 | 2601.83 | 2600.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 2641.70 | 2613.68 | 2606.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 2656.10 | 2657.24 | 2636.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:30:00 | 2656.70 | 2657.24 | 2636.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2646.00 | 2656.00 | 2645.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2648.80 | 2656.00 | 2645.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2635.00 | 2651.80 | 2644.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 2635.00 | 2651.80 | 2644.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2616.00 | 2644.64 | 2641.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 2616.00 | 2644.64 | 2641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 2615.00 | 2638.71 | 2639.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 2596.90 | 2625.69 | 2633.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 2563.90 | 2563.37 | 2581.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 10:45:00 | 2565.50 | 2563.37 | 2581.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2582.20 | 2565.02 | 2577.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2582.20 | 2565.02 | 2577.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2589.50 | 2569.91 | 2578.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 2589.50 | 2569.91 | 2578.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2582.70 | 2573.05 | 2578.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 2582.00 | 2573.05 | 2578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 2589.10 | 2576.26 | 2579.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 2589.30 | 2576.26 | 2579.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2576.70 | 2576.35 | 2578.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 2562.70 | 2573.58 | 2577.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 2590.00 | 2578.06 | 2576.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2590.00 | 2578.06 | 2576.48 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 2550.00 | 2572.56 | 2574.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 2543.00 | 2560.04 | 2567.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 2569.00 | 2555.55 | 2563.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2561.00 | 2556.64 | 2562.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 2558.40 | 2556.64 | 2562.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2581.30 | 2553.00 | 2557.23 | SL hit (close>static) qty=1.00 sl=2572.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 2568.50 | 2560.90 | 2560.38 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2549.20 | 2559.79 | 2560.04 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 2566.90 | 2560.01 | 2559.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 2583.00 | 2570.90 | 2565.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 2591.40 | 2591.18 | 2580.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2591.80 | 2591.30 | 2581.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 2596.20 | 2591.30 | 2581.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 2594.90 | 2592.02 | 2582.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:45:00 | 2604.30 | 2594.24 | 2584.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 2612.20 | 2598.36 | 2589.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2570.00 | 2599.33 | 2594.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 2570.00 | 2599.33 | 2594.97 | SL hit (close<static) qty=1.00 sl=2571.60 alert=retest2 |

### Cycle 221 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 2554.50 | 2590.36 | 2591.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 09:15:00 | 2547.90 | 2573.06 | 2581.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 2579.00 | 2565.73 | 2573.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 2599.00 | 2572.38 | 2575.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2567.90 | 2572.38 | 2575.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2546.00 | 2567.10 | 2573.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2543.00 | 2562.88 | 2569.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 2523.00 | 2532.21 | 2535.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 14:15:00 | 2415.85 | 2434.58 | 2455.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 15:15:00 | 2396.85 | 2425.38 | 2449.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 2340.30 | 2334.00 | 2359.33 | SL hit (close>ema200) qty=0.50 sl=2334.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2397.00 | 2370.36 | 2369.45 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 2353.10 | 2366.91 | 2367.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 2345.30 | 2362.59 | 2365.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:45:00 | 2366.30 | 2361.67 | 2364.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2368.00 | 2362.94 | 2364.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 2368.00 | 2362.94 | 2364.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2403.80 | 2373.36 | 2369.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 2416.70 | 2382.03 | 2373.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2511.40 | 2405.47 | 2403.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 2451.40 | 2472.60 | 2457.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 2403.40 | 2442.11 | 2447.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2403.40 | 2442.11 | 2447.28 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 2487.30 | 2450.17 | 2446.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2499.40 | 2468.48 | 2456.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 2572.50 | 2582.28 | 2548.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:30:00 | 2557.60 | 2582.28 | 2548.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 2566.60 | 2577.13 | 2556.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 2527.50 | 2577.13 | 2556.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2560.00 | 2573.71 | 2557.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 2560.00 | 2573.71 | 2557.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 2555.00 | 2569.96 | 2556.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 2501.00 | 2569.96 | 2556.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2517.60 | 2559.49 | 2553.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 2497.40 | 2559.49 | 2553.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2530.30 | 2548.55 | 2549.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 2514.70 | 2541.78 | 2546.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 2545.30 | 2521.45 | 2534.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2515.00 | 2499.94 | 2514.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 2513.00 | 2499.94 | 2514.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 2508.20 | 2501.59 | 2513.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 2498.10 | 2517.03 | 2517.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 2504.90 | 2477.48 | 2475.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2504.90 | 2477.48 | 2475.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 2543.20 | 2490.62 | 2481.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2657.00 | 2665.08 | 2628.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:30:00 | 2659.30 | 2665.08 | 2628.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 2639.70 | 2660.01 | 2629.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 2669.30 | 2642.52 | 2634.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2610.00 | 2640.30 | 2635.36 | SL hit (close<static) qty=1.00 sl=2623.50 alert=retest2 |

### Cycle 229 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2554.70 | 2630.35 | 2634.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 2527.60 | 2609.80 | 2624.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2432.20 | 2428.51 | 2460.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 11:15:00 | 2460.10 | 2438.17 | 2459.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 2460.10 | 2438.17 | 2459.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 2460.10 | 2438.17 | 2459.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 2471.90 | 2444.91 | 2460.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 2471.90 | 2444.91 | 2460.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 2488.40 | 2453.61 | 2463.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 2488.40 | 2453.61 | 2463.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 2503.40 | 2469.68 | 2468.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 2532.30 | 2485.01 | 2476.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 2479.00 | 2487.66 | 2479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2479.00 | 2485.93 | 2479.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2422.70 | 2485.93 | 2479.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 2420.00 | 2472.74 | 2474.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2389.00 | 2452.74 | 2463.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 2407.60 | 2401.34 | 2429.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:45:00 | 2408.00 | 2401.34 | 2429.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2425.10 | 2393.86 | 2415.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2425.10 | 2393.86 | 2415.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2403.30 | 2395.75 | 2414.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:30:00 | 2366.40 | 2389.59 | 2404.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2363.20 | 2389.59 | 2404.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:30:00 | 2367.10 | 2357.08 | 2375.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:15:00 | 2365.90 | 2359.67 | 2374.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 2290.00 | 2296.46 | 2315.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2207.20 | 2296.46 | 2315.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2248.08 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2245.04 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2248.74 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2247.61 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 2215.60 | 2213.82 | 2244.40 | SL hit (close>ema200) qty=0.50 sl=2213.82 alert=retest2 |

### Cycle 232 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2341.00 | 2263.04 | 2257.81 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2232.20 | 2281.45 | 2286.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 2173.70 | 2242.53 | 2256.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2231.90 | 2216.21 | 2236.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2231.90 | 2216.21 | 2236.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 2258.40 | 2224.64 | 2238.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 2258.40 | 2224.64 | 2238.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 2235.00 | 2226.72 | 2238.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 2222.00 | 2225.77 | 2236.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2347.30 | 2231.53 | 2221.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 2347.30 | 2231.53 | 2221.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 2380.00 | 2304.97 | 2263.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2348.60 | 2349.68 | 2311.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 2347.60 | 2349.68 | 2311.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2396.20 | 2422.80 | 2385.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 2420.00 | 2404.57 | 2392.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:15:00 | 2418.90 | 2405.91 | 2393.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 2422.00 | 2408.05 | 2399.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 2375.00 | 2392.22 | 2394.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 2375.00 | 2392.22 | 2394.49 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2417.40 | 2397.70 | 2396.61 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 2381.00 | 2394.90 | 2396.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 2356.60 | 2386.61 | 2392.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 2395.00 | 2380.69 | 2387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 2371.00 | 2378.75 | 2385.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:45:00 | 2360.40 | 2375.54 | 2383.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 2387.30 | 2376.36 | 2381.88 | SL hit (close>static) qty=1.00 sl=2386.70 alert=retest2 |

### Cycle 238 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 2395.00 | 2386.61 | 2385.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 15:15:00 | 2400.00 | 2389.29 | 2386.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 2394.40 | 2397.87 | 2392.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 2400.00 | 2398.30 | 2393.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 2445.10 | 2399.16 | 2394.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 2373.90 | 2394.26 | 2396.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 2373.90 | 2394.26 | 2396.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 2342.00 | 2383.81 | 2391.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 2346.00 | 2339.44 | 2358.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 2346.00 | 2339.44 | 2358.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2395.90 | 2350.73 | 2361.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2395.90 | 2350.73 | 2361.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2387.80 | 2358.15 | 2364.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 2380.20 | 2358.15 | 2364.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 2370.00 | 2363.49 | 2362.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2370.00 | 2363.49 | 2362.91 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 2353.60 | 2362.30 | 2362.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 2325.70 | 2354.98 | 2359.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 2342.50 | 2328.28 | 2340.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2358.80 | 2334.66 | 2341.01 | EMA400 retest candle locked (from downside) |

### Cycle 242 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 2362.20 | 2347.23 | 2346.06 | EMA200 above EMA400 |

### Cycle 243 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 2328.00 | 2343.38 | 2344.41 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 2364.50 | 2348.70 | 2346.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 2375.00 | 2353.96 | 2349.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 2331.00 | 2349.37 | 2347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 245 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 2334.10 | 2346.31 | 2346.40 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 2347.90 | 2346.63 | 2346.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 2361.10 | 2349.52 | 2347.86 | Break + close above crossover candle high |

### Cycle 247 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 2328.00 | 2345.22 | 2346.05 | EMA200 below EMA400 |

### Cycle 248 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 2363.90 | 2348.04 | 2347.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2370.20 | 2352.47 | 2349.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 2339.50 | 2349.88 | 2348.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 2349.10 | 2349.72 | 2348.41 | EMA400 retest candle locked (from upside) |

### Cycle 249 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 2334.90 | 2346.76 | 2347.19 | EMA200 below EMA400 |

### Cycle 250 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 2359.50 | 2348.79 | 2348.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 2435.10 | 2367.76 | 2356.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 09:45:00 | 1418.15 | 2023-06-14 09:15:00 | 1559.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-30 11:00:00 | 1419.20 | 2023-06-14 09:15:00 | 1561.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-30 11:45:00 | 1423.30 | 2023-06-14 09:15:00 | 1565.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-30 12:15:00 | 1420.95 | 2023-06-14 09:15:00 | 1563.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-30 15:00:00 | 1427.85 | 2023-06-14 09:15:00 | 1570.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-30 09:15:00 | 1591.80 | 2023-07-07 15:15:00 | 1623.05 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2023-07-03 09:15:00 | 1598.80 | 2023-07-07 15:15:00 | 1623.05 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2023-07-18 09:15:00 | 1705.45 | 2023-07-18 11:15:00 | 1680.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-07-18 09:45:00 | 1706.80 | 2023-07-18 11:15:00 | 1680.95 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-07-20 13:30:00 | 1684.95 | 2023-07-24 09:15:00 | 1698.35 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-07-20 14:00:00 | 1686.80 | 2023-07-24 09:15:00 | 1698.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2023-07-20 14:45:00 | 1685.15 | 2023-07-24 09:15:00 | 1698.35 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-07-21 09:30:00 | 1685.95 | 2023-07-24 09:15:00 | 1698.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-07-26 09:15:00 | 1707.05 | 2023-07-26 09:15:00 | 1686.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2023-08-04 12:30:00 | 1696.00 | 2023-08-04 15:15:00 | 1679.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-08-04 13:15:00 | 1695.95 | 2023-08-04 15:15:00 | 1679.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-08-04 13:45:00 | 1697.95 | 2023-08-04 15:15:00 | 1679.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-08-21 10:45:00 | 1603.85 | 2023-08-21 15:15:00 | 1627.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest1 | 2023-08-23 10:00:00 | 1649.80 | 2023-08-23 15:15:00 | 1627.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2023-08-23 12:15:00 | 1647.20 | 2023-08-23 15:15:00 | 1627.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2023-08-23 13:00:00 | 1645.85 | 2023-08-23 15:15:00 | 1627.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-08-24 09:45:00 | 1639.30 | 2023-08-25 14:15:00 | 1612.05 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-08-24 10:30:00 | 1638.90 | 2023-08-25 14:15:00 | 1612.05 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-09-04 09:15:00 | 1667.60 | 2023-09-04 10:15:00 | 1629.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-09-08 13:15:00 | 1617.00 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2023-09-08 14:15:00 | 1616.00 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2023-09-12 10:00:00 | 1616.50 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2023-09-13 10:15:00 | 1606.55 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2023-09-13 11:15:00 | 1593.05 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-09-14 10:00:00 | 1585.40 | 2023-09-15 12:15:00 | 1607.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-09-21 09:15:00 | 1665.55 | 2023-09-21 15:15:00 | 1607.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2023-09-26 12:45:00 | 1590.35 | 2023-10-03 09:15:00 | 1602.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-09-26 15:15:00 | 1590.10 | 2023-10-03 09:15:00 | 1602.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-09-28 10:45:00 | 1587.80 | 2023-10-03 09:15:00 | 1602.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-09-29 09:15:00 | 1591.00 | 2023-10-03 09:15:00 | 1602.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-10-18 13:00:00 | 1579.25 | 2023-10-18 14:15:00 | 1615.65 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2023-10-18 13:45:00 | 1579.80 | 2023-10-18 14:15:00 | 1615.65 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2023-11-01 10:15:00 | 1582.25 | 2023-11-01 11:15:00 | 1604.85 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-11-16 11:15:00 | 1594.85 | 2023-11-17 09:15:00 | 1623.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-11-16 14:45:00 | 1595.10 | 2023-11-17 09:15:00 | 1623.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2023-11-16 15:15:00 | 1594.00 | 2023-11-17 09:15:00 | 1623.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-11-30 12:45:00 | 1650.05 | 2023-12-07 13:15:00 | 1694.55 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2023-12-14 09:15:00 | 1708.00 | 2023-12-14 13:15:00 | 1690.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-12-18 13:15:00 | 1804.90 | 2023-12-27 11:15:00 | 1780.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-12-20 14:45:00 | 1758.95 | 2023-12-27 11:15:00 | 1780.00 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2023-12-20 15:15:00 | 1767.00 | 2023-12-27 11:15:00 | 1780.00 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-01-02 15:00:00 | 1891.00 | 2024-01-03 09:15:00 | 1873.25 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-01-09 09:15:00 | 2218.00 | 2024-01-10 09:15:00 | 2040.55 | STOP_HIT | 1.00 | -8.00% |
| BUY | retest2 | 2024-01-09 13:30:00 | 2116.40 | 2024-01-10 09:15:00 | 2040.55 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-01-16 13:30:00 | 1955.35 | 2024-01-17 14:15:00 | 2042.65 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2024-01-16 14:00:00 | 1965.00 | 2024-01-17 14:15:00 | 2042.65 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2024-02-15 10:45:00 | 1806.85 | 2024-02-19 11:15:00 | 1833.25 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-02-16 10:00:00 | 1803.40 | 2024-02-19 11:15:00 | 1833.25 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-02-19 09:30:00 | 1800.00 | 2024-02-19 11:15:00 | 1833.25 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-02-26 11:30:00 | 1855.85 | 2024-02-28 11:15:00 | 1824.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-02-27 10:30:00 | 1851.25 | 2024-02-28 11:15:00 | 1824.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-02-27 12:30:00 | 1851.50 | 2024-02-28 11:15:00 | 1824.25 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-03-04 15:15:00 | 1830.00 | 2024-03-07 14:15:00 | 1823.80 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-03-05 09:30:00 | 1825.00 | 2024-03-07 14:15:00 | 1823.80 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-03-21 13:00:00 | 1776.45 | 2024-03-22 09:15:00 | 1788.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-03-21 14:15:00 | 1776.65 | 2024-03-22 09:15:00 | 1788.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-03-21 14:45:00 | 1777.10 | 2024-03-22 09:15:00 | 1788.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-03-26 09:15:00 | 1802.85 | 2024-03-26 11:15:00 | 1776.70 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-03-26 10:15:00 | 1814.95 | 2024-03-26 11:15:00 | 1776.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-03-28 11:15:00 | 1797.25 | 2024-04-12 11:15:00 | 1895.75 | STOP_HIT | 1.00 | 5.48% |
| BUY | retest2 | 2024-03-28 12:15:00 | 1807.80 | 2024-04-12 11:15:00 | 1895.75 | STOP_HIT | 1.00 | 4.87% |
| BUY | retest2 | 2024-04-30 09:15:00 | 2005.80 | 2024-05-02 10:15:00 | 1954.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-04-30 15:15:00 | 1991.00 | 2024-05-02 10:15:00 | 1954.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-05-08 15:15:00 | 1929.90 | 2024-05-09 09:15:00 | 1962.05 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-05-15 09:15:00 | 2048.00 | 2024-05-16 14:15:00 | 2025.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-05-15 10:15:00 | 2028.05 | 2024-05-16 14:15:00 | 2025.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-05-15 11:30:00 | 2031.45 | 2024-05-16 14:15:00 | 2025.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-05-31 09:15:00 | 2202.90 | 2024-06-04 10:15:00 | 2157.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-05-31 10:45:00 | 2207.45 | 2024-06-04 10:15:00 | 2157.40 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-06-04 10:00:00 | 2206.05 | 2024-06-04 10:15:00 | 2157.40 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-18 10:15:00 | 2553.25 | 2024-07-18 11:15:00 | 2647.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-07-24 11:15:00 | 2571.25 | 2024-07-26 10:15:00 | 2594.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-07-25 10:00:00 | 2565.15 | 2024-07-26 10:15:00 | 2594.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest1 | 2024-08-05 09:15:00 | 2461.05 | 2024-08-06 12:15:00 | 2522.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-08-14 09:30:00 | 2499.90 | 2024-08-20 14:15:00 | 2511.10 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-08-16 09:45:00 | 2520.45 | 2024-08-20 14:15:00 | 2511.10 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-08-16 13:00:00 | 2519.20 | 2024-08-20 14:15:00 | 2511.10 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-08-20 13:30:00 | 2518.45 | 2024-08-20 14:15:00 | 2511.10 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-08-30 15:00:00 | 2466.15 | 2024-09-03 12:15:00 | 2509.95 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-09-02 09:30:00 | 2467.15 | 2024-09-03 12:15:00 | 2509.95 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-06 09:15:00 | 2541.85 | 2024-09-09 11:15:00 | 2507.05 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-09 11:15:00 | 2539.05 | 2024-09-09 11:15:00 | 2507.05 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-09-10 10:15:00 | 2496.50 | 2024-09-18 13:15:00 | 2453.45 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-09-10 12:00:00 | 2497.65 | 2024-09-18 13:15:00 | 2453.45 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2024-09-30 12:30:00 | 2375.10 | 2024-10-01 10:15:00 | 2402.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-15 10:30:00 | 2351.65 | 2024-10-15 12:15:00 | 2316.55 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-10-17 12:00:00 | 2386.70 | 2024-10-22 11:15:00 | 2356.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-10-18 09:45:00 | 2386.00 | 2024-10-22 11:15:00 | 2356.65 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-10-18 12:30:00 | 2388.00 | 2024-10-22 11:15:00 | 2356.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-10-21 09:15:00 | 2391.65 | 2024-10-22 11:15:00 | 2356.65 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-10-25 14:00:00 | 2377.95 | 2024-10-28 09:15:00 | 2353.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-10-25 14:45:00 | 2378.00 | 2024-10-28 09:15:00 | 2353.35 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-12-12 09:15:00 | 2333.00 | 2024-12-18 09:15:00 | 2216.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 2333.00 | 2024-12-19 11:15:00 | 2230.35 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2025-01-02 13:30:00 | 2191.15 | 2025-01-06 09:15:00 | 2159.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-01-02 14:45:00 | 2189.95 | 2025-01-06 09:15:00 | 2159.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-01-02 15:15:00 | 2192.20 | 2025-01-06 09:15:00 | 2159.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-01-08 09:15:00 | 2110.80 | 2025-01-13 13:15:00 | 2005.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 2115.80 | 2025-01-13 13:15:00 | 2010.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 2116.00 | 2025-01-13 13:15:00 | 2010.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 2113.15 | 2025-01-13 13:15:00 | 2007.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:45:00 | 2097.85 | 2025-01-13 14:15:00 | 1992.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 2110.80 | 2025-01-14 13:15:00 | 2038.95 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-01-08 10:15:00 | 2115.80 | 2025-01-14 13:15:00 | 2038.95 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-01-08 12:45:00 | 2116.00 | 2025-01-14 13:15:00 | 2038.95 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2025-01-09 12:15:00 | 2113.15 | 2025-01-14 13:15:00 | 2038.95 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-01-09 14:45:00 | 2097.85 | 2025-01-14 13:15:00 | 2038.95 | STOP_HIT | 0.50 | 2.81% |
| BUY | retest2 | 2025-01-22 11:30:00 | 2119.80 | 2025-01-24 09:15:00 | 2079.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-01-23 09:45:00 | 2120.00 | 2025-01-24 09:15:00 | 2079.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-01-23 12:30:00 | 2116.30 | 2025-01-24 09:15:00 | 2079.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-01-30 13:15:00 | 1981.65 | 2025-02-01 09:15:00 | 2006.85 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-01-31 13:45:00 | 1986.40 | 2025-02-01 09:15:00 | 2006.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-31 15:15:00 | 1985.00 | 2025-02-01 09:15:00 | 2006.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest1 | 2025-02-18 09:15:00 | 1794.80 | 2025-02-19 09:15:00 | 1847.85 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-02-21 12:15:00 | 1879.95 | 2025-02-24 12:15:00 | 1853.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-02-21 15:00:00 | 1881.15 | 2025-02-24 12:15:00 | 1853.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-03-03 10:45:00 | 1715.60 | 2025-03-05 09:15:00 | 1784.65 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-03-03 15:00:00 | 1710.00 | 2025-03-05 09:15:00 | 1784.65 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-03-04 10:15:00 | 1710.00 | 2025-03-05 09:15:00 | 1784.65 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-03-12 09:30:00 | 1886.70 | 2025-03-27 11:15:00 | 1961.35 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2025-03-12 13:15:00 | 1890.50 | 2025-03-27 11:15:00 | 1961.35 | STOP_HIT | 1.00 | 3.75% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1962.85 | 2025-04-04 09:15:00 | 1864.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1962.85 | 2025-04-07 09:15:00 | 1766.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-15 09:15:00 | 1917.50 | 2025-04-24 14:15:00 | 1949.00 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2025-04-29 10:30:00 | 1900.10 | 2025-05-07 09:15:00 | 1922.70 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1901.40 | 2025-05-07 09:15:00 | 1922.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-30 10:30:00 | 1897.70 | 2025-05-07 09:15:00 | 1922.70 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-05-09 11:30:00 | 2101.60 | 2025-05-21 09:15:00 | 2311.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 2101.70 | 2025-05-21 09:15:00 | 2311.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 09:15:00 | 2512.70 | 2025-06-12 11:15:00 | 2485.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-17 09:15:00 | 2431.60 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-06-17 10:30:00 | 2438.70 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-06-17 12:30:00 | 2441.50 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2025-06-18 11:15:00 | 2437.20 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-06-18 12:15:00 | 2421.20 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2025-06-18 14:15:00 | 2423.30 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.93% |
| SELL | retest2 | 2025-06-18 15:15:00 | 2419.30 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2025-06-19 12:15:00 | 2423.90 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-06-19 13:15:00 | 2408.50 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.58% |
| SELL | retest2 | 2025-06-19 15:15:00 | 2400.10 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.95% |
| BUY | retest2 | 2025-06-24 12:15:00 | 2528.00 | 2025-06-25 14:15:00 | 2780.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 15:00:00 | 2513.80 | 2025-06-25 14:15:00 | 2765.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-23 11:30:00 | 2615.00 | 2025-08-01 13:15:00 | 2484.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 12:15:00 | 2619.60 | 2025-08-01 13:15:00 | 2488.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 2601.70 | 2025-08-01 15:15:00 | 2471.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 11:30:00 | 2615.00 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-07-23 12:15:00 | 2619.60 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-07-25 09:30:00 | 2601.70 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2598.00 | 2025-08-18 10:15:00 | 2857.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 2913.30 | 2025-09-02 09:15:00 | 2828.10 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-09-02 09:30:00 | 2883.70 | 2025-09-02 10:15:00 | 2845.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-02 12:00:00 | 2885.20 | 2025-09-02 14:15:00 | 2877.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-02 12:45:00 | 2882.00 | 2025-09-02 14:15:00 | 2877.70 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-09-04 09:15:00 | 2946.60 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-09-04 10:45:00 | 2927.10 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-09-04 11:15:00 | 2930.30 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2847.30 | 2025-09-25 14:15:00 | 2704.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 2849.20 | 2025-09-25 14:15:00 | 2706.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2847.30 | 2025-09-26 10:15:00 | 2753.00 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-09-24 10:15:00 | 2849.20 | 2025-09-26 10:15:00 | 2753.00 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2939.70 | 2025-10-24 11:15:00 | 2910.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-24 10:30:00 | 2917.70 | 2025-10-24 11:15:00 | 2910.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-10-29 09:45:00 | 2876.60 | 2025-11-03 15:15:00 | 2884.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-07 15:00:00 | 2862.10 | 2025-11-12 11:15:00 | 2876.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-12 10:45:00 | 2862.20 | 2025-11-12 11:15:00 | 2876.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-11-20 11:00:00 | 2736.00 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-20 12:30:00 | 2727.20 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-21 09:30:00 | 2757.80 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2662.80 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-11-25 10:00:00 | 2663.00 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-27 09:15:00 | 2661.50 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-11-27 11:00:00 | 2661.70 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-11-28 11:15:00 | 2636.40 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-11-28 12:30:00 | 2638.20 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-11-28 15:15:00 | 2635.10 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-12-05 09:15:00 | 2632.90 | 2025-12-05 13:15:00 | 2669.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2618.30 | 2025-12-12 14:15:00 | 2642.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2607.50 | 2025-12-12 14:15:00 | 2642.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-30 15:00:00 | 2562.70 | 2025-12-31 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-02 11:15:00 | 2558.40 | 2026-01-05 09:15:00 | 2581.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-08 12:15:00 | 2596.20 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-08 13:00:00 | 2594.90 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-01-08 13:45:00 | 2604.30 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-09 09:45:00 | 2612.20 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2543.00 | 2026-01-22 14:15:00 | 2415.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2523.00 | 2026-01-22 15:15:00 | 2396.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2543.00 | 2026-01-28 10:15:00 | 2340.30 | STOP_HIT | 0.50 | 7.97% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2523.00 | 2026-01-28 10:15:00 | 2340.30 | STOP_HIT | 0.50 | 7.24% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2511.40 | 2026-02-05 10:15:00 | 2403.40 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-02-04 12:30:00 | 2451.40 | 2026-02-05 10:15:00 | 2403.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2498.10 | 2026-02-23 10:15:00 | 2504.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-27 15:15:00 | 2669.30 | 2026-03-02 09:15:00 | 2610.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-02 15:00:00 | 2666.10 | 2026-03-04 09:15:00 | 2554.70 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-17 10:30:00 | 2366.40 | 2026-03-23 09:15:00 | 2248.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2363.20 | 2026-03-23 09:15:00 | 2245.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:30:00 | 2367.10 | 2026-03-23 09:15:00 | 2248.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 13:15:00 | 2365.90 | 2026-03-23 09:15:00 | 2247.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:30:00 | 2366.40 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.37% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2363.20 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.25% |
| SELL | retest2 | 2026-03-18 11:30:00 | 2367.10 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.40% |
| SELL | retest2 | 2026-03-18 13:15:00 | 2365.90 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.35% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2207.20 | 2026-03-25 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2026-04-06 10:00:00 | 2222.00 | 2026-04-08 09:15:00 | 2347.30 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2026-04-15 09:30:00 | 2420.00 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-04-15 11:15:00 | 2418.90 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-16 09:15:00 | 2422.00 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-20 14:45:00 | 2360.40 | 2026-04-21 10:15:00 | 2387.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-23 09:15:00 | 2445.10 | 2026-04-24 09:15:00 | 2373.90 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-04-27 14:15:00 | 2380.20 | 2026-04-29 10:15:00 | 2370.00 | STOP_HIT | 1.00 | 0.43% |
