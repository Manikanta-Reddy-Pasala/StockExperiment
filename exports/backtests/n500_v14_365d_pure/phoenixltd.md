# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 24 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 17
- **Target hits / Stop hits / Partials:** 7 / 20 / 3
- **Avg / median % per leg:** 1.63% / -1.17%
- **Sum % (uncompounded):** 48.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 7 | 4 | 0 | 5.70% | 62.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 7 | 63.6% | 7 | 4 | 0 | 5.70% | 62.7% |
| SELL (all) | 19 | 6 | 31.6% | 0 | 16 | 3 | -0.73% | -13.9% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.38% | 20.3% |
| SELL @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.63% | -34.2% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.38% | 20.3% |
| retest2 (combined) | 24 | 7 | 29.2% | 7 | 17 | 0 | 1.19% | 28.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 1644.70 | 1590.34 | 1590.23 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 1538.00 | 1592.38 | 1592.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 1499.50 | 1591.45 | 1591.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 10:15:00 | 1528.00 | 1524.00 | 1550.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1508.00 | 1524.39 | 1549.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 13:45:00 | 1513.30 | 1524.02 | 1548.76 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 14:45:00 | 1511.20 | 1523.88 | 1548.57 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:15:00 | 1437.63 | 1507.18 | 1534.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:15:00 | 1435.64 | 1507.18 | 1534.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 1432.60 | 1503.44 | 1531.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1484.10 | 1481.15 | 1513.94 | SL hit (close>ema200) qty=0.50 sl=1481.15 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1484.10 | 1481.15 | 1513.94 | SL hit (close>ema200) qty=0.50 sl=1481.15 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1484.10 | 1481.15 | 1513.94 | SL hit (close>ema200) qty=0.50 sl=1481.15 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1532.30 | 1483.29 | 1512.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 1532.30 | 1483.29 | 1512.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1526.80 | 1483.73 | 1513.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 1518.90 | 1509.74 | 1522.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1566.90 | 1510.12 | 1521.44 | SL hit (close>static) qty=1.00 sl=1537.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 1517.30 | 1513.93 | 1522.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:00:00 | 1515.00 | 1513.96 | 1522.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 1516.30 | 1512.85 | 1521.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1527.80 | 1513.00 | 1521.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 1523.00 | 1513.22 | 1521.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1511.30 | 1513.83 | 1521.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1540.80 | 1514.31 | 1521.92 | SL hit (close>static) qty=1.00 sl=1537.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1540.80 | 1514.31 | 1521.92 | SL hit (close>static) qty=1.00 sl=1537.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1540.80 | 1514.31 | 1521.92 | SL hit (close>static) qty=1.00 sl=1537.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1540.80 | 1514.31 | 1521.92 | SL hit (close>static) qty=1.00 sl=1535.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1540.80 | 1514.31 | 1521.92 | SL hit (close>static) qty=1.00 sl=1535.30 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1602.60 | 1528.51 | 1528.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1616.10 | 1532.31 | 1530.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1558.40 | 1563.55 | 1548.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 1558.40 | 1563.55 | 1548.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1555.00 | 1563.39 | 1548.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 11:45:00 | 1567.20 | 1563.41 | 1548.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 12:45:00 | 1567.00 | 1563.41 | 1548.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 1579.60 | 1562.90 | 1548.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:45:00 | 1570.40 | 1562.99 | 1548.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1555.70 | 1562.99 | 1549.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:30:00 | 1550.30 | 1562.99 | 1549.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1549.60 | 1562.85 | 1549.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1549.60 | 1562.85 | 1549.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1551.00 | 1562.73 | 1549.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1541.90 | 1562.73 | 1549.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1547.40 | 1562.58 | 1549.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:30:00 | 1562.50 | 1562.11 | 1549.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 1565.90 | 1562.04 | 1549.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1562.50 | 1562.05 | 1549.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 1563.10 | 1562.10 | 1549.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1551.70 | 1562.21 | 1549.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | SL hit (close<static) qty=1.00 sl=1544.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | SL hit (close<static) qty=1.00 sl=1544.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | SL hit (close<static) qty=1.00 sl=1544.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | SL hit (close<static) qty=1.00 sl=1544.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 1573.10 | 1561.74 | 1549.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 1567.00 | 1561.74 | 1550.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:45:00 | 1570.90 | 1561.89 | 1550.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 1718.75 | 1622.82 | 1590.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 1722.49 | 1622.82 | 1590.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 1718.75 | 1622.82 | 1590.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 1719.41 | 1622.82 | 1590.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-11-03 09:15:00 | 1723.70 | 1641.23 | 1604.55 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-11-03 10:15:00 | 1730.41 | 1642.06 | 1605.15 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-11-03 10:15:00 | 1727.99 | 1642.06 | 1605.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 1748.70 | 1758.36 | 1758.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 1769.20 | 1758.45 | 1758.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 1800.20 | 1758.97 | 1758.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1720.40 | 1758.39 | 1758.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1720.10 | 1756.03 | 1757.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1599.30 | 1587.42 | 1641.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1709.20 | 1588.76 | 1641.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1709.20 | 1588.76 | 1641.84 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1812.40 | 1679.02 | 1678.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1834.10 | 1709.11 | 1694.94 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 13:45:00 | 1587.40 | 2025-06-03 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-22 14:15:00 | 1582.80 | 2025-06-03 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1587.30 | 2025-06-06 15:15:00 | 1609.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-28 15:15:00 | 1584.50 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-05-30 09:15:00 | 1564.50 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2025-06-03 11:00:00 | 1564.00 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-06-05 14:45:00 | 1565.10 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest1 | 2025-07-28 09:15:00 | 1508.00 | 2025-08-06 10:15:00 | 1437.63 | PARTIAL | 0.50 | 4.67% |
| SELL | retest1 | 2025-07-28 13:45:00 | 1513.30 | 2025-08-06 10:15:00 | 1435.64 | PARTIAL | 0.50 | 5.13% |
| SELL | retest1 | 2025-07-28 14:45:00 | 1511.20 | 2025-08-07 09:15:00 | 1432.60 | PARTIAL | 0.50 | 5.20% |
| SELL | retest1 | 2025-07-28 09:15:00 | 1508.00 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2025-07-28 13:45:00 | 1513.30 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-07-28 14:45:00 | 1511.20 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2025-08-28 12:45:00 | 1518.90 | 2025-09-02 09:15:00 | 1566.90 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-03 12:45:00 | 1517.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-04 10:00:00 | 1515.00 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1516.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-08 11:45:00 | 1523.00 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1511.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-26 11:45:00 | 1567.20 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-26 12:45:00 | 1567.00 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-29 09:15:00 | 1579.60 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-29 09:45:00 | 1570.40 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-30 14:30:00 | 1562.50 | 2025-10-28 09:15:00 | 1718.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 09:30:00 | 1565.90 | 2025-10-28 09:15:00 | 1722.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 11:00:00 | 1562.50 | 2025-10-28 09:15:00 | 1718.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 11:45:00 | 1563.10 | 2025-10-28 09:15:00 | 1719.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 14:30:00 | 1573.10 | 2025-11-03 09:15:00 | 1723.70 | TARGET_HIT | 1.00 | 9.57% |
| BUY | retest2 | 2025-10-06 11:15:00 | 1567.00 | 2025-11-03 10:15:00 | 1730.41 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2025-10-06 12:45:00 | 1570.90 | 2025-11-03 10:15:00 | 1727.99 | TARGET_HIT | 1.00 | 10.00% |
