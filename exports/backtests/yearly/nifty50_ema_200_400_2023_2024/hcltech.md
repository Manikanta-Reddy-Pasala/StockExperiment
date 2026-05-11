# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2022-04-07 09:15:00 → 2026-05-08 15:15:00 (7054 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 9 |
| TARGET_HIT | 24 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 21
- **Target hits / Stop hits / Partials:** 24 / 21 / 9
- **Avg / median % per leg:** 4.20% / 5.00%
- **Sum % (uncompounded):** 226.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 15 | 60.0% | 15 | 10 | 0 | 4.63% | 115.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 15 | 60.0% | 15 | 10 | 0 | 4.63% | 115.9% |
| SELL (all) | 29 | 18 | 62.1% | 9 | 11 | 9 | 3.82% | 110.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 18 | 62.1% | 9 | 11 | 9 | 3.82% | 110.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 33 | 61.1% | 24 | 21 | 9 | 4.20% | 226.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 1130.00 | 1081.60 | 1081.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 1132.00 | 1087.37 | 1084.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 14:15:00 | 1108.70 | 1110.89 | 1099.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 15:00:00 | 1108.70 | 1110.89 | 1099.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 1132.25 | 1153.07 | 1131.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 10:00:00 | 1165.20 | 1144.48 | 1130.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 11:15:00 | 1166.85 | 1145.59 | 1131.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 12:30:00 | 1162.65 | 1145.89 | 1131.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 13:30:00 | 1163.60 | 1146.11 | 1131.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 1134.30 | 1148.04 | 1134.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-21 11:15:00 | 1118.85 | 1147.52 | 1133.99 | SL hit (close<static) qty=1.00 sl=1122.45 alert=retest2 |

### Cycle 2 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 1467.10 | 1560.06 | 1560.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 1447.00 | 1558.02 | 1559.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1373.35 | 1368.69 | 1420.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:45:00 | 1373.95 | 1368.69 | 1420.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1420.25 | 1370.42 | 1419.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:45:00 | 1423.80 | 1370.42 | 1419.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 1423.85 | 1370.95 | 1419.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:00:00 | 1423.85 | 1370.95 | 1419.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 1429.35 | 1371.53 | 1419.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:45:00 | 1429.95 | 1371.53 | 1419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 1427.65 | 1372.09 | 1419.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:45:00 | 1430.35 | 1372.09 | 1419.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 1416.05 | 1374.78 | 1419.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:30:00 | 1415.75 | 1374.78 | 1419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 1418.60 | 1375.22 | 1419.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:45:00 | 1419.10 | 1375.22 | 1419.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 1419.70 | 1375.66 | 1419.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:15:00 | 1413.85 | 1375.66 | 1419.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1419.35 | 1376.09 | 1419.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 1419.35 | 1376.09 | 1419.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1418.35 | 1376.51 | 1419.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1423.00 | 1376.51 | 1419.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1421.05 | 1376.96 | 1419.76 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 1517.50 | 1438.37 | 1438.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 1521.70 | 1439.20 | 1438.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1555.00 | 1555.91 | 1513.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 1555.00 | 1555.91 | 1513.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1762.70 | 1818.67 | 1766.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 1762.70 | 1818.67 | 1766.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1772.50 | 1818.21 | 1766.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1773.85 | 1817.77 | 1766.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 1750.00 | 1815.97 | 1766.38 | SL hit (close<static) qty=1.00 sl=1756.55 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 1723.85 | 1862.17 | 1862.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 1710.95 | 1860.66 | 1861.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 09:15:00 | 1643.55 | 1614.37 | 1680.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:45:00 | 1654.05 | 1614.37 | 1680.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1569.00 | 1510.51 | 1587.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:45:00 | 1577.50 | 1510.51 | 1587.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1599.00 | 1511.39 | 1587.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 1599.00 | 1511.39 | 1587.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1593.50 | 1512.21 | 1587.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 1586.60 | 1515.45 | 1587.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 12:00:00 | 1586.60 | 1517.61 | 1587.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:45:00 | 1583.20 | 1520.93 | 1587.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:00:00 | 1588.70 | 1534.91 | 1584.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1577.70 | 1535.33 | 1584.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:00:00 | 1574.30 | 1535.72 | 1584.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 1574.10 | 1536.11 | 1584.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:15:00 | 1574.00 | 1536.11 | 1584.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 14:45:00 | 1572.70 | 1536.79 | 1584.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1581.90 | 1539.26 | 1583.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 1581.90 | 1539.26 | 1583.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1576.90 | 1541.62 | 1583.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 1581.40 | 1541.62 | 1583.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 1597.70 | 1542.50 | 1583.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 1597.70 | 1542.50 | 1583.05 | SL hit (close>static) qty=1.00 sl=1589.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1650.80 | 1606.38 | 1606.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1668.90 | 1622.83 | 1615.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.70 | 1692.97 | 1665.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.37 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.50 | 1646.92 | 1648.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.10 | 1476.51 | 1514.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.73 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.17 | 1495.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1651.20 | 1639.61 | 1601.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 1488.30 | 1639.35 | 1632.78 | SL hit (close<static) qty=1.00 sl=1490.50 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:45:00 | 1395.90 | 1395.18 | 1457.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 1432.60 | 1399.27 | 1454.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 1465.70 | 1405.31 | 1454.03 | SL hit (close>static) qty=1.00 sl=1464.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 10:30:00 | 1069.10 | 2023-05-18 11:15:00 | 1079.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-07-17 10:00:00 | 1165.20 | 2023-07-21 11:15:00 | 1118.85 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2023-07-18 11:15:00 | 1166.85 | 2023-07-21 11:15:00 | 1118.85 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2023-07-18 12:30:00 | 1162.65 | 2023-07-21 11:15:00 | 1118.85 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2023-07-18 13:30:00 | 1163.60 | 2023-07-21 11:15:00 | 1118.85 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2023-08-04 09:30:00 | 1144.85 | 2023-09-07 11:15:00 | 1255.48 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2023-08-08 14:45:00 | 1142.00 | 2023-09-07 11:15:00 | 1254.00 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2023-08-09 14:15:00 | 1141.35 | 2023-09-07 14:15:00 | 1256.20 | TARGET_HIT | 1.00 | 10.06% |
| BUY | retest2 | 2023-08-10 13:00:00 | 1140.00 | 2023-09-08 09:15:00 | 1259.34 | TARGET_HIT | 1.00 | 10.47% |
| BUY | retest2 | 2023-08-28 11:15:00 | 1150.85 | 2023-09-08 09:15:00 | 1265.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 12:45:00 | 1149.75 | 2023-09-08 09:15:00 | 1264.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 14:15:00 | 1149.85 | 2023-09-08 09:15:00 | 1264.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 15:15:00 | 1151.00 | 2023-09-08 09:15:00 | 1266.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-16 09:15:00 | 1267.25 | 2023-12-14 09:15:00 | 1393.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-19 11:00:00 | 1268.95 | 2023-12-14 09:15:00 | 1395.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 13:30:00 | 1266.60 | 2023-12-14 09:15:00 | 1393.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 14:30:00 | 1267.30 | 2023-12-14 09:15:00 | 1394.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1773.85 | 2024-11-04 10:15:00 | 1750.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-11-05 09:15:00 | 1776.00 | 2024-12-13 12:15:00 | 1953.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-05 10:30:00 | 1774.65 | 2024-12-13 12:15:00 | 1952.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-05 12:30:00 | 1775.40 | 2024-12-13 12:15:00 | 1952.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-14 11:30:00 | 1829.95 | 2025-01-16 10:15:00 | 1796.05 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-01-15 09:15:00 | 1834.85 | 2025-01-16 10:15:00 | 1796.05 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-01-15 12:15:00 | 1825.05 | 2025-01-16 10:15:00 | 1796.05 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-15 12:45:00 | 1826.30 | 2025-01-16 10:15:00 | 1796.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-04-24 09:15:00 | 1586.60 | 2025-05-08 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-04-24 12:00:00 | 1586.60 | 2025-05-08 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1583.20 | 2025-05-08 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-05 10:00:00 | 1588.70 | 2025-05-08 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-05-05 12:00:00 | 1574.30 | 2025-05-12 09:15:00 | 1627.50 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-05-05 12:30:00 | 1574.10 | 2025-05-12 09:15:00 | 1627.50 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-05-05 13:15:00 | 1574.00 | 2025-05-12 09:15:00 | 1627.50 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-05-05 14:45:00 | 1572.70 | 2025-05-12 09:15:00 | 1627.50 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-05-09 09:15:00 | 1558.90 | 2025-05-12 09:15:00 | 1627.50 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1651.20 | 2026-02-12 13:15:00 | 1488.30 | STOP_HIT | 1.00 | -9.87% |
| SELL | retest2 | 2026-04-08 10:15:00 | 1432.60 | 2026-04-09 14:15:00 | 1465.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 09:15:00 | 1358.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 09:15:00 | 1362.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 09:15:00 | 1350.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 09:15:00 | 1368.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 09:15:00 | 1369.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 09:15:00 | 1371.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 09:15:00 | 1364.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-22 09:15:00 | 1353.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 10:15:00 | 1299.24 | TARGET_HIT | 0.50 | 9.14% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 11:15:00 | 1290.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 11:15:00 | 1296.09 | TARGET_HIT | 0.50 | 8.84% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 11:15:00 | 1297.44 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 11:15:00 | 1293.03 | TARGET_HIT | 0.50 | 10.31% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 12:15:00 | 1287.00 | TARGET_HIT | 0.50 | 10.85% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 12:15:00 | 1281.96 | TARGET_HIT | 0.50 | 10.77% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-23 09:15:00 | 1279.62 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-04-24 09:15:00 | 1242.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-05-08 09:15:00 | 1177.20 | TARGET_HIT | 0.50 | 10.00% |
