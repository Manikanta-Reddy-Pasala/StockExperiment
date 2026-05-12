# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 9 |
| TARGET_HIT | 20 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 20
- **Target hits / Stop hits / Partials:** 20 / 20 / 9
- **Avg / median % per leg:** 4.16% / 5.00%
- **Sum % (uncompounded):** 203.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 11 | 52.4% | 11 | 10 | 0 | 4.38% | 92.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 11 | 52.4% | 11 | 10 | 0 | 4.38% | 92.1% |
| SELL (all) | 28 | 18 | 64.3% | 9 | 10 | 9 | 3.99% | 111.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 18 | 64.3% | 9 | 10 | 9 | 3.99% | 111.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 29 | 59.2% | 20 | 20 | 9 | 4.16% | 203.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 1517.50 | 1438.37 | 1438.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 1521.70 | 1439.20 | 1438.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1555.00 | 1555.91 | 1513.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 1555.00 | 1555.91 | 1513.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1762.70 | 1818.67 | 1766.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 1762.70 | 1818.67 | 1766.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1772.50 | 1818.21 | 1766.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1773.85 | 1817.77 | 1766.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 1750.00 | 1815.97 | 1766.38 | SL hit (close<static) qty=1.00 sl=1756.55 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-27 13:15:00)

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

### Cycle 3 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1650.80 | 1606.38 | 1606.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1668.90 | 1622.83 | 1615.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.70 | 1692.97 | 1665.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 1691.70 | 1692.97 | 1665.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1658.40 | 1692.30 | 1667.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1658.40 | 1691.96 | 1667.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 1663.50 | 1691.29 | 1667.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 1664.60 | 1690.98 | 1667.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 1647.10 | 1689.98 | 1667.05 | SL hit (close<static) qty=1.00 sl=1649.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.50 | 1646.92 | 1648.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.10 | 1476.51 | 1514.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 15:00:00 | 1483.10 | 1476.51 | 1514.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1477.40 | 1443.05 | 1478.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1481.00 | 1443.42 | 1478.65 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.73 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.17 | 1495.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 1501.60 | 1501.08 | 1498.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1508.90 | 1501.16 | 1498.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 1510.40 | 1501.16 | 1498.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 1510.10 | 1501.31 | 1498.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 1511.10 | 1501.31 | 1498.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 1510.70 | 1501.40 | 1498.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-19 11:15:00 | 1661.44 | 1536.80 | 1518.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-02-13 14:15:00)

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
| BUY | retest2 | 2025-07-10 13:15:00 | 1663.50 | 2025-07-11 09:15:00 | 1647.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-10 14:15:00 | 1664.60 | 2025-07-11 09:15:00 | 1647.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-07 11:15:00 | 1510.40 | 2025-11-19 11:15:00 | 1661.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 12:30:00 | 1510.10 | 2025-11-19 11:15:00 | 1661.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 13:15:00 | 1511.10 | 2025-11-19 12:15:00 | 1662.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 14:00:00 | 1510.70 | 2025-11-19 12:15:00 | 1661.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 12:15:00 | 1608.90 | 2026-02-03 09:15:00 | 1769.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 13:15:00 | 1608.30 | 2026-02-03 09:15:00 | 1769.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 14:45:00 | 1610.10 | 2026-02-03 09:15:00 | 1771.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1611.90 | 2026-02-03 09:15:00 | 1773.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1636.60 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1642.00 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-05 09:45:00 | 1625.10 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -1.88% |
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
