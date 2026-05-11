# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1659.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 20 / 28
- **Target hits / Stop hits / Partials:** 5 / 31 / 12
- **Avg / median % per leg:** 1.53% / -0.75%
- **Sum % (uncompounded):** 73.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 2 | 6.9% | 2 | 27 | 0 | -0.57% | -16.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 2 | 6.9% | 2 | 27 | 0 | -0.57% | -16.5% |
| SELL (all) | 19 | 18 | 94.7% | 3 | 4 | 12 | 4.74% | 90.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 18 | 94.7% | 3 | 4 | 12 | 4.74% | 90.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 20 | 41.7% | 5 | 31 | 12 | 1.53% | 73.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 897.00 | 941.23 | 941.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 892.30 | 940.74 | 941.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 930.25 | 912.54 | 925.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 930.25 | 912.54 | 925.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 930.25 | 912.54 | 925.26 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 1006.35 | 929.83 | 929.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 14:15:00 | 1028.35 | 938.83 | 934.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 10:15:00 | 989.90 | 992.16 | 968.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 14:15:00 | 972.55 | 993.12 | 972.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 972.55 | 993.12 | 972.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1325.55 | 1257.31 | 1213.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-18 09:15:00 | 1458.11 | 1279.99 | 1229.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 1705.00 | 1843.06 | 1843.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 1699.65 | 1840.27 | 1842.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 1627.65 | 1622.24 | 1686.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 13:00:00 | 1627.65 | 1622.24 | 1686.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1629.60 | 1566.18 | 1625.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 1629.60 | 1566.18 | 1625.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 1644.25 | 1566.96 | 1625.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:00:00 | 1644.25 | 1566.96 | 1625.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 1697.25 | 1568.26 | 1625.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:30:00 | 1701.50 | 1568.26 | 1625.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1655.30 | 1569.12 | 1625.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 1565.00 | 1569.72 | 1625.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 1486.75 | 1566.01 | 1618.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 11:15:00 | 1408.50 | 1550.66 | 1605.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1622.00 | 1427.11 | 1426.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1633.40 | 1440.20 | 1433.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 1512.10 | 1519.08 | 1483.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:45:00 | 1514.00 | 1519.08 | 1483.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1484.80 | 1518.22 | 1484.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 1485.60 | 1518.22 | 1484.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1499.10 | 1518.03 | 1484.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:45:00 | 1500.50 | 1517.61 | 1484.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1502.60 | 1517.25 | 1484.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 1480.10 | 1516.42 | 1484.81 | SL hit (close<static) qty=1.00 sl=1484.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 1880.30 | 1917.33 | 1917.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1863.00 | 1912.87 | 1915.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1835.60 | 1808.11 | 1851.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 1835.60 | 1808.11 | 1851.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1832.90 | 1806.97 | 1845.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 1846.40 | 1806.97 | 1845.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1776.00 | 1745.99 | 1798.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1819.70 | 1745.99 | 1798.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1827.70 | 1746.81 | 1798.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1827.70 | 1746.81 | 1798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1813.20 | 1747.47 | 1798.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 1802.70 | 1748.60 | 1798.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1851.40 | 1753.22 | 1799.19 | SL hit (close>static) qty=1.00 sl=1829.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 1325.55 | 2024-04-18 09:15:00 | 1458.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1565.00 | 2025-02-10 13:15:00 | 1486.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1565.00 | 2025-02-13 11:15:00 | 1408.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-16 13:45:00 | 1500.50 | 2025-06-17 12:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-17 09:15:00 | 1502.60 | 2025-06-17 12:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-19 09:30:00 | 1505.00 | 2025-06-19 15:15:00 | 1478.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-06-20 11:00:00 | 1501.10 | 2025-06-23 09:15:00 | 1482.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1488.50 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-24 11:45:00 | 1486.20 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1498.00 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1488.20 | 2025-07-04 09:15:00 | 1637.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 12:15:00 | 1910.00 | 2025-10-24 09:15:00 | 1880.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-27 09:45:00 | 1905.40 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-27 11:15:00 | 1904.60 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1908.20 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-31 09:30:00 | 1931.20 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-11-06 13:00:00 | 1926.30 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-10 09:30:00 | 1930.50 | 2025-11-21 14:15:00 | 1888.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-11 13:15:00 | 1930.00 | 2025-11-21 14:15:00 | 1888.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-11-28 10:00:00 | 1945.20 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-28 11:00:00 | 1937.40 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-28 15:00:00 | 1941.70 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-12-01 11:30:00 | 1937.00 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-12 15:00:00 | 1923.70 | 2025-12-15 09:15:00 | 1908.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1926.00 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-16 11:00:00 | 1922.30 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-16 11:45:00 | 1923.00 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-18 13:00:00 | 1923.60 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-19 09:30:00 | 1923.90 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-29 15:00:00 | 1925.00 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1919.90 | 2026-01-06 12:15:00 | 1893.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1802.70 | 2026-02-26 11:15:00 | 1851.40 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1783.20 | 2026-03-04 12:15:00 | 1694.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1783.20 | 2026-03-06 09:15:00 | 1763.90 | STOP_HIT | 0.50 | 1.08% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1802.00 | 2026-03-16 15:15:00 | 1711.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 1800.00 | 2026-03-16 15:15:00 | 1710.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:15:00 | 1760.30 | 2026-03-19 09:15:00 | 1672.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1748.10 | 2026-03-19 09:15:00 | 1660.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1802.00 | 2026-03-19 15:15:00 | 1621.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 1800.00 | 2026-03-19 15:15:00 | 1620.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-13 13:15:00 | 1760.30 | 2026-03-20 15:15:00 | 1739.00 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1748.10 | 2026-03-20 15:15:00 | 1739.00 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1762.00 | 2026-04-28 11:15:00 | 1673.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 10:30:00 | 1756.00 | 2026-04-28 15:15:00 | 1668.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 09:15:00 | 1726.20 | 2026-04-29 14:15:00 | 1646.26 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2026-04-20 11:45:00 | 1727.10 | 2026-04-30 09:15:00 | 1643.88 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-04-21 12:30:00 | 1732.90 | 2026-04-30 10:15:00 | 1639.89 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1730.40 | 2026-04-30 10:15:00 | 1640.74 | PARTIAL | 0.50 | 5.18% |
