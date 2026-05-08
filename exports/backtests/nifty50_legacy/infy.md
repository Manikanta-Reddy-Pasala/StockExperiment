# INFY (INFY)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1179.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 13 |
| PENDING | 38 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 25 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 3 / 29
- **Target hits / Stop hits / Partials:** 0 / 31 / 1
- **Avg / median % per leg:** -2.12% / -2.56%
- **Sum % (uncompounded):** -67.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 14 | 1 | -1.23% | -18.5% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -4.02% | -20.1% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | 0.16% | 1.6% |
| SELL (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.91% | -49.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.05% | -6.0% |
| SELL @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.72% | -43.5% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -4.36% | -26.2% |
| retest2 (combined) | 26 | 2 | 7.7% | 0 | 25 | 1 | -1.61% | -41.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 1369.15 | 1420.37 | 1420.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 1362.35 | 1410.58 | 1415.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 14:15:00 | 1412.15 | 1406.07 | 1412.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 15:15:00 | 1412.00 | 1406.13 | 1412.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 1412.00 | 1406.13 | 1412.40 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 1458.45 | 1417.79 | 1417.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1463.90 | 1428.13 | 1423.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 10:15:00 | 1439.35 | 1445.20 | 1433.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 1440.00 | 1445.06 | 1433.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 1440.00 | 1445.06 | 1433.90 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-13 13:15:00 | 1443.20 | 1445.04 | 1433.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 14:15:00 | 1449.10 | 1445.08 | 1434.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:15:00 | 1666.46 | 1555.95 | 1514.64 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-22 09:15:00 | 1647.50 | 1650.59 | 1599.80 | SL hit (close<ema200) qty=0.50 sl=1650.59 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 1505.90 | 1597.96 | 1598.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 1490.25 | 1594.19 | 1596.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 1453.75 | 1452.22 | 1489.07 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-29 14:15:00 | 1450.45 | 1456.52 | 1485.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-29 15:15:00 | 1452.95 | 1456.49 | 1484.91 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-30 09:15:00 | 1435.25 | 1456.28 | 1484.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 1435.05 | 1456.07 | 1484.41 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1471.65 | 1445.30 | 1473.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 1521.80 | 1446.33 | 1473.63 | SL hit (close>ema400) qty=1.00 sl=1473.63 alert=retest1 |

### Cycle 4 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1564.80 | 1489.42 | 1489.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 1568.00 | 1490.20 | 1489.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 11:15:00 | 1733.60 | 1734.33 | 1651.14 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-08-05 13:15:00 | 1753.20 | 1734.55 | 1652.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 14:15:00 | 1751.85 | 1734.72 | 1652.58 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1881.25 | 1918.32 | 1870.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1858.70 | 1916.23 | 1870.42 | SL hit (close<ema400) qty=1.00 sl=1870.42 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-22 14:15:00 | 1905.60 | 1852.55 | 1850.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 1889.20 | 1852.91 | 1851.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-25 13:15:00 | 1888.30 | 1854.79 | 1852.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 1888.85 | 1855.13 | 1852.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 1861.25 | 1865.18 | 1857.86 | SL hit (close<static) qty=1.00 sl=1869.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 1861.25 | 1865.18 | 1857.86 | SL hit (close<static) qty=1.00 sl=1869.45 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-03 10:15:00 | 1893.30 | 1865.48 | 1858.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:15:00 | 1895.65 | 1865.78 | 1858.84 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-03 13:15:00 | 1892.75 | 1866.25 | 1859.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:15:00 | 1892.85 | 1866.51 | 1859.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1897.75 | 1917.98 | 1896.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 1845.95 | 1916.18 | 1896.23 | SL hit (close<static) qty=1.00 sl=1869.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 1845.95 | 1916.18 | 1896.23 | SL hit (close<static) qty=1.00 sl=1869.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-02 10:15:00 | 1935.10 | 1912.33 | 1895.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1937.50 | 1912.58 | 1895.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 13:15:00 | 1929.00 | 1918.15 | 1901.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 1937.00 | 1918.33 | 1901.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-09 15:15:00 | 1920.10 | 1918.98 | 1902.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1933.20 | 1919.13 | 1902.58 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1821.70 | 1927.22 | 1909.78 | SL hit (close<static) qty=1.00 sl=1886.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1821.70 | 1927.22 | 1909.78 | SL hit (close<static) qty=1.00 sl=1886.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1821.70 | 1927.22 | 1909.78 | SL hit (close<static) qty=1.00 sl=1886.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1823.65 | 1894.83 | 1895.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 1817.65 | 1874.23 | 1882.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1519.10 | 1516.96 | 1605.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 12:15:00 | 1608.00 | 1516.43 | 1587.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 1608.00 | 1516.43 | 1587.45 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-13 14:15:00 | 1565.80 | 1523.02 | 1587.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-13 15:15:00 | 1568.60 | 1523.47 | 1587.60 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-19 11:15:00 | 1564.20 | 1537.26 | 1587.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 1565.50 | 1537.54 | 1587.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-20 12:15:00 | 1565.30 | 1539.55 | 1586.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 1562.60 | 1539.78 | 1586.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-21 15:15:00 | 1566.90 | 1541.98 | 1585.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1557.00 | 1542.13 | 1585.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-05-23 14:15:00 | 1564.40 | 1544.49 | 1584.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 1566.10 | 1544.71 | 1584.24 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1575.30 | 1548.74 | 1583.48 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-28 10:15:00 | 1573.40 | 1548.99 | 1583.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-28 11:15:00 | 1578.30 | 1549.28 | 1583.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-28 14:15:00 | 1570.50 | 1550.04 | 1583.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 1571.80 | 1550.26 | 1583.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1592.40 | 1550.68 | 1583.27 | SL hit (close>static) qty=1.00 sl=1587.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-30 09:15:00 | 1561.10 | 1552.65 | 1583.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1561.90 | 1552.74 | 1583.05 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-09 10:15:00 | 1574.30 | 1553.57 | 1577.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1574.90 | 1553.78 | 1577.72 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-09 13:15:00 | 1574.40 | 1554.21 | 1577.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 1572.80 | 1554.40 | 1577.68 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1585.00 | 1554.88 | 1577.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1593.90 | 1555.27 | 1577.77 | SL hit (close>static) qty=1.00 sl=1587.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1593.90 | 1555.27 | 1577.77 | SL hit (close>static) qty=1.00 sl=1587.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1593.90 | 1555.27 | 1577.77 | SL hit (close>static) qty=1.00 sl=1587.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1623.70 | 1559.23 | 1578.80 | SL hit (close>static) qty=1.00 sl=1609.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1623.70 | 1559.23 | 1578.80 | SL hit (close>static) qty=1.00 sl=1609.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1623.70 | 1559.23 | 1578.80 | SL hit (close>static) qty=1.00 sl=1609.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1623.70 | 1559.23 | 1578.80 | SL hit (close>static) qty=1.00 sl=1609.60 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 1606.30 | 1591.66 | 1591.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.40 | 1592.60 | 1592.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.10 | 1605.27 | 1599.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 1594.10 | 1605.27 | 1599.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1594.10 | 1605.27 | 1599.32 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-16 12:15:00 | 1608.40 | 1601.14 | 1597.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 1609.50 | 1601.22 | 1597.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-16 15:15:00 | 1607.90 | 1601.33 | 1597.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-17 09:15:00 | 1596.10 | 1601.28 | 1597.91 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 1588.50 | 1601.04 | 1597.84 | SL hit (close<static) qty=1.00 sl=1590.60 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.20 | 1594.51 | 1594.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.50 | 1495.72 | 1531.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1535.20 | 1496.21 | 1529.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.20 | 1496.21 | 1529.19 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 1506.50 | 1500.09 | 1529.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 1509.60 | 1500.19 | 1529.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1509.60 | 1491.78 | 1515.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1510.00 | 1491.96 | 1515.59 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 1506.70 | 1495.35 | 1515.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1505.40 | 1495.45 | 1515.80 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.07 | 1515.75 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.07 | 1515.75 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.07 | 1515.75 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 1507.50 | 1503.64 | 1516.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1501.10 | 1503.62 | 1516.91 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1490.30 | 1483.07 | 1500.29 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 1488.40 | 1486.37 | 1500.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 1485.50 | 1486.36 | 1500.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 15:15:00 | 1487.20 | 1486.83 | 1500.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1473.00 | 1486.69 | 1500.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.04 | SL hit (close>static) qty=1.00 sl=1509.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.04 | SL hit (close>static) qty=1.00 sl=1509.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1541.50 | 1481.50 | 1495.27 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-31 10:15:00 | 1486.60 | 1490.55 | 1497.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1485.80 | 1490.50 | 1497.74 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1518.20 | 1486.44 | 1494.41 | SL hit (close>static) qty=1.00 sl=1509.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 12:15:00 | 1487.10 | 1498.21 | 1499.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-18 13:15:00 | 1491.50 | 1498.14 | 1499.48 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 1484.50 | 1498.00 | 1499.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 1486.80 | 1497.89 | 1499.34 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1528.00 | 1498.19 | 1499.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1528.00 | 1498.19 | 1499.48 | SL hit (close>static) qty=1.00 sl=1509.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.20 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.57 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.00 | 1606.82 | 1574.46 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 1639.50 | 1608.38 | 1578.04 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 14:15:00 | 1641.20 | 1608.71 | 1578.35 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 1673.50 | 1608.28 | 1583.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1684.10 | 1609.03 | 1583.80 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 14:15:00 | 1641.00 | 1634.72 | 1606.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 15:15:00 | 1641.20 | 1634.79 | 1606.39 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1662.20 | 1634.66 | 1607.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1671.80 | 1635.03 | 1607.76 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | SL hit (close<ema400) qty=1.00 sl=1608.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | SL hit (close<ema400) qty=1.00 sl=1608.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | SL hit (close<ema400) qty=1.00 sl=1608.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.55 | 1608.84 | SL hit (close<ema400) qty=1.00 sl=1608.84 alert=retest1 |

### Cycle 9 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.00 | 1587.61 | 1587.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1585.74 | 1586.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.00 | 1311.02 | 1382.66 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1289.30 | 1314.96 | 1375.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1287.40 | 1314.68 | 1374.95 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1271.60 | 1311.05 | 1360.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1264.30 | 1310.59 | 1360.43 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-13 14:15:00 | 1449.10 | 2024-01-23 09:15:00 | 1666.46 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-13 14:15:00 | 1449.10 | 2024-02-22 09:15:00 | 1647.50 | STOP_HIT | 0.50 | 13.69% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1435.05 | 2024-06-07 09:15:00 | 1521.80 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest1 | 2024-08-05 14:15:00 | 1751.85 | 2024-10-21 09:15:00 | 1858.70 | STOP_HIT | 1.00 | 6.10% |
| BUY | retest2 | 2024-11-22 15:15:00 | 1889.20 | 2024-11-28 12:15:00 | 1861.25 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-11-25 14:15:00 | 1888.85 | 2024-11-28 12:15:00 | 1861.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-03 11:15:00 | 1895.65 | 2024-12-31 09:15:00 | 1845.95 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-12-03 14:15:00 | 1892.85 | 2024-12-31 09:15:00 | 1845.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-01-02 11:15:00 | 1937.50 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2025-01-08 14:15:00 | 1937.00 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -5.95% |
| BUY | retest2 | 2025-01-10 09:15:00 | 1933.20 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2025-05-19 12:15:00 | 1565.50 | 2025-05-29 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-20 13:15:00 | 1562.60 | 2025-06-10 10:15:00 | 1593.90 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1557.00 | 2025-06-10 10:15:00 | 1593.90 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-05-23 15:15:00 | 1566.10 | 2025-06-10 10:15:00 | 1593.90 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-28 15:15:00 | 1571.80 | 2025-06-11 12:15:00 | 1623.70 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1561.90 | 2025-06-11 12:15:00 | 1623.70 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-06-09 11:15:00 | 1574.90 | 2025-06-11 12:15:00 | 1623.70 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-06-09 14:15:00 | 1572.80 | 2025-06-11 12:15:00 | 1623.70 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-07-16 13:15:00 | 1609.50 | 2025-07-17 12:15:00 | 1588.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-28 10:15:00 | 1509.60 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-11 10:15:00 | 1510.00 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-09-15 10:15:00 | 1505.40 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1501.10 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-10-13 11:15:00 | 1485.50 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-10-15 09:15:00 | 1473.00 | 2025-10-23 10:15:00 | 1541.50 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2025-10-31 11:15:00 | 1485.80 | 2025-11-10 10:15:00 | 1518.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-18 15:15:00 | 1486.80 | 2025-11-19 09:15:00 | 1528.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2026-01-07 14:15:00 | 1641.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest1 | 2026-01-16 10:15:00 | 1684.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -7.93% |
| BUY | retest1 | 2026-01-30 15:15:00 | 1641.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest1 | 2026-02-03 10:15:00 | 1671.80 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -7.25% |
