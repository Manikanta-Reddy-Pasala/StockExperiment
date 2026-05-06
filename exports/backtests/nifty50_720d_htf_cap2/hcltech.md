# HCLTECH (HCLTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1189.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 19 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 11 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 0 / 13 / 6
- **Avg / median % per leg:** 4.15% / 10.11%
- **Sum % (uncompounded):** 78.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 8 | 61.5% | 0 | 9 | 4 | 4.25% | 55.3% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 12.94% | 103.5% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -9.65% | -48.2% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 3.93% | 23.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.39% | -0.4% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 3 | 2 | 4.79% | 24.0% |
| retest1 (combined) | 9 | 8 | 88.9% | 0 | 5 | 4 | 11.46% | 103.1% |
| retest2 (combined) | 10 | 2 | 20.0% | 0 | 8 | 2 | -2.43% | -24.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 1517.35 | 1438.59 | 1438.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 1521.70 | 1439.41 | 1438.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1555.00 | 1556.05 | 1513.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-06 09:15:00 | 1597.40 | 1556.70 | 1515.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:15:00 | 1594.40 | 1557.07 | 1515.39 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 09:15:00 | 1600.15 | 1558.93 | 1517.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:15:00 | 1600.30 | 1559.34 | 1517.98 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-08 11:15:00 | 1599.00 | 1562.06 | 1520.99 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-08 12:15:00 | 1568.05 | 1562.12 | 1521.22 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-09 09:15:00 | 1589.25 | 1562.37 | 1522.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1595.90 | 1562.70 | 1522.53 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-12 10:15:00 | 1593.20 | 1564.66 | 1524.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 1594.85 | 1564.96 | 1525.26 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-10-10 09:15:00 | 1833.56 | 1764.32 | 1707.76 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-10-10 09:15:00 | 1835.29 | 1764.32 | 1707.76 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-10-10 09:15:00 | 1834.08 | 1764.32 | 1707.76 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-10-11 09:15:00 | 1840.35 | 1767.97 | 1711.57 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1762.70 | 1818.74 | 1766.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-06 09:15:00 | 1821.85 | 1811.47 | 1766.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 1834.05 | 1811.69 | 1767.09 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-16 13:15:00 | 1793.50 | 1904.67 | 1880.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 1791.90 | 1903.55 | 1880.36 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1762.15 | 1901.13 | 1879.38 | SL hit qty=1.00 sl=1762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1762.15 | 1901.13 | 1879.38 | SL hit qty=1.00 sl=1762.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-17 14:15:00 | 1789.10 | 1895.22 | 1876.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 15:15:00 | 1788.90 | 1894.16 | 1876.48 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-20 13:15:00 | 1791.75 | 1888.73 | 1874.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 1795.00 | 1887.79 | 1873.77 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1796.20 | 1886.88 | 1873.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-21 09:15:00 | 1810.35 | 1886.12 | 1873.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1805.20 | 1885.31 | 1872.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 1793.35 | 1869.11 | 1865.62 | SL hit qty=1.00 sl=1793.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 1762.15 | 1867.22 | 1864.70 | SL hit qty=1.00 sl=1762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 1762.15 | 1867.22 | 1864.70 | SL hit qty=1.00 sl=1762.15 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-27 13:15:00 | 1723.60 | 1862.22 | 1862.23 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1594.40 | 1727.84 | 1772.74 | SL hit qty=0.50 sl=1594.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1600.30 | 1727.84 | 1772.74 | SL hit qty=0.50 sl=1600.30 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1595.90 | 1727.84 | 1772.74 | SL hit qty=0.50 sl=1595.90 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1594.85 | 1727.84 | 1772.74 | SL hit qty=0.50 sl=1594.85 alert=retest1 |
| CROSSOVER_SKIP | 2025-05-28 10:15:00 | 1650.00 | 1606.86 | 1606.85 | HTF filter: close below htf_sma |

### Cycle 2 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.35 | 1649.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.40 | 1647.01 | 1648.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.00 | 1476.57 | 1514.67 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-09-19 12:15:00 | 1470.50 | 1478.10 | 1512.03 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 13:15:00 | 1472.90 | 1478.05 | 1511.83 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.00 | 1443.08 | 1478.70 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1478.70 | 1443.08 | 1478.70 | SL hit qty=1.00 sl=1478.70 alert=retest1 |
| CROSSOVER_SKIP | 2025-11-03 12:15:00 | 1535.20 | 1495.77 | 1495.73 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 1405.90 | 1633.54 | 1629.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 1447.10 | 1631.68 | 1628.21 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-13 14:15:00 | 1455.10 | 1625.00 | 1624.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1458.00 | 1623.33 | 1624.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1458.00 | 1623.33 | 1624.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 1452.80 | 1616.72 | 1620.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.05 | 1456.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1437.90 | 1399.28 | 1453.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.28 | 1453.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1433.70 | 1406.13 | 1453.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1429.30 | 1406.36 | 1453.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1427.70 | 1408.33 | 1453.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1423.40 | 1408.48 | 1453.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1464.00 | 1409.83 | 1452.38 | SL hit qty=1.00 sl=1464.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1464.00 | 1409.83 | 1452.38 | SL hit qty=1.00 sl=1464.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 1435.40 | 1412.86 | 1452.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 1443.00 | 1413.16 | 1452.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 10:15:00 | 1432.90 | 1414.60 | 1451.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 1433.20 | 1414.79 | 1451.70 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-20 09:15:00 | 1430.00 | 1415.86 | 1451.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:15:00 | 1429.60 | 1416.00 | 1451.22 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 300m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1449.40 | 1416.91 | 1450.98 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1305.50 | 1416.72 | 1450.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1299.80 | 1415.55 | 1449.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1218.22 | 1394.98 | 1436.22 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-24 11:15:00 | 1215.16 | 1394.98 | 1436.22 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-06 10:15:00 | 1594.40 | 2024-10-10 09:15:00 | 1833.56 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-08-07 10:15:00 | 1600.30 | 2024-10-10 09:15:00 | 1835.29 | PARTIAL | 0.50 | 14.68% |
| BUY | retest1 | 2024-08-09 10:15:00 | 1595.90 | 2024-10-10 09:15:00 | 1834.08 | PARTIAL | 0.50 | 14.92% |
| BUY | retest1 | 2024-08-12 11:15:00 | 1594.85 | 2024-10-11 09:15:00 | 1840.35 | PARTIAL | 0.50 | 15.39% |
| BUY | retest1 | 2024-08-06 10:15:00 | 1594.40 | 2025-01-17 09:15:00 | 1762.15 | STOP_HIT | 0.50 | 10.52% |
| BUY | retest1 | 2024-08-07 10:15:00 | 1600.30 | 2025-01-17 09:15:00 | 1762.15 | STOP_HIT | 0.50 | 10.11% |
| BUY | retest1 | 2024-08-09 10:15:00 | 1595.90 | 2025-01-24 14:15:00 | 1793.35 | STOP_HIT | 0.50 | 12.37% |
| BUY | retest1 | 2024-08-12 11:15:00 | 1594.85 | 2025-01-27 09:15:00 | 1762.15 | STOP_HIT | 0.50 | 10.49% |
| BUY | retest2 | 2024-11-06 10:15:00 | 1834.05 | 2025-01-27 09:15:00 | 1762.15 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-01-16 14:15:00 | 1791.90 | 2025-02-28 09:15:00 | 1594.40 | STOP_HIT | 1.00 | -11.02% |
| BUY | retest2 | 2025-01-17 15:15:00 | 1788.90 | 2025-02-28 09:15:00 | 1600.30 | STOP_HIT | 1.00 | -10.54% |
| BUY | retest2 | 2025-01-20 14:15:00 | 1795.00 | 2025-02-28 09:15:00 | 1595.90 | STOP_HIT | 1.00 | -11.09% |
| BUY | retest2 | 2025-01-21 10:15:00 | 1805.20 | 2025-02-28 09:15:00 | 1594.85 | STOP_HIT | 1.00 | -11.65% |
| SELL | retest1 | 2025-09-19 13:15:00 | 1472.90 | 2025-10-09 09:15:00 | 1478.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-02-13 10:15:00 | 1447.10 | 2026-02-13 15:15:00 | 1458.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-04-10 10:15:00 | 1429.30 | 2026-04-15 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1423.40 | 2026-04-15 09:15:00 | 1464.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2026-04-17 11:15:00 | 1433.20 | 2026-04-24 11:15:00 | 1218.22 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-04-20 14:15:00 | 1429.60 | 2026-04-24 11:15:00 | 1215.16 | PARTIAL | 0.50 | 15.00% |
