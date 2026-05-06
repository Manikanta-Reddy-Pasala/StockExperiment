# INFY (INFY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1167.20
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
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 27 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 21 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 19
- **Target hits / Stop hits / Partials:** 0 / 21 / 1
- **Avg / median % per leg:** -0.05% / -1.24%
- **Sum % (uncompounded):** -1.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 14 | 1 | 0.82% | 12.3% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 6.76% | 6.8% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 0 | 13 | 1 | 0.40% | 5.6% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.91% | -13.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.91% | -13.4% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 6.76% | 6.8% |
| retest2 (combined) | 21 | 2 | 9.5% | 0 | 20 | 1 | -0.37% | -7.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 1459.20 | 1417.38 | 1417.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1463.90 | 1428.13 | 1423.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 10:15:00 | 1439.35 | 1445.20 | 1433.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 1440.00 | 1445.06 | 1433.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 1440.00 | 1445.06 | 1433.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-13 13:15:00 | 1443.20 | 1445.04 | 1433.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 14:15:00 | 1449.10 | 1445.08 | 1433.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-01-23 09:15:00 | 1666.46 | 1555.95 | 1514.61 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-03-28 10:15:00 | 1505.90 | 1597.96 | 1598.08 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 1449.10 | 1538.89 | 1563.11 | SL hit qty=0.50 sl=1449.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-23 12:15:00 | 1440.90 | 1507.95 | 1542.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:15:00 | 1445.20 | 1507.32 | 1542.45 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 1433.20 | 1504.63 | 1540.40 | SL hit qty=1.00 sl=1433.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-25 13:15:00 | 1443.25 | 1498.03 | 1535.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-25 14:15:00 | 1438.05 | 1497.43 | 1534.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-09 11:15:00 | 1449.35 | 1466.36 | 1507.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 12:15:00 | 1442.50 | 1466.12 | 1506.93 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 1433.20 | 1464.91 | 1505.50 | SL hit qty=1.00 sl=1433.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-16 09:15:00 | 1443.45 | 1454.70 | 1494.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-16 10:15:00 | 1439.60 | 1454.55 | 1494.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-16 11:15:00 | 1447.35 | 1454.47 | 1494.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-16 12:15:00 | 1436.75 | 1454.30 | 1493.93 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-16 14:15:00 | 1452.95 | 1454.11 | 1493.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 1450.75 | 1454.07 | 1493.23 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1442.10 | 1453.95 | 1492.97 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 1433.20 | 1453.26 | 1491.28 | SL hit qty=1.00 sl=1433.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-22 11:15:00 | 1453.75 | 1452.22 | 1489.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 1458.25 | 1452.28 | 1488.92 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-29 15:15:00 | 1452.95 | 1456.49 | 1484.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-30 09:15:00 | 1435.25 | 1456.28 | 1484.66 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1439.60 | 1456.28 | 1484.66 | SL hit qty=1.00 sl=1439.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-06 11:15:00 | 1452.05 | 1444.60 | 1473.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 12:15:00 | 1464.20 | 1444.79 | 1473.42 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 1564.80 | 1489.42 | 1489.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1564.80 | 1489.42 | 1489.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 1568.00 | 1490.20 | 1489.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 11:15:00 | 1733.60 | 1734.33 | 1651.14 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-05 13:15:00 | 1753.20 | 1734.55 | 1652.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 14:15:00 | 1751.85 | 1734.72 | 1652.58 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1881.25 | 1918.32 | 1870.29 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1870.29 | 1918.32 | 1870.29 | SL hit qty=1.00 sl=1870.29 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-22 14:15:00 | 1905.60 | 1852.55 | 1850.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 1889.20 | 1852.91 | 1851.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-25 13:15:00 | 1888.30 | 1854.79 | 1852.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 1888.85 | 1855.13 | 1852.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1869.45 | 1865.15 | 1857.78 | SL hit qty=1.00 sl=1869.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1869.45 | 1865.15 | 1857.78 | SL hit qty=1.00 sl=1869.45 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-03 10:15:00 | 1893.30 | 1865.48 | 1858.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:15:00 | 1895.65 | 1865.78 | 1858.84 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-03 13:15:00 | 1892.75 | 1866.25 | 1859.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:15:00 | 1892.85 | 1866.51 | 1859.32 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1897.75 | 1917.98 | 1896.40 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 1869.45 | 1916.18 | 1896.23 | SL hit qty=1.00 sl=1869.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 1869.45 | 1916.18 | 1896.23 | SL hit qty=1.00 sl=1869.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-02 10:15:00 | 1935.10 | 1912.33 | 1895.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1937.50 | 1912.58 | 1895.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 13:15:00 | 1929.00 | 1918.15 | 1901.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 1937.00 | 1918.33 | 1901.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-09 15:15:00 | 1920.10 | 1918.98 | 1902.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1933.20 | 1919.13 | 1902.58 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1886.50 | 1927.22 | 1909.78 | SL hit qty=1.00 sl=1886.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1886.50 | 1927.22 | 1909.78 | SL hit qty=1.00 sl=1886.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1886.50 | 1927.22 | 1909.78 | SL hit qty=1.00 sl=1886.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-27 15:15:00 | 1823.65 | 1894.83 | 1895.13 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-07-01 12:15:00 | 1606.30 | 1591.66 | 1591.61 | HTF filter: close below htf_sma |

### Cycle 3 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.20 | 1594.51 | 1594.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.50 | 1495.72 | 1531.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1535.20 | 1496.21 | 1529.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.20 | 1496.21 | 1529.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 1506.50 | 1500.09 | 1529.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 1509.60 | 1500.19 | 1529.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1509.60 | 1491.78 | 1515.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1510.00 | 1491.96 | 1515.59 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1535.70 | 1493.28 | 1515.57 | SL hit qty=1.00 sl=1535.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1535.70 | 1493.28 | 1515.57 | SL hit qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 1506.70 | 1495.35 | 1515.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1505.40 | 1495.45 | 1515.80 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1535.70 | 1499.07 | 1515.75 | SL hit qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 1507.50 | 1503.64 | 1516.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1501.10 | 1503.62 | 1516.91 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1490.30 | 1483.07 | 1500.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 1488.40 | 1486.37 | 1500.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 1485.50 | 1486.36 | 1500.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1509.00 | 1486.60 | 1500.45 | SL hit qty=1.00 sl=1509.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-14 15:15:00 | 1487.20 | 1486.83 | 1500.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1473.00 | 1486.69 | 1500.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1509.00 | 1480.90 | 1495.04 | SL hit qty=1.00 sl=1509.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1535.70 | 1481.50 | 1495.27 | SL hit qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-31 10:15:00 | 1486.60 | 1490.55 | 1497.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1485.80 | 1490.50 | 1497.74 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1509.00 | 1486.44 | 1494.41 | SL hit qty=1.00 sl=1509.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 12:15:00 | 1487.10 | 1498.21 | 1499.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-18 13:15:00 | 1491.50 | 1498.14 | 1499.48 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 1484.50 | 1498.00 | 1499.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 1486.80 | 1497.89 | 1499.34 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2025-11-19 09:15:00 | 1528.00 | 1498.19 | 1499.48 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2025-11-20 09:15:00 | 1540.20 | 1500.97 | 1500.85 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2026-02-12 10:15:00 | 1405.00 | 1587.61 | 1587.74 | slope filter: EMA200 not falling 2.00% over 1400 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-13 14:15:00 | 1449.10 | 2024-01-23 09:15:00 | 1666.46 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-13 14:15:00 | 1449.10 | 2024-04-16 09:15:00 | 1449.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-04-23 13:15:00 | 1445.20 | 2024-04-24 10:15:00 | 1433.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-09 12:15:00 | 1442.50 | 2024-05-10 09:15:00 | 1433.20 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-05-16 15:15:00 | 1450.75 | 2024-05-21 09:15:00 | 1433.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-05-22 12:15:00 | 1458.25 | 2024-05-30 09:15:00 | 1439.60 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-06-06 12:15:00 | 1464.20 | 2024-06-27 11:15:00 | 1564.80 | STOP_HIT | 1.00 | 6.87% |
| BUY | retest1 | 2024-08-05 14:15:00 | 1751.85 | 2024-10-18 11:15:00 | 1870.29 | STOP_HIT | 1.00 | 6.76% |
| BUY | retest2 | 2024-11-22 15:15:00 | 1889.20 | 2024-11-28 10:15:00 | 1869.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-11-25 14:15:00 | 1888.85 | 2024-11-28 10:15:00 | 1869.45 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-12-03 11:15:00 | 1895.65 | 2024-12-31 09:15:00 | 1869.45 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-03 14:15:00 | 1892.85 | 2024-12-31 09:15:00 | 1869.45 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-01-02 11:15:00 | 1937.50 | 2025-01-17 09:15:00 | 1886.50 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-01-08 14:15:00 | 1937.00 | 2025-01-17 09:15:00 | 1886.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-01-10 09:15:00 | 1933.20 | 2025-01-17 09:15:00 | 1886.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-28 10:15:00 | 1509.60 | 2025-09-12 09:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-11 10:15:00 | 1510.00 | 2025-09-12 09:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-15 10:15:00 | 1505.40 | 2025-09-18 09:15:00 | 1535.70 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1501.10 | 2025-10-14 09:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-13 11:15:00 | 1485.50 | 2025-10-23 09:15:00 | 1509.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-10-15 09:15:00 | 1473.00 | 2025-10-23 10:15:00 | 1535.70 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-10-31 11:15:00 | 1485.80 | 2025-11-10 10:15:00 | 1509.00 | STOP_HIT | 1.00 | -1.56% |
