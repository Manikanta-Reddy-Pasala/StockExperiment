# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2024-12-19 09:40:00 → 2026-05-08 15:25:00 (25583 bars)
- **Last close:** 1686.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 8
- **Avg / median % per leg:** 0.35% / 0.00%
- **Sum % (uncompounded):** 8.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.49% | 6.9% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.49% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.14% | 1.4% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.14% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 10 | 41.7% | 2 | 14 | 8 | 0.35% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 1922.10 | 1931.81 | 0.00 | ORB-short ORB[1925.00,1945.40] vol=2.2x ATR=7.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:10:00 | 1910.54 | 1929.60 | 0.00 | T1 1.5R @ 1910.54 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 1922.10 | 1931.49 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 1952.60 | 1946.69 | 0.00 | ORB-long ORB[1929.15,1949.00] vol=1.9x ATR=9.54 |
| Stop hit — per-position SL triggered | 2025-01-09 11:20:00 | 1943.06 | 1948.82 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-01-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:05:00 | 1966.00 | 1951.46 | 0.00 | ORB-long ORB[1941.00,1958.90] vol=1.6x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:20:00 | 1979.34 | 1965.28 | 0.00 | T1 1.5R @ 1979.34 |
| Stop hit — per-position SL triggered | 2025-01-15 10:30:00 | 1966.00 | 1965.63 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1886.90 | 1910.83 | 0.00 | ORB-short ORB[1906.55,1930.00] vol=2.3x ATR=9.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 12:55:00 | 1873.07 | 1890.83 | 0.00 | T1 1.5R @ 1873.07 |
| Stop hit — per-position SL triggered | 2025-01-21 13:45:00 | 1886.90 | 1887.90 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:05:00 | 1867.95 | 1875.05 | 0.00 | ORB-short ORB[1872.00,1899.65] vol=1.6x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:35:00 | 1861.60 | 1873.82 | 0.00 | T1 1.5R @ 1861.60 |
| Stop hit — per-position SL triggered | 2025-01-22 12:10:00 | 1867.95 | 1857.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 1758.30 | 1726.62 | 0.00 | ORB-long ORB[1694.95,1704.95] vol=1.8x ATR=9.36 |
| Stop hit — per-position SL triggered | 2025-02-07 10:20:00 | 1748.94 | 1733.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-12 11:00:00 | 1675.00 | 1650.28 | 0.00 | ORB-long ORB[1651.35,1671.00] vol=1.6x ATR=8.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 11:15:00 | 1688.31 | 1652.34 | 0.00 | T1 1.5R @ 1688.31 |
| Target hit | 2025-02-12 15:20:00 | 1749.90 | 1702.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-02-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 10:30:00 | 1845.00 | 1804.62 | 0.00 | ORB-long ORB[1792.40,1809.90] vol=3.1x ATR=13.53 |
| Stop hit — per-position SL triggered | 2025-02-21 10:35:00 | 1831.47 | 1806.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:40:00 | 1777.95 | 1771.99 | 0.00 | ORB-long ORB[1752.00,1777.20] vol=3.1x ATR=11.06 |
| Stop hit — per-position SL triggered | 2025-03-05 10:10:00 | 1766.89 | 1772.32 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-03-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 10:10:00 | 1779.75 | 1780.80 | 0.00 | ORB-short ORB[1781.00,1801.15] vol=2.2x ATR=9.00 |
| Stop hit — per-position SL triggered | 2025-03-06 10:25:00 | 1788.75 | 1781.15 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 1506.90 | 1482.46 | 0.00 | ORB-long ORB[1463.15,1484.95] vol=1.8x ATR=9.06 |
| Stop hit — per-position SL triggered | 2025-03-21 09:40:00 | 1497.84 | 1484.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-03-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:00:00 | 1502.25 | 1485.08 | 0.00 | ORB-long ORB[1472.80,1494.90] vol=2.0x ATR=7.85 |
| Stop hit — per-position SL triggered | 2025-03-26 11:05:00 | 1494.40 | 1491.50 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 1408.00 | 1414.79 | 0.00 | ORB-short ORB[1415.10,1430.00] vol=5.0x ATR=6.55 |
| Stop hit — per-position SL triggered | 2025-04-16 10:30:00 | 1414.55 | 1412.21 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1449.80 | 1440.87 | 0.00 | ORB-long ORB[1426.20,1442.00] vol=2.9x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:35:00 | 1458.34 | 1444.38 | 0.00 | T1 1.5R @ 1458.34 |
| Stop hit — per-position SL triggered | 2025-04-22 09:45:00 | 1449.80 | 1445.81 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 1448.00 | 1462.43 | 0.00 | ORB-short ORB[1464.40,1484.10] vol=1.7x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 1437.24 | 1453.77 | 0.00 | T1 1.5R @ 1437.24 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 1448.00 | 1451.21 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 1487.30 | 1478.45 | 0.00 | ORB-long ORB[1466.60,1482.70] vol=2.4x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 09:55:00 | 1495.94 | 1484.91 | 0.00 | T1 1.5R @ 1495.94 |
| Target hit | 2025-05-08 13:00:00 | 1536.00 | 1540.27 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-03 10:05:00 | 1922.10 | 2025-01-03 10:10:00 | 1910.54 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-01-03 10:05:00 | 1922.10 | 2025-01-03 10:15:00 | 1922.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 10:50:00 | 1952.60 | 2025-01-09 11:20:00 | 1943.06 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-15 10:05:00 | 1966.00 | 2025-01-15 10:20:00 | 1979.34 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-15 10:05:00 | 1966.00 | 2025-01-15 10:30:00 | 1966.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 10:15:00 | 1886.90 | 2025-01-21 12:55:00 | 1873.07 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-01-21 10:15:00 | 1886.90 | 2025-01-21 13:45:00 | 1886.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 11:05:00 | 1867.95 | 2025-01-22 11:35:00 | 1861.60 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-22 11:05:00 | 1867.95 | 2025-01-22 12:10:00 | 1867.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 10:10:00 | 1758.30 | 2025-02-07 10:20:00 | 1748.94 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-02-12 11:00:00 | 1675.00 | 2025-02-12 11:15:00 | 1688.31 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-02-12 11:00:00 | 1675.00 | 2025-02-12 15:20:00 | 1749.90 | TARGET_HIT | 0.50 | 4.47% |
| BUY | retest1 | 2025-02-21 10:30:00 | 1845.00 | 2025-02-21 10:35:00 | 1831.47 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-03-05 09:40:00 | 1777.95 | 2025-03-05 10:10:00 | 1766.89 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2025-03-06 10:10:00 | 1779.75 | 2025-03-06 10:25:00 | 1788.75 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-03-21 09:35:00 | 1506.90 | 2025-03-21 09:40:00 | 1497.84 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-03-26 10:00:00 | 1502.25 | 2025-03-26 11:05:00 | 1494.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-04-16 09:45:00 | 1408.00 | 2025-04-16 10:30:00 | 1414.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1449.80 | 2025-04-22 09:35:00 | 1458.34 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1449.80 | 2025-04-22 09:45:00 | 1449.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:40:00 | 1448.00 | 2025-04-25 09:55:00 | 1437.24 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-04-25 09:40:00 | 1448.00 | 2025-04-25 10:10:00 | 1448.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:35:00 | 1487.30 | 2025-05-08 09:55:00 | 1495.94 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-05-08 09:35:00 | 1487.30 | 2025-05-08 13:00:00 | 1536.00 | TARGET_HIT | 0.50 | 3.27% |
