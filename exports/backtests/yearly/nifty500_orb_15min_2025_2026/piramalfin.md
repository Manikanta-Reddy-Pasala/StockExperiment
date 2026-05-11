# Piramal Finance Ltd. (PIRAMALFIN)

## Backtest Summary

- **Window:** 2025-12-08 09:15:00 → 2026-05-08 15:25:00 (7650 bars)
- **Last close:** 2015.00
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 10
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 7.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.30% | 4.5% |
| BUY @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.30% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.24% | 3.3% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.24% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 14 | 48.3% | 4 | 15 | 10 | 0.27% | 7.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:55:00 | 1453.00 | 1467.39 | 0.00 | ORB-short ORB[1465.00,1485.90] vol=3.2x ATR=9.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:05:00 | 1438.44 | 1459.23 | 0.00 | T1 1.5R @ 1438.44 |
| Stop hit — per-position SL triggered | 2025-12-08 14:25:00 | 1453.00 | 1446.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:35:00 | 1498.40 | 1505.59 | 0.00 | ORB-short ORB[1504.40,1515.10] vol=2.0x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-12-16 09:40:00 | 1502.61 | 1505.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-12-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:40:00 | 1652.00 | 1640.06 | 0.00 | ORB-long ORB[1605.40,1624.00] vol=2.0x ATR=9.80 |
| Stop hit — per-position SL triggered | 2025-12-22 10:00:00 | 1642.20 | 1643.24 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:50:00 | 1640.00 | 1631.53 | 0.00 | ORB-long ORB[1612.80,1627.60] vol=1.8x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 12:55:00 | 1647.57 | 1635.73 | 0.00 | T1 1.5R @ 1647.57 |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 1640.00 | 1639.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 1629.00 | 1607.33 | 0.00 | ORB-long ORB[1603.10,1618.80] vol=3.5x ATR=7.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:55:00 | 1639.65 | 1612.57 | 0.00 | T1 1.5R @ 1639.65 |
| Stop hit — per-position SL triggered | 2025-12-31 13:05:00 | 1629.00 | 1631.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:50:00 | 1874.90 | 1849.20 | 0.00 | ORB-long ORB[1826.00,1847.30] vol=5.4x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:25:00 | 1884.75 | 1861.05 | 0.00 | T1 1.5R @ 1884.75 |
| Target hit | 2026-01-16 15:20:00 | 1903.90 | 1881.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:05:00 | 1804.80 | 1821.38 | 0.00 | ORB-short ORB[1817.10,1843.90] vol=2.4x ATR=7.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:30:00 | 1793.68 | 1818.19 | 0.00 | T1 1.5R @ 1793.68 |
| Stop hit — per-position SL triggered | 2026-01-22 12:20:00 | 1804.80 | 1814.11 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:55:00 | 1750.00 | 1761.72 | 0.00 | ORB-short ORB[1757.40,1779.90] vol=2.2x ATR=4.70 |
| Stop hit — per-position SL triggered | 2026-01-29 12:30:00 | 1754.70 | 1756.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 1750.90 | 1737.51 | 0.00 | ORB-long ORB[1718.60,1741.90] vol=3.6x ATR=6.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1761.27 | 1741.36 | 0.00 | T1 1.5R @ 1761.27 |
| Stop hit — per-position SL triggered | 2026-02-10 13:35:00 | 1750.90 | 1757.66 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 1751.90 | 1762.85 | 0.00 | ORB-short ORB[1762.30,1784.90] vol=2.9x ATR=6.20 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 1758.10 | 1762.44 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 1790.40 | 1772.29 | 0.00 | ORB-long ORB[1747.30,1770.70] vol=2.8x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 1782.96 | 1776.34 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 1756.00 | 1769.01 | 0.00 | ORB-short ORB[1772.80,1792.20] vol=5.5x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:40:00 | 1747.01 | 1766.35 | 0.00 | T1 1.5R @ 1747.01 |
| Stop hit — per-position SL triggered | 2026-03-06 13:05:00 | 1756.00 | 1754.77 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 1802.00 | 1777.02 | 0.00 | ORB-long ORB[1770.10,1797.00] vol=2.2x ATR=8.22 |
| Stop hit — per-position SL triggered | 2026-03-17 11:10:00 | 1793.78 | 1778.30 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 11:00:00 | 1719.30 | 1731.75 | 0.00 | ORB-short ORB[1722.00,1744.90] vol=8.2x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 14:50:00 | 1709.80 | 1723.49 | 0.00 | T1 1.5R @ 1709.80 |
| Target hit | 2026-04-07 15:20:00 | 1696.80 | 1719.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1746.80 | 1753.11 | 0.00 | ORB-short ORB[1750.00,1767.80] vol=2.0x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:20:00 | 1736.76 | 1743.91 | 0.00 | T1 1.5R @ 1736.76 |
| Target hit | 2026-04-15 10:30:00 | 1744.50 | 1743.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 1654.50 | 1647.45 | 0.00 | ORB-long ORB[1632.20,1650.00] vol=1.9x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:30:00 | 1661.85 | 1650.12 | 0.00 | T1 1.5R @ 1661.85 |
| Target hit | 2026-04-21 15:20:00 | 1696.00 | 1685.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1945.00 | 1927.12 | 0.00 | ORB-long ORB[1906.70,1934.90] vol=3.0x ATR=9.27 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 1935.73 | 1931.59 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 1915.70 | 1924.98 | 0.00 | ORB-short ORB[1925.00,1950.00] vol=1.8x ATR=8.63 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 1924.33 | 1921.86 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1950.00 | 1937.21 | 0.00 | ORB-long ORB[1918.00,1936.50] vol=3.4x ATR=5.49 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 1944.51 | 1944.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-12-08 10:55:00 | 1453.00 | 2025-12-08 12:05:00 | 1438.44 | PARTIAL | 0.50 | 1.00% |
| SELL | retest1 | 2025-12-08 10:55:00 | 1453.00 | 2025-12-08 14:25:00 | 1453.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 09:35:00 | 1498.40 | 2025-12-16 09:40:00 | 1502.61 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-22 09:40:00 | 1652.00 | 2025-12-22 10:00:00 | 1642.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-12-26 10:50:00 | 1640.00 | 2025-12-26 12:55:00 | 1647.57 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-26 10:50:00 | 1640.00 | 2025-12-26 14:15:00 | 1640.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:50:00 | 1629.00 | 2025-12-31 10:55:00 | 1639.65 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-12-31 10:50:00 | 1629.00 | 2025-12-31 13:05:00 | 1629.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:50:00 | 1874.90 | 2026-01-16 11:25:00 | 1884.75 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-01-16 10:50:00 | 1874.90 | 2026-01-16 15:20:00 | 1903.90 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2026-01-22 11:05:00 | 1804.80 | 2026-01-22 11:30:00 | 1793.68 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-01-22 11:05:00 | 1804.80 | 2026-01-22 12:20:00 | 1804.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 10:55:00 | 1750.00 | 2026-01-29 12:30:00 | 1754.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-10 09:50:00 | 1750.90 | 2026-02-10 10:10:00 | 1761.27 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-10 09:50:00 | 1750.90 | 2026-02-10 13:35:00 | 1750.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:15:00 | 1751.90 | 2026-02-13 10:30:00 | 1758.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-25 10:25:00 | 1790.40 | 2026-02-25 10:55:00 | 1782.96 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1756.00 | 2026-03-06 11:40:00 | 1747.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1756.00 | 2026-03-06 13:05:00 | 1756.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 11:05:00 | 1802.00 | 2026-03-17 11:10:00 | 1793.78 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1719.30 | 2026-04-07 14:50:00 | 1709.80 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1719.30 | 2026-04-07 15:20:00 | 1696.80 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2026-04-15 09:35:00 | 1746.80 | 2026-04-15 10:20:00 | 1736.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-15 09:35:00 | 1746.80 | 2026-04-15 10:30:00 | 1744.50 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1654.50 | 2026-04-21 11:30:00 | 1661.85 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1654.50 | 2026-04-21 15:20:00 | 1696.00 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1945.00 | 2026-05-06 09:55:00 | 1935.73 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-07 10:45:00 | 1915.70 | 2026-05-07 11:40:00 | 1924.33 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1950.00 | 2026-05-08 09:45:00 | 1944.51 | STOP_HIT | 1.00 | -0.28% |
