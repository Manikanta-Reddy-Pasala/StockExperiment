# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2024-11-07 09:15:00 → 2026-05-08 15:25:00 (27688 bars)
- **Last close:** 3905.00
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 12
- **Target hits / Stop hits / Partials:** 5 / 12 / 12
- **Avg / median % per leg:** 0.41% / 0.56%
- **Sum % (uncompounded):** 11.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.19% | -0.9% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.19% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 16 | 66.7% | 5 | 8 | 11 | 0.53% | 12.8% |
| SELL @ 2nd Alert (retest1) | 24 | 16 | 66.7% | 5 | 8 | 11 | 0.53% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 17 | 58.6% | 5 | 12 | 12 | 0.41% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:30:00 | 1566.67 | 1582.30 | 0.00 | ORB-short ORB[1574.37,1594.90] vol=2.5x ATR=8.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:00:00 | 1554.46 | 1574.38 | 0.00 | T1 1.5R @ 1554.46 |
| Stop hit — per-position SL triggered | 2024-11-22 10:35:00 | 1566.67 | 1571.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1560.65 | 1546.29 | 0.00 | ORB-long ORB[1528.83,1548.25] vol=3.7x ATR=7.28 |
| Stop hit — per-position SL triggered | 2024-11-29 10:10:00 | 1553.37 | 1554.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 1910.18 | 1897.69 | 0.00 | ORB-long ORB[1884.68,1898.33] vol=2.4x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:50:00 | 1917.97 | 1900.64 | 0.00 | T1 1.5R @ 1917.97 |
| Stop hit — per-position SL triggered | 2024-12-17 11:35:00 | 1910.18 | 1905.62 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1796.27 | 1811.98 | 0.00 | ORB-short ORB[1808.70,1825.62] vol=3.4x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:10:00 | 1786.22 | 1803.90 | 0.00 | T1 1.5R @ 1786.22 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 1796.27 | 1803.45 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1796.98 | 1813.76 | 0.00 | ORB-short ORB[1815.00,1830.53] vol=2.9x ATR=6.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 09:50:00 | 1786.49 | 1804.24 | 0.00 | T1 1.5R @ 1786.49 |
| Target hit | 2024-12-27 15:20:00 | 1759.93 | 1781.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:55:00 | 1795.67 | 1781.53 | 0.00 | ORB-long ORB[1765.33,1781.90] vol=2.1x ATR=7.45 |
| Stop hit — per-position SL triggered | 2025-01-01 10:00:00 | 1788.22 | 1782.20 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:00:00 | 1802.08 | 1802.84 | 0.00 | ORB-short ORB[1802.57,1823.13] vol=1.8x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-01-02 11:10:00 | 1807.04 | 1802.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:15:00 | 1821.67 | 1839.52 | 0.00 | ORB-short ORB[1827.68,1847.07] vol=2.3x ATR=6.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 12:35:00 | 1811.30 | 1834.67 | 0.00 | T1 1.5R @ 1811.30 |
| Stop hit — per-position SL triggered | 2025-01-03 13:25:00 | 1821.67 | 1832.63 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 1731.05 | 1763.36 | 0.00 | ORB-short ORB[1769.02,1792.50] vol=1.7x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:45:00 | 1719.41 | 1753.72 | 0.00 | T1 1.5R @ 1719.41 |
| Stop hit — per-position SL triggered | 2025-01-06 12:35:00 | 1731.05 | 1747.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-01-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:30:00 | 1962.00 | 1974.97 | 0.00 | ORB-short ORB[1966.67,1992.33] vol=1.6x ATR=7.86 |
| Stop hit — per-position SL triggered | 2025-01-17 09:35:00 | 1969.86 | 1973.78 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 1973.67 | 1983.42 | 0.00 | ORB-short ORB[1974.33,2003.23] vol=1.5x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:35:00 | 1961.50 | 1976.36 | 0.00 | T1 1.5R @ 1961.50 |
| Target hit | 2025-01-21 11:45:00 | 1951.97 | 1947.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 1763.30 | 1750.58 | 0.00 | ORB-long ORB[1740.38,1757.67] vol=1.5x ATR=8.32 |
| Stop hit — per-position SL triggered | 2025-01-31 12:45:00 | 1754.98 | 1758.95 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-02-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:40:00 | 1739.40 | 1761.76 | 0.00 | ORB-short ORB[1753.08,1778.33] vol=1.6x ATR=11.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1722.58 | 1751.36 | 0.00 | T1 1.5R @ 1722.58 |
| Target hit | 2025-02-14 15:20:00 | 1702.93 | 1730.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 1538.03 | 1552.33 | 0.00 | ORB-short ORB[1550.55,1562.33] vol=1.7x ATR=7.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:45:00 | 1526.60 | 1545.07 | 0.00 | T1 1.5R @ 1526.60 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 1538.03 | 1543.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-04-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:45:00 | 1854.55 | 1871.01 | 0.00 | ORB-short ORB[1861.10,1886.55] vol=1.6x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:55:00 | 1841.47 | 1865.90 | 0.00 | T1 1.5R @ 1841.47 |
| Target hit | 2025-04-04 10:55:00 | 1853.83 | 1852.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 2091.33 | 2100.14 | 0.00 | ORB-short ORB[2094.00,2111.83] vol=1.6x ATR=7.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:35:00 | 2079.67 | 2097.83 | 0.00 | T1 1.5R @ 2079.67 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 2091.33 | 2096.63 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 2143.00 | 2177.23 | 0.00 | ORB-short ORB[2168.67,2198.33] vol=2.3x ATR=10.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 2127.46 | 2165.03 | 0.00 | T1 1.5R @ 2127.46 |
| Target hit | 2025-04-25 13:55:00 | 2127.33 | 2117.06 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-22 09:30:00 | 1566.67 | 2024-11-22 10:00:00 | 1554.46 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-11-22 09:30:00 | 1566.67 | 2024-11-22 10:35:00 | 1566.67 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 09:30:00 | 1560.65 | 2024-11-29 10:10:00 | 1553.37 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-12-17 10:45:00 | 1910.18 | 2024-12-17 10:50:00 | 1917.97 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-17 10:45:00 | 1910.18 | 2024-12-17 11:35:00 | 1910.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:30:00 | 1796.27 | 2024-12-26 10:10:00 | 1786.22 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-12-26 09:30:00 | 1796.27 | 2024-12-26 10:15:00 | 1796.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 09:30:00 | 1796.98 | 2024-12-27 09:50:00 | 1786.49 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-27 09:30:00 | 1796.98 | 2024-12-27 15:20:00 | 1759.93 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2025-01-01 09:55:00 | 1795.67 | 2025-01-01 10:00:00 | 1788.22 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-02 11:00:00 | 1802.08 | 2025-01-02 11:10:00 | 1807.04 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-03 11:15:00 | 1821.67 | 2025-01-03 12:35:00 | 1811.30 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-03 11:15:00 | 1821.67 | 2025-01-03 13:25:00 | 1821.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1731.05 | 2025-01-06 11:45:00 | 1719.41 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1731.05 | 2025-01-06 12:35:00 | 1731.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 09:30:00 | 1962.00 | 2025-01-17 09:35:00 | 1969.86 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-21 09:30:00 | 1973.67 | 2025-01-21 09:35:00 | 1961.50 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-01-21 09:30:00 | 1973.67 | 2025-01-21 11:45:00 | 1951.97 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2025-01-31 09:40:00 | 1763.30 | 2025-01-31 12:45:00 | 1754.98 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-02-14 10:40:00 | 1739.40 | 2025-02-14 12:15:00 | 1722.58 | PARTIAL | 0.50 | 0.97% |
| SELL | retest1 | 2025-02-14 10:40:00 | 1739.40 | 2025-02-14 15:20:00 | 1702.93 | TARGET_HIT | 0.50 | 2.10% |
| SELL | retest1 | 2025-03-26 09:35:00 | 1538.03 | 2025-03-26 09:45:00 | 1526.60 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-03-26 09:35:00 | 1538.03 | 2025-03-26 09:55:00 | 1538.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-04 09:45:00 | 1854.55 | 2025-04-04 09:55:00 | 1841.47 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-04-04 09:45:00 | 1854.55 | 2025-04-04 10:55:00 | 1853.83 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-04-23 09:30:00 | 2091.33 | 2025-04-23 09:35:00 | 2079.67 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-23 09:30:00 | 2091.33 | 2025-04-23 09:45:00 | 2091.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2143.00 | 2025-04-25 09:45:00 | 2127.46 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2143.00 | 2025-04-25 13:55:00 | 2127.33 | TARGET_HIT | 0.50 | 0.73% |
