# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-03-30 15:25:00 (34996 bars)
- **Last close:** 1191.90
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
| ENTRY1 | 30 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 7 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 24
- **Target hits / Stop hits / Partials:** 7 / 23 / 14
- **Avg / median % per leg:** 0.28% / 0.00%
- **Sum % (uncompounded):** 12.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 15 | 53.6% | 6 | 12 | 10 | 0.46% | 12.8% |
| BUY @ 2nd Alert (retest1) | 28 | 15 | 53.6% | 6 | 12 | 10 | 0.46% | 12.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.04% | -0.6% |
| SELL @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.04% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 44 | 20 | 45.5% | 7 | 23 | 14 | 0.28% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:05:00 | 1805.95 | 1793.53 | 0.00 | ORB-long ORB[1780.90,1800.00] vol=2.7x ATR=7.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:40:00 | 1816.60 | 1797.13 | 0.00 | T1 1.5R @ 1816.60 |
| Stop hit — per-position SL triggered | 2024-05-23 12:10:00 | 1805.95 | 1798.04 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1843.40 | 1830.84 | 0.00 | ORB-long ORB[1805.00,1830.90] vol=2.1x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:35:00 | 1855.19 | 1839.98 | 0.00 | T1 1.5R @ 1855.19 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 1843.40 | 1839.30 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 10:45:00 | 1794.60 | 1782.86 | 0.00 | ORB-long ORB[1762.00,1787.15] vol=1.5x ATR=6.92 |
| Stop hit — per-position SL triggered | 2024-05-31 10:55:00 | 1787.68 | 1783.12 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 11:05:00 | 1811.65 | 1827.67 | 0.00 | ORB-short ORB[1816.10,1838.00] vol=2.1x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:15:00 | 1803.54 | 1826.45 | 0.00 | T1 1.5R @ 1803.54 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 1811.65 | 1823.28 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 1841.10 | 1829.22 | 0.00 | ORB-long ORB[1812.00,1837.90] vol=2.0x ATR=9.01 |
| Stop hit — per-position SL triggered | 2024-06-10 09:50:00 | 1832.09 | 1834.99 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:15:00 | 1842.55 | 1836.53 | 0.00 | ORB-long ORB[1825.75,1842.05] vol=4.4x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:35:00 | 1850.71 | 1838.74 | 0.00 | T1 1.5R @ 1850.71 |
| Target hit | 2024-06-12 15:20:00 | 1853.10 | 1851.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:00:00 | 1883.85 | 1873.80 | 0.00 | ORB-long ORB[1858.05,1872.50] vol=6.0x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:05:00 | 1894.82 | 1876.86 | 0.00 | T1 1.5R @ 1894.82 |
| Target hit | 2024-06-13 11:10:00 | 1888.05 | 1888.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 1982.70 | 1970.74 | 0.00 | ORB-long ORB[1950.85,1974.90] vol=3.2x ATR=6.79 |
| Stop hit — per-position SL triggered | 2024-06-20 10:30:00 | 1975.91 | 1974.20 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:15:00 | 1989.80 | 2003.13 | 0.00 | ORB-short ORB[1991.85,2015.00] vol=3.2x ATR=9.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:40:00 | 1975.64 | 1995.42 | 0.00 | T1 1.5R @ 1975.64 |
| Stop hit — per-position SL triggered | 2024-06-27 12:25:00 | 1989.80 | 1990.64 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1975.15 | 1972.33 | 0.00 | ORB-long ORB[1951.00,1974.35] vol=1.7x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:45:00 | 1983.65 | 1973.54 | 0.00 | T1 1.5R @ 1983.65 |
| Target hit | 2024-07-04 10:45:00 | 1971.90 | 1973.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 2079.80 | 2069.21 | 0.00 | ORB-long ORB[2049.95,2074.70] vol=1.8x ATR=10.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:35:00 | 2095.75 | 2079.37 | 0.00 | T1 1.5R @ 2095.75 |
| Stop hit — per-position SL triggered | 2024-07-05 09:40:00 | 2079.80 | 2079.41 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:50:00 | 2146.20 | 2136.62 | 0.00 | ORB-long ORB[2104.05,2132.40] vol=1.8x ATR=7.72 |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 2138.48 | 2138.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:15:00 | 2076.00 | 2096.47 | 0.00 | ORB-short ORB[2091.25,2118.10] vol=1.6x ATR=8.91 |
| Stop hit — per-position SL triggered | 2024-07-19 10:20:00 | 2084.91 | 2095.37 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 1892.30 | 1902.46 | 0.00 | ORB-short ORB[1916.00,1934.05] vol=2.8x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 11:00:00 | 1881.71 | 1900.74 | 0.00 | T1 1.5R @ 1881.71 |
| Target hit | 2024-08-14 15:20:00 | 1875.70 | 1887.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 1979.80 | 1967.59 | 0.00 | ORB-long ORB[1950.10,1977.60] vol=2.8x ATR=9.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:10:00 | 1993.83 | 1973.71 | 0.00 | T1 1.5R @ 1993.83 |
| Target hit | 2024-08-19 15:20:00 | 2129.70 | 2083.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:40:00 | 2347.00 | 2325.94 | 0.00 | ORB-long ORB[2300.05,2324.80] vol=6.7x ATR=9.15 |
| Stop hit — per-position SL triggered | 2024-08-30 09:50:00 | 2337.85 | 2330.15 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:40:00 | 2379.90 | 2367.28 | 0.00 | ORB-long ORB[2345.05,2374.10] vol=1.6x ATR=10.63 |
| Stop hit — per-position SL triggered | 2024-09-02 11:40:00 | 2369.27 | 2374.90 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 2365.75 | 2346.50 | 0.00 | ORB-long ORB[2331.00,2360.00] vol=3.2x ATR=10.56 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 2355.19 | 2347.37 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:40:00 | 2327.45 | 2311.48 | 0.00 | ORB-long ORB[2290.00,2316.25] vol=1.7x ATR=9.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:05:00 | 2342.16 | 2313.95 | 0.00 | T1 1.5R @ 2342.16 |
| Stop hit — per-position SL triggered | 2024-10-08 11:35:00 | 2327.45 | 2318.44 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:10:00 | 2458.90 | 2475.70 | 0.00 | ORB-short ORB[2465.00,2495.95] vol=3.6x ATR=9.96 |
| Stop hit — per-position SL triggered | 2024-10-21 12:35:00 | 2468.86 | 2473.34 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:00:00 | 2652.00 | 2680.29 | 0.00 | ORB-short ORB[2671.10,2711.00] vol=1.6x ATR=14.54 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 2666.54 | 2678.82 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:15:00 | 2507.50 | 2519.78 | 0.00 | ORB-short ORB[2507.75,2540.20] vol=3.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-01-21 11:20:00 | 2513.28 | 2519.64 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:55:00 | 2396.80 | 2380.14 | 0.00 | ORB-long ORB[2366.00,2396.00] vol=2.3x ATR=13.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:00:00 | 2416.78 | 2395.27 | 0.00 | T1 1.5R @ 2416.78 |
| Target hit | 2025-01-23 12:05:00 | 2402.00 | 2403.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2025-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:45:00 | 2340.05 | 2322.22 | 0.00 | ORB-long ORB[2302.70,2323.85] vol=3.3x ATR=12.64 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 2327.41 | 2325.87 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-02-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:10:00 | 2327.20 | 2332.00 | 0.00 | ORB-short ORB[2338.55,2360.00] vol=2.7x ATR=10.71 |
| Stop hit — per-position SL triggered | 2025-02-01 10:40:00 | 2337.91 | 2331.97 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 10:25:00 | 2157.85 | 2172.52 | 0.00 | ORB-short ORB[2163.40,2196.05] vol=3.2x ATR=12.91 |
| Stop hit — per-position SL triggered | 2025-02-20 10:35:00 | 2170.76 | 2171.29 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 09:35:00 | 2230.00 | 2243.11 | 0.00 | ORB-short ORB[2235.00,2258.45] vol=1.9x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:25:00 | 2219.37 | 2233.93 | 0.00 | T1 1.5R @ 2219.37 |
| Stop hit — per-position SL triggered | 2025-03-19 11:55:00 | 2230.00 | 2227.34 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 2185.55 | 2206.99 | 0.00 | ORB-short ORB[2205.05,2228.50] vol=1.6x ATR=10.13 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 2195.68 | 2200.01 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:45:00 | 2269.70 | 2252.63 | 0.00 | ORB-long ORB[2239.60,2265.00] vol=1.7x ATR=7.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 12:05:00 | 2281.38 | 2266.47 | 0.00 | T1 1.5R @ 2281.38 |
| Target hit | 2025-04-16 12:55:00 | 2307.70 | 2314.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 2436.80 | 2484.93 | 0.00 | ORB-short ORB[2503.10,2534.70] vol=2.8x ATR=13.53 |
| Stop hit — per-position SL triggered | 2025-04-25 10:05:00 | 2450.33 | 2477.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-23 11:05:00 | 1805.95 | 2024-05-23 11:40:00 | 1816.60 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-05-23 11:05:00 | 1805.95 | 2024-05-23 12:10:00 | 1805.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1843.40 | 2024-05-27 09:35:00 | 1855.19 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1843.40 | 2024-05-27 09:40:00 | 1843.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-31 10:45:00 | 1794.60 | 2024-05-31 10:55:00 | 1787.68 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-06-07 11:05:00 | 1811.65 | 2024-06-07 11:15:00 | 1803.54 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-07 11:05:00 | 1811.65 | 2024-06-07 11:25:00 | 1811.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-10 09:30:00 | 1841.10 | 2024-06-10 09:50:00 | 1832.09 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-12 11:15:00 | 1842.55 | 2024-06-12 11:35:00 | 1850.71 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-12 11:15:00 | 1842.55 | 2024-06-12 15:20:00 | 1853.10 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-13 10:00:00 | 1883.85 | 2024-06-13 10:05:00 | 1894.82 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-13 10:00:00 | 1883.85 | 2024-06-13 11:10:00 | 1888.05 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-06-20 10:10:00 | 1982.70 | 2024-06-20 10:30:00 | 1975.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-27 10:15:00 | 1989.80 | 2024-06-27 10:40:00 | 1975.64 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-06-27 10:15:00 | 1989.80 | 2024-06-27 12:25:00 | 1989.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:30:00 | 1975.15 | 2024-07-04 10:45:00 | 1983.65 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-04 10:30:00 | 1975.15 | 2024-07-04 10:45:00 | 1971.90 | TARGET_HIT | 0.50 | -0.16% |
| BUY | retest1 | 2024-07-05 09:30:00 | 2079.80 | 2024-07-05 09:35:00 | 2095.75 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-07-05 09:30:00 | 2079.80 | 2024-07-05 09:40:00 | 2079.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:50:00 | 2146.20 | 2024-07-16 11:15:00 | 2138.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-19 10:15:00 | 2076.00 | 2024-07-19 10:20:00 | 2084.91 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-14 10:50:00 | 1892.30 | 2024-08-14 11:00:00 | 1881.71 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-08-14 10:50:00 | 1892.30 | 2024-08-14 15:20:00 | 1875.70 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2024-08-19 09:40:00 | 1979.80 | 2024-08-19 10:10:00 | 1993.83 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-08-19 09:40:00 | 1979.80 | 2024-08-19 15:20:00 | 2129.70 | TARGET_HIT | 0.50 | 7.57% |
| BUY | retest1 | 2024-08-30 09:40:00 | 2347.00 | 2024-08-30 09:50:00 | 2337.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-02 09:40:00 | 2379.90 | 2024-09-02 11:40:00 | 2369.27 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-24 10:55:00 | 2365.75 | 2024-09-24 11:05:00 | 2355.19 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-08 10:40:00 | 2327.45 | 2024-10-08 11:05:00 | 2342.16 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-10-08 10:40:00 | 2327.45 | 2024-10-08 11:35:00 | 2327.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 11:10:00 | 2458.90 | 2024-10-21 12:35:00 | 2468.86 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-10 10:00:00 | 2652.00 | 2025-01-10 10:05:00 | 2666.54 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-01-21 11:15:00 | 2507.50 | 2025-01-21 11:20:00 | 2513.28 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-23 10:55:00 | 2396.80 | 2025-01-23 11:00:00 | 2416.78 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-01-23 10:55:00 | 2396.80 | 2025-01-23 12:05:00 | 2402.00 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-01-31 09:45:00 | 2340.05 | 2025-01-31 10:05:00 | 2327.41 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-02-01 10:10:00 | 2327.20 | 2025-02-01 10:40:00 | 2337.91 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-02-20 10:25:00 | 2157.85 | 2025-02-20 10:35:00 | 2170.76 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2025-03-19 09:35:00 | 2230.00 | 2025-03-19 10:25:00 | 2219.37 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-03-19 09:35:00 | 2230.00 | 2025-03-19 11:55:00 | 2230.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-26 09:40:00 | 2185.55 | 2025-03-26 09:55:00 | 2195.68 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-16 10:45:00 | 2269.70 | 2025-04-16 12:05:00 | 2281.38 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-04-16 10:45:00 | 2269.70 | 2025-04-16 12:55:00 | 2307.70 | TARGET_HIT | 0.50 | 1.67% |
| SELL | retest1 | 2025-04-25 09:50:00 | 2436.80 | 2025-04-25 10:05:00 | 2450.33 | STOP_HIT | 1.00 | -0.56% |
