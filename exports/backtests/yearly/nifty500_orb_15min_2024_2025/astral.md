# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1567.40
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 43 |
| TARGET_HIT | 20 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 65
- **Target hits / Stop hits / Partials:** 20 / 65 / 43
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 22.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 29 | 46.8% | 8 | 33 | 21 | 0.17% | 10.8% |
| BUY @ 2nd Alert (retest1) | 62 | 29 | 46.8% | 8 | 33 | 21 | 0.17% | 10.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 34 | 51.5% | 12 | 32 | 22 | 0.17% | 11.3% |
| SELL @ 2nd Alert (retest1) | 66 | 34 | 51.5% | 12 | 32 | 22 | 0.17% | 11.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 63 | 49.2% | 20 | 65 | 43 | 0.17% | 22.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:50:00 | 2230.00 | 2227.00 | 0.00 | ORB-long ORB[2209.90,2229.00] vol=2.2x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:05:00 | 2240.44 | 2229.24 | 0.00 | T1 1.5R @ 2240.44 |
| Target hit | 2024-05-16 11:15:00 | 2245.10 | 2247.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:45:00 | 2301.30 | 2284.36 | 0.00 | ORB-long ORB[2266.40,2293.00] vol=2.0x ATR=10.60 |
| Stop hit — per-position SL triggered | 2024-05-17 09:55:00 | 2290.70 | 2288.95 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:55:00 | 2113.60 | 2095.70 | 0.00 | ORB-long ORB[2078.30,2103.90] vol=2.0x ATR=8.70 |
| Stop hit — per-position SL triggered | 2024-05-22 12:05:00 | 2104.90 | 2100.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 2130.80 | 2123.58 | 0.00 | ORB-long ORB[2106.05,2127.50] vol=3.7x ATR=8.21 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 2122.59 | 2124.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:55:00 | 2169.45 | 2160.80 | 0.00 | ORB-long ORB[2142.00,2168.00] vol=1.7x ATR=9.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:10:00 | 2183.53 | 2164.26 | 0.00 | T1 1.5R @ 2183.53 |
| Target hit | 2024-05-27 12:00:00 | 2171.15 | 2171.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-05-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:25:00 | 2100.95 | 2109.57 | 0.00 | ORB-short ORB[2101.40,2123.00] vol=1.6x ATR=7.01 |
| Stop hit — per-position SL triggered | 2024-05-30 10:55:00 | 2107.96 | 2109.16 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:15:00 | 2163.90 | 2160.10 | 0.00 | ORB-long ORB[2147.75,2163.75] vol=2.8x ATR=4.83 |
| Stop hit — per-position SL triggered | 2024-06-07 12:05:00 | 2159.07 | 2160.45 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 2158.20 | 2170.24 | 0.00 | ORB-short ORB[2158.80,2185.00] vol=1.9x ATR=8.70 |
| Stop hit — per-position SL triggered | 2024-06-10 09:35:00 | 2166.90 | 2169.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:40:00 | 2211.90 | 2191.08 | 0.00 | ORB-long ORB[2150.00,2172.45] vol=1.8x ATR=9.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:35:00 | 2225.89 | 2200.94 | 0.00 | T1 1.5R @ 2225.89 |
| Stop hit — per-position SL triggered | 2024-06-11 12:00:00 | 2211.90 | 2204.57 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:55:00 | 2247.50 | 2238.63 | 0.00 | ORB-long ORB[2233.25,2247.45] vol=2.5x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-06-13 11:10:00 | 2242.32 | 2239.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 2288.80 | 2267.66 | 0.00 | ORB-long ORB[2242.45,2272.00] vol=4.3x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-06-14 09:35:00 | 2281.74 | 2270.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:35:00 | 2295.10 | 2281.43 | 0.00 | ORB-long ORB[2256.00,2287.95] vol=1.9x ATR=7.40 |
| Stop hit — per-position SL triggered | 2024-06-18 10:40:00 | 2287.70 | 2282.55 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 11:15:00 | 2216.05 | 2222.04 | 0.00 | ORB-short ORB[2221.45,2250.95] vol=2.1x ATR=6.70 |
| Stop hit — per-position SL triggered | 2024-06-19 12:05:00 | 2222.75 | 2221.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:05:00 | 2331.80 | 2291.85 | 0.00 | ORB-long ORB[2243.35,2278.05] vol=1.9x ATR=11.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:50:00 | 2348.67 | 2306.20 | 0.00 | T1 1.5R @ 2348.67 |
| Stop hit — per-position SL triggered | 2024-06-24 12:05:00 | 2331.80 | 2308.92 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 2420.90 | 2410.60 | 0.00 | ORB-long ORB[2390.00,2418.40] vol=3.4x ATR=8.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:35:00 | 2434.01 | 2417.12 | 0.00 | T1 1.5R @ 2434.01 |
| Stop hit — per-position SL triggered | 2024-06-26 10:10:00 | 2420.90 | 2430.11 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:00:00 | 2422.00 | 2400.07 | 0.00 | ORB-long ORB[2377.35,2396.60] vol=1.6x ATR=8.38 |
| Stop hit — per-position SL triggered | 2024-06-27 10:30:00 | 2413.62 | 2404.19 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:50:00 | 2401.15 | 2410.21 | 0.00 | ORB-short ORB[2408.00,2425.00] vol=2.0x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:10:00 | 2392.99 | 2408.06 | 0.00 | T1 1.5R @ 2392.99 |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 2401.15 | 2407.22 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:30:00 | 2385.25 | 2398.12 | 0.00 | ORB-short ORB[2399.35,2414.20] vol=2.2x ATR=9.27 |
| Stop hit — per-position SL triggered | 2024-07-08 10:35:00 | 2394.52 | 2397.67 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 2332.65 | 2339.91 | 0.00 | ORB-short ORB[2335.30,2352.00] vol=1.5x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 2323.45 | 2336.38 | 0.00 | T1 1.5R @ 2323.45 |
| Target hit | 2024-07-10 11:35:00 | 2325.45 | 2323.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2024-07-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:20:00 | 2325.25 | 2341.65 | 0.00 | ORB-short ORB[2336.00,2354.15] vol=1.7x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:15:00 | 2315.91 | 2333.11 | 0.00 | T1 1.5R @ 2315.91 |
| Stop hit — per-position SL triggered | 2024-07-11 11:55:00 | 2325.25 | 2329.89 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:15:00 | 2321.20 | 2326.44 | 0.00 | ORB-short ORB[2324.10,2346.45] vol=7.4x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 12:25:00 | 2312.94 | 2324.69 | 0.00 | T1 1.5R @ 2312.94 |
| Target hit | 2024-07-12 15:20:00 | 2300.30 | 2311.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 2268.85 | 2274.60 | 0.00 | ORB-short ORB[2273.00,2300.00] vol=3.0x ATR=6.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:00:00 | 2258.76 | 2273.58 | 0.00 | T1 1.5R @ 2258.76 |
| Target hit | 2024-07-15 14:40:00 | 2263.40 | 2262.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — SELL (started 2024-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:10:00 | 2242.35 | 2256.01 | 0.00 | ORB-short ORB[2249.55,2276.95] vol=2.1x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:45:00 | 2232.87 | 2251.84 | 0.00 | T1 1.5R @ 2232.87 |
| Stop hit — per-position SL triggered | 2024-07-18 12:05:00 | 2242.35 | 2250.74 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:40:00 | 2223.70 | 2236.25 | 0.00 | ORB-short ORB[2232.15,2255.70] vol=2.2x ATR=6.61 |
| Stop hit — per-position SL triggered | 2024-07-19 11:05:00 | 2230.31 | 2233.92 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 2231.55 | 2247.55 | 0.00 | ORB-short ORB[2250.00,2274.00] vol=6.1x ATR=7.56 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 2239.11 | 2247.12 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 2120.00 | 2111.09 | 0.00 | ORB-long ORB[2098.00,2114.50] vol=2.4x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:35:00 | 2129.90 | 2116.39 | 0.00 | T1 1.5R @ 2129.90 |
| Stop hit — per-position SL triggered | 2024-08-07 13:30:00 | 2120.00 | 2119.40 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:35:00 | 1869.70 | 1874.96 | 0.00 | ORB-short ORB[1873.65,1892.65] vol=2.1x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-08-16 10:45:00 | 1875.17 | 1874.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:40:00 | 1878.00 | 1890.21 | 0.00 | ORB-short ORB[1882.95,1905.60] vol=1.8x ATR=6.29 |
| Stop hit — per-position SL triggered | 2024-08-19 11:25:00 | 1884.29 | 1888.00 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 1889.70 | 1892.72 | 0.00 | ORB-short ORB[1890.00,1908.85] vol=2.0x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:10:00 | 1884.27 | 1892.47 | 0.00 | T1 1.5R @ 1884.27 |
| Stop hit — per-position SL triggered | 2024-08-21 11:50:00 | 1889.70 | 1891.71 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:50:00 | 1921.75 | 1913.83 | 0.00 | ORB-long ORB[1895.05,1919.70] vol=2.2x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:15:00 | 1929.56 | 1919.50 | 0.00 | T1 1.5R @ 1929.56 |
| Target hit | 2024-08-22 15:20:00 | 1959.70 | 1938.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2024-09-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:45:00 | 1919.75 | 1921.21 | 0.00 | ORB-short ORB[1920.00,1935.00] vol=1.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-09-02 10:50:00 | 1923.95 | 1921.36 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:15:00 | 1923.50 | 1932.84 | 0.00 | ORB-short ORB[1932.00,1947.00] vol=2.7x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-09-05 11:45:00 | 1926.88 | 1931.55 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:00:00 | 1914.15 | 1923.21 | 0.00 | ORB-short ORB[1931.00,1939.90] vol=2.2x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:15:00 | 1906.88 | 1921.60 | 0.00 | T1 1.5R @ 1906.88 |
| Stop hit — per-position SL triggered | 2024-09-06 12:10:00 | 1914.15 | 1918.10 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:10:00 | 1938.05 | 1928.26 | 0.00 | ORB-long ORB[1919.95,1931.80] vol=6.0x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 12:45:00 | 1943.88 | 1931.74 | 0.00 | T1 1.5R @ 1943.88 |
| Target hit | 2024-09-10 15:20:00 | 1942.20 | 1939.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-09-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:30:00 | 1917.90 | 1933.34 | 0.00 | ORB-short ORB[1928.05,1940.80] vol=1.6x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:50:00 | 1910.91 | 1928.65 | 0.00 | T1 1.5R @ 1910.91 |
| Target hit | 2024-09-16 15:20:00 | 1906.25 | 1910.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:50:00 | 1911.60 | 1910.65 | 0.00 | ORB-long ORB[1902.90,1911.40] vol=1.6x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-09-18 10:00:00 | 1907.95 | 1910.75 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:00:00 | 1927.45 | 1917.14 | 0.00 | ORB-long ORB[1903.00,1923.00] vol=2.8x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:05:00 | 1937.65 | 1924.90 | 0.00 | T1 1.5R @ 1937.65 |
| Stop hit — per-position SL triggered | 2024-09-20 13:25:00 | 1927.45 | 1929.12 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 2000.80 | 1989.72 | 0.00 | ORB-long ORB[1973.95,1988.70] vol=3.3x ATR=5.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:25:00 | 2009.48 | 1997.28 | 0.00 | T1 1.5R @ 2009.48 |
| Stop hit — per-position SL triggered | 2024-09-24 11:20:00 | 2000.80 | 2001.17 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:35:00 | 2018.35 | 2002.09 | 0.00 | ORB-long ORB[1986.15,2011.15] vol=1.8x ATR=6.09 |
| Stop hit — per-position SL triggered | 2024-09-27 09:45:00 | 2012.26 | 2006.36 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:40:00 | 1978.35 | 1991.79 | 0.00 | ORB-short ORB[1988.00,2013.50] vol=1.8x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-09-30 10:00:00 | 1984.16 | 1987.68 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 1870.45 | 1893.85 | 0.00 | ORB-short ORB[1893.80,1907.55] vol=1.5x ATR=7.29 |
| Stop hit — per-position SL triggered | 2024-10-07 10:40:00 | 1877.74 | 1892.37 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:20:00 | 1899.00 | 1900.45 | 0.00 | ORB-short ORB[1902.05,1915.00] vol=5.8x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:20:00 | 1892.24 | 1898.96 | 0.00 | T1 1.5R @ 1892.24 |
| Stop hit — per-position SL triggered | 2024-10-10 12:00:00 | 1899.00 | 1898.82 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:50:00 | 1892.35 | 1897.91 | 0.00 | ORB-short ORB[1895.45,1907.10] vol=1.6x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:35:00 | 1887.41 | 1895.87 | 0.00 | T1 1.5R @ 1887.41 |
| Stop hit — per-position SL triggered | 2024-10-16 14:25:00 | 1892.35 | 1890.93 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:30:00 | 1870.00 | 1879.71 | 0.00 | ORB-short ORB[1880.00,1890.60] vol=1.5x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:50:00 | 1863.61 | 1875.98 | 0.00 | T1 1.5R @ 1863.61 |
| Target hit | 2024-10-17 14:50:00 | 1863.45 | 1862.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 1791.00 | 1797.13 | 0.00 | ORB-short ORB[1795.00,1804.25] vol=2.2x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-10-29 09:40:00 | 1796.27 | 1796.18 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:50:00 | 1783.55 | 1783.07 | 0.00 | ORB-long ORB[1768.55,1782.50] vol=5.5x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:30:00 | 1789.88 | 1784.19 | 0.00 | T1 1.5R @ 1789.88 |
| Target hit | 2024-11-06 15:20:00 | 1810.10 | 1791.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 1730.60 | 1722.98 | 0.00 | ORB-long ORB[1711.85,1729.80] vol=2.9x ATR=6.45 |
| Stop hit — per-position SL triggered | 2024-11-12 09:40:00 | 1724.15 | 1724.04 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:50:00 | 1727.55 | 1714.51 | 0.00 | ORB-long ORB[1708.00,1725.00] vol=2.5x ATR=5.15 |
| Stop hit — per-position SL triggered | 2024-11-21 11:00:00 | 1722.40 | 1716.21 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:00:00 | 1792.90 | 1783.58 | 0.00 | ORB-long ORB[1772.55,1785.65] vol=1.5x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 10:05:00 | 1801.05 | 1788.47 | 0.00 | T1 1.5R @ 1801.05 |
| Stop hit — per-position SL triggered | 2024-11-25 12:00:00 | 1792.90 | 1795.81 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 1825.05 | 1811.19 | 0.00 | ORB-long ORB[1792.95,1810.00] vol=1.7x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:50:00 | 1833.35 | 1814.19 | 0.00 | T1 1.5R @ 1833.35 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 1825.05 | 1825.82 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:15:00 | 1802.95 | 1791.11 | 0.00 | ORB-long ORB[1771.25,1793.15] vol=1.8x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:40:00 | 1809.71 | 1794.82 | 0.00 | T1 1.5R @ 1809.71 |
| Target hit | 2024-12-02 15:20:00 | 1815.50 | 1801.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 1839.90 | 1830.44 | 0.00 | ORB-long ORB[1822.60,1832.85] vol=2.2x ATR=4.93 |
| Stop hit — per-position SL triggered | 2024-12-03 09:35:00 | 1834.97 | 1832.66 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:05:00 | 1840.00 | 1848.21 | 0.00 | ORB-short ORB[1848.00,1858.85] vol=4.0x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-12-04 11:10:00 | 1843.82 | 1847.29 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:40:00 | 1852.00 | 1848.91 | 0.00 | ORB-long ORB[1842.90,1851.45] vol=4.7x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 13:25:00 | 1857.11 | 1850.32 | 0.00 | T1 1.5R @ 1857.11 |
| Stop hit — per-position SL triggered | 2024-12-11 14:50:00 | 1852.00 | 1851.89 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:55:00 | 1841.85 | 1844.67 | 0.00 | ORB-short ORB[1846.80,1857.00] vol=15.9x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-12-12 11:05:00 | 1845.13 | 1844.69 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 1797.30 | 1809.29 | 0.00 | ORB-short ORB[1811.65,1829.10] vol=1.8x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:35:00 | 1789.88 | 1805.34 | 0.00 | T1 1.5R @ 1789.88 |
| Stop hit — per-position SL triggered | 2024-12-13 11:00:00 | 1797.30 | 1803.79 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:15:00 | 1828.55 | 1834.17 | 0.00 | ORB-short ORB[1830.00,1846.50] vol=2.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-12-16 11:35:00 | 1832.35 | 1833.37 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 1740.80 | 1728.69 | 0.00 | ORB-long ORB[1716.80,1728.90] vol=2.8x ATR=4.82 |
| Stop hit — per-position SL triggered | 2024-12-24 11:00:00 | 1735.98 | 1735.70 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:50:00 | 1668.50 | 1672.60 | 0.00 | ORB-short ORB[1670.00,1679.00] vol=2.1x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-12-27 11:05:00 | 1671.83 | 1671.35 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 10:25:00 | 1649.90 | 1654.07 | 0.00 | ORB-short ORB[1649.95,1662.05] vol=2.2x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 10:35:00 | 1642.19 | 1651.81 | 0.00 | T1 1.5R @ 1642.19 |
| Target hit | 2024-12-31 12:00:00 | 1644.45 | 1643.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 1648.00 | 1650.97 | 0.00 | ORB-short ORB[1648.85,1657.75] vol=1.8x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:50:00 | 1641.96 | 1649.00 | 0.00 | T1 1.5R @ 1641.96 |
| Stop hit — per-position SL triggered | 2025-01-01 10:00:00 | 1648.00 | 1648.65 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:45:00 | 1644.05 | 1648.69 | 0.00 | ORB-short ORB[1645.50,1657.95] vol=1.7x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 12:10:00 | 1639.12 | 1643.82 | 0.00 | T1 1.5R @ 1639.12 |
| Target hit | 2025-01-02 13:05:00 | 1643.45 | 1643.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — SELL (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 1641.05 | 1644.99 | 0.00 | ORB-short ORB[1642.30,1654.95] vol=1.8x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:10:00 | 1635.98 | 1643.83 | 0.00 | T1 1.5R @ 1635.98 |
| Target hit | 2025-01-03 10:55:00 | 1639.85 | 1638.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 1598.95 | 1616.88 | 0.00 | ORB-short ORB[1615.00,1628.00] vol=2.6x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:05:00 | 1592.55 | 1612.97 | 0.00 | T1 1.5R @ 1592.55 |
| Target hit | 2025-01-06 15:20:00 | 1558.70 | 1581.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-01-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:05:00 | 1560.85 | 1553.52 | 0.00 | ORB-long ORB[1546.15,1558.00] vol=3.7x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:25:00 | 1568.62 | 1556.57 | 0.00 | T1 1.5R @ 1568.62 |
| Target hit | 2025-01-10 14:35:00 | 1562.45 | 1563.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — SELL (started 2025-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:30:00 | 1528.05 | 1537.19 | 0.00 | ORB-short ORB[1531.10,1550.00] vol=1.9x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:35:00 | 1518.66 | 1533.29 | 0.00 | T1 1.5R @ 1518.66 |
| Stop hit — per-position SL triggered | 2025-01-13 09:45:00 | 1528.05 | 1531.63 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:00:00 | 1515.10 | 1506.63 | 0.00 | ORB-long ORB[1492.05,1506.55] vol=2.0x ATR=4.11 |
| Stop hit — per-position SL triggered | 2025-01-17 10:30:00 | 1510.99 | 1508.41 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1479.00 | 1490.74 | 0.00 | ORB-short ORB[1483.00,1497.20] vol=1.5x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-01-21 11:00:00 | 1483.52 | 1486.06 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:00:00 | 1473.30 | 1466.52 | 0.00 | ORB-long ORB[1447.10,1468.25] vol=5.4x ATR=5.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:35:00 | 1481.97 | 1469.81 | 0.00 | T1 1.5R @ 1481.97 |
| Stop hit — per-position SL triggered | 2025-01-23 11:05:00 | 1473.30 | 1470.35 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:45:00 | 1468.00 | 1480.98 | 0.00 | ORB-short ORB[1490.95,1504.45] vol=2.1x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-01-24 11:40:00 | 1472.90 | 1477.83 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:55:00 | 1492.45 | 1483.30 | 0.00 | ORB-long ORB[1474.85,1485.90] vol=2.2x ATR=3.97 |
| Stop hit — per-position SL triggered | 2025-01-30 10:15:00 | 1488.48 | 1486.15 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:45:00 | 1500.35 | 1503.47 | 0.00 | ORB-short ORB[1506.05,1522.95] vol=2.0x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:50:00 | 1493.43 | 1500.49 | 0.00 | T1 1.5R @ 1493.43 |
| Target hit | 2025-02-06 15:20:00 | 1479.00 | 1488.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2025-02-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:05:00 | 1346.50 | 1362.09 | 0.00 | ORB-short ORB[1360.15,1375.40] vol=1.9x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 1350.99 | 1361.59 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:45:00 | 1379.95 | 1374.97 | 0.00 | ORB-long ORB[1363.80,1378.45] vol=1.8x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:00:00 | 1386.58 | 1376.01 | 0.00 | T1 1.5R @ 1386.58 |
| Target hit | 2025-02-20 15:20:00 | 1400.65 | 1389.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2025-02-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:25:00 | 1379.00 | 1396.27 | 0.00 | ORB-short ORB[1394.95,1414.00] vol=1.7x ATR=5.78 |
| Stop hit — per-position SL triggered | 2025-02-21 10:30:00 | 1384.78 | 1395.27 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 1317.00 | 1324.28 | 0.00 | ORB-short ORB[1319.95,1337.55] vol=1.5x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-03-06 09:35:00 | 1321.60 | 1323.91 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 1280.00 | 1270.06 | 0.00 | ORB-long ORB[1260.00,1273.45] vol=2.2x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-03-20 09:40:00 | 1276.23 | 1272.91 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:10:00 | 1328.90 | 1319.24 | 0.00 | ORB-long ORB[1301.00,1318.00] vol=1.5x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:25:00 | 1336.72 | 1322.39 | 0.00 | T1 1.5R @ 1336.72 |
| Stop hit — per-position SL triggered | 2025-03-24 11:50:00 | 1328.90 | 1329.20 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 10:05:00 | 1303.90 | 1301.57 | 0.00 | ORB-long ORB[1281.00,1293.45] vol=2.0x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 10:15:00 | 1311.63 | 1302.53 | 0.00 | T1 1.5R @ 1311.63 |
| Stop hit — per-position SL triggered | 2025-04-01 10:35:00 | 1303.90 | 1303.92 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-04-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 10:00:00 | 1342.05 | 1348.05 | 0.00 | ORB-short ORB[1343.10,1358.00] vol=1.7x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:30:00 | 1334.98 | 1345.52 | 0.00 | T1 1.5R @ 1334.98 |
| Target hit | 2025-04-04 15:05:00 | 1338.60 | 1338.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 81 — BUY (started 2025-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:05:00 | 1328.90 | 1317.31 | 0.00 | ORB-long ORB[1306.00,1316.80] vol=1.8x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-04-21 10:15:00 | 1324.69 | 1320.04 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1359.00 | 1351.47 | 0.00 | ORB-long ORB[1338.00,1354.60] vol=2.9x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:50:00 | 1365.48 | 1355.25 | 0.00 | T1 1.5R @ 1365.48 |
| Stop hit — per-position SL triggered | 2025-04-22 09:55:00 | 1359.00 | 1355.61 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 1366.30 | 1359.36 | 0.00 | ORB-long ORB[1348.80,1365.60] vol=1.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-04-30 10:40:00 | 1361.02 | 1365.06 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:45:00 | 1323.70 | 1333.19 | 0.00 | ORB-short ORB[1330.70,1343.70] vol=2.0x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:20:00 | 1317.01 | 1326.08 | 0.00 | T1 1.5R @ 1317.01 |
| Target hit | 2025-05-06 15:20:00 | 1295.90 | 1312.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — BUY (started 2025-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:05:00 | 1315.60 | 1305.85 | 0.00 | ORB-long ORB[1296.20,1309.50] vol=3.3x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-05-08 11:50:00 | 1311.50 | 1307.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:50:00 | 2230.00 | 2024-05-16 10:05:00 | 2240.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-16 09:50:00 | 2230.00 | 2024-05-16 11:15:00 | 2245.10 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-05-17 09:45:00 | 2301.30 | 2024-05-17 09:55:00 | 2290.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-05-22 10:55:00 | 2113.60 | 2024-05-22 12:05:00 | 2104.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-23 09:40:00 | 2130.80 | 2024-05-23 09:50:00 | 2122.59 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-27 09:55:00 | 2169.45 | 2024-05-27 10:10:00 | 2183.53 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-05-27 09:55:00 | 2169.45 | 2024-05-27 12:00:00 | 2171.15 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-05-30 10:25:00 | 2100.95 | 2024-05-30 10:55:00 | 2107.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-07 11:15:00 | 2163.90 | 2024-06-07 12:05:00 | 2159.07 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-06-10 09:30:00 | 2158.20 | 2024-06-10 09:35:00 | 2166.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-11 10:40:00 | 2211.90 | 2024-06-11 11:35:00 | 2225.89 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-11 10:40:00 | 2211.90 | 2024-06-11 12:00:00 | 2211.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 10:55:00 | 2247.50 | 2024-06-13 11:10:00 | 2242.32 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-14 09:30:00 | 2288.80 | 2024-06-14 09:35:00 | 2281.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-18 10:35:00 | 2295.10 | 2024-06-18 10:40:00 | 2287.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-19 11:15:00 | 2216.05 | 2024-06-19 12:05:00 | 2222.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-24 11:05:00 | 2331.80 | 2024-06-24 11:50:00 | 2348.67 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-06-24 11:05:00 | 2331.80 | 2024-06-24 12:05:00 | 2331.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:30:00 | 2420.90 | 2024-06-26 09:35:00 | 2434.01 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-26 09:30:00 | 2420.90 | 2024-06-26 10:10:00 | 2420.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:00:00 | 2422.00 | 2024-06-27 10:30:00 | 2413.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-03 10:50:00 | 2401.15 | 2024-07-03 11:10:00 | 2392.99 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-07-03 10:50:00 | 2401.15 | 2024-07-03 11:15:00 | 2401.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 10:30:00 | 2385.25 | 2024-07-08 10:35:00 | 2394.52 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-10 09:45:00 | 2332.65 | 2024-07-10 10:05:00 | 2323.45 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 09:45:00 | 2332.65 | 2024-07-10 11:35:00 | 2325.45 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-07-11 10:20:00 | 2325.25 | 2024-07-11 11:15:00 | 2315.91 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-11 10:20:00 | 2325.25 | 2024-07-11 11:55:00 | 2325.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 11:15:00 | 2321.20 | 2024-07-12 12:25:00 | 2312.94 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-12 11:15:00 | 2321.20 | 2024-07-12 15:20:00 | 2300.30 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-07-15 10:50:00 | 2268.85 | 2024-07-15 11:00:00 | 2258.76 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-15 10:50:00 | 2268.85 | 2024-07-15 14:40:00 | 2263.40 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-07-18 11:10:00 | 2242.35 | 2024-07-18 11:45:00 | 2232.87 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-18 11:10:00 | 2242.35 | 2024-07-18 12:05:00 | 2242.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:40:00 | 2223.70 | 2024-07-19 11:05:00 | 2230.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-23 11:15:00 | 2231.55 | 2024-07-23 11:20:00 | 2239.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-07 10:25:00 | 2120.00 | 2024-08-07 11:35:00 | 2129.90 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-08-07 10:25:00 | 2120.00 | 2024-08-07 13:30:00 | 2120.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-16 10:35:00 | 1869.70 | 2024-08-16 10:45:00 | 1875.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-19 10:40:00 | 1878.00 | 2024-08-19 11:25:00 | 1884.29 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-21 11:00:00 | 1889.70 | 2024-08-21 11:10:00 | 1884.27 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-08-21 11:00:00 | 1889.70 | 2024-08-21 11:50:00 | 1889.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:50:00 | 1921.75 | 2024-08-22 10:15:00 | 1929.56 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-22 09:50:00 | 1921.75 | 2024-08-22 15:20:00 | 1959.70 | TARGET_HIT | 0.50 | 1.97% |
| SELL | retest1 | 2024-09-02 10:45:00 | 1919.75 | 2024-09-02 10:50:00 | 1923.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-05 11:15:00 | 1923.50 | 2024-09-05 11:45:00 | 1926.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-06 11:00:00 | 1914.15 | 2024-09-06 11:15:00 | 1906.88 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-06 11:00:00 | 1914.15 | 2024-09-06 12:10:00 | 1914.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 11:10:00 | 1938.05 | 2024-09-10 12:45:00 | 1943.88 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-09-10 11:10:00 | 1938.05 | 2024-09-10 15:20:00 | 1942.20 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2024-09-16 10:30:00 | 1917.90 | 2024-09-16 10:50:00 | 1910.91 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-16 10:30:00 | 1917.90 | 2024-09-16 15:20:00 | 1906.25 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-09-18 09:50:00 | 1911.60 | 2024-09-18 10:00:00 | 1907.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-20 10:00:00 | 1927.45 | 2024-09-20 11:05:00 | 1937.65 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-09-20 10:00:00 | 1927.45 | 2024-09-20 13:25:00 | 1927.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 09:50:00 | 2000.80 | 2024-09-24 10:25:00 | 2009.48 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-24 09:50:00 | 2000.80 | 2024-09-24 11:20:00 | 2000.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 09:35:00 | 2018.35 | 2024-09-27 09:45:00 | 2012.26 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-30 09:40:00 | 1978.35 | 2024-09-30 10:00:00 | 1984.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-07 10:35:00 | 1870.45 | 2024-10-07 10:40:00 | 1877.74 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-10 10:20:00 | 1899.00 | 2024-10-10 11:20:00 | 1892.24 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-10 10:20:00 | 1899.00 | 2024-10-10 12:00:00 | 1899.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 10:50:00 | 1892.35 | 2024-10-16 11:35:00 | 1887.41 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-10-16 10:50:00 | 1892.35 | 2024-10-16 14:25:00 | 1892.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:30:00 | 1870.00 | 2024-10-17 10:50:00 | 1863.61 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-17 10:30:00 | 1870.00 | 2024-10-17 14:50:00 | 1863.45 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-29 09:35:00 | 1791.00 | 2024-10-29 09:40:00 | 1796.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-06 10:50:00 | 1783.55 | 2024-11-06 11:30:00 | 1789.88 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-11-06 10:50:00 | 1783.55 | 2024-11-06 15:20:00 | 1810.10 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2024-11-12 09:30:00 | 1730.60 | 2024-11-12 09:40:00 | 1724.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-21 10:50:00 | 1727.55 | 2024-11-21 11:00:00 | 1722.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-25 10:00:00 | 1792.90 | 2024-11-25 10:05:00 | 1801.05 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-25 10:00:00 | 1792.90 | 2024-11-25 12:00:00 | 1792.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 09:45:00 | 1825.05 | 2024-11-28 09:50:00 | 1833.35 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-28 09:45:00 | 1825.05 | 2024-11-28 10:30:00 | 1825.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 11:15:00 | 1802.95 | 2024-12-02 11:40:00 | 1809.71 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-02 11:15:00 | 1802.95 | 2024-12-02 15:20:00 | 1815.50 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-03 09:30:00 | 1839.90 | 2024-12-03 09:35:00 | 1834.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-04 11:05:00 | 1840.00 | 2024-12-04 11:10:00 | 1843.82 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-11 10:40:00 | 1852.00 | 2024-12-11 13:25:00 | 1857.11 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-12-11 10:40:00 | 1852.00 | 2024-12-11 14:50:00 | 1852.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:55:00 | 1841.85 | 2024-12-12 11:05:00 | 1845.13 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-13 10:15:00 | 1797.30 | 2024-12-13 10:35:00 | 1789.88 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-13 10:15:00 | 1797.30 | 2024-12-13 11:00:00 | 1797.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:15:00 | 1828.55 | 2024-12-16 11:35:00 | 1832.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-24 09:50:00 | 1740.80 | 2024-12-24 11:00:00 | 1735.98 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-27 10:50:00 | 1668.50 | 2024-12-27 11:05:00 | 1671.83 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-31 10:25:00 | 1649.90 | 2024-12-31 10:35:00 | 1642.19 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-31 10:25:00 | 1649.90 | 2024-12-31 12:00:00 | 1644.45 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-01 09:30:00 | 1648.00 | 2025-01-01 09:50:00 | 1641.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-01-01 09:30:00 | 1648.00 | 2025-01-01 10:00:00 | 1648.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 09:45:00 | 1644.05 | 2025-01-02 12:10:00 | 1639.12 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-01-02 09:45:00 | 1644.05 | 2025-01-02 13:05:00 | 1643.45 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-01-03 10:05:00 | 1641.05 | 2025-01-03 10:10:00 | 1635.98 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-03 10:05:00 | 1641.05 | 2025-01-03 10:55:00 | 1639.85 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1598.95 | 2025-01-06 11:05:00 | 1592.55 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-01-06 10:55:00 | 1598.95 | 2025-01-06 15:20:00 | 1558.70 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2025-01-10 10:05:00 | 1560.85 | 2025-01-10 10:25:00 | 1568.62 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-10 10:05:00 | 1560.85 | 2025-01-10 14:35:00 | 1562.45 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-01-13 09:30:00 | 1528.05 | 2025-01-13 09:35:00 | 1518.66 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-13 09:30:00 | 1528.05 | 2025-01-13 09:45:00 | 1528.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 10:00:00 | 1515.10 | 2025-01-17 10:30:00 | 1510.99 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-21 10:15:00 | 1479.00 | 2025-01-21 11:00:00 | 1483.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-23 10:00:00 | 1473.30 | 2025-01-23 10:35:00 | 1481.97 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-01-23 10:00:00 | 1473.30 | 2025-01-23 11:05:00 | 1473.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 10:45:00 | 1468.00 | 2025-01-24 11:40:00 | 1472.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-30 09:55:00 | 1492.45 | 2025-01-30 10:15:00 | 1488.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-06 10:45:00 | 1500.35 | 2025-02-06 11:50:00 | 1493.43 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-06 10:45:00 | 1500.35 | 2025-02-06 15:20:00 | 1479.00 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2025-02-14 11:05:00 | 1346.50 | 2025-02-14 11:10:00 | 1350.99 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-20 10:45:00 | 1379.95 | 2025-02-20 11:00:00 | 1386.58 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-02-20 10:45:00 | 1379.95 | 2025-02-20 15:20:00 | 1400.65 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2025-02-21 10:25:00 | 1379.00 | 2025-02-21 10:30:00 | 1384.78 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-06 09:30:00 | 1317.00 | 2025-03-06 09:35:00 | 1321.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-20 09:30:00 | 1280.00 | 2025-03-20 09:40:00 | 1276.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-24 10:10:00 | 1328.90 | 2025-03-24 10:25:00 | 1336.72 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-24 10:10:00 | 1328.90 | 2025-03-24 11:50:00 | 1328.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-01 10:05:00 | 1303.90 | 2025-04-01 10:15:00 | 1311.63 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-01 10:05:00 | 1303.90 | 2025-04-01 10:35:00 | 1303.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-04 10:00:00 | 1342.05 | 2025-04-04 10:30:00 | 1334.98 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-04 10:00:00 | 1342.05 | 2025-04-04 15:05:00 | 1338.60 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-04-21 10:05:00 | 1328.90 | 2025-04-21 10:15:00 | 1324.69 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1359.00 | 2025-04-22 09:50:00 | 1365.48 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1359.00 | 2025-04-22 09:55:00 | 1359.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 09:30:00 | 1366.30 | 2025-04-30 10:40:00 | 1361.02 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-05-06 09:45:00 | 1323.70 | 2025-05-06 11:20:00 | 1317.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-05-06 09:45:00 | 1323.70 | 2025-05-06 15:20:00 | 1295.90 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2025-05-08 11:05:00 | 1315.60 | 2025-05-08 11:50:00 | 1311.50 | STOP_HIT | 1.00 | -0.31% |
