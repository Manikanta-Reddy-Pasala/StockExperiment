# Mankind Pharma Ltd. (MANKIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2423.00
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 16
- **Target hits / Stop hits / Partials:** 6 / 16 / 10
- **Avg / median % per leg:** 0.41% / 0.29%
- **Sum % (uncompounded):** 13.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 12 | 57.1% | 5 | 9 | 7 | 0.56% | 11.8% |
| BUY @ 2nd Alert (retest1) | 21 | 12 | 57.1% | 5 | 9 | 7 | 0.56% | 11.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.13% | 1.4% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.13% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 16 | 50.0% | 6 | 16 | 10 | 0.41% | 13.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:30:00 | 2082.20 | 2073.05 | 0.00 | ORB-long ORB[2051.70,2074.00] vol=3.1x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-02-16 10:35:00 | 2077.78 | 2073.23 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 2070.40 | 2079.44 | 0.00 | ORB-short ORB[2075.20,2088.30] vol=3.1x ATR=3.76 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 2074.16 | 2078.89 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:15:00 | 2054.00 | 2057.52 | 0.00 | ORB-short ORB[2055.00,2070.00] vol=1.9x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:20:00 | 2048.14 | 2056.67 | 0.00 | T1 1.5R @ 2048.14 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 2054.00 | 2056.14 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 2087.60 | 2080.27 | 0.00 | ORB-long ORB[2069.50,2085.00] vol=2.0x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-02-19 10:55:00 | 2081.61 | 2080.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:40:00 | 2055.60 | 2042.51 | 0.00 | ORB-long ORB[2030.70,2045.40] vol=2.1x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-02-24 11:50:00 | 2051.10 | 2047.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 2073.30 | 2064.62 | 0.00 | ORB-long ORB[2052.80,2067.50] vol=1.6x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:05:00 | 2080.73 | 2075.46 | 0.00 | T1 1.5R @ 2080.73 |
| Target hit | 2026-02-25 15:20:00 | 2156.20 | 2124.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 2183.00 | 2170.02 | 0.00 | ORB-long ORB[2145.90,2170.80] vol=2.0x ATR=10.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 2198.54 | 2183.47 | 0.00 | T1 1.5R @ 2198.54 |
| Target hit | 2026-02-26 12:25:00 | 2239.40 | 2242.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 2233.30 | 2230.37 | 0.00 | ORB-long ORB[2202.80,2228.00] vol=4.9x ATR=6.45 |
| Stop hit — per-position SL triggered | 2026-03-05 10:25:00 | 2226.85 | 2228.42 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2200.40 | 2208.28 | 0.00 | ORB-short ORB[2214.50,2231.70] vol=1.6x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:00:00 | 2191.77 | 2206.14 | 0.00 | T1 1.5R @ 2191.77 |
| Stop hit — per-position SL triggered | 2026-03-06 13:40:00 | 2200.40 | 2196.46 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:10:00 | 2118.00 | 2134.73 | 0.00 | ORB-short ORB[2131.00,2153.80] vol=2.9x ATR=8.34 |
| Stop hit — per-position SL triggered | 2026-03-09 10:35:00 | 2126.34 | 2127.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 2201.50 | 2194.01 | 0.00 | ORB-long ORB[2169.20,2200.00] vol=1.8x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:30:00 | 2213.44 | 2195.10 | 0.00 | T1 1.5R @ 2213.44 |
| Target hit | 2026-03-10 15:20:00 | 2243.30 | 2211.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 2186.70 | 2186.97 | 0.00 | ORB-short ORB[2189.10,2205.00] vol=12.1x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 2176.29 | 2186.84 | 0.00 | T1 1.5R @ 2176.29 |
| Target hit | 2026-03-13 15:20:00 | 2142.10 | 2170.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 2094.20 | 2113.70 | 0.00 | ORB-short ORB[2111.00,2141.60] vol=2.9x ATR=7.21 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 2101.41 | 2113.03 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 1934.30 | 1952.23 | 0.00 | ORB-short ORB[1961.50,1984.80] vol=2.2x ATR=8.95 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1943.25 | 1951.27 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 1920.40 | 1939.78 | 0.00 | ORB-short ORB[1929.00,1954.20] vol=1.9x ATR=7.30 |
| Stop hit — per-position SL triggered | 2026-03-24 11:10:00 | 1927.70 | 1939.07 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 2068.20 | 2060.49 | 0.00 | ORB-long ORB[2047.00,2067.40] vol=1.8x ATR=5.97 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 2062.23 | 2061.56 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 2119.00 | 2109.14 | 0.00 | ORB-long ORB[2085.60,2114.80] vol=2.4x ATR=7.22 |
| Stop hit — per-position SL triggered | 2026-04-15 10:10:00 | 2111.78 | 2112.67 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 2120.00 | 2118.13 | 0.00 | ORB-long ORB[2102.80,2116.80] vol=2.7x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:20:00 | 2128.74 | 2119.06 | 0.00 | T1 1.5R @ 2128.74 |
| Stop hit — per-position SL triggered | 2026-04-17 10:40:00 | 2120.00 | 2121.70 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:15:00 | 2186.00 | 2166.43 | 0.00 | ORB-long ORB[2137.00,2162.00] vol=2.5x ATR=6.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:50:00 | 2195.66 | 2174.49 | 0.00 | T1 1.5R @ 2195.66 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 2186.00 | 2177.51 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 2209.00 | 2198.51 | 0.00 | ORB-long ORB[2181.00,2208.30] vol=2.3x ATR=7.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 2219.92 | 2203.83 | 0.00 | T1 1.5R @ 2219.92 |
| Target hit | 2026-04-22 15:20:00 | 2235.00 | 2226.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 2273.70 | 2259.10 | 0.00 | ORB-long ORB[2222.60,2255.00] vol=3.0x ATR=7.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:45:00 | 2285.04 | 2283.93 | 0.00 | T1 1.5R @ 2285.04 |
| Target hit | 2026-04-23 10:40:00 | 2289.10 | 2296.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 2296.20 | 2283.66 | 0.00 | ORB-long ORB[2267.30,2284.90] vol=1.6x ATR=8.45 |
| Stop hit — per-position SL triggered | 2026-04-29 10:10:00 | 2287.75 | 2287.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 10:30:00 | 2082.20 | 2026-02-16 10:35:00 | 2077.78 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-17 10:50:00 | 2070.40 | 2026-02-17 11:00:00 | 2074.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-18 10:15:00 | 2054.00 | 2026-02-18 10:20:00 | 2048.14 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-18 10:15:00 | 2054.00 | 2026-02-18 10:50:00 | 2054.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 10:45:00 | 2087.60 | 2026-02-19 10:55:00 | 2081.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-24 10:40:00 | 2055.60 | 2026-02-24 11:50:00 | 2051.10 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 09:55:00 | 2073.30 | 2026-02-25 10:05:00 | 2080.73 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-25 09:55:00 | 2073.30 | 2026-02-25 15:20:00 | 2156.20 | TARGET_HIT | 0.50 | 4.00% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2183.00 | 2026-02-26 09:45:00 | 2198.54 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2183.00 | 2026-02-26 12:25:00 | 2239.40 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2026-03-05 09:55:00 | 2233.30 | 2026-03-05 10:25:00 | 2226.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2200.40 | 2026-03-06 11:00:00 | 2191.77 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2200.40 | 2026-03-06 13:40:00 | 2200.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-09 10:10:00 | 2118.00 | 2026-03-09 10:35:00 | 2126.34 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-10 11:00:00 | 2201.50 | 2026-03-10 11:30:00 | 2213.44 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-10 11:00:00 | 2201.50 | 2026-03-10 15:20:00 | 2243.30 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2026-03-13 09:50:00 | 2186.70 | 2026-03-13 10:10:00 | 2176.29 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-13 09:50:00 | 2186.70 | 2026-03-13 15:20:00 | 2142.10 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2026-03-16 11:00:00 | 2094.20 | 2026-03-16 11:05:00 | 2101.41 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-23 11:00:00 | 1934.30 | 2026-03-23 11:05:00 | 1943.25 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-24 10:55:00 | 1920.40 | 2026-03-24 11:10:00 | 1927.70 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:45:00 | 2068.20 | 2026-04-10 10:05:00 | 2062.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-15 09:40:00 | 2119.00 | 2026-04-15 10:10:00 | 2111.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 10:10:00 | 2120.00 | 2026-04-17 10:20:00 | 2128.74 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-17 10:10:00 | 2120.00 | 2026-04-17 10:40:00 | 2120.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:15:00 | 2186.00 | 2026-04-21 10:50:00 | 2195.66 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-21 10:15:00 | 2186.00 | 2026-04-21 11:35:00 | 2186.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:45:00 | 2209.00 | 2026-04-22 09:50:00 | 2219.92 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 09:45:00 | 2209.00 | 2026-04-22 15:20:00 | 2235.00 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2026-04-23 09:40:00 | 2273.70 | 2026-04-23 09:45:00 | 2285.04 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-23 09:40:00 | 2273.70 | 2026-04-23 10:40:00 | 2289.10 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-29 09:40:00 | 2296.20 | 2026-04-29 10:10:00 | 2287.75 | STOP_HIT | 1.00 | -0.37% |
