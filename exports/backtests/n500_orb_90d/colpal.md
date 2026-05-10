# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2193.70
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 5
- **Avg / median % per leg:** -0.05% / -0.20%
- **Sum % (uncompounded):** -1.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.06% | 0.7% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.06% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 1 | 10.0% | 0 | 9 | 1 | -0.20% | -2.0% |
| SELL @ 2nd Alert (retest1) | 10 | 1 | 10.0% | 0 | 9 | 1 | -0.20% | -2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 7 | 30.4% | 2 | 16 | 5 | -0.05% | -1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 2149.80 | 2138.72 | 0.00 | ORB-long ORB[2134.60,2147.30] vol=2.7x ATR=7.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:20:00 | 2161.26 | 2148.77 | 0.00 | T1 1.5R @ 2161.26 |
| Target hit | 2026-02-09 14:30:00 | 2150.60 | 2152.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 2112.90 | 2118.50 | 0.00 | ORB-short ORB[2119.00,2128.20] vol=2.5x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:30:00 | 2106.75 | 2115.43 | 0.00 | T1 1.5R @ 2106.75 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 2112.90 | 2112.45 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 2122.00 | 2115.86 | 0.00 | ORB-long ORB[2104.70,2120.60] vol=9.1x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:25:00 | 2127.77 | 2116.64 | 0.00 | T1 1.5R @ 2127.77 |
| Stop hit — per-position SL triggered | 2026-02-16 11:45:00 | 2122.00 | 2116.88 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 2149.80 | 2134.60 | 0.00 | ORB-long ORB[2118.60,2138.50] vol=1.9x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:10:00 | 2155.91 | 2136.34 | 0.00 | T1 1.5R @ 2155.91 |
| Stop hit — per-position SL triggered | 2026-02-17 11:50:00 | 2149.80 | 2141.58 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 2167.80 | 2158.03 | 0.00 | ORB-long ORB[2148.70,2164.00] vol=1.6x ATR=4.40 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 2163.40 | 2159.41 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 2201.00 | 2193.04 | 0.00 | ORB-long ORB[2167.20,2187.50] vol=2.0x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:45:00 | 2210.26 | 2195.53 | 0.00 | T1 1.5R @ 2210.26 |
| Stop hit — per-position SL triggered | 2026-02-20 12:05:00 | 2201.00 | 2196.26 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 2238.10 | 2245.26 | 0.00 | ORB-short ORB[2239.10,2252.20] vol=1.9x ATR=6.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:10:00 | 2244.47 | 2244.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 2181.50 | 2189.00 | 0.00 | ORB-short ORB[2184.80,2204.00] vol=1.8x ATR=5.89 |
| Stop hit — per-position SL triggered | 2026-03-05 09:50:00 | 2187.39 | 2188.02 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:30:00 | 1985.00 | 1975.13 | 0.00 | ORB-long ORB[1959.10,1980.60] vol=2.0x ATR=7.02 |
| Stop hit — per-position SL triggered | 2026-03-13 09:40:00 | 1977.98 | 1976.39 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 1952.10 | 1941.08 | 0.00 | ORB-long ORB[1932.70,1949.90] vol=1.8x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 1946.11 | 1945.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:45:00 | 1913.80 | 1918.56 | 0.00 | ORB-short ORB[1916.20,1935.20] vol=2.4x ATR=5.34 |
| Stop hit — per-position SL triggered | 2026-03-19 10:10:00 | 1919.14 | 1917.43 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:10:00 | 1844.00 | 1858.98 | 0.00 | ORB-short ORB[1866.80,1880.60] vol=2.0x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-03-24 11:20:00 | 1849.09 | 1858.22 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:30:00 | 1933.90 | 1919.32 | 0.00 | ORB-long ORB[1908.30,1926.70] vol=2.1x ATR=6.31 |
| Target hit | 2026-04-10 15:20:00 | 1939.70 | 1930.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1918.00 | 1932.67 | 0.00 | ORB-short ORB[1929.30,1949.00] vol=1.6x ATR=6.09 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 1924.09 | 1931.64 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 2135.50 | 2127.97 | 0.00 | ORB-long ORB[2111.90,2129.00] vol=2.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 2130.49 | 2128.05 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 2111.00 | 2116.84 | 0.00 | ORB-short ORB[2114.20,2136.40] vol=2.5x ATR=4.62 |
| Stop hit — per-position SL triggered | 2026-04-28 12:35:00 | 2115.62 | 2115.21 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:10:00 | 2097.20 | 2119.81 | 0.00 | ORB-short ORB[2110.00,2131.40] vol=1.6x ATR=7.97 |
| Stop hit — per-position SL triggered | 2026-04-30 11:35:00 | 2105.17 | 2118.26 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:10:00 | 2166.50 | 2174.76 | 0.00 | ORB-short ORB[2167.30,2184.90] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-05-05 12:10:00 | 2171.28 | 2173.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 2149.80 | 2026-02-09 12:20:00 | 2161.26 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-09 10:25:00 | 2149.80 | 2026-02-09 14:30:00 | 2150.60 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2026-02-13 10:10:00 | 2112.90 | 2026-02-13 10:30:00 | 2106.75 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-13 10:10:00 | 2112.90 | 2026-02-13 10:55:00 | 2112.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 11:00:00 | 2122.00 | 2026-02-16 11:25:00 | 2127.77 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-16 11:00:00 | 2122.00 | 2026-02-16 11:45:00 | 2122.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:05:00 | 2149.80 | 2026-02-17 11:10:00 | 2155.91 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-17 11:05:00 | 2149.80 | 2026-02-17 11:50:00 | 2149.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:40:00 | 2167.80 | 2026-02-18 09:50:00 | 2163.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-20 10:35:00 | 2201.00 | 2026-02-20 11:45:00 | 2210.26 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-20 10:35:00 | 2201.00 | 2026-02-20 12:05:00 | 2201.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:55:00 | 2238.10 | 2026-02-25 10:10:00 | 2244.47 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-05 09:30:00 | 2181.50 | 2026-03-05 09:50:00 | 2187.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-13 09:30:00 | 1985.00 | 2026-03-13 09:40:00 | 1977.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-17 10:20:00 | 1952.10 | 2026-03-17 11:15:00 | 1946.11 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-19 09:45:00 | 1913.80 | 2026-03-19 10:10:00 | 1919.14 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-24 11:10:00 | 1844.00 | 2026-03-24 11:20:00 | 1849.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-10 10:30:00 | 1933.90 | 2026-04-10 15:20:00 | 1939.70 | TARGET_HIT | 1.00 | 0.30% |
| SELL | retest1 | 2026-04-15 09:35:00 | 1918.00 | 2026-04-15 09:40:00 | 1924.09 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 10:55:00 | 2135.50 | 2026-04-21 11:00:00 | 2130.49 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-28 10:55:00 | 2111.00 | 2026-04-28 12:35:00 | 2115.62 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-30 11:10:00 | 2097.20 | 2026-04-30 11:35:00 | 2105.17 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-05 11:10:00 | 2166.50 | 2026-05-05 12:10:00 | 2171.28 | STOP_HIT | 1.00 | -0.22% |
