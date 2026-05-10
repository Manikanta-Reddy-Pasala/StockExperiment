# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2091.00
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
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.04% | 0.3% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.04% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.16% | 2.1% |
| SELL @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.16% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 4 | 12 | 6 | 0.11% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 2199.90 | 2208.49 | 0.00 | ORB-short ORB[2213.10,2244.10] vol=5.1x ATR=5.74 |
| Stop hit — per-position SL triggered | 2026-02-12 11:05:00 | 2205.64 | 2208.24 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2217.30 | 2207.57 | 0.00 | ORB-long ORB[2192.90,2216.90] vol=1.7x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 2226.82 | 2213.08 | 0.00 | T1 1.5R @ 2226.82 |
| Target hit | 2026-02-17 15:05:00 | 2225.00 | 2225.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 2197.00 | 2201.77 | 0.00 | ORB-short ORB[2200.00,2228.00] vol=1.8x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:50:00 | 2187.32 | 2199.45 | 0.00 | T1 1.5R @ 2187.32 |
| Target hit | 2026-02-18 15:15:00 | 2184.80 | 2177.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 2159.00 | 2163.52 | 0.00 | ORB-short ORB[2161.10,2185.80] vol=2.0x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 2164.66 | 2163.39 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 2166.20 | 2159.24 | 0.00 | ORB-long ORB[2143.50,2157.80] vol=1.9x ATR=4.60 |
| Stop hit — per-position SL triggered | 2026-02-20 10:45:00 | 2161.60 | 2159.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 2153.70 | 2161.07 | 0.00 | ORB-short ORB[2154.90,2183.40] vol=2.9x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 15:00:00 | 2145.67 | 2154.65 | 0.00 | T1 1.5R @ 2145.67 |
| Target hit | 2026-02-27 15:20:00 | 2129.90 | 2150.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 10:15:00 | 2100.80 | 2095.96 | 0.00 | ORB-long ORB[2080.00,2100.00] vol=1.6x ATR=6.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 15:00:00 | 2111.26 | 2100.01 | 0.00 | T1 1.5R @ 2111.26 |
| Target hit | 2026-03-02 15:20:00 | 2117.50 | 2101.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 2046.80 | 2055.30 | 0.00 | ORB-short ORB[2061.00,2090.20] vol=2.0x ATR=7.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:25:00 | 2035.88 | 2051.17 | 0.00 | T1 1.5R @ 2035.88 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 2046.80 | 2047.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 2102.90 | 2082.13 | 0.00 | ORB-long ORB[2073.70,2093.60] vol=2.4x ATR=10.00 |
| Stop hit — per-position SL triggered | 2026-03-06 09:40:00 | 2092.90 | 2082.19 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:50:00 | 1997.90 | 2009.61 | 0.00 | ORB-short ORB[2009.00,2034.80] vol=2.9x ATR=6.38 |
| Stop hit — per-position SL triggered | 2026-03-27 10:55:00 | 2004.28 | 2009.14 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 2100.00 | 2111.50 | 0.00 | ORB-short ORB[2106.80,2129.90] vol=2.1x ATR=5.17 |
| Stop hit — per-position SL triggered | 2026-04-15 11:00:00 | 2105.17 | 2110.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 2171.40 | 2162.13 | 0.00 | ORB-long ORB[2141.30,2167.50] vol=2.6x ATR=8.78 |
| Stop hit — per-position SL triggered | 2026-04-17 12:30:00 | 2162.62 | 2168.90 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 2195.50 | 2183.55 | 0.00 | ORB-long ORB[2162.00,2190.50] vol=1.5x ATR=7.04 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 2188.46 | 2184.63 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2106.50 | 2116.93 | 0.00 | ORB-short ORB[2107.10,2132.00] vol=2.7x ATR=5.86 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 2112.36 | 2116.61 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 2114.00 | 2125.16 | 0.00 | ORB-short ORB[2121.00,2140.00] vol=1.8x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 2105.17 | 2122.83 | 0.00 | T1 1.5R @ 2105.17 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 2114.00 | 2122.27 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 2112.30 | 2097.38 | 0.00 | ORB-long ORB[2080.00,2104.90] vol=4.4x ATR=6.86 |
| Stop hit — per-position SL triggered | 2026-05-08 10:50:00 | 2105.44 | 2097.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:55:00 | 2199.90 | 2026-02-12 11:05:00 | 2205.64 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-17 10:30:00 | 2217.30 | 2026-02-17 11:15:00 | 2226.82 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:30:00 | 2217.30 | 2026-02-17 15:05:00 | 2225.00 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-18 10:20:00 | 2197.00 | 2026-02-18 10:50:00 | 2187.32 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-18 10:20:00 | 2197.00 | 2026-02-18 15:15:00 | 2184.80 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-19 09:40:00 | 2159.00 | 2026-02-19 09:55:00 | 2164.66 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-20 10:30:00 | 2166.20 | 2026-02-20 10:45:00 | 2161.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2153.70 | 2026-02-27 15:00:00 | 2145.67 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2153.70 | 2026-02-27 15:20:00 | 2129.90 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2026-03-02 10:15:00 | 2100.80 | 2026-03-02 15:00:00 | 2111.26 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-02 10:15:00 | 2100.80 | 2026-03-02 15:20:00 | 2117.50 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2026-03-05 10:05:00 | 2046.80 | 2026-03-05 10:25:00 | 2035.88 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-05 10:05:00 | 2046.80 | 2026-03-05 11:05:00 | 2046.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 09:35:00 | 2102.90 | 2026-03-06 09:40:00 | 2092.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-27 10:50:00 | 1997.90 | 2026-03-27 10:55:00 | 2004.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-15 10:45:00 | 2100.00 | 2026-04-15 11:00:00 | 2105.17 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-17 09:30:00 | 2171.40 | 2026-04-17 12:30:00 | 2162.62 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-21 09:55:00 | 2195.50 | 2026-04-21 10:05:00 | 2188.46 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-27 11:15:00 | 2106.50 | 2026-04-27 11:25:00 | 2112.36 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-29 10:50:00 | 2114.00 | 2026-04-29 11:15:00 | 2105.17 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-29 10:50:00 | 2114.00 | 2026-04-29 11:20:00 | 2114.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 10:45:00 | 2112.30 | 2026-05-08 10:50:00 | 2105.44 | STOP_HIT | 1.00 | -0.32% |
