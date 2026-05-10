# Adani Enterprises Ltd. (ADANIENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2502.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 8
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 5.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.65% | 5.2% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.65% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 5 | 33.3% | 0 | 10 | 5 | 0.01% | 0.2% |
| SELL @ 2nd Alert (retest1) | 15 | 5 | 33.3% | 0 | 10 | 5 | 0.01% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 10 | 43.5% | 2 | 13 | 8 | 0.24% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 2242.40 | 2246.56 | 0.00 | ORB-short ORB[2244.10,2259.00] vol=1.5x ATR=5.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:55:00 | 2233.60 | 2245.19 | 0.00 | T1 1.5R @ 2233.60 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 2242.40 | 2245.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:35:00 | 2211.60 | 2216.84 | 0.00 | ORB-short ORB[2216.00,2229.50] vol=1.6x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 2204.56 | 2215.05 | 0.00 | T1 1.5R @ 2204.56 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 2211.60 | 2212.91 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 2191.10 | 2187.72 | 0.00 | ORB-long ORB[2170.00,2181.90] vol=4.3x ATR=7.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 2202.22 | 2188.70 | 0.00 | T1 1.5R @ 2202.22 |
| Target hit | 2026-02-17 15:20:00 | 2244.00 | 2217.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 2195.80 | 2220.27 | 0.00 | ORB-short ORB[2228.10,2254.00] vol=1.7x ATR=6.56 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 2202.36 | 2213.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 2172.80 | 2194.03 | 0.00 | ORB-short ORB[2197.50,2218.90] vol=1.7x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 2164.45 | 2188.19 | 0.00 | T1 1.5R @ 2164.45 |
| Stop hit — per-position SL triggered | 2026-02-19 11:20:00 | 2172.80 | 2187.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:35:00 | 2193.00 | 2179.47 | 0.00 | ORB-long ORB[2155.30,2184.00] vol=2.9x ATR=7.95 |
| Stop hit — per-position SL triggered | 2026-02-23 10:40:00 | 2185.05 | 2188.71 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 2181.20 | 2197.73 | 0.00 | ORB-short ORB[2188.40,2211.30] vol=2.3x ATR=5.58 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 2186.78 | 2197.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 2214.50 | 2220.41 | 0.00 | ORB-short ORB[2217.60,2232.40] vol=1.6x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:00:00 | 2206.84 | 2219.19 | 0.00 | T1 1.5R @ 2206.84 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 2214.50 | 2219.15 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 2194.30 | 2205.18 | 0.00 | ORB-short ORB[2202.10,2216.40] vol=1.5x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 2185.43 | 2200.56 | 0.00 | T1 1.5R @ 2185.43 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 2194.30 | 2199.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2064.10 | 2078.18 | 0.00 | ORB-short ORB[2065.10,2088.50] vol=2.4x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-03-06 11:50:00 | 2070.09 | 2073.87 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 1990.20 | 2010.61 | 0.00 | ORB-short ORB[2009.10,2030.00] vol=1.9x ATR=8.61 |
| Stop hit — per-position SL triggered | 2026-03-10 14:10:00 | 1998.81 | 1996.55 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:35:00 | 2004.00 | 1970.23 | 0.00 | ORB-long ORB[1947.10,1966.60] vol=1.5x ATR=8.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:55:00 | 2016.74 | 1977.74 | 0.00 | T1 1.5R @ 2016.74 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 2004.00 | 1986.40 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2164.00 | 2171.97 | 0.00 | ORB-short ORB[2166.20,2186.00] vol=2.8x ATR=7.61 |
| Stop hit — per-position SL triggered | 2026-04-16 09:45:00 | 2171.61 | 2172.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:20:00 | 2258.90 | 2248.74 | 0.00 | ORB-long ORB[2221.80,2254.90] vol=2.6x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:50:00 | 2271.43 | 2250.95 | 0.00 | T1 1.5R @ 2271.43 |
| Target hit | 2026-04-23 15:20:00 | 2300.00 | 2284.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 2437.90 | 2426.08 | 0.00 | ORB-long ORB[2404.20,2431.70] vol=2.9x ATR=9.17 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 2428.73 | 2426.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:45:00 | 2242.40 | 2026-02-10 09:55:00 | 2233.60 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-10 09:45:00 | 2242.40 | 2026-02-10 10:00:00 | 2242.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:35:00 | 2211.60 | 2026-02-12 11:15:00 | 2204.56 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-12 10:35:00 | 2211.60 | 2026-02-12 12:30:00 | 2211.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:40:00 | 2191.10 | 2026-02-17 11:15:00 | 2202.22 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-17 10:40:00 | 2191.10 | 2026-02-17 15:20:00 | 2244.00 | TARGET_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-02-18 10:35:00 | 2195.80 | 2026-02-18 11:00:00 | 2202.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:45:00 | 2172.80 | 2026-02-19 11:15:00 | 2164.45 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 10:45:00 | 2172.80 | 2026-02-19 11:20:00 | 2172.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 09:35:00 | 2193.00 | 2026-02-23 10:40:00 | 2185.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-25 11:15:00 | 2181.20 | 2026-02-25 11:20:00 | 2186.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-26 10:45:00 | 2214.50 | 2026-02-26 11:00:00 | 2206.84 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-26 10:45:00 | 2214.50 | 2026-02-26 11:05:00 | 2214.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2194.30 | 2026-02-27 10:15:00 | 2185.43 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2194.30 | 2026-02-27 10:25:00 | 2194.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2064.10 | 2026-03-06 11:50:00 | 2070.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 09:35:00 | 1990.20 | 2026-03-10 14:10:00 | 1998.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-12 10:35:00 | 2004.00 | 2026-03-12 10:55:00 | 2016.74 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-12 10:35:00 | 2004.00 | 2026-03-12 11:20:00 | 2004.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:30:00 | 2164.00 | 2026-04-16 09:45:00 | 2171.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-23 10:20:00 | 2258.90 | 2026-04-23 10:50:00 | 2271.43 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-23 10:20:00 | 2258.90 | 2026-04-23 15:20:00 | 2300.00 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2026-04-29 10:40:00 | 2437.90 | 2026-04-29 10:45:00 | 2428.73 | STOP_HIT | 1.00 | -0.38% |
