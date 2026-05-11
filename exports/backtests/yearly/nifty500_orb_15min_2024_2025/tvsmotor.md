# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-08-08 15:25:00 (4596 bars)
- **Last close:** 2527.00
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
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 7
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.21% | 3.5% |
| BUY @ 2nd Alert (retest1) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.21% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.16% | -1.0% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.16% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 11 | 47.8% | 4 | 12 | 7 | 0.11% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:40:00 | 2096.55 | 2083.37 | 0.00 | ORB-long ORB[2065.00,2084.25] vol=2.0x ATR=8.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 09:55:00 | 2109.65 | 2091.31 | 0.00 | T1 1.5R @ 2109.65 |
| Target hit | 2024-05-14 10:20:00 | 2102.05 | 2103.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 2090.50 | 2117.35 | 0.00 | ORB-short ORB[2108.65,2131.65] vol=1.9x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 2097.08 | 2116.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:05:00 | 2140.00 | 2151.78 | 0.00 | ORB-short ORB[2150.05,2170.00] vol=3.8x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:30:00 | 2132.46 | 2149.88 | 0.00 | T1 1.5R @ 2132.46 |
| Stop hit — per-position SL triggered | 2024-05-22 15:00:00 | 2140.00 | 2139.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:30:00 | 2178.85 | 2170.70 | 0.00 | ORB-long ORB[2160.00,2176.55] vol=1.5x ATR=6.19 |
| Stop hit — per-position SL triggered | 2024-05-24 10:00:00 | 2172.66 | 2175.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:50:00 | 2264.95 | 2243.46 | 0.00 | ORB-long ORB[2226.65,2258.00] vol=2.2x ATR=7.75 |
| Stop hit — per-position SL triggered | 2024-05-28 11:30:00 | 2257.20 | 2252.73 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:00:00 | 2468.00 | 2451.77 | 0.00 | ORB-long ORB[2442.00,2457.80] vol=2.1x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:10:00 | 2477.13 | 2459.13 | 0.00 | T1 1.5R @ 2477.13 |
| Target hit | 2024-06-14 15:20:00 | 2496.35 | 2486.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:05:00 | 2477.50 | 2463.08 | 0.00 | ORB-long ORB[2436.00,2449.70] vol=1.7x ATR=6.64 |
| Stop hit — per-position SL triggered | 2024-06-25 10:20:00 | 2470.86 | 2466.54 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 2362.50 | 2354.28 | 0.00 | ORB-long ORB[2336.05,2360.95] vol=1.5x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:45:00 | 2372.31 | 2358.56 | 0.00 | T1 1.5R @ 2372.31 |
| Stop hit — per-position SL triggered | 2024-07-03 09:50:00 | 2362.50 | 2358.86 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:05:00 | 2397.15 | 2417.19 | 0.00 | ORB-short ORB[2420.30,2445.95] vol=2.7x ATR=6.38 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 2403.53 | 2415.04 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 2412.65 | 2448.59 | 0.00 | ORB-short ORB[2439.25,2470.00] vol=1.7x ATR=10.03 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 2422.68 | 2447.51 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 11:05:00 | 2380.00 | 2392.68 | 0.00 | ORB-short ORB[2387.40,2417.70] vol=2.1x ATR=8.21 |
| Stop hit — per-position SL triggered | 2024-07-19 12:00:00 | 2388.21 | 2388.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:05:00 | 2438.30 | 2420.61 | 0.00 | ORB-long ORB[2408.05,2434.70] vol=2.7x ATR=7.41 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 2430.89 | 2423.71 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 2448.55 | 2432.00 | 0.00 | ORB-long ORB[2421.00,2445.55] vol=3.0x ATR=6.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:45:00 | 2458.49 | 2436.39 | 0.00 | T1 1.5R @ 2458.49 |
| Stop hit — per-position SL triggered | 2024-07-25 11:50:00 | 2448.55 | 2436.49 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 2497.25 | 2474.11 | 0.00 | ORB-long ORB[2438.75,2452.00] vol=6.6x ATR=8.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:55:00 | 2510.60 | 2486.54 | 0.00 | T1 1.5R @ 2510.60 |
| Target hit | 2024-07-26 10:45:00 | 2500.05 | 2501.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-07-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:10:00 | 2498.05 | 2486.84 | 0.00 | ORB-long ORB[2470.60,2488.10] vol=1.8x ATR=5.60 |
| Stop hit — per-position SL triggered | 2024-07-30 10:25:00 | 2492.45 | 2489.18 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 2519.40 | 2512.95 | 0.00 | ORB-long ORB[2492.10,2517.75] vol=2.2x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:40:00 | 2528.10 | 2517.70 | 0.00 | T1 1.5R @ 2528.10 |
| Target hit | 2024-07-31 13:55:00 | 2538.10 | 2543.19 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:40:00 | 2096.55 | 2024-05-14 09:55:00 | 2109.65 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-05-14 09:40:00 | 2096.55 | 2024-05-14 10:20:00 | 2102.05 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-05-16 11:15:00 | 2090.50 | 2024-05-16 11:25:00 | 2097.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-22 11:05:00 | 2140.00 | 2024-05-22 11:30:00 | 2132.46 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-22 11:05:00 | 2140.00 | 2024-05-22 15:00:00 | 2140.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-24 09:30:00 | 2178.85 | 2024-05-24 10:00:00 | 2172.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-28 10:50:00 | 2264.95 | 2024-05-28 11:30:00 | 2257.20 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-14 11:00:00 | 2468.00 | 2024-06-14 11:10:00 | 2477.13 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-14 11:00:00 | 2468.00 | 2024-06-14 15:20:00 | 2496.35 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-06-25 10:05:00 | 2477.50 | 2024-06-25 10:20:00 | 2470.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-03 09:40:00 | 2362.50 | 2024-07-03 09:45:00 | 2372.31 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-03 09:40:00 | 2362.50 | 2024-07-03 09:50:00 | 2362.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:05:00 | 2397.15 | 2024-07-08 11:10:00 | 2403.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-10 10:35:00 | 2412.65 | 2024-07-10 10:40:00 | 2422.68 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-19 11:05:00 | 2380.00 | 2024-07-19 12:00:00 | 2388.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-23 11:05:00 | 2438.30 | 2024-07-23 11:15:00 | 2430.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-25 11:05:00 | 2448.55 | 2024-07-25 11:45:00 | 2458.49 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-25 11:05:00 | 2448.55 | 2024-07-25 11:50:00 | 2448.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:50:00 | 2497.25 | 2024-07-26 09:55:00 | 2510.60 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-26 09:50:00 | 2497.25 | 2024-07-26 10:45:00 | 2500.05 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-07-30 10:10:00 | 2498.05 | 2024-07-30 10:25:00 | 2492.45 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-31 09:30:00 | 2519.40 | 2024-07-31 09:40:00 | 2528.10 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-31 09:30:00 | 2519.40 | 2024-07-31 13:55:00 | 2538.10 | TARGET_HIT | 0.50 | 0.74% |
