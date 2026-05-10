# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2424.80
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 1.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.03% | 0.2% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.03% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.10% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2083.00 | 2097.18 | 0.00 | ORB-short ORB[2085.00,2113.60] vol=1.5x ATR=9.02 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 2092.02 | 2096.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:05:00 | 2046.10 | 2025.77 | 0.00 | ORB-long ORB[2020.20,2044.00] vol=3.6x ATR=6.02 |
| Stop hit — per-position SL triggered | 2026-03-10 11:10:00 | 2040.08 | 2030.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:15:00 | 2045.00 | 2068.77 | 0.00 | ORB-short ORB[2066.80,2094.30] vol=1.9x ATR=7.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 2033.39 | 2065.65 | 0.00 | T1 1.5R @ 2033.39 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 2045.00 | 2061.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:10:00 | 2015.50 | 2025.81 | 0.00 | ORB-short ORB[2016.00,2040.40] vol=1.8x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-03-17 13:30:00 | 2021.81 | 2022.46 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2107.00 | 2115.88 | 0.00 | ORB-short ORB[2108.80,2129.00] vol=1.7x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 2096.23 | 2110.95 | 0.00 | T1 1.5R @ 2096.23 |
| Target hit | 2026-04-16 10:20:00 | 2103.00 | 2102.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:50:00 | 2136.00 | 2119.47 | 0.00 | ORB-long ORB[2102.40,2130.90] vol=2.4x ATR=8.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:00:00 | 2148.91 | 2127.78 | 0.00 | T1 1.5R @ 2148.91 |
| Target hit | 2026-04-22 15:05:00 | 2161.40 | 2164.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 2134.30 | 2157.48 | 0.00 | ORB-short ORB[2145.60,2175.00] vol=2.0x ATR=6.63 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 2140.93 | 2157.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 2274.00 | 2262.49 | 0.00 | ORB-long ORB[2246.50,2271.60] vol=2.0x ATR=9.88 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 2264.12 | 2263.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 2286.70 | 2268.60 | 0.00 | ORB-long ORB[2247.10,2280.00] vol=1.6x ATR=8.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:40:00 | 2300.01 | 2282.45 | 0.00 | T1 1.5R @ 2300.01 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 2286.70 | 2285.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 2363.80 | 2330.62 | 0.00 | ORB-long ORB[2308.00,2330.00] vol=3.5x ATR=11.27 |
| Stop hit — per-position SL triggered | 2026-05-07 10:20:00 | 2352.53 | 2334.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 2083.00 | 2026-02-13 09:40:00 | 2092.02 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-10 11:05:00 | 2046.10 | 2026-03-10 11:10:00 | 2040.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-13 10:15:00 | 2045.00 | 2026-03-13 10:20:00 | 2033.39 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 10:15:00 | 2045.00 | 2026-03-13 10:50:00 | 2045.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 11:10:00 | 2015.50 | 2026-03-17 13:30:00 | 2021.81 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-16 09:30:00 | 2107.00 | 2026-04-16 09:40:00 | 2096.23 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-16 09:30:00 | 2107.00 | 2026-04-16 10:20:00 | 2103.00 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2026-04-22 09:50:00 | 2136.00 | 2026-04-22 10:00:00 | 2148.91 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-22 09:50:00 | 2136.00 | 2026-04-22 15:05:00 | 2161.40 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2026-04-23 11:10:00 | 2134.30 | 2026-04-23 11:15:00 | 2140.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-04 10:00:00 | 2274.00 | 2026-05-04 10:10:00 | 2264.12 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-06 09:35:00 | 2286.70 | 2026-05-06 09:40:00 | 2300.01 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-05-06 09:35:00 | 2286.70 | 2026-05-06 10:05:00 | 2286.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:15:00 | 2363.80 | 2026-05-07 10:20:00 | 2352.53 | STOP_HIT | 1.00 | -0.48% |
