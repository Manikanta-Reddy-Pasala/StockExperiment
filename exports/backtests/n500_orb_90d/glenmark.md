# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2361.20
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 4
- **Avg / median % per leg:** 0.13% / -0.24%
- **Sum % (uncompounded):** 2.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 2 | 8 | 2 | 0.07% | 0.8% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 2 | 8 | 2 | 0.07% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 3 | 11 | 4 | 0.13% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 1971.10 | 1962.82 | 0.00 | ORB-long ORB[1941.40,1965.00] vol=2.6x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-02-09 13:30:00 | 1964.19 | 1968.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:15:00 | 1986.10 | 1975.25 | 0.00 | ORB-long ORB[1949.90,1979.00] vol=2.3x ATR=5.51 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 1980.59 | 1976.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 2039.30 | 2027.25 | 0.00 | ORB-long ORB[2018.20,2034.30] vol=2.4x ATR=6.01 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 2033.29 | 2033.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 2042.00 | 2038.39 | 0.00 | ORB-long ORB[2025.60,2035.70] vol=2.9x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 2037.03 | 2038.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 2021.80 | 2032.12 | 0.00 | ORB-short ORB[2022.30,2038.00] vol=1.8x ATR=5.00 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 2026.80 | 2030.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 2032.60 | 2016.70 | 0.00 | ORB-long ORB[1999.20,2023.60] vol=4.2x ATR=7.08 |
| Stop hit — per-position SL triggered | 2026-02-20 11:35:00 | 2025.52 | 2021.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 2081.60 | 2076.09 | 0.00 | ORB-long ORB[2061.00,2081.00] vol=3.5x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-02-25 11:40:00 | 2075.32 | 2077.92 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 2102.20 | 2093.59 | 0.00 | ORB-long ORB[2080.00,2099.90] vol=1.5x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:40:00 | 2111.30 | 2098.63 | 0.00 | T1 1.5R @ 2111.30 |
| Target hit | 2026-02-26 10:20:00 | 2115.80 | 2117.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 2231.20 | 2255.38 | 0.00 | ORB-short ORB[2251.40,2273.20] vol=1.6x ATR=9.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:35:00 | 2216.76 | 2244.59 | 0.00 | T1 1.5R @ 2216.76 |
| Stop hit — per-position SL triggered | 2026-03-12 09:45:00 | 2231.20 | 2239.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2163.10 | 2170.86 | 0.00 | ORB-short ORB[2165.70,2182.00] vol=1.9x ATR=6.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:30:00 | 2152.86 | 2170.49 | 0.00 | T1 1.5R @ 2152.86 |
| Target hit | 2026-03-17 15:20:00 | 2141.90 | 2158.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-03-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:25:00 | 2149.50 | 2140.52 | 0.00 | ORB-long ORB[2110.10,2141.00] vol=1.6x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:55:00 | 2162.10 | 2144.29 | 0.00 | T1 1.5R @ 2162.10 |
| Target hit | 2026-03-20 15:20:00 | 2184.50 | 2166.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 2143.10 | 2169.09 | 0.00 | ORB-short ORB[2158.50,2186.20] vol=1.7x ATR=8.06 |
| Stop hit — per-position SL triggered | 2026-04-09 09:35:00 | 2151.16 | 2167.68 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 2198.40 | 2176.19 | 0.00 | ORB-long ORB[2156.80,2184.70] vol=1.5x ATR=9.03 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 2189.37 | 2180.17 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:50:00 | 2271.00 | 2256.90 | 0.00 | ORB-long ORB[2240.90,2265.90] vol=1.7x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 2265.35 | 2259.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 1971.10 | 2026-02-09 13:30:00 | 1964.19 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-11 10:15:00 | 1986.10 | 2026-02-11 10:20:00 | 1980.59 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-13 09:40:00 | 2039.30 | 2026-02-13 09:55:00 | 2033.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:10:00 | 2042.00 | 2026-02-17 10:40:00 | 2037.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-18 10:40:00 | 2021.80 | 2026-02-18 10:55:00 | 2026.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-20 10:50:00 | 2032.60 | 2026-02-20 11:35:00 | 2025.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-25 10:15:00 | 2081.60 | 2026-02-25 11:40:00 | 2075.32 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2102.20 | 2026-02-26 09:40:00 | 2111.30 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2102.20 | 2026-02-26 10:20:00 | 2115.80 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-12 09:30:00 | 2231.20 | 2026-03-12 09:35:00 | 2216.76 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-12 09:30:00 | 2231.20 | 2026-03-12 09:45:00 | 2231.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 11:15:00 | 2163.10 | 2026-03-17 11:30:00 | 2152.86 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-17 11:15:00 | 2163.10 | 2026-03-17 15:20:00 | 2141.90 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2026-03-20 10:25:00 | 2149.50 | 2026-03-20 10:55:00 | 2162.10 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-03-20 10:25:00 | 2149.50 | 2026-03-20 15:20:00 | 2184.50 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2026-04-09 09:30:00 | 2143.10 | 2026-04-09 09:35:00 | 2151.16 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:35:00 | 2198.40 | 2026-04-10 09:55:00 | 2189.37 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-17 10:50:00 | 2271.00 | 2026-04-17 11:15:00 | 2265.35 | STOP_HIT | 1.00 | -0.25% |
