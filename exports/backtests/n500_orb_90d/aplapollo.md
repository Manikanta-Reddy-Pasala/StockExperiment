# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1950.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 3
- **Avg / median % per leg:** 0.16% / -0.21%
- **Sum % (uncompounded):** 2.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.32% | 2.6% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.32% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.00% | -0.0% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.00% | -0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 2 | 11 | 3 | 0.16% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 2254.70 | 2248.33 | 0.00 | ORB-long ORB[2230.00,2250.80] vol=1.6x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-02-11 10:55:00 | 2250.23 | 2249.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 2271.50 | 2276.55 | 0.00 | ORB-short ORB[2271.60,2293.30] vol=9.0x ATR=5.53 |
| Stop hit — per-position SL triggered | 2026-02-12 09:50:00 | 2277.03 | 2276.27 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 2209.50 | 2230.92 | 0.00 | ORB-short ORB[2234.30,2247.70] vol=3.4x ATR=4.71 |
| Stop hit — per-position SL triggered | 2026-02-26 11:10:00 | 2214.21 | 2228.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 2221.70 | 2230.87 | 0.00 | ORB-short ORB[2224.40,2241.80] vol=1.6x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 2228.05 | 2229.60 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:25:00 | 2185.70 | 2172.22 | 0.00 | ORB-long ORB[2143.40,2174.40] vol=2.2x ATR=6.57 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 2179.13 | 2173.29 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:00:00 | 1928.50 | 1912.87 | 0.00 | ORB-long ORB[1897.40,1922.30] vol=2.5x ATR=6.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 1921.95 | 1913.76 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:50:00 | 1982.30 | 1971.91 | 0.00 | ORB-long ORB[1948.00,1977.00] vol=2.4x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:50:00 | 1994.49 | 1977.81 | 0.00 | T1 1.5R @ 1994.49 |
| Stop hit — per-position SL triggered | 2026-03-20 10:55:00 | 1982.30 | 1978.41 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 2013.90 | 2003.84 | 0.00 | ORB-long ORB[1981.60,2008.30] vol=1.8x ATR=8.59 |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 2005.31 | 2009.54 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 11:00:00 | 1995.80 | 1981.08 | 0.00 | ORB-long ORB[1964.20,1993.60] vol=1.5x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 12:20:00 | 2008.39 | 1986.14 | 0.00 | T1 1.5R @ 2008.39 |
| Target hit | 2026-04-08 15:20:00 | 2047.30 | 2016.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:45:00 | 2098.30 | 2106.78 | 0.00 | ORB-short ORB[2098.40,2123.90] vol=1.6x ATR=7.48 |
| Stop hit — per-position SL triggered | 2026-04-20 09:55:00 | 2105.78 | 2106.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 2135.00 | 2146.49 | 0.00 | ORB-short ORB[2150.30,2173.00] vol=2.4x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:25:00 | 2127.78 | 2143.30 | 0.00 | T1 1.5R @ 2127.78 |
| Target hit | 2026-04-21 15:20:00 | 2106.60 | 2119.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 1986.40 | 2007.90 | 0.00 | ORB-short ORB[2020.40,2039.50] vol=5.8x ATR=6.62 |
| Stop hit — per-position SL triggered | 2026-04-24 12:05:00 | 1993.02 | 2001.38 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 1974.00 | 1984.46 | 0.00 | ORB-short ORB[1976.30,1998.40] vol=1.6x ATR=4.74 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 1978.74 | 1984.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:40:00 | 2254.70 | 2026-02-11 10:55:00 | 2250.23 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-12 09:30:00 | 2271.50 | 2026-02-12 09:50:00 | 2277.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-26 10:55:00 | 2209.50 | 2026-02-26 11:10:00 | 2214.21 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 10:05:00 | 2221.70 | 2026-02-27 10:20:00 | 2228.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-06 10:25:00 | 2185.70 | 2026-03-06 10:35:00 | 2179.13 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-17 11:00:00 | 1928.50 | 2026-03-17 11:20:00 | 1921.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-20 09:50:00 | 1982.30 | 2026-03-20 10:50:00 | 1994.49 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-03-20 09:50:00 | 1982.30 | 2026-03-20 10:55:00 | 1982.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:30:00 | 2013.90 | 2026-03-25 11:15:00 | 2005.31 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-08 11:00:00 | 1995.80 | 2026-04-08 12:20:00 | 2008.39 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-08 11:00:00 | 1995.80 | 2026-04-08 15:20:00 | 2047.30 | TARGET_HIT | 0.50 | 2.58% |
| SELL | retest1 | 2026-04-20 09:45:00 | 2098.30 | 2026-04-20 09:55:00 | 2105.78 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-21 11:00:00 | 2135.00 | 2026-04-21 11:25:00 | 2127.78 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-21 11:00:00 | 2135.00 | 2026-04-21 15:20:00 | 2106.60 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-04-24 10:45:00 | 1986.40 | 2026-04-24 12:05:00 | 1993.02 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-29 11:05:00 | 1974.00 | 2026-04-29 11:15:00 | 1978.74 | STOP_HIT | 1.00 | -0.24% |
