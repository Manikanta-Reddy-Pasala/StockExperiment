# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2265.10
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 4
- **Avg / median % per leg:** 0.39% / 0.48%
- **Sum % (uncompounded):** 4.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.52% | 4.2% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.52% | 4.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.39% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 2650.00 | 2664.17 | 0.00 | ORB-short ORB[2665.40,2694.60] vol=1.6x ATR=9.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:45:00 | 2635.78 | 2658.88 | 0.00 | T1 1.5R @ 2635.78 |
| Target hit | 2026-02-12 15:05:00 | 2642.00 | 2636.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 2258.00 | 2273.10 | 0.00 | ORB-short ORB[2267.00,2296.10] vol=1.8x ATR=13.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:25:00 | 2237.94 | 2260.05 | 0.00 | T1 1.5R @ 2237.94 |
| Target hit | 2026-03-13 15:20:00 | 2235.30 | 2252.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 2309.10 | 2321.94 | 0.00 | ORB-short ORB[2322.50,2347.90] vol=1.5x ATR=8.44 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 2317.54 | 2320.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 2281.80 | 2295.44 | 0.00 | ORB-short ORB[2293.00,2311.10] vol=1.5x ATR=5.88 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 2287.68 | 2294.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 2256.00 | 2266.83 | 0.00 | ORB-short ORB[2261.20,2284.00] vol=1.5x ATR=7.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 2245.10 | 2258.65 | 0.00 | T1 1.5R @ 2245.10 |
| Target hit | 2026-04-24 14:55:00 | 2220.00 | 2215.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 2188.70 | 2183.66 | 0.00 | ORB-long ORB[2172.50,2185.50] vol=2.3x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:45:00 | 2199.18 | 2187.48 | 0.00 | T1 1.5R @ 2199.18 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 2188.70 | 2188.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 2290.90 | 2277.33 | 0.00 | ORB-long ORB[2250.10,2272.40] vol=1.9x ATR=7.96 |
| Stop hit — per-position SL triggered | 2026-05-08 11:05:00 | 2282.94 | 2278.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:20:00 | 2650.00 | 2026-02-12 10:45:00 | 2635.78 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-12 10:20:00 | 2650.00 | 2026-02-12 15:05:00 | 2642.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-13 09:55:00 | 2258.00 | 2026-03-13 13:25:00 | 2237.94 | PARTIAL | 0.50 | 0.89% |
| SELL | retest1 | 2026-03-13 09:55:00 | 2258.00 | 2026-03-13 15:20:00 | 2235.30 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-04-17 10:05:00 | 2309.10 | 2026-04-17 10:15:00 | 2317.54 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-23 09:40:00 | 2281.80 | 2026-04-23 09:50:00 | 2287.68 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 09:35:00 | 2256.00 | 2026-04-24 09:50:00 | 2245.10 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-24 09:35:00 | 2256.00 | 2026-04-24 14:55:00 | 2220.00 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2026-05-04 09:40:00 | 2188.70 | 2026-05-04 09:45:00 | 2199.18 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-04 09:40:00 | 2188.70 | 2026-05-04 10:00:00 | 2188.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 11:00:00 | 2290.90 | 2026-05-08 11:05:00 | 2282.94 | STOP_HIT | 1.00 | -0.35% |
