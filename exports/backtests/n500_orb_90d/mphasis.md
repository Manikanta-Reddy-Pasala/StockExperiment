# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2214.50
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 4
- **Avg / median % per leg:** 0.29% / 0.42%
- **Sum % (uncompounded):** 3.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.47% | 3.7% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.47% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.29% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 2636.60 | 2629.99 | 0.00 | ORB-long ORB[2600.70,2633.90] vol=1.5x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:45:00 | 2648.38 | 2634.10 | 0.00 | T1 1.5R @ 2648.38 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 2636.60 | 2641.23 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:10:00 | 2433.00 | 2436.81 | 0.00 | ORB-short ORB[2436.60,2467.50] vol=1.6x ATR=9.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 2418.73 | 2432.33 | 0.00 | T1 1.5R @ 2418.73 |
| Target hit | 2026-02-19 15:20:00 | 2372.00 | 2406.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 2235.30 | 2253.25 | 0.00 | ORB-short ORB[2266.70,2299.00] vol=3.1x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 12:20:00 | 2223.13 | 2242.41 | 0.00 | T1 1.5R @ 2223.13 |
| Target hit | 2026-03-05 14:40:00 | 2226.00 | 2224.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2026-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:55:00 | 2188.90 | 2172.49 | 0.00 | ORB-long ORB[2157.00,2184.90] vol=2.1x ATR=7.38 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 2181.52 | 2173.44 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 2085.50 | 2100.68 | 0.00 | ORB-short ORB[2090.00,2118.70] vol=1.8x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2093.78 | 2099.60 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:00:00 | 2070.90 | 2081.77 | 0.00 | ORB-short ORB[2080.20,2109.70] vol=1.9x ATR=9.06 |
| Stop hit — per-position SL triggered | 2026-03-20 10:10:00 | 2079.96 | 2081.50 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 2292.00 | 2324.65 | 0.00 | ORB-short ORB[2327.50,2360.20] vol=3.0x ATR=7.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 2280.18 | 2322.11 | 0.00 | T1 1.5R @ 2280.18 |
| Stop hit — per-position SL triggered | 2026-04-23 11:35:00 | 2292.00 | 2320.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 2308.30 | 2290.11 | 0.00 | ORB-long ORB[2268.70,2289.20] vol=2.9x ATR=8.71 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 2299.59 | 2292.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:05:00 | 2636.60 | 2026-02-10 10:45:00 | 2648.38 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 10:05:00 | 2636.60 | 2026-02-10 11:10:00 | 2636.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:10:00 | 2433.00 | 2026-02-19 11:15:00 | 2418.73 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-19 10:10:00 | 2433.00 | 2026-02-19 15:20:00 | 2372.00 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2026-03-05 10:50:00 | 2235.30 | 2026-03-05 12:20:00 | 2223.13 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-05 10:50:00 | 2235.30 | 2026-03-05 14:40:00 | 2226.00 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-12 10:55:00 | 2188.90 | 2026-03-12 11:00:00 | 2181.52 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-16 11:00:00 | 2085.50 | 2026-03-16 11:15:00 | 2093.78 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-20 10:00:00 | 2070.90 | 2026-03-20 10:10:00 | 2079.96 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-23 11:10:00 | 2292.00 | 2026-04-23 11:20:00 | 2280.18 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-23 11:10:00 | 2292.00 | 2026-04-23 11:35:00 | 2292.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:50:00 | 2308.30 | 2026-04-29 09:55:00 | 2299.59 | STOP_HIT | 1.00 | -0.38% |
