# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2656.80
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 3
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 2.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.16% | 1.0% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.16% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.20% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2363.70 | 2379.64 | 0.00 | ORB-short ORB[2372.00,2405.00] vol=2.0x ATR=9.12 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 2372.82 | 2376.85 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 2381.00 | 2396.91 | 0.00 | ORB-short ORB[2394.90,2417.00] vol=1.7x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:10:00 | 2372.94 | 2393.70 | 0.00 | T1 1.5R @ 2372.94 |
| Target hit | 2026-02-19 15:20:00 | 2332.90 | 2375.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 2329.90 | 2343.94 | 0.00 | ORB-short ORB[2345.10,2370.00] vol=1.8x ATR=7.52 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 2337.42 | 2343.46 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 2221.90 | 2232.77 | 0.00 | ORB-short ORB[2229.00,2248.00] vol=1.6x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-27 12:25:00 | 2228.04 | 2226.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 2242.90 | 2216.21 | 0.00 | ORB-long ORB[2178.80,2210.90] vol=6.1x ATR=12.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:55:00 | 2261.69 | 2230.08 | 0.00 | T1 1.5R @ 2261.69 |
| Target hit | 2026-03-05 10:45:00 | 2264.10 | 2265.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2399.70 | 2379.54 | 0.00 | ORB-long ORB[2363.00,2383.90] vol=1.9x ATR=10.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:35:00 | 2414.84 | 2388.40 | 0.00 | T1 1.5R @ 2414.84 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 2399.70 | 2391.30 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 2693.40 | 2675.82 | 0.00 | ORB-long ORB[2650.60,2683.80] vol=1.7x ATR=10.25 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 2683.15 | 2676.88 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 2676.40 | 2695.78 | 0.00 | ORB-short ORB[2686.50,2720.70] vol=2.2x ATR=11.16 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 2687.56 | 2693.02 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 2642.40 | 2624.81 | 0.00 | ORB-long ORB[2596.00,2633.90] vol=1.8x ATR=14.07 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 2628.33 | 2626.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 2363.70 | 2026-02-13 09:40:00 | 2372.82 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-19 11:15:00 | 2381.00 | 2026-02-19 12:10:00 | 2372.94 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 11:15:00 | 2381.00 | 2026-02-19 15:20:00 | 2332.90 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2026-02-23 10:50:00 | 2329.90 | 2026-02-23 10:55:00 | 2337.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2221.90 | 2026-02-27 12:25:00 | 2228.04 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-05 09:50:00 | 2242.90 | 2026-03-05 09:55:00 | 2261.69 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2026-03-05 09:50:00 | 2242.90 | 2026-03-05 10:45:00 | 2264.10 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2399.70 | 2026-03-18 09:35:00 | 2414.84 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2399.70 | 2026-03-18 09:45:00 | 2399.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:05:00 | 2693.40 | 2026-04-22 10:10:00 | 2683.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-24 09:30:00 | 2676.40 | 2026-04-24 09:40:00 | 2687.56 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-05 09:30:00 | 2642.40 | 2026-05-05 09:40:00 | 2628.33 | STOP_HIT | 1.00 | -0.53% |
