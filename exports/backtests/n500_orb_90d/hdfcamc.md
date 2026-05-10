# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2843.90
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 4
- **Avg / median % per leg:** 0.13% / 0.27%
- **Sum % (uncompounded):** 1.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.13% | 1.2% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.13% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.14% | 0.6% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.14% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.13% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 2781.50 | 2762.34 | 0.00 | ORB-long ORB[2735.00,2761.00] vol=2.3x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 2792.53 | 2767.11 | 0.00 | T1 1.5R @ 2792.53 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 2781.50 | 2769.55 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 2864.80 | 2848.22 | 0.00 | ORB-long ORB[2823.20,2852.50] vol=1.7x ATR=6.36 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 2858.44 | 2849.36 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 2715.20 | 2711.86 | 0.00 | ORB-long ORB[2682.80,2711.30] vol=3.5x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:00:00 | 2725.76 | 2715.52 | 0.00 | T1 1.5R @ 2725.76 |
| Target hit | 2026-02-25 12:45:00 | 2722.50 | 2728.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:50:00 | 2442.10 | 2434.45 | 0.00 | ORB-long ORB[2408.40,2440.00] vol=1.5x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:30:00 | 2456.04 | 2437.58 | 0.00 | T1 1.5R @ 2456.04 |
| Target hit | 2026-03-12 14:25:00 | 2453.30 | 2453.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 2395.60 | 2382.89 | 0.00 | ORB-long ORB[2355.50,2382.00] vol=3.0x ATR=9.53 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 2386.07 | 2383.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:20:00 | 2299.00 | 2327.48 | 0.00 | ORB-short ORB[2329.30,2361.30] vol=1.8x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:40:00 | 2282.53 | 2314.42 | 0.00 | T1 1.5R @ 2282.53 |
| Target hit | 2026-03-23 14:10:00 | 2285.00 | 2283.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 2509.00 | 2523.01 | 0.00 | ORB-short ORB[2516.20,2542.00] vol=1.6x ATR=11.71 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 2520.71 | 2514.01 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 2771.20 | 2778.73 | 0.00 | ORB-short ORB[2773.00,2811.40] vol=1.8x ATR=8.08 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 2779.28 | 2777.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 2801.00 | 2780.50 | 0.00 | ORB-long ORB[2762.60,2794.90] vol=3.3x ATR=8.33 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 2792.67 | 2782.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 2781.50 | 2026-02-10 09:45:00 | 2792.53 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-10 09:40:00 | 2781.50 | 2026-02-10 09:50:00 | 2781.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:25:00 | 2864.80 | 2026-02-17 10:30:00 | 2858.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 10:45:00 | 2715.20 | 2026-02-25 11:00:00 | 2725.76 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-25 10:45:00 | 2715.20 | 2026-02-25 12:45:00 | 2722.50 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-03-12 10:50:00 | 2442.10 | 2026-03-12 11:30:00 | 2456.04 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-12 10:50:00 | 2442.10 | 2026-03-12 14:25:00 | 2453.30 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-17 10:25:00 | 2395.60 | 2026-03-17 10:35:00 | 2386.07 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-23 10:20:00 | 2299.00 | 2026-03-23 11:40:00 | 2282.53 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-03-23 10:20:00 | 2299.00 | 2026-03-23 14:10:00 | 2285.00 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-09 09:30:00 | 2509.00 | 2026-04-09 10:15:00 | 2520.71 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-22 10:40:00 | 2771.20 | 2026-04-22 11:05:00 | 2779.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 10:45:00 | 2801.00 | 2026-04-29 11:00:00 | 2792.67 | STOP_HIT | 1.00 | -0.30% |
