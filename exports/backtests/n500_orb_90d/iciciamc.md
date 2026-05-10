# ICICI Prudential Asset Management Company Ltd. (ICICIAMC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3240.00
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
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 5
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.01% | 0.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.01% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.12% | 1.2% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.12% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.08% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 3071.40 | 3043.12 | 0.00 | ORB-long ORB[3006.20,3050.00] vol=1.6x ATR=12.10 |
| Stop hit — per-position SL triggered | 2026-02-10 11:40:00 | 3059.30 | 3049.11 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 3049.20 | 3031.18 | 0.00 | ORB-long ORB[2992.00,3033.50] vol=5.0x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:45:00 | 3060.15 | 3039.23 | 0.00 | T1 1.5R @ 3060.15 |
| Target hit | 2026-02-18 15:20:00 | 3075.00 | 3053.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 3051.20 | 3041.18 | 0.00 | ORB-long ORB[3000.00,3044.50] vol=1.9x ATR=9.30 |
| Stop hit — per-position SL triggered | 2026-02-20 11:35:00 | 3041.90 | 3042.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 2873.80 | 2892.05 | 0.00 | ORB-short ORB[2900.10,2931.30] vol=4.4x ATR=9.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:55:00 | 2859.17 | 2883.31 | 0.00 | T1 1.5R @ 2859.17 |
| Stop hit — per-position SL triggered | 2026-03-13 12:05:00 | 2873.80 | 2882.30 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 2844.40 | 2857.67 | 0.00 | ORB-short ORB[2853.10,2879.70] vol=2.4x ATR=14.08 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2858.48 | 2854.91 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:50:00 | 2923.10 | 2937.49 | 0.00 | ORB-short ORB[2946.00,2974.30] vol=4.0x ATR=9.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:10:00 | 2908.49 | 2935.05 | 0.00 | T1 1.5R @ 2908.49 |
| Stop hit — per-position SL triggered | 2026-03-19 11:50:00 | 2923.10 | 2927.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 2841.10 | 2829.44 | 0.00 | ORB-long ORB[2800.10,2840.00] vol=1.6x ATR=13.36 |
| Stop hit — per-position SL triggered | 2026-03-25 09:40:00 | 2827.74 | 2830.19 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:45:00 | 3279.80 | 3305.02 | 0.00 | ORB-short ORB[3314.20,3349.00] vol=1.5x ATR=11.79 |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 3291.59 | 3302.19 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 3291.90 | 3302.63 | 0.00 | ORB-short ORB[3293.00,3329.50] vol=1.6x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:30:00 | 3282.10 | 3301.47 | 0.00 | T1 1.5R @ 3282.10 |
| Stop hit — per-position SL triggered | 2026-05-06 11:40:00 | 3291.90 | 3301.32 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 3290.00 | 3303.63 | 0.00 | ORB-short ORB[3297.00,3317.00] vol=2.2x ATR=9.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:35:00 | 3275.87 | 3299.13 | 0.00 | T1 1.5R @ 3275.87 |
| Target hit | 2026-05-07 10:45:00 | 3280.40 | 3272.28 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:45:00 | 3071.40 | 2026-02-10 11:40:00 | 3059.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-18 10:50:00 | 3049.20 | 2026-02-18 12:45:00 | 3060.15 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-18 10:50:00 | 3049.20 | 2026-02-18 15:20:00 | 3075.00 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2026-02-20 11:00:00 | 3051.20 | 2026-02-20 11:35:00 | 3041.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-13 10:55:00 | 2873.80 | 2026-03-13 11:55:00 | 2859.17 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-13 10:55:00 | 2873.80 | 2026-03-13 12:05:00 | 2873.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:35:00 | 2844.40 | 2026-03-16 11:15:00 | 2858.48 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-19 10:50:00 | 2923.10 | 2026-03-19 11:10:00 | 2908.49 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-19 10:50:00 | 2923.10 | 2026-03-19 11:50:00 | 2923.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:30:00 | 2841.10 | 2026-03-25 09:40:00 | 2827.74 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-27 10:45:00 | 3279.80 | 2026-04-27 11:15:00 | 3291.59 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-06 11:15:00 | 3291.90 | 2026-05-06 11:30:00 | 3282.10 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-06 11:15:00 | 3291.90 | 2026-05-06 11:40:00 | 3291.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 09:30:00 | 3290.00 | 2026-05-07 09:35:00 | 3275.87 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-07 09:30:00 | 3290.00 | 2026-05-07 10:45:00 | 3280.40 | TARGET_HIT | 0.50 | 0.29% |
