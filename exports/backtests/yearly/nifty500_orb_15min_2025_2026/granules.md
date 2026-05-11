# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 750.10
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
| ENTRY1 | 88 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 21 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 67
- **Target hits / Stop hits / Partials:** 21 / 67 / 36
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 20.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 24 | 36.9% | 9 | 41 | 15 | 0.03% | 1.9% |
| BUY @ 2nd Alert (retest1) | 65 | 24 | 36.9% | 9 | 41 | 15 | 0.03% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 33 | 55.9% | 12 | 26 | 21 | 0.31% | 18.1% |
| SELL @ 2nd Alert (retest1) | 59 | 33 | 55.9% | 12 | 26 | 21 | 0.31% | 18.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 124 | 57 | 46.0% | 21 | 67 | 36 | 0.16% | 20.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:50:00 | 507.35 | 503.09 | 0.00 | ORB-long ORB[500.25,504.90] vol=3.2x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-05-16 11:05:00 | 505.36 | 503.41 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 525.60 | 523.51 | 0.00 | ORB-long ORB[518.25,524.70] vol=4.0x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-05-26 09:35:00 | 523.65 | 523.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 516.25 | 520.02 | 0.00 | ORB-short ORB[521.55,525.65] vol=2.1x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:40:00 | 513.20 | 518.61 | 0.00 | T1 1.5R @ 513.20 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 516.25 | 517.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:00:00 | 526.80 | 529.22 | 0.00 | ORB-short ORB[528.55,534.10] vol=1.9x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-06-06 11:25:00 | 528.18 | 529.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:35:00 | 531.75 | 535.78 | 0.00 | ORB-short ORB[535.30,540.90] vol=2.1x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:00:00 | 529.11 | 532.43 | 0.00 | T1 1.5R @ 529.11 |
| Target hit | 2025-06-12 15:20:00 | 524.00 | 528.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 489.75 | 492.98 | 0.00 | ORB-short ORB[491.00,496.95] vol=1.7x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-06-19 11:00:00 | 491.51 | 492.69 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:50:00 | 485.20 | 482.43 | 0.00 | ORB-long ORB[478.25,484.20] vol=1.8x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-06-20 10:55:00 | 483.59 | 482.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:55:00 | 494.95 | 489.66 | 0.00 | ORB-long ORB[482.05,489.35] vol=1.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 11:10:00 | 498.20 | 490.22 | 0.00 | T1 1.5R @ 498.20 |
| Target hit | 2025-06-23 15:20:00 | 497.70 | 493.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-06-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 11:10:00 | 495.80 | 499.09 | 0.00 | ORB-short ORB[497.80,504.35] vol=2.1x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:30:00 | 493.45 | 498.75 | 0.00 | T1 1.5R @ 493.45 |
| Target hit | 2025-06-24 15:20:00 | 487.80 | 494.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 490.30 | 492.09 | 0.00 | ORB-short ORB[493.05,497.20] vol=4.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-06-26 11:20:00 | 491.44 | 492.03 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:55:00 | 501.00 | 499.45 | 0.00 | ORB-long ORB[496.05,500.55] vol=2.0x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:40:00 | 502.77 | 500.20 | 0.00 | T1 1.5R @ 502.77 |
| Stop hit — per-position SL triggered | 2025-06-27 12:00:00 | 501.00 | 500.31 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:05:00 | 494.20 | 497.04 | 0.00 | ORB-short ORB[494.80,499.75] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-06-30 11:20:00 | 495.38 | 497.00 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:50:00 | 490.05 | 493.36 | 0.00 | ORB-short ORB[493.05,498.05] vol=2.0x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-07-01 10:05:00 | 491.22 | 492.80 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:10:00 | 494.00 | 490.67 | 0.00 | ORB-long ORB[487.50,492.60] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-07-03 10:45:00 | 492.60 | 491.31 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:50:00 | 494.90 | 492.45 | 0.00 | ORB-long ORB[490.50,492.30] vol=2.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 09:55:00 | 496.70 | 493.82 | 0.00 | T1 1.5R @ 496.70 |
| Stop hit — per-position SL triggered | 2025-07-04 10:05:00 | 494.90 | 494.35 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:30:00 | 488.65 | 490.56 | 0.00 | ORB-short ORB[490.00,493.00] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:55:00 | 486.55 | 488.96 | 0.00 | T1 1.5R @ 486.55 |
| Target hit | 2025-07-08 15:05:00 | 482.25 | 481.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2025-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:50:00 | 475.85 | 479.11 | 0.00 | ORB-short ORB[478.70,482.95] vol=1.7x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:05:00 | 474.41 | 478.34 | 0.00 | T1 1.5R @ 474.41 |
| Target hit | 2025-07-10 15:20:00 | 473.95 | 474.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 472.70 | 470.12 | 0.00 | ORB-long ORB[467.00,470.45] vol=3.0x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:45:00 | 474.85 | 471.17 | 0.00 | T1 1.5R @ 474.85 |
| Target hit | 2025-07-14 15:20:00 | 485.45 | 481.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 487.35 | 489.83 | 0.00 | ORB-short ORB[490.35,495.70] vol=3.1x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:00:00 | 485.00 | 488.23 | 0.00 | T1 1.5R @ 485.00 |
| Target hit | 2025-07-22 13:35:00 | 484.40 | 483.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2025-07-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:55:00 | 474.90 | 476.43 | 0.00 | ORB-short ORB[476.25,480.05] vol=2.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:05:00 | 473.11 | 476.26 | 0.00 | T1 1.5R @ 473.11 |
| Stop hit — per-position SL triggered | 2025-07-23 11:45:00 | 474.90 | 475.48 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 483.00 | 480.78 | 0.00 | ORB-long ORB[475.95,482.65] vol=2.3x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-07-24 09:35:00 | 481.71 | 480.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 476.10 | 471.96 | 0.00 | ORB-long ORB[466.50,472.65] vol=2.0x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-07-28 09:40:00 | 474.25 | 472.97 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:00:00 | 473.35 | 469.70 | 0.00 | ORB-long ORB[465.00,471.15] vol=2.2x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-29 10:10:00 | 471.56 | 470.15 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:50:00 | 480.00 | 484.01 | 0.00 | ORB-short ORB[482.15,487.40] vol=1.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:00:00 | 477.23 | 481.09 | 0.00 | T1 1.5R @ 477.23 |
| Stop hit — per-position SL triggered | 2025-07-31 10:10:00 | 480.00 | 480.75 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 451.50 | 454.46 | 0.00 | ORB-short ORB[452.75,458.20] vol=1.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 09:45:00 | 448.95 | 452.94 | 0.00 | T1 1.5R @ 448.95 |
| Target hit | 2025-08-05 15:05:00 | 448.00 | 447.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — BUY (started 2025-08-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:55:00 | 445.75 | 443.12 | 0.00 | ORB-long ORB[440.75,443.50] vol=2.5x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:25:00 | 448.06 | 444.42 | 0.00 | T1 1.5R @ 448.06 |
| Stop hit — per-position SL triggered | 2025-08-12 12:45:00 | 445.75 | 447.14 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:00:00 | 462.00 | 456.63 | 0.00 | ORB-long ORB[451.95,457.95] vol=1.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-08-13 10:45:00 | 459.66 | 459.47 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:00:00 | 459.70 | 460.90 | 0.00 | ORB-short ORB[460.00,464.00] vol=1.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:35:00 | 458.21 | 460.40 | 0.00 | T1 1.5R @ 458.21 |
| Stop hit — per-position SL triggered | 2025-08-19 14:45:00 | 459.70 | 459.22 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 454.65 | 456.84 | 0.00 | ORB-short ORB[455.05,461.75] vol=1.9x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:40:00 | 452.83 | 456.07 | 0.00 | T1 1.5R @ 452.83 |
| Stop hit — per-position SL triggered | 2025-08-20 09:45:00 | 454.65 | 455.93 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 456.35 | 458.19 | 0.00 | ORB-short ORB[457.40,462.80] vol=2.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-08-21 09:35:00 | 457.49 | 458.03 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:50:00 | 463.05 | 460.20 | 0.00 | ORB-long ORB[456.30,462.30] vol=4.2x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:55:00 | 465.30 | 462.29 | 0.00 | T1 1.5R @ 465.30 |
| Target hit | 2025-08-26 11:20:00 | 469.40 | 469.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 513.80 | 510.15 | 0.00 | ORB-long ORB[507.30,510.85] vol=1.7x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-09-03 09:35:00 | 511.97 | 510.30 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 511.90 | 515.88 | 0.00 | ORB-short ORB[513.00,519.00] vol=2.4x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 513.82 | 515.37 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:30:00 | 523.40 | 521.27 | 0.00 | ORB-long ORB[517.45,523.20] vol=1.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-09-09 10:40:00 | 521.63 | 521.38 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:30:00 | 529.75 | 526.92 | 0.00 | ORB-long ORB[524.55,529.30] vol=3.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-09-12 11:10:00 | 528.00 | 528.30 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:15:00 | 539.50 | 536.53 | 0.00 | ORB-long ORB[533.40,538.35] vol=3.0x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-09-16 10:30:00 | 537.98 | 536.87 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:55:00 | 543.70 | 539.57 | 0.00 | ORB-long ORB[534.95,541.50] vol=5.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-17 11:00:00 | 542.32 | 539.79 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:45:00 | 537.85 | 540.71 | 0.00 | ORB-short ORB[541.45,546.50] vol=3.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:15:00 | 535.94 | 540.14 | 0.00 | T1 1.5R @ 535.94 |
| Stop hit — per-position SL triggered | 2025-09-19 12:00:00 | 537.85 | 539.50 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 535.90 | 541.77 | 0.00 | ORB-short ORB[540.00,546.55] vol=2.1x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:20:00 | 533.26 | 540.61 | 0.00 | T1 1.5R @ 533.26 |
| Target hit | 2025-09-22 15:20:00 | 526.80 | 532.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 525.45 | 523.20 | 0.00 | ORB-long ORB[519.05,525.40] vol=1.6x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:50:00 | 528.95 | 525.77 | 0.00 | T1 1.5R @ 528.95 |
| Target hit | 2025-09-24 10:10:00 | 526.95 | 527.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2025-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:00:00 | 523.15 | 519.13 | 0.00 | ORB-long ORB[512.25,519.95] vol=2.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-09-29 11:10:00 | 521.27 | 519.24 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:10:00 | 553.00 | 549.84 | 0.00 | ORB-long ORB[542.00,549.90] vol=1.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-10-03 10:30:00 | 550.46 | 549.92 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:05:00 | 554.25 | 557.20 | 0.00 | ORB-short ORB[554.50,560.75] vol=2.1x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-10-06 12:05:00 | 556.06 | 556.91 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 570.25 | 567.54 | 0.00 | ORB-long ORB[564.25,569.00] vol=2.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 568.15 | 567.67 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 565.20 | 565.76 | 0.00 | ORB-short ORB[568.00,571.95] vol=4.4x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-10-20 11:40:00 | 566.50 | 565.77 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:55:00 | 572.55 | 569.87 | 0.00 | ORB-long ORB[565.00,569.00] vol=7.2x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 570.92 | 570.73 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:35:00 | 573.95 | 571.44 | 0.00 | ORB-long ORB[568.00,573.15] vol=2.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:40:00 | 576.83 | 572.57 | 0.00 | T1 1.5R @ 576.83 |
| Target hit | 2025-10-29 12:00:00 | 575.20 | 575.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-10-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:05:00 | 574.85 | 572.29 | 0.00 | ORB-long ORB[569.20,572.35] vol=1.6x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-10-30 10:10:00 | 573.17 | 572.31 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:20:00 | 571.10 | 572.61 | 0.00 | ORB-short ORB[572.95,574.80] vol=2.4x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:55:00 | 569.40 | 571.83 | 0.00 | T1 1.5R @ 569.40 |
| Stop hit — per-position SL triggered | 2025-10-31 11:00:00 | 571.10 | 571.82 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:40:00 | 573.80 | 570.05 | 0.00 | ORB-long ORB[564.30,569.55] vol=2.7x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-11-03 09:55:00 | 572.28 | 571.78 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:30:00 | 570.70 | 573.30 | 0.00 | ORB-short ORB[573.50,578.05] vol=2.4x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:10:00 | 568.34 | 572.91 | 0.00 | T1 1.5R @ 568.34 |
| Target hit | 2025-11-04 15:20:00 | 567.40 | 569.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 565.05 | 567.09 | 0.00 | ORB-short ORB[567.05,570.00] vol=2.5x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:05:00 | 562.75 | 565.42 | 0.00 | T1 1.5R @ 562.75 |
| Target hit | 2025-11-06 10:30:00 | 564.15 | 564.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 558.40 | 563.12 | 0.00 | ORB-short ORB[558.75,565.55] vol=2.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 555.85 | 562.58 | 0.00 | T1 1.5R @ 555.85 |
| Stop hit — per-position SL triggered | 2025-12-08 11:20:00 | 558.40 | 562.16 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 565.65 | 562.66 | 0.00 | ORB-long ORB[557.50,564.00] vol=1.7x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-12-10 10:40:00 | 564.01 | 563.09 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:40:00 | 551.05 | 553.49 | 0.00 | ORB-short ORB[553.05,557.20] vol=4.1x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-12-11 09:45:00 | 553.10 | 553.23 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:10:00 | 572.00 | 570.19 | 0.00 | ORB-long ORB[563.40,570.75] vol=3.9x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 569.89 | 570.31 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:35:00 | 570.00 | 573.70 | 0.00 | ORB-short ORB[573.10,577.75] vol=1.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-12-15 10:50:00 | 571.56 | 573.64 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:35:00 | 574.50 | 572.84 | 0.00 | ORB-long ORB[570.00,574.25] vol=1.6x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:40:00 | 576.89 | 574.87 | 0.00 | T1 1.5R @ 576.89 |
| Target hit | 2025-12-17 10:40:00 | 577.20 | 577.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2025-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:00:00 | 580.45 | 577.50 | 0.00 | ORB-long ORB[573.30,579.90] vol=2.2x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-12-18 11:05:00 | 579.01 | 578.08 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 611.80 | 614.99 | 0.00 | ORB-short ORB[612.60,619.15] vol=1.5x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:00:00 | 609.15 | 614.39 | 0.00 | T1 1.5R @ 609.15 |
| Target hit | 2025-12-29 15:20:00 | 602.35 | 605.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:35:00 | 595.35 | 597.57 | 0.00 | ORB-short ORB[597.00,601.80] vol=2.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-12-31 11:05:00 | 597.06 | 597.27 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:40:00 | 610.00 | 606.87 | 0.00 | ORB-long ORB[600.05,609.00] vol=2.5x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:45:00 | 612.92 | 608.36 | 0.00 | T1 1.5R @ 612.92 |
| Target hit | 2026-01-01 10:50:00 | 612.05 | 612.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — SELL (started 2026-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:35:00 | 608.50 | 611.54 | 0.00 | ORB-short ORB[610.10,617.50] vol=1.8x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-01-05 09:40:00 | 609.88 | 611.45 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:05:00 | 608.30 | 606.26 | 0.00 | ORB-long ORB[601.40,607.60] vol=2.9x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:55:00 | 611.22 | 606.94 | 0.00 | T1 1.5R @ 611.22 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 608.30 | 607.27 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:40:00 | 617.85 | 615.72 | 0.00 | ORB-long ORB[610.25,616.80] vol=2.2x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-01-07 09:45:00 | 615.26 | 615.73 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:45:00 | 612.70 | 607.36 | 0.00 | ORB-long ORB[603.60,612.00] vol=1.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-01-09 10:10:00 | 610.48 | 609.31 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:15:00 | 594.95 | 598.44 | 0.00 | ORB-short ORB[596.25,602.00] vol=1.6x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 12:00:00 | 592.85 | 597.83 | 0.00 | T1 1.5R @ 592.85 |
| Target hit | 2026-01-16 15:20:00 | 579.20 | 589.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:05:00 | 564.00 | 570.69 | 0.00 | ORB-short ORB[570.40,576.90] vol=1.8x ATR=2.05 |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 566.05 | 570.49 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:05:00 | 548.50 | 553.45 | 0.00 | ORB-short ORB[552.35,558.70] vol=1.6x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:10:00 | 544.11 | 552.58 | 0.00 | T1 1.5R @ 544.11 |
| Target hit | 2026-01-21 12:55:00 | 543.00 | 542.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — SELL (started 2026-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:10:00 | 556.75 | 559.01 | 0.00 | ORB-short ORB[557.70,565.00] vol=3.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-01-29 11:30:00 | 558.13 | 558.93 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 564.85 | 569.14 | 0.00 | ORB-short ORB[570.00,577.95] vol=4.7x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 562.57 | 568.40 | 0.00 | T1 1.5R @ 562.57 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 564.85 | 567.54 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 569.00 | 565.07 | 0.00 | ORB-long ORB[562.80,568.00] vol=2.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 566.76 | 566.39 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 572.85 | 569.71 | 0.00 | ORB-long ORB[564.35,570.80] vol=1.6x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 570.98 | 570.65 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:25:00 | 577.25 | 575.24 | 0.00 | ORB-long ORB[572.00,576.40] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:55:00 | 579.59 | 575.56 | 0.00 | T1 1.5R @ 579.59 |
| Stop hit — per-position SL triggered | 2026-02-20 11:45:00 | 577.25 | 575.90 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 597.25 | 595.62 | 0.00 | ORB-long ORB[591.25,596.30] vol=2.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 595.06 | 595.93 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 603.20 | 601.11 | 0.00 | ORB-long ORB[596.20,600.75] vol=9.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 601.62 | 601.59 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 600.90 | 599.99 | 0.00 | ORB-long ORB[597.05,600.00] vol=4.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-02-26 09:35:00 | 599.72 | 599.81 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 570.45 | 565.93 | 0.00 | ORB-long ORB[560.50,568.30] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-16 10:20:00 | 567.75 | 568.08 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:25:00 | 577.00 | 581.97 | 0.00 | ORB-short ORB[582.35,588.00] vol=2.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-03-23 10:35:00 | 579.04 | 581.69 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:55:00 | 635.25 | 639.01 | 0.00 | ORB-short ORB[640.90,648.75] vol=1.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-04-09 10:00:00 | 638.10 | 638.83 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 642.30 | 639.81 | 0.00 | ORB-long ORB[634.00,641.00] vol=3.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 640.34 | 640.15 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 654.00 | 652.15 | 0.00 | ORB-long ORB[646.85,652.00] vol=3.3x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 651.75 | 652.16 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 669.40 | 664.93 | 0.00 | ORB-long ORB[661.10,668.70] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 667.36 | 665.81 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 681.00 | 675.59 | 0.00 | ORB-long ORB[666.90,675.40] vol=3.2x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:35:00 | 685.60 | 679.69 | 0.00 | T1 1.5R @ 685.60 |
| Target hit | 2026-04-22 10:50:00 | 681.85 | 682.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 685.25 | 680.99 | 0.00 | ORB-long ORB[670.30,680.40] vol=1.6x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:45:00 | 689.91 | 686.35 | 0.00 | T1 1.5R @ 689.91 |
| Target hit | 2026-04-23 10:05:00 | 686.85 | 687.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 86 — BUY (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 703.50 | 701.57 | 0.00 | ORB-long ORB[695.00,702.00] vol=1.9x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 706.61 | 702.63 | 0.00 | T1 1.5R @ 706.61 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 703.50 | 703.16 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 715.55 | 712.50 | 0.00 | ORB-long ORB[706.75,714.00] vol=2.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-04-29 10:05:00 | 712.76 | 712.60 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 723.45 | 718.58 | 0.00 | ORB-long ORB[712.40,720.00] vol=3.7x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 720.74 | 716.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 10:50:00 | 507.35 | 2025-05-16 11:05:00 | 505.36 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-26 09:30:00 | 525.60 | 2025-05-26 09:35:00 | 523.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-04 09:35:00 | 516.25 | 2025-06-04 09:40:00 | 513.20 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-06-04 09:35:00 | 516.25 | 2025-06-04 10:00:00 | 516.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 11:00:00 | 526.80 | 2025-06-06 11:25:00 | 528.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-12 10:35:00 | 531.75 | 2025-06-12 13:00:00 | 529.11 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-06-12 10:35:00 | 531.75 | 2025-06-12 15:20:00 | 524.00 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-06-19 10:35:00 | 489.75 | 2025-06-19 11:00:00 | 491.51 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-20 10:50:00 | 485.20 | 2025-06-20 10:55:00 | 483.59 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-23 10:55:00 | 494.95 | 2025-06-23 11:10:00 | 498.20 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-06-23 10:55:00 | 494.95 | 2025-06-23 15:20:00 | 497.70 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-24 11:10:00 | 495.80 | 2025-06-24 11:30:00 | 493.45 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-06-24 11:10:00 | 495.80 | 2025-06-24 15:20:00 | 487.80 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2025-06-26 11:15:00 | 490.30 | 2025-06-26 11:20:00 | 491.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-27 10:55:00 | 501.00 | 2025-06-27 11:40:00 | 502.77 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-06-27 10:55:00 | 501.00 | 2025-06-27 12:00:00 | 501.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-30 11:05:00 | 494.20 | 2025-06-30 11:20:00 | 495.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-01 09:50:00 | 490.05 | 2025-07-01 10:05:00 | 491.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-03 10:10:00 | 494.00 | 2025-07-03 10:45:00 | 492.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-04 09:50:00 | 494.90 | 2025-07-04 09:55:00 | 496.70 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-04 09:50:00 | 494.90 | 2025-07-04 10:05:00 | 494.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 09:30:00 | 488.65 | 2025-07-08 09:55:00 | 486.55 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-08 09:30:00 | 488.65 | 2025-07-08 15:05:00 | 482.25 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-07-10 10:50:00 | 475.85 | 2025-07-10 11:05:00 | 474.41 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-10 10:50:00 | 475.85 | 2025-07-10 15:20:00 | 473.95 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-07-14 09:40:00 | 472.70 | 2025-07-14 09:45:00 | 474.85 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-14 09:40:00 | 472.70 | 2025-07-14 15:20:00 | 485.45 | TARGET_HIT | 0.50 | 2.70% |
| SELL | retest1 | 2025-07-22 09:30:00 | 487.35 | 2025-07-22 10:00:00 | 485.00 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-07-22 09:30:00 | 487.35 | 2025-07-22 13:35:00 | 484.40 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-23 10:55:00 | 474.90 | 2025-07-23 11:05:00 | 473.11 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-23 10:55:00 | 474.90 | 2025-07-23 11:45:00 | 474.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 09:30:00 | 483.00 | 2025-07-24 09:35:00 | 481.71 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-28 09:30:00 | 476.10 | 2025-07-28 09:40:00 | 474.25 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-29 10:00:00 | 473.35 | 2025-07-29 10:10:00 | 471.56 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-31 09:50:00 | 480.00 | 2025-07-31 10:00:00 | 477.23 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-07-31 09:50:00 | 480.00 | 2025-07-31 10:10:00 | 480.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-05 09:30:00 | 451.50 | 2025-08-05 09:45:00 | 448.95 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-08-05 09:30:00 | 451.50 | 2025-08-05 15:05:00 | 448.00 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-08-12 09:55:00 | 445.75 | 2025-08-12 10:25:00 | 448.06 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-08-12 09:55:00 | 445.75 | 2025-08-12 12:45:00 | 445.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 10:00:00 | 462.00 | 2025-08-13 10:45:00 | 459.66 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-08-19 11:00:00 | 459.70 | 2025-08-19 11:35:00 | 458.21 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-19 11:00:00 | 459.70 | 2025-08-19 14:45:00 | 459.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-20 09:30:00 | 454.65 | 2025-08-20 09:40:00 | 452.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-20 09:30:00 | 454.65 | 2025-08-20 09:45:00 | 454.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 09:30:00 | 456.35 | 2025-08-21 09:35:00 | 457.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-26 09:50:00 | 463.05 | 2025-08-26 09:55:00 | 465.30 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-08-26 09:50:00 | 463.05 | 2025-08-26 11:20:00 | 469.40 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2025-09-03 09:30:00 | 513.80 | 2025-09-03 09:35:00 | 511.97 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-05 10:10:00 | 511.90 | 2025-09-05 10:20:00 | 513.82 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-09 10:30:00 | 523.40 | 2025-09-09 10:40:00 | 521.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-12 10:30:00 | 529.75 | 2025-09-12 11:10:00 | 528.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-16 10:15:00 | 539.50 | 2025-09-16 10:30:00 | 537.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-17 10:55:00 | 543.70 | 2025-09-17 11:00:00 | 542.32 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-19 10:45:00 | 537.85 | 2025-09-19 11:15:00 | 535.94 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-19 10:45:00 | 537.85 | 2025-09-19 12:00:00 | 537.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-22 11:00:00 | 535.90 | 2025-09-22 11:20:00 | 533.26 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-22 11:00:00 | 535.90 | 2025-09-22 15:20:00 | 526.80 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-09-24 09:35:00 | 525.45 | 2025-09-24 09:50:00 | 528.95 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-09-24 09:35:00 | 525.45 | 2025-09-24 10:10:00 | 526.95 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-29 11:00:00 | 523.15 | 2025-09-29 11:10:00 | 521.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-03 10:10:00 | 553.00 | 2025-10-03 10:30:00 | 550.46 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-10-06 11:05:00 | 554.25 | 2025-10-06 12:05:00 | 556.06 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-10 09:30:00 | 570.25 | 2025-10-10 09:35:00 | 568.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-20 11:15:00 | 565.20 | 2025-10-20 11:40:00 | 566.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-28 09:55:00 | 572.55 | 2025-10-28 10:15:00 | 570.92 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-29 09:35:00 | 573.95 | 2025-10-29 09:40:00 | 576.83 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-29 09:35:00 | 573.95 | 2025-10-29 12:00:00 | 575.20 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-10-30 10:05:00 | 574.85 | 2025-10-30 10:10:00 | 573.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-31 10:20:00 | 571.10 | 2025-10-31 10:55:00 | 569.40 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-31 10:20:00 | 571.10 | 2025-10-31 11:00:00 | 571.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 09:40:00 | 573.80 | 2025-11-03 09:55:00 | 572.28 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-04 10:30:00 | 570.70 | 2025-11-04 11:10:00 | 568.34 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-04 10:30:00 | 570.70 | 2025-11-04 15:20:00 | 567.40 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-06 09:30:00 | 565.05 | 2025-11-06 10:05:00 | 562.75 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-06 09:30:00 | 565.05 | 2025-11-06 10:30:00 | 564.15 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-12-08 11:05:00 | 558.40 | 2025-12-08 11:15:00 | 555.85 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-08 11:05:00 | 558.40 | 2025-12-08 11:20:00 | 558.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 10:15:00 | 565.65 | 2025-12-10 10:40:00 | 564.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-11 09:40:00 | 551.05 | 2025-12-11 09:45:00 | 553.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-12 10:10:00 | 572.00 | 2025-12-12 10:15:00 | 569.89 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-12-15 10:35:00 | 570.00 | 2025-12-15 10:50:00 | 571.56 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-17 09:35:00 | 574.50 | 2025-12-17 09:40:00 | 576.89 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-17 09:35:00 | 574.50 | 2025-12-17 10:40:00 | 577.20 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-12-18 11:00:00 | 580.45 | 2025-12-18 11:05:00 | 579.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-29 09:55:00 | 611.80 | 2025-12-29 10:00:00 | 609.15 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-29 09:55:00 | 611.80 | 2025-12-29 15:20:00 | 602.35 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2025-12-31 10:35:00 | 595.35 | 2025-12-31 11:05:00 | 597.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-01 09:40:00 | 610.00 | 2026-01-01 09:45:00 | 612.92 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-01 09:40:00 | 610.00 | 2026-01-01 10:50:00 | 612.05 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-05 09:35:00 | 608.50 | 2026-01-05 09:40:00 | 609.88 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-06 10:05:00 | 608.30 | 2026-01-06 10:55:00 | 611.22 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-06 10:05:00 | 608.30 | 2026-01-06 11:20:00 | 608.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:40:00 | 617.85 | 2026-01-07 09:45:00 | 615.26 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-01-09 09:45:00 | 612.70 | 2026-01-09 10:10:00 | 610.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-16 11:15:00 | 594.95 | 2026-01-16 12:00:00 | 592.85 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-16 11:15:00 | 594.95 | 2026-01-16 15:20:00 | 579.20 | TARGET_HIT | 0.50 | 2.65% |
| SELL | retest1 | 2026-01-20 11:05:00 | 564.00 | 2026-01-20 11:15:00 | 566.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-21 10:05:00 | 548.50 | 2026-01-21 10:10:00 | 544.11 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2026-01-21 10:05:00 | 548.50 | 2026-01-21 12:55:00 | 543.00 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-01-29 11:10:00 | 556.75 | 2026-01-29 11:30:00 | 558.13 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-12 10:45:00 | 564.85 | 2026-02-12 11:20:00 | 562.57 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-12 10:45:00 | 564.85 | 2026-02-12 12:30:00 | 564.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 09:40:00 | 569.00 | 2026-02-13 09:55:00 | 566.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-18 09:35:00 | 572.85 | 2026-02-18 09:50:00 | 570.98 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-20 10:25:00 | 577.25 | 2026-02-20 10:55:00 | 579.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-20 10:25:00 | 577.25 | 2026-02-20 11:45:00 | 577.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 09:30:00 | 597.25 | 2026-02-24 09:40:00 | 595.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-25 10:40:00 | 603.20 | 2026-02-25 11:30:00 | 601.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-26 09:30:00 | 600.90 | 2026-02-26 09:35:00 | 599.72 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-16 09:40:00 | 570.45 | 2026-03-16 10:20:00 | 567.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-23 10:25:00 | 577.00 | 2026-03-23 10:35:00 | 579.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-09 09:55:00 | 635.25 | 2026-04-09 10:00:00 | 638.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:35:00 | 642.30 | 2026-04-15 09:45:00 | 640.34 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-17 09:30:00 | 654.00 | 2026-04-17 09:40:00 | 651.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-21 10:55:00 | 669.40 | 2026-04-21 11:00:00 | 667.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 09:30:00 | 681.00 | 2026-04-22 09:35:00 | 685.60 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-22 09:30:00 | 681.00 | 2026-04-22 10:50:00 | 681.85 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-04-23 09:35:00 | 685.25 | 2026-04-23 09:45:00 | 689.91 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-23 09:35:00 | 685.25 | 2026-04-23 10:05:00 | 686.85 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2026-04-28 11:05:00 | 703.50 | 2026-04-28 11:20:00 | 706.61 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-28 11:05:00 | 703.50 | 2026-04-28 11:45:00 | 703.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:00:00 | 715.55 | 2026-04-29 10:05:00 | 712.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-06 09:30:00 | 723.45 | 2026-05-06 09:35:00 | 720.74 | STOP_HIT | 1.00 | -0.37% |
