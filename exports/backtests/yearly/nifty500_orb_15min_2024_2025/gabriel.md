# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1136.50
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 10 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 33
- **Target hits / Stop hits / Partials:** 10 / 33 / 18
- **Avg / median % per leg:** 0.37% / 0.00%
- **Sum % (uncompounded):** 22.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 11 | 42.3% | 4 | 15 | 7 | 0.47% | 12.1% |
| BUY @ 2nd Alert (retest1) | 26 | 11 | 42.3% | 4 | 15 | 7 | 0.47% | 12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 17 | 48.6% | 6 | 18 | 11 | 0.31% | 10.7% |
| SELL @ 2nd Alert (retest1) | 35 | 17 | 48.6% | 6 | 18 | 11 | 0.31% | 10.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 28 | 45.9% | 10 | 33 | 18 | 0.37% | 22.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 373.65 | 372.94 | 0.00 | ORB-long ORB[369.90,373.00] vol=2.5x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 372.31 | 372.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 368.45 | 372.48 | 0.00 | ORB-short ORB[372.25,375.90] vol=2.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 370.45 | 372.04 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:00:00 | 379.35 | 382.56 | 0.00 | ORB-short ORB[380.05,384.80] vol=1.5x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:05:00 | 377.20 | 381.91 | 0.00 | T1 1.5R @ 377.20 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 379.35 | 381.62 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 363.10 | 365.03 | 0.00 | ORB-short ORB[363.80,367.70] vol=4.0x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:45:00 | 360.88 | 364.43 | 0.00 | T1 1.5R @ 360.88 |
| Target hit | 2024-05-31 15:20:00 | 357.55 | 358.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 385.00 | 382.39 | 0.00 | ORB-long ORB[379.55,382.80] vol=2.5x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-06-07 09:35:00 | 383.45 | 382.85 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:35:00 | 386.40 | 383.08 | 0.00 | ORB-long ORB[382.55,385.50] vol=2.2x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:40:00 | 388.48 | 384.86 | 0.00 | T1 1.5R @ 388.48 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 386.40 | 385.30 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 11:10:00 | 423.40 | 425.23 | 0.00 | ORB-short ORB[423.75,428.40] vol=1.9x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-06-21 11:20:00 | 424.86 | 425.16 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:05:00 | 436.45 | 435.35 | 0.00 | ORB-long ORB[420.40,427.00] vol=1.5x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-06-24 10:15:00 | 433.22 | 435.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 498.35 | 492.33 | 0.00 | ORB-long ORB[483.00,489.60] vol=10.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 09:35:00 | 502.88 | 496.10 | 0.00 | T1 1.5R @ 502.88 |
| Stop hit — per-position SL triggered | 2024-07-04 09:40:00 | 498.35 | 496.25 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:45:00 | 491.55 | 493.75 | 0.00 | ORB-short ORB[492.30,496.20] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-07-12 11:00:00 | 493.18 | 492.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:00:00 | 497.90 | 492.72 | 0.00 | ORB-long ORB[488.00,493.90] vol=3.1x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:05:00 | 500.54 | 495.04 | 0.00 | T1 1.5R @ 500.54 |
| Target hit | 2024-07-16 14:05:00 | 499.75 | 500.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 489.60 | 493.86 | 0.00 | ORB-short ORB[494.00,499.30] vol=5.8x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:55:00 | 486.17 | 490.56 | 0.00 | T1 1.5R @ 486.17 |
| Stop hit — per-position SL triggered | 2024-07-18 12:10:00 | 489.60 | 490.05 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 476.50 | 483.81 | 0.00 | ORB-short ORB[483.30,489.90] vol=1.6x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 472.38 | 481.58 | 0.00 | T1 1.5R @ 472.38 |
| Stop hit — per-position SL triggered | 2024-07-19 13:30:00 | 476.50 | 477.87 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:35:00 | 481.40 | 476.52 | 0.00 | ORB-long ORB[474.90,479.45] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-07-23 10:45:00 | 479.32 | 477.26 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 495.50 | 497.42 | 0.00 | ORB-short ORB[496.45,502.50] vol=3.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-07-29 10:10:00 | 497.54 | 496.97 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 493.30 | 497.70 | 0.00 | ORB-short ORB[498.05,502.35] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-08-01 10:55:00 | 494.79 | 497.67 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:50:00 | 500.75 | 504.82 | 0.00 | ORB-short ORB[504.05,511.45] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 502.70 | 504.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 531.50 | 535.53 | 0.00 | ORB-short ORB[534.10,539.45] vol=1.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-08-22 09:40:00 | 533.95 | 534.86 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 532.20 | 534.29 | 0.00 | ORB-short ORB[533.00,537.25] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-08-23 09:40:00 | 533.84 | 537.51 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:05:00 | 525.00 | 518.23 | 0.00 | ORB-long ORB[513.60,517.50] vol=5.7x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:10:00 | 528.89 | 521.34 | 0.00 | T1 1.5R @ 528.89 |
| Target hit | 2024-08-30 15:20:00 | 547.00 | 540.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-09-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:45:00 | 517.80 | 515.64 | 0.00 | ORB-long ORB[510.05,517.30] vol=1.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 516.11 | 515.72 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:05:00 | 519.10 | 522.45 | 0.00 | ORB-short ORB[521.20,526.80] vol=7.4x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 12:45:00 | 516.78 | 521.45 | 0.00 | T1 1.5R @ 516.78 |
| Target hit | 2024-09-18 15:20:00 | 513.55 | 518.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 09:35:00 | 529.20 | 533.47 | 0.00 | ORB-short ORB[531.90,538.85] vol=3.1x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:50:00 | 524.97 | 531.11 | 0.00 | T1 1.5R @ 524.97 |
| Target hit | 2024-09-23 15:20:00 | 526.90 | 529.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2024-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:50:00 | 525.00 | 526.47 | 0.00 | ORB-short ORB[525.55,531.85] vol=3.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-09-25 10:00:00 | 526.82 | 526.38 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:35:00 | 526.95 | 521.69 | 0.00 | ORB-long ORB[517.30,523.05] vol=8.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-09-27 10:40:00 | 524.79 | 522.13 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 453.70 | 455.98 | 0.00 | ORB-short ORB[453.80,460.20] vol=1.5x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-10-14 09:55:00 | 455.32 | 453.92 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:00:00 | 439.45 | 440.24 | 0.00 | ORB-short ORB[440.20,445.35] vol=1.6x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:00:00 | 436.78 | 439.95 | 0.00 | T1 1.5R @ 436.78 |
| Target hit | 2024-10-21 15:20:00 | 432.85 | 438.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-10-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:40:00 | 427.65 | 434.47 | 0.00 | ORB-short ORB[434.00,439.60] vol=3.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:50:00 | 424.93 | 432.58 | 0.00 | T1 1.5R @ 424.93 |
| Stop hit — per-position SL triggered | 2024-10-29 11:45:00 | 427.65 | 428.26 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-11-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:00:00 | 461.90 | 466.46 | 0.00 | ORB-short ORB[464.50,470.10] vol=2.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:30:00 | 459.07 | 465.28 | 0.00 | T1 1.5R @ 459.07 |
| Stop hit — per-position SL triggered | 2024-11-07 10:50:00 | 461.90 | 464.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 432.40 | 429.91 | 0.00 | ORB-long ORB[424.05,429.60] vol=4.9x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-11-19 09:40:00 | 430.28 | 430.25 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 442.75 | 439.65 | 0.00 | ORB-long ORB[434.15,439.60] vol=5.9x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:35:00 | 445.84 | 453.62 | 0.00 | T1 1.5R @ 445.84 |
| Target hit | 2024-12-06 15:20:00 | 478.45 | 474.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:15:00 | 502.20 | 498.13 | 0.00 | ORB-long ORB[495.00,501.90] vol=2.2x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:35:00 | 505.40 | 499.01 | 0.00 | T1 1.5R @ 505.40 |
| Stop hit — per-position SL triggered | 2024-12-19 12:00:00 | 502.20 | 499.55 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 480.85 | 482.95 | 0.00 | ORB-short ORB[481.40,484.95] vol=2.4x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-12-26 09:35:00 | 482.52 | 482.76 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 475.70 | 478.14 | 0.00 | ORB-short ORB[476.70,480.95] vol=2.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-12-27 12:05:00 | 477.12 | 477.69 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 11:00:00 | 470.85 | 466.04 | 0.00 | ORB-long ORB[461.20,468.05] vol=3.0x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-01-07 11:05:00 | 468.39 | 466.12 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 435.60 | 439.02 | 0.00 | ORB-short ORB[436.85,443.05] vol=2.4x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:20:00 | 431.84 | 437.81 | 0.00 | T1 1.5R @ 431.84 |
| Target hit | 2025-01-13 15:20:00 | 427.00 | 432.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-02-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:25:00 | 493.95 | 491.75 | 0.00 | ORB-long ORB[486.65,493.40] vol=1.7x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-02-06 10:50:00 | 491.26 | 492.35 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:10:00 | 472.85 | 468.71 | 0.00 | ORB-long ORB[461.40,467.90] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-02-20 11:25:00 | 471.06 | 468.88 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 10:05:00 | 491.70 | 488.35 | 0.00 | ORB-long ORB[484.55,490.85] vol=1.6x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-02-27 11:00:00 | 489.06 | 489.31 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:05:00 | 564.40 | 561.60 | 0.00 | ORB-long ORB[555.65,562.45] vol=1.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-04-16 11:30:00 | 562.16 | 561.66 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 576.10 | 580.19 | 0.00 | ORB-short ORB[576.70,583.90] vol=2.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 578.44 | 579.35 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:50:00 | 558.95 | 563.47 | 0.00 | ORB-short ORB[563.40,571.45] vol=1.5x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:15:00 | 554.96 | 561.12 | 0.00 | T1 1.5R @ 554.96 |
| Target hit | 2025-04-29 15:20:00 | 549.85 | 555.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:35:00 | 543.90 | 540.84 | 0.00 | ORB-long ORB[537.00,543.20] vol=1.9x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 15:00:00 | 548.65 | 544.24 | 0.00 | T1 1.5R @ 548.65 |
| Target hit | 2025-05-05 15:20:00 | 544.35 | 544.53 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 09:35:00 | 373.65 | 2024-05-17 09:40:00 | 372.31 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-22 09:40:00 | 368.45 | 2024-05-22 09:45:00 | 370.45 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-05-28 10:00:00 | 379.35 | 2024-05-28 10:05:00 | 377.20 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-05-28 10:00:00 | 379.35 | 2024-05-28 10:15:00 | 379.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 09:30:00 | 363.10 | 2024-05-31 09:45:00 | 360.88 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-05-31 09:30:00 | 363.10 | 2024-05-31 15:20:00 | 357.55 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2024-06-07 09:30:00 | 385.00 | 2024-06-07 09:35:00 | 383.45 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-11 10:35:00 | 386.40 | 2024-06-11 10:40:00 | 388.48 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-11 10:35:00 | 386.40 | 2024-06-11 11:00:00 | 386.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 11:10:00 | 423.40 | 2024-06-21 11:20:00 | 424.86 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-24 10:05:00 | 436.45 | 2024-06-24 10:15:00 | 433.22 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-07-04 09:30:00 | 498.35 | 2024-07-04 09:35:00 | 502.88 | PARTIAL | 0.50 | 0.91% |
| BUY | retest1 | 2024-07-04 09:30:00 | 498.35 | 2024-07-04 09:40:00 | 498.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 09:45:00 | 491.55 | 2024-07-12 11:00:00 | 493.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-16 10:00:00 | 497.90 | 2024-07-16 10:05:00 | 500.54 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-16 10:00:00 | 497.90 | 2024-07-16 14:05:00 | 499.75 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-18 09:30:00 | 489.60 | 2024-07-18 10:55:00 | 486.17 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-07-18 09:30:00 | 489.60 | 2024-07-18 12:10:00 | 489.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:00:00 | 476.50 | 2024-07-19 10:10:00 | 472.38 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2024-07-19 10:00:00 | 476.50 | 2024-07-19 13:30:00 | 476.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 10:35:00 | 481.40 | 2024-07-23 10:45:00 | 479.32 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-29 09:30:00 | 495.50 | 2024-07-29 10:10:00 | 497.54 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-01 10:50:00 | 493.30 | 2024-08-01 10:55:00 | 494.79 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-13 10:50:00 | 500.75 | 2024-08-13 11:10:00 | 502.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-22 09:30:00 | 531.50 | 2024-08-22 09:40:00 | 533.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-23 09:30:00 | 532.20 | 2024-08-23 09:40:00 | 533.84 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-30 10:05:00 | 525.00 | 2024-08-30 10:10:00 | 528.89 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-08-30 10:05:00 | 525.00 | 2024-08-30 15:20:00 | 547.00 | TARGET_HIT | 0.50 | 4.19% |
| BUY | retest1 | 2024-09-13 09:45:00 | 517.80 | 2024-09-13 09:50:00 | 516.11 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-18 11:05:00 | 519.10 | 2024-09-18 12:45:00 | 516.78 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-18 11:05:00 | 519.10 | 2024-09-18 15:20:00 | 513.55 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2024-09-23 09:35:00 | 529.20 | 2024-09-23 11:50:00 | 524.97 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-09-23 09:35:00 | 529.20 | 2024-09-23 15:20:00 | 526.90 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-25 09:50:00 | 525.00 | 2024-09-25 10:00:00 | 526.82 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-27 10:35:00 | 526.95 | 2024-09-27 10:40:00 | 524.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-14 09:30:00 | 453.70 | 2024-10-14 09:55:00 | 455.32 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-21 11:00:00 | 439.45 | 2024-10-21 12:00:00 | 436.78 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-21 11:00:00 | 439.45 | 2024-10-21 15:20:00 | 432.85 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-10-29 10:40:00 | 427.65 | 2024-10-29 10:50:00 | 424.93 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-10-29 10:40:00 | 427.65 | 2024-10-29 11:45:00 | 427.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:00:00 | 461.90 | 2024-11-07 10:30:00 | 459.07 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-07 10:00:00 | 461.90 | 2024-11-07 10:50:00 | 461.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:30:00 | 432.40 | 2024-11-19 09:40:00 | 430.28 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-06 09:30:00 | 442.75 | 2024-12-06 09:35:00 | 445.84 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-06 09:30:00 | 442.75 | 2024-12-06 15:20:00 | 478.45 | TARGET_HIT | 0.50 | 8.06% |
| BUY | retest1 | 2024-12-19 11:15:00 | 502.20 | 2024-12-19 11:35:00 | 505.40 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-12-19 11:15:00 | 502.20 | 2024-12-19 12:00:00 | 502.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:30:00 | 480.85 | 2024-12-26 09:35:00 | 482.52 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-27 10:55:00 | 475.70 | 2024-12-27 12:05:00 | 477.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-07 11:00:00 | 470.85 | 2025-01-07 11:05:00 | 468.39 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-01-13 11:10:00 | 435.60 | 2025-01-13 12:20:00 | 431.84 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2025-01-13 11:10:00 | 435.60 | 2025-01-13 15:20:00 | 427.00 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2025-02-06 10:25:00 | 493.95 | 2025-02-06 10:50:00 | 491.26 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-02-20 11:10:00 | 472.85 | 2025-02-20 11:25:00 | 471.06 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-27 10:05:00 | 491.70 | 2025-02-27 11:00:00 | 489.06 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-04-16 11:05:00 | 564.40 | 2025-04-16 11:30:00 | 562.16 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-23 09:30:00 | 576.10 | 2025-04-23 09:40:00 | 578.44 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-29 09:50:00 | 558.95 | 2025-04-29 10:15:00 | 554.96 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-04-29 09:50:00 | 558.95 | 2025-04-29 15:20:00 | 549.85 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2025-05-05 09:35:00 | 543.90 | 2025-05-05 15:00:00 | 548.65 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2025-05-05 09:35:00 | 543.90 | 2025-05-05 15:20:00 | 544.35 | TARGET_HIT | 0.50 | 0.08% |
