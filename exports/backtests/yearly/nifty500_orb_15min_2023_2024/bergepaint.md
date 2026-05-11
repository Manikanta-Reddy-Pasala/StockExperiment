# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53779 bars)
- **Last close:** 515.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 8 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 81
- **Target hits / Stop hits / Partials:** 8 / 81 / 39
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 14.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 23 | 41.8% | 4 | 32 | 19 | 0.24% | 13.4% |
| BUY @ 2nd Alert (retest1) | 55 | 23 | 41.8% | 4 | 32 | 19 | 0.24% | 13.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 24 | 32.9% | 4 | 49 | 20 | 0.01% | 0.9% |
| SELL @ 2nd Alert (retest1) | 73 | 24 | 32.9% | 4 | 49 | 20 | 0.01% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 47 | 36.7% | 8 | 81 | 39 | 0.11% | 14.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:55:00 | 528.17 | 526.14 | 0.00 | ORB-long ORB[522.79,528.04] vol=1.6x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 10:10:00 | 530.63 | 527.32 | 0.00 | T1 1.5R @ 530.63 |
| Stop hit — per-position SL triggered | 2023-05-15 10:20:00 | 528.17 | 527.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 11:10:00 | 520.13 | 517.43 | 0.00 | ORB-long ORB[514.79,518.92] vol=3.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-05-18 11:20:00 | 518.76 | 517.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 520.46 | 522.32 | 0.00 | ORB-short ORB[520.83,524.79] vol=1.9x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:00:00 | 517.95 | 521.30 | 0.00 | T1 1.5R @ 517.95 |
| Stop hit — per-position SL triggered | 2023-05-19 11:15:00 | 520.46 | 519.61 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:50:00 | 523.42 | 519.77 | 0.00 | ORB-long ORB[512.54,520.29] vol=2.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-05-22 11:00:00 | 522.01 | 519.94 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:55:00 | 526.83 | 524.21 | 0.00 | ORB-long ORB[520.46,524.96] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-05-24 10:05:00 | 525.68 | 524.44 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 10:30:00 | 529.54 | 527.74 | 0.00 | ORB-long ORB[525.21,527.79] vol=2.2x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-05-25 10:40:00 | 528.27 | 527.79 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:20:00 | 537.50 | 535.17 | 0.00 | ORB-long ORB[532.79,536.67] vol=1.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-05-26 10:30:00 | 536.17 | 535.23 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:35:00 | 532.17 | 535.61 | 0.00 | ORB-short ORB[534.17,537.46] vol=1.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2023-05-30 10:45:00 | 533.40 | 535.17 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:20:00 | 539.92 | 542.71 | 0.00 | ORB-short ORB[540.21,544.13] vol=2.3x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-06-06 10:30:00 | 541.11 | 542.50 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 11:05:00 | 543.08 | 544.92 | 0.00 | ORB-short ORB[544.58,547.88] vol=1.6x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:35:00 | 541.73 | 544.37 | 0.00 | T1 1.5R @ 541.73 |
| Target hit | 2023-06-08 15:20:00 | 541.83 | 542.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 09:45:00 | 548.83 | 546.64 | 0.00 | ORB-long ORB[542.63,547.50] vol=2.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:50:00 | 550.82 | 549.44 | 0.00 | T1 1.5R @ 550.82 |
| Stop hit — per-position SL triggered | 2023-06-09 10:05:00 | 548.83 | 550.89 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:40:00 | 563.71 | 561.52 | 0.00 | ORB-long ORB[556.25,563.13] vol=1.8x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:25:00 | 565.98 | 563.19 | 0.00 | T1 1.5R @ 565.98 |
| Stop hit — per-position SL triggered | 2023-06-15 10:55:00 | 563.71 | 563.31 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:40:00 | 567.83 | 564.74 | 0.00 | ORB-long ORB[559.42,565.25] vol=1.9x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-06-26 09:50:00 | 565.88 | 565.28 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:55:00 | 561.08 | 563.87 | 0.00 | ORB-short ORB[563.42,568.71] vol=2.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-07-03 10:00:00 | 562.45 | 563.52 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 11:10:00 | 561.50 | 563.35 | 0.00 | ORB-short ORB[563.38,566.67] vol=2.8x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 12:35:00 | 559.93 | 562.62 | 0.00 | T1 1.5R @ 559.93 |
| Target hit | 2023-07-04 15:20:00 | 558.79 | 560.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2023-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:35:00 | 564.67 | 562.83 | 0.00 | ORB-long ORB[558.33,563.92] vol=2.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2023-07-05 10:55:00 | 563.63 | 563.08 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:35:00 | 569.17 | 566.64 | 0.00 | ORB-long ORB[562.42,568.33] vol=3.0x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 09:45:00 | 571.49 | 568.25 | 0.00 | T1 1.5R @ 571.49 |
| Target hit | 2023-07-07 10:45:00 | 570.13 | 570.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2023-07-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:35:00 | 564.13 | 561.50 | 0.00 | ORB-long ORB[556.21,560.46] vol=1.7x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 11:10:00 | 565.79 | 562.27 | 0.00 | T1 1.5R @ 565.79 |
| Stop hit — per-position SL triggered | 2023-07-11 11:15:00 | 564.13 | 562.30 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:30:00 | 550.58 | 553.63 | 0.00 | ORB-short ORB[552.25,559.04] vol=1.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-07-12 09:35:00 | 552.17 | 553.42 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:30:00 | 564.00 | 561.44 | 0.00 | ORB-long ORB[556.92,562.79] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 09:35:00 | 566.34 | 562.81 | 0.00 | T1 1.5R @ 566.34 |
| Stop hit — per-position SL triggered | 2023-07-17 10:20:00 | 564.00 | 564.29 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 563.29 | 566.34 | 0.00 | ORB-short ORB[565.04,570.33] vol=1.8x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-07-19 09:45:00 | 564.86 | 565.45 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 11:00:00 | 570.63 | 573.23 | 0.00 | ORB-short ORB[571.29,578.75] vol=6.1x ATR=1.76 |
| Stop hit — per-position SL triggered | 2023-07-24 12:05:00 | 572.39 | 572.66 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:50:00 | 588.92 | 590.49 | 0.00 | ORB-short ORB[589.00,593.29] vol=1.7x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:10:00 | 586.57 | 589.74 | 0.00 | T1 1.5R @ 586.57 |
| Stop hit — per-position SL triggered | 2023-08-08 14:50:00 | 588.92 | 587.67 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:40:00 | 582.88 | 586.64 | 0.00 | ORB-short ORB[585.42,591.50] vol=2.1x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-08-18 09:45:00 | 584.61 | 586.50 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 10:25:00 | 584.21 | 585.82 | 0.00 | ORB-short ORB[584.67,589.75] vol=2.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-08-28 10:50:00 | 585.63 | 585.06 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:35:00 | 595.42 | 593.69 | 0.00 | ORB-long ORB[590.00,595.29] vol=1.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-08-29 09:40:00 | 594.20 | 593.87 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:50:00 | 591.71 | 590.22 | 0.00 | ORB-long ORB[585.21,590.33] vol=4.1x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 11:20:00 | 593.65 | 590.68 | 0.00 | T1 1.5R @ 593.65 |
| Stop hit — per-position SL triggered | 2023-09-05 11:50:00 | 591.71 | 591.49 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 11:00:00 | 590.29 | 586.79 | 0.00 | ORB-long ORB[583.38,589.50] vol=2.0x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-09-06 11:20:00 | 589.03 | 587.09 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 11:15:00 | 591.29 | 594.32 | 0.00 | ORB-short ORB[591.33,597.00] vol=5.1x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-09-07 11:45:00 | 592.45 | 593.82 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 596.71 | 602.72 | 0.00 | ORB-short ORB[600.83,607.25] vol=2.0x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:45:00 | 593.90 | 601.77 | 0.00 | T1 1.5R @ 593.90 |
| Stop hit — per-position SL triggered | 2023-09-12 10:05:00 | 596.71 | 599.39 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 09:30:00 | 599.17 | 597.00 | 0.00 | ORB-long ORB[591.88,598.92] vol=2.0x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 10:00:00 | 602.56 | 598.82 | 0.00 | T1 1.5R @ 602.56 |
| Stop hit — per-position SL triggered | 2023-09-13 10:20:00 | 599.17 | 599.21 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 11:10:00 | 752.40 | 749.64 | 0.00 | ORB-long ORB[743.05,750.40] vol=2.0x ATR=3.46 |
| Stop hit — per-position SL triggered | 2023-09-21 12:15:00 | 748.94 | 749.93 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 09:55:00 | 569.40 | 573.25 | 0.00 | ORB-short ORB[572.90,578.00] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-11-10 10:05:00 | 570.91 | 572.87 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 574.05 | 575.60 | 0.00 | ORB-short ORB[574.50,577.75] vol=2.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-11-13 10:35:00 | 575.32 | 575.45 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-11-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:05:00 | 577.00 | 575.38 | 0.00 | ORB-long ORB[572.70,576.75] vol=1.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-11-16 12:05:00 | 576.08 | 575.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-11-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 10:55:00 | 585.00 | 583.05 | 0.00 | ORB-long ORB[580.00,584.35] vol=2.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-11-21 11:40:00 | 583.88 | 583.51 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:55:00 | 574.20 | 575.36 | 0.00 | ORB-short ORB[575.05,578.50] vol=3.3x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 12:25:00 | 572.36 | 574.79 | 0.00 | T1 1.5R @ 572.36 |
| Stop hit — per-position SL triggered | 2023-11-24 14:50:00 | 574.20 | 573.92 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-11-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:35:00 | 576.30 | 573.75 | 0.00 | ORB-long ORB[571.35,573.90] vol=3.0x ATR=1.20 |
| Stop hit — per-position SL triggered | 2023-11-30 09:40:00 | 575.10 | 573.85 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:40:00 | 591.70 | 589.81 | 0.00 | ORB-long ORB[585.50,591.40] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-12-05 10:15:00 | 590.25 | 590.27 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-12-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:30:00 | 585.45 | 587.75 | 0.00 | ORB-short ORB[588.95,593.40] vol=5.5x ATR=1.50 |
| Stop hit — per-position SL triggered | 2023-12-06 10:45:00 | 586.95 | 587.71 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-12-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:30:00 | 594.35 | 588.97 | 0.00 | ORB-long ORB[583.00,589.00] vol=3.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2023-12-07 09:35:00 | 592.18 | 589.78 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:50:00 | 582.90 | 585.67 | 0.00 | ORB-short ORB[587.50,590.85] vol=2.6x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:00:00 | 580.95 | 585.10 | 0.00 | T1 1.5R @ 580.95 |
| Target hit | 2023-12-08 15:20:00 | 579.45 | 581.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 11:15:00 | 576.45 | 578.30 | 0.00 | ORB-short ORB[576.55,580.55] vol=2.3x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 12:05:00 | 574.83 | 577.58 | 0.00 | T1 1.5R @ 574.83 |
| Stop hit — per-position SL triggered | 2023-12-11 15:05:00 | 576.45 | 576.48 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-12-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:45:00 | 574.35 | 577.02 | 0.00 | ORB-short ORB[576.50,580.85] vol=2.4x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 12:45:00 | 572.91 | 575.31 | 0.00 | T1 1.5R @ 572.91 |
| Stop hit — per-position SL triggered | 2023-12-12 12:50:00 | 574.35 | 575.33 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:30:00 | 567.65 | 572.02 | 0.00 | ORB-short ORB[572.30,577.95] vol=1.9x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 11:45:00 | 565.37 | 569.97 | 0.00 | T1 1.5R @ 565.37 |
| Stop hit — per-position SL triggered | 2023-12-13 12:25:00 | 567.65 | 568.49 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:10:00 | 589.00 | 587.16 | 0.00 | ORB-long ORB[581.60,587.80] vol=1.7x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 12:25:00 | 590.81 | 588.12 | 0.00 | T1 1.5R @ 590.81 |
| Target hit | 2023-12-18 15:20:00 | 593.80 | 591.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2023-12-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:50:00 | 593.85 | 594.49 | 0.00 | ORB-short ORB[594.30,600.00] vol=1.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-12-19 11:25:00 | 595.21 | 594.41 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:50:00 | 592.00 | 594.29 | 0.00 | ORB-short ORB[592.10,597.95] vol=5.8x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-12-20 11:10:00 | 593.67 | 593.86 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:50:00 | 580.10 | 577.30 | 0.00 | ORB-long ORB[574.25,577.60] vol=4.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 11:05:00 | 581.91 | 579.17 | 0.00 | T1 1.5R @ 581.91 |
| Stop hit — per-position SL triggered | 2023-12-22 11:10:00 | 580.10 | 579.20 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 11:15:00 | 586.30 | 584.80 | 0.00 | ORB-long ORB[582.35,586.00] vol=2.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 12:15:00 | 588.33 | 585.41 | 0.00 | T1 1.5R @ 588.33 |
| Stop hit — per-position SL triggered | 2023-12-26 14:20:00 | 586.30 | 585.80 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-12-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:50:00 | 589.20 | 586.78 | 0.00 | ORB-long ORB[585.15,588.80] vol=2.9x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:55:00 | 591.08 | 587.49 | 0.00 | T1 1.5R @ 591.08 |
| Stop hit — per-position SL triggered | 2023-12-28 11:45:00 | 589.20 | 588.77 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 11:05:00 | 600.80 | 597.75 | 0.00 | ORB-long ORB[595.00,599.95] vol=3.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:15:00 | 603.51 | 598.54 | 0.00 | T1 1.5R @ 603.51 |
| Stop hit — per-position SL triggered | 2023-12-29 12:35:00 | 600.80 | 600.93 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:40:00 | 595.10 | 599.09 | 0.00 | ORB-short ORB[600.10,603.00] vol=2.2x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-01-02 11:20:00 | 596.72 | 598.27 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 601.35 | 603.42 | 0.00 | ORB-short ORB[602.30,606.95] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-01-03 09:50:00 | 602.93 | 603.25 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-01-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:45:00 | 592.90 | 596.58 | 0.00 | ORB-short ORB[597.00,601.30] vol=6.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-01-04 09:50:00 | 594.71 | 596.20 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-01-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:00:00 | 601.30 | 599.35 | 0.00 | ORB-long ORB[596.05,600.00] vol=2.1x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:30:00 | 603.34 | 600.47 | 0.00 | T1 1.5R @ 603.34 |
| Stop hit — per-position SL triggered | 2024-01-05 10:45:00 | 601.30 | 601.09 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:05:00 | 587.95 | 592.16 | 0.00 | ORB-short ORB[591.85,596.90] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-01-08 10:30:00 | 589.65 | 591.28 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 10:30:00 | 582.00 | 584.27 | 0.00 | ORB-short ORB[582.20,586.05] vol=1.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-01-09 11:05:00 | 583.38 | 583.93 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:10:00 | 555.80 | 559.11 | 0.00 | ORB-short ORB[558.50,563.00] vol=1.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:20:00 | 553.79 | 558.63 | 0.00 | T1 1.5R @ 553.79 |
| Stop hit — per-position SL triggered | 2024-01-25 10:30:00 | 555.80 | 558.16 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 10:45:00 | 559.50 | 562.47 | 0.00 | ORB-short ORB[561.80,566.10] vol=2.3x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-01-30 11:25:00 | 560.72 | 561.81 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:10:00 | 563.30 | 565.12 | 0.00 | ORB-short ORB[564.00,567.20] vol=1.9x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-02-01 11:15:00 | 564.26 | 565.13 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:55:00 | 566.55 | 570.57 | 0.00 | ORB-short ORB[570.95,574.80] vol=2.3x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-02-05 11:00:00 | 568.12 | 570.35 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 10:55:00 | 558.00 | 560.46 | 0.00 | ORB-short ORB[560.05,567.10] vol=2.0x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 12:45:00 | 555.54 | 557.91 | 0.00 | T1 1.5R @ 555.54 |
| Target hit | 2024-02-06 15:20:00 | 555.45 | 557.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-02-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:40:00 | 554.00 | 556.15 | 0.00 | ORB-short ORB[554.45,558.55] vol=1.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:00:00 | 551.87 | 555.73 | 0.00 | T1 1.5R @ 551.87 |
| Stop hit — per-position SL triggered | 2024-02-08 11:25:00 | 554.00 | 555.08 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 11:05:00 | 546.80 | 550.37 | 0.00 | ORB-short ORB[550.65,554.20] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-02-12 11:10:00 | 548.41 | 550.22 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:45:00 | 554.50 | 550.42 | 0.00 | ORB-long ORB[547.40,552.00] vol=2.4x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 12:35:00 | 556.94 | 552.56 | 0.00 | T1 1.5R @ 556.94 |
| Target hit | 2024-02-13 15:20:00 | 558.90 | 554.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2024-02-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:45:00 | 556.05 | 558.84 | 0.00 | ORB-short ORB[557.85,562.90] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-02-15 09:50:00 | 557.66 | 558.60 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-02-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:30:00 | 565.20 | 564.17 | 0.00 | ORB-long ORB[561.50,565.00] vol=1.7x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:45:00 | 567.04 | 565.00 | 0.00 | T1 1.5R @ 567.04 |
| Stop hit — per-position SL triggered | 2024-02-21 10:10:00 | 565.20 | 565.46 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 11:05:00 | 565.00 | 567.55 | 0.00 | ORB-short ORB[568.00,572.85] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-02-23 12:05:00 | 566.34 | 566.92 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 558.10 | 559.45 | 0.00 | ORB-short ORB[559.80,564.60] vol=3.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-02-28 10:55:00 | 559.73 | 559.40 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-02-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 10:35:00 | 566.65 | 561.85 | 0.00 | ORB-long ORB[556.00,564.00] vol=2.2x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 10:50:00 | 570.07 | 565.51 | 0.00 | T1 1.5R @ 570.07 |
| Target hit | 2024-02-29 15:20:00 | 614.45 | 591.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 571.15 | 573.39 | 0.00 | ORB-short ORB[572.75,575.85] vol=1.5x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:00:00 | 568.69 | 572.41 | 0.00 | T1 1.5R @ 568.69 |
| Stop hit — per-position SL triggered | 2024-03-06 10:10:00 | 571.15 | 572.13 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:45:00 | 568.10 | 571.19 | 0.00 | ORB-short ORB[570.45,576.30] vol=2.1x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-03-12 10:55:00 | 569.74 | 570.99 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:50:00 | 545.45 | 548.25 | 0.00 | ORB-short ORB[546.10,552.50] vol=2.0x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:05:00 | 542.95 | 547.29 | 0.00 | T1 1.5R @ 542.95 |
| Stop hit — per-position SL triggered | 2024-03-19 10:10:00 | 545.45 | 547.13 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-03-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:55:00 | 554.30 | 552.33 | 0.00 | ORB-long ORB[549.10,554.15] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-03-21 11:05:00 | 553.00 | 552.40 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 10:55:00 | 563.50 | 565.19 | 0.00 | ORB-short ORB[564.45,567.90] vol=2.8x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-04-02 11:15:00 | 564.64 | 565.05 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 11:10:00 | 559.95 | 558.33 | 0.00 | ORB-long ORB[555.00,559.80] vol=2.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-04-05 11:35:00 | 558.89 | 558.75 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 558.05 | 560.76 | 0.00 | ORB-short ORB[560.05,563.00] vol=2.2x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-04-12 09:50:00 | 559.37 | 560.53 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-15 09:35:00 | 544.70 | 546.38 | 0.00 | ORB-short ORB[545.10,552.55] vol=2.3x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-04-15 09:40:00 | 546.52 | 546.33 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 09:55:00 | 505.80 | 509.95 | 0.00 | ORB-short ORB[509.05,516.00] vol=1.7x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:35:00 | 502.94 | 507.89 | 0.00 | T1 1.5R @ 502.94 |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 505.80 | 506.55 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:45:00 | 503.50 | 507.69 | 0.00 | ORB-short ORB[507.45,511.50] vol=2.1x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-04-23 11:40:00 | 504.85 | 505.70 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:30:00 | 504.80 | 505.79 | 0.00 | ORB-short ORB[505.00,508.20] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-04-25 09:35:00 | 505.59 | 505.75 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 11:00:00 | 505.00 | 505.56 | 0.00 | ORB-short ORB[505.10,507.80] vol=3.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 12:15:00 | 504.13 | 505.33 | 0.00 | T1 1.5R @ 504.13 |
| Stop hit — per-position SL triggered | 2024-04-29 15:20:00 | 505.05 | 504.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2024-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:30:00 | 511.65 | 509.59 | 0.00 | ORB-long ORB[505.75,510.55] vol=3.3x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 09:50:00 | 513.17 | 511.00 | 0.00 | T1 1.5R @ 513.17 |
| Stop hit — per-position SL triggered | 2024-04-30 11:05:00 | 511.65 | 512.62 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-05-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 11:00:00 | 516.90 | 520.02 | 0.00 | ORB-short ORB[520.55,526.90] vol=1.5x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:05:00 | 514.58 | 519.42 | 0.00 | T1 1.5R @ 514.58 |
| Stop hit — per-position SL triggered | 2024-05-03 11:45:00 | 516.90 | 518.68 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 520.05 | 522.02 | 0.00 | ORB-short ORB[520.75,527.75] vol=2.9x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:50:00 | 517.20 | 521.27 | 0.00 | T1 1.5R @ 517.20 |
| Stop hit — per-position SL triggered | 2024-05-06 10:20:00 | 520.05 | 520.88 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:30:00 | 516.15 | 515.24 | 0.00 | ORB-long ORB[512.30,516.00] vol=3.4x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:40:00 | 518.43 | 515.67 | 0.00 | T1 1.5R @ 518.43 |
| Stop hit — per-position SL triggered | 2024-05-07 10:20:00 | 516.15 | 516.34 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-05-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 10:40:00 | 508.95 | 511.75 | 0.00 | ORB-short ORB[509.30,514.30] vol=2.4x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 11:05:00 | 506.81 | 510.67 | 0.00 | T1 1.5R @ 506.81 |
| Stop hit — per-position SL triggered | 2024-05-08 12:15:00 | 508.95 | 509.45 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-05-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:05:00 | 489.05 | 494.95 | 0.00 | ORB-short ORB[496.25,501.65] vol=2.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-05-09 11:20:00 | 490.56 | 494.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:55:00 | 528.17 | 2023-05-15 10:10:00 | 530.63 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-05-15 09:55:00 | 528.17 | 2023-05-15 10:20:00 | 528.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-18 11:10:00 | 520.13 | 2023-05-18 11:20:00 | 518.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-05-19 09:30:00 | 520.46 | 2023-05-19 10:00:00 | 517.95 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-05-19 09:30:00 | 520.46 | 2023-05-19 11:15:00 | 520.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 10:50:00 | 523.42 | 2023-05-22 11:00:00 | 522.01 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-24 09:55:00 | 526.83 | 2023-05-24 10:05:00 | 525.68 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-05-25 10:30:00 | 529.54 | 2023-05-25 10:40:00 | 528.27 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-26 10:20:00 | 537.50 | 2023-05-26 10:30:00 | 536.17 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-05-30 10:35:00 | 532.17 | 2023-05-30 10:45:00 | 533.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-06 10:20:00 | 539.92 | 2023-06-06 10:30:00 | 541.11 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-08 11:05:00 | 543.08 | 2023-06-08 11:35:00 | 541.73 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-06-08 11:05:00 | 543.08 | 2023-06-08 15:20:00 | 541.83 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2023-06-09 09:45:00 | 548.83 | 2023-06-09 09:50:00 | 550.82 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-06-09 09:45:00 | 548.83 | 2023-06-09 10:05:00 | 548.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-15 09:40:00 | 563.71 | 2023-06-15 10:25:00 | 565.98 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-06-15 09:40:00 | 563.71 | 2023-06-15 10:55:00 | 563.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-26 09:40:00 | 567.83 | 2023-06-26 09:50:00 | 565.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-03 09:55:00 | 561.08 | 2023-07-03 10:00:00 | 562.45 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-04 11:10:00 | 561.50 | 2023-07-04 12:35:00 | 559.93 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-07-04 11:10:00 | 561.50 | 2023-07-04 15:20:00 | 558.79 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-05 10:35:00 | 564.67 | 2023-07-05 10:55:00 | 563.63 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-07 09:35:00 | 569.17 | 2023-07-07 09:45:00 | 571.49 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-07 09:35:00 | 569.17 | 2023-07-07 10:45:00 | 570.13 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2023-07-11 10:35:00 | 564.13 | 2023-07-11 11:10:00 | 565.79 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-11 10:35:00 | 564.13 | 2023-07-11 11:15:00 | 564.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-12 09:30:00 | 550.58 | 2023-07-12 09:35:00 | 552.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-17 09:30:00 | 564.00 | 2023-07-17 09:35:00 | 566.34 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-07-17 09:30:00 | 564.00 | 2023-07-17 10:20:00 | 564.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-19 09:30:00 | 563.29 | 2023-07-19 09:45:00 | 564.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-24 11:00:00 | 570.63 | 2023-07-24 12:05:00 | 572.39 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-08 09:50:00 | 588.92 | 2023-08-08 10:10:00 | 586.57 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-08-08 09:50:00 | 588.92 | 2023-08-08 14:50:00 | 588.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-18 09:40:00 | 582.88 | 2023-08-18 09:45:00 | 584.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-28 10:25:00 | 584.21 | 2023-08-28 10:50:00 | 585.63 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-29 09:35:00 | 595.42 | 2023-08-29 09:40:00 | 594.20 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-05 10:50:00 | 591.71 | 2023-09-05 11:20:00 | 593.65 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-09-05 10:50:00 | 591.71 | 2023-09-05 11:50:00 | 591.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-06 11:00:00 | 590.29 | 2023-09-06 11:20:00 | 589.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-09-07 11:15:00 | 591.29 | 2023-09-07 11:45:00 | 592.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-12 09:35:00 | 596.71 | 2023-09-12 09:45:00 | 593.90 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-12 09:35:00 | 596.71 | 2023-09-12 10:05:00 | 596.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-13 09:30:00 | 599.17 | 2023-09-13 10:00:00 | 602.56 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-09-13 09:30:00 | 599.17 | 2023-09-13 10:20:00 | 599.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-21 11:10:00 | 752.40 | 2023-09-21 12:15:00 | 748.94 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-11-10 09:55:00 | 569.40 | 2023-11-10 10:05:00 | 570.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-11-13 10:20:00 | 574.05 | 2023-11-13 10:35:00 | 575.32 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-16 11:05:00 | 577.00 | 2023-11-16 12:05:00 | 576.08 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-21 10:55:00 | 585.00 | 2023-11-21 11:40:00 | 583.88 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-24 10:55:00 | 574.20 | 2023-11-24 12:25:00 | 572.36 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-11-24 10:55:00 | 574.20 | 2023-11-24 14:50:00 | 574.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 09:35:00 | 576.30 | 2023-11-30 09:40:00 | 575.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-05 09:40:00 | 591.70 | 2023-12-05 10:15:00 | 590.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-06 10:30:00 | 585.45 | 2023-12-06 10:45:00 | 586.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-07 09:30:00 | 594.35 | 2023-12-07 09:35:00 | 592.18 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-12-08 10:50:00 | 582.90 | 2023-12-08 11:00:00 | 580.95 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-12-08 10:50:00 | 582.90 | 2023-12-08 15:20:00 | 579.45 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2023-12-11 11:15:00 | 576.45 | 2023-12-11 12:05:00 | 574.83 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-12-11 11:15:00 | 576.45 | 2023-12-11 15:05:00 | 576.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-12 10:45:00 | 574.35 | 2023-12-12 12:45:00 | 572.91 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-12-12 10:45:00 | 574.35 | 2023-12-12 12:50:00 | 574.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 10:30:00 | 567.65 | 2023-12-13 11:45:00 | 565.37 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-13 10:30:00 | 567.65 | 2023-12-13 12:25:00 | 567.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-18 11:10:00 | 589.00 | 2023-12-18 12:25:00 | 590.81 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-12-18 11:10:00 | 589.00 | 2023-12-18 15:20:00 | 593.80 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2023-12-19 10:50:00 | 593.85 | 2023-12-19 11:25:00 | 595.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-20 10:50:00 | 592.00 | 2023-12-20 11:10:00 | 593.67 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-22 10:50:00 | 580.10 | 2023-12-22 11:05:00 | 581.91 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-12-22 10:50:00 | 580.10 | 2023-12-22 11:10:00 | 580.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 11:15:00 | 586.30 | 2023-12-26 12:15:00 | 588.33 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-12-26 11:15:00 | 586.30 | 2023-12-26 14:20:00 | 586.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-28 10:50:00 | 589.20 | 2023-12-28 10:55:00 | 591.08 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-12-28 10:50:00 | 589.20 | 2023-12-28 11:45:00 | 589.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 11:05:00 | 600.80 | 2023-12-29 11:15:00 | 603.51 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-12-29 11:05:00 | 600.80 | 2023-12-29 12:35:00 | 600.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 10:40:00 | 595.10 | 2024-01-02 11:20:00 | 596.72 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-03 09:40:00 | 601.35 | 2024-01-03 09:50:00 | 602.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-04 09:45:00 | 592.90 | 2024-01-04 09:50:00 | 594.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-01-05 10:00:00 | 601.30 | 2024-01-05 10:30:00 | 603.34 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-01-05 10:00:00 | 601.30 | 2024-01-05 10:45:00 | 601.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 10:05:00 | 587.95 | 2024-01-08 10:30:00 | 589.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-09 10:30:00 | 582.00 | 2024-01-09 11:05:00 | 583.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-25 10:10:00 | 555.80 | 2024-01-25 10:20:00 | 553.79 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-01-25 10:10:00 | 555.80 | 2024-01-25 10:30:00 | 555.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-30 10:45:00 | 559.50 | 2024-01-30 11:25:00 | 560.72 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-01 11:10:00 | 563.30 | 2024-02-01 11:15:00 | 564.26 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-02-05 10:55:00 | 566.55 | 2024-02-05 11:00:00 | 568.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-06 10:55:00 | 558.00 | 2024-02-06 12:45:00 | 555.54 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-06 10:55:00 | 558.00 | 2024-02-06 15:20:00 | 555.45 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-02-08 10:40:00 | 554.00 | 2024-02-08 11:00:00 | 551.87 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-02-08 10:40:00 | 554.00 | 2024-02-08 11:25:00 | 554.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-12 11:05:00 | 546.80 | 2024-02-12 11:10:00 | 548.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-13 10:45:00 | 554.50 | 2024-02-13 12:35:00 | 556.94 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-13 10:45:00 | 554.50 | 2024-02-13 15:20:00 | 558.90 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2024-02-15 09:45:00 | 556.05 | 2024-02-15 09:50:00 | 557.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-21 09:30:00 | 565.20 | 2024-02-21 09:45:00 | 567.04 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-02-21 09:30:00 | 565.20 | 2024-02-21 10:10:00 | 565.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-23 11:05:00 | 565.00 | 2024-02-23 12:05:00 | 566.34 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-02-28 10:50:00 | 558.10 | 2024-02-28 10:55:00 | 559.73 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-29 10:35:00 | 566.65 | 2024-02-29 10:50:00 | 570.07 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-02-29 10:35:00 | 566.65 | 2024-02-29 15:20:00 | 614.45 | TARGET_HIT | 0.50 | 8.44% |
| SELL | retest1 | 2024-03-06 09:55:00 | 571.15 | 2024-03-06 10:00:00 | 568.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-03-06 09:55:00 | 571.15 | 2024-03-06 10:10:00 | 571.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-12 10:45:00 | 568.10 | 2024-03-12 10:55:00 | 569.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-19 09:50:00 | 545.45 | 2024-03-19 10:05:00 | 542.95 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-03-19 09:50:00 | 545.45 | 2024-03-19 10:10:00 | 545.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:55:00 | 554.30 | 2024-03-21 11:05:00 | 553.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-02 10:55:00 | 563.50 | 2024-04-02 11:15:00 | 564.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-04-05 11:10:00 | 559.95 | 2024-04-05 11:35:00 | 558.89 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-04-12 09:40:00 | 558.05 | 2024-04-12 09:50:00 | 559.37 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-15 09:35:00 | 544.70 | 2024-04-15 09:40:00 | 546.52 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-22 09:55:00 | 505.80 | 2024-04-22 10:35:00 | 502.94 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-04-22 09:55:00 | 505.80 | 2024-04-22 11:15:00 | 505.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-23 10:45:00 | 503.50 | 2024-04-23 11:40:00 | 504.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-25 09:30:00 | 504.80 | 2024-04-25 09:35:00 | 505.59 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-04-29 11:00:00 | 505.00 | 2024-04-29 12:15:00 | 504.13 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2024-04-29 11:00:00 | 505.00 | 2024-04-29 15:20:00 | 505.05 | STOP_HIT | 0.50 | -0.01% |
| BUY | retest1 | 2024-04-30 09:30:00 | 511.65 | 2024-04-30 09:50:00 | 513.17 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-30 09:30:00 | 511.65 | 2024-04-30 11:05:00 | 511.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 11:00:00 | 516.90 | 2024-05-03 11:05:00 | 514.58 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-03 11:00:00 | 516.90 | 2024-05-03 11:45:00 | 516.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-06 09:35:00 | 520.05 | 2024-05-06 09:50:00 | 517.20 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-06 09:35:00 | 520.05 | 2024-05-06 10:20:00 | 520.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-07 09:30:00 | 516.15 | 2024-05-07 09:40:00 | 518.43 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-07 09:30:00 | 516.15 | 2024-05-07 10:20:00 | 516.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-08 10:40:00 | 508.95 | 2024-05-08 11:05:00 | 506.81 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-08 10:40:00 | 508.95 | 2024-05-08 12:15:00 | 508.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 11:05:00 | 489.05 | 2024-05-09 11:20:00 | 490.56 | STOP_HIT | 1.00 | -0.31% |
