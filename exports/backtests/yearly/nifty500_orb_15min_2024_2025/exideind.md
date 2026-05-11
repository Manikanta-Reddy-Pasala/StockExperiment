# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 361.75
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
| ENTRY1 | 69 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 62
- **Target hits / Stop hits / Partials:** 7 / 62 / 19
- **Avg / median % per leg:** 0.07% / -0.23%
- **Sum % (uncompounded):** 5.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 13 | 27.1% | 3 | 35 | 10 | 0.06% | 2.8% |
| BUY @ 2nd Alert (retest1) | 48 | 13 | 27.1% | 3 | 35 | 10 | 0.06% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 13 | 32.5% | 4 | 27 | 9 | 0.08% | 3.2% |
| SELL @ 2nd Alert (retest1) | 40 | 13 | 32.5% | 4 | 27 | 9 | 0.08% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 26 | 29.5% | 7 | 62 | 19 | 0.07% | 5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:15:00 | 468.60 | 464.27 | 0.00 | ORB-long ORB[460.30,466.55] vol=2.3x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:30:00 | 471.92 | 467.17 | 0.00 | T1 1.5R @ 471.92 |
| Stop hit — per-position SL triggered | 2024-05-16 10:50:00 | 468.60 | 469.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 467.70 | 472.92 | 0.00 | ORB-short ORB[472.05,477.30] vol=2.4x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 470.21 | 472.18 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 462.25 | 467.08 | 0.00 | ORB-short ORB[464.70,470.85] vol=1.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-05-23 11:00:00 | 463.99 | 465.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:00:00 | 462.10 | 464.10 | 0.00 | ORB-short ORB[463.65,467.70] vol=2.4x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-05-24 10:20:00 | 463.82 | 463.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:10:00 | 536.00 | 528.87 | 0.00 | ORB-long ORB[526.05,531.45] vol=3.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 533.90 | 529.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:55:00 | 532.50 | 529.51 | 0.00 | ORB-long ORB[523.00,529.10] vol=1.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:10:00 | 535.34 | 530.28 | 0.00 | T1 1.5R @ 535.34 |
| Stop hit — per-position SL triggered | 2024-06-11 12:35:00 | 532.50 | 533.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:25:00 | 539.20 | 534.31 | 0.00 | ORB-long ORB[529.80,534.90] vol=2.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-06-12 10:30:00 | 537.31 | 534.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 554.40 | 549.88 | 0.00 | ORB-long ORB[543.20,550.85] vol=3.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-06-18 09:50:00 | 552.40 | 552.11 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 569.85 | 572.77 | 0.00 | ORB-short ORB[570.00,574.90] vol=1.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-06-21 10:55:00 | 571.85 | 572.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:45:00 | 569.65 | 566.58 | 0.00 | ORB-long ORB[563.00,567.35] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-07-01 11:10:00 | 567.82 | 566.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:35:00 | 563.45 | 564.59 | 0.00 | ORB-short ORB[563.70,567.35] vol=2.5x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:55:00 | 561.21 | 563.89 | 0.00 | T1 1.5R @ 561.21 |
| Stop hit — per-position SL triggered | 2024-07-03 10:25:00 | 563.45 | 563.50 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:05:00 | 569.30 | 567.70 | 0.00 | ORB-long ORB[566.25,569.05] vol=3.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-07-05 11:10:00 | 568.15 | 567.72 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 566.95 | 571.68 | 0.00 | ORB-short ORB[568.50,576.40] vol=1.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 568.99 | 571.45 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 11:15:00 | 574.45 | 572.65 | 0.00 | ORB-long ORB[569.00,573.90] vol=9.7x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-07-09 11:20:00 | 572.33 | 572.72 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:40:00 | 566.45 | 562.51 | 0.00 | ORB-long ORB[558.55,562.50] vol=4.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-07-15 09:45:00 | 564.62 | 563.04 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:05:00 | 568.30 | 565.44 | 0.00 | ORB-long ORB[561.60,565.55] vol=3.4x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-07-16 10:10:00 | 566.87 | 565.56 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 549.00 | 552.57 | 0.00 | ORB-short ORB[550.60,556.90] vol=3.0x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 551.19 | 551.88 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:35:00 | 551.05 | 548.00 | 0.00 | ORB-long ORB[543.45,549.15] vol=4.5x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-07-23 09:40:00 | 549.14 | 548.21 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:10:00 | 539.70 | 537.20 | 0.00 | ORB-long ORB[532.60,538.30] vol=1.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 538.45 | 537.26 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:45:00 | 545.00 | 542.54 | 0.00 | ORB-long ORB[539.40,544.00] vol=2.5x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-07-26 10:10:00 | 543.01 | 544.11 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 523.55 | 525.00 | 0.00 | ORB-short ORB[524.90,528.00] vol=1.7x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-08-01 09:40:00 | 524.96 | 525.10 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 503.35 | 501.02 | 0.00 | ORB-long ORB[496.95,501.95] vol=2.0x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:45:00 | 506.22 | 503.00 | 0.00 | T1 1.5R @ 506.22 |
| Stop hit — per-position SL triggered | 2024-08-13 09:55:00 | 503.35 | 503.16 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:45:00 | 492.40 | 494.95 | 0.00 | ORB-short ORB[495.40,498.95] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-08-20 11:00:00 | 493.44 | 494.79 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 512.60 | 510.51 | 0.00 | ORB-long ORB[508.00,511.75] vol=2.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-08-22 09:35:00 | 511.36 | 510.76 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:15:00 | 504.55 | 501.40 | 0.00 | ORB-long ORB[497.25,501.50] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-08-27 10:40:00 | 503.40 | 501.99 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:40:00 | 494.15 | 493.36 | 0.00 | ORB-long ORB[491.10,493.95] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-08-30 12:05:00 | 492.88 | 493.54 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:45:00 | 489.55 | 491.53 | 0.00 | ORB-short ORB[490.50,493.00] vol=2.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-09-03 11:00:00 | 490.48 | 491.40 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:35:00 | 485.75 | 490.53 | 0.00 | ORB-short ORB[492.70,497.85] vol=2.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-09-06 10:40:00 | 487.51 | 490.39 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 483.70 | 480.90 | 0.00 | ORB-long ORB[477.05,480.20] vol=3.2x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-09-11 09:50:00 | 482.25 | 481.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 11:15:00 | 482.75 | 480.32 | 0.00 | ORB-long ORB[477.50,482.00] vol=5.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 481.61 | 480.49 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 479.30 | 480.68 | 0.00 | ORB-short ORB[479.45,482.25] vol=1.7x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-09-18 09:40:00 | 480.33 | 480.66 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 471.10 | 473.68 | 0.00 | ORB-short ORB[473.50,476.80] vol=2.1x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:50:00 | 468.93 | 473.04 | 0.00 | T1 1.5R @ 468.93 |
| Target hit | 2024-09-19 15:20:00 | 461.00 | 460.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:15:00 | 474.05 | 474.92 | 0.00 | ORB-short ORB[474.60,478.50] vol=1.8x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:40:00 | 472.61 | 474.70 | 0.00 | T1 1.5R @ 472.61 |
| Target hit | 2024-09-25 14:50:00 | 472.85 | 472.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-09-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:50:00 | 479.35 | 475.47 | 0.00 | ORB-long ORB[470.05,476.40] vol=2.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:00:00 | 481.77 | 477.53 | 0.00 | T1 1.5R @ 481.77 |
| Target hit | 2024-09-27 15:20:00 | 498.45 | 493.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:45:00 | 502.65 | 507.68 | 0.00 | ORB-short ORB[509.00,515.70] vol=1.5x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:05:00 | 499.39 | 505.31 | 0.00 | T1 1.5R @ 499.39 |
| Stop hit — per-position SL triggered | 2024-10-17 10:10:00 | 502.65 | 505.07 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 491.40 | 494.16 | 0.00 | ORB-short ORB[491.50,498.65] vol=1.9x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:45:00 | 488.92 | 492.93 | 0.00 | T1 1.5R @ 488.92 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 491.40 | 492.50 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:05:00 | 458.50 | 464.92 | 0.00 | ORB-short ORB[463.85,469.65] vol=2.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-10-29 10:10:00 | 460.46 | 464.41 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:45:00 | 450.10 | 447.99 | 0.00 | ORB-long ORB[443.55,449.25] vol=3.9x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-11-06 10:50:00 | 448.50 | 448.09 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:40:00 | 414.35 | 417.54 | 0.00 | ORB-short ORB[416.95,422.60] vol=4.2x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-11-14 09:55:00 | 416.62 | 416.75 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:30:00 | 438.90 | 434.06 | 0.00 | ORB-long ORB[425.65,432.25] vol=2.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-11-27 10:45:00 | 436.95 | 434.84 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 458.15 | 455.51 | 0.00 | ORB-long ORB[452.00,456.00] vol=2.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-12-03 10:05:00 | 456.97 | 456.80 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 461.00 | 459.09 | 0.00 | ORB-long ORB[455.00,459.85] vol=4.3x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 459.68 | 459.58 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 450.85 | 452.65 | 0.00 | ORB-short ORB[453.00,456.80] vol=1.7x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:40:00 | 449.00 | 452.08 | 0.00 | T1 1.5R @ 449.00 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 450.85 | 451.93 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:40:00 | 457.80 | 455.16 | 0.00 | ORB-long ORB[453.00,456.65] vol=2.4x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:50:00 | 460.10 | 456.72 | 0.00 | T1 1.5R @ 460.10 |
| Target hit | 2024-12-06 15:20:00 | 462.45 | 461.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:55:00 | 464.80 | 467.67 | 0.00 | ORB-short ORB[466.80,469.70] vol=1.8x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-12-11 11:10:00 | 465.87 | 467.44 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 453.85 | 455.75 | 0.00 | ORB-short ORB[454.15,457.00] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-12-16 11:10:00 | 454.91 | 455.37 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 441.90 | 437.66 | 0.00 | ORB-long ORB[433.40,439.05] vol=1.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-12-19 10:05:00 | 439.81 | 438.30 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 415.90 | 417.64 | 0.00 | ORB-short ORB[417.20,422.70] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 417.17 | 417.59 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 09:30:00 | 389.45 | 386.43 | 0.00 | ORB-long ORB[383.15,388.35] vol=1.6x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-01-13 09:35:00 | 387.47 | 386.56 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:35:00 | 386.70 | 388.66 | 0.00 | ORB-short ORB[387.55,393.35] vol=3.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-01-20 09:40:00 | 387.98 | 388.60 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 384.00 | 389.71 | 0.00 | ORB-short ORB[390.10,394.40] vol=1.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-01-21 10:45:00 | 385.24 | 388.27 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:20:00 | 384.60 | 379.69 | 0.00 | ORB-long ORB[373.20,376.45] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 383.19 | 381.69 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:50:00 | 384.95 | 382.69 | 0.00 | ORB-long ORB[379.30,382.80] vol=4.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-02-05 11:05:00 | 384.00 | 383.19 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 380.50 | 382.98 | 0.00 | ORB-short ORB[382.10,386.85] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 381.61 | 382.54 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:45:00 | 369.00 | 366.61 | 0.00 | ORB-long ORB[363.20,368.60] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-02-13 11:35:00 | 367.46 | 367.59 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:55:00 | 368.90 | 365.34 | 0.00 | ORB-long ORB[360.55,365.80] vol=2.3x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:15:00 | 370.57 | 366.25 | 0.00 | T1 1.5R @ 370.57 |
| Stop hit — per-position SL triggered | 2025-02-20 11:50:00 | 368.90 | 367.09 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 349.20 | 347.06 | 0.00 | ORB-long ORB[342.55,347.00] vol=2.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 348.11 | 347.23 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 358.10 | 355.82 | 0.00 | ORB-long ORB[353.05,358.05] vol=2.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:45:00 | 359.94 | 357.32 | 0.00 | T1 1.5R @ 359.94 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 358.10 | 357.37 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:30:00 | 353.25 | 354.65 | 0.00 | ORB-short ORB[353.30,358.00] vol=2.0x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-04-09 09:35:00 | 354.54 | 354.56 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:00:00 | 376.05 | 373.90 | 0.00 | ORB-long ORB[370.60,374.60] vol=3.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:10:00 | 377.94 | 374.95 | 0.00 | T1 1.5R @ 377.94 |
| Stop hit — per-position SL triggered | 2025-04-15 10:45:00 | 376.05 | 375.48 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 09:30:00 | 377.65 | 380.45 | 0.00 | ORB-short ORB[379.35,383.50] vol=1.8x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-04-16 09:35:00 | 378.69 | 380.33 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:05:00 | 371.95 | 374.78 | 0.00 | ORB-short ORB[374.20,378.80] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:15:00 | 369.85 | 373.12 | 0.00 | T1 1.5R @ 369.85 |
| Stop hit — per-position SL triggered | 2025-04-17 10:20:00 | 371.95 | 373.01 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 379.20 | 377.14 | 0.00 | ORB-long ORB[374.45,378.30] vol=1.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:55:00 | 380.97 | 378.33 | 0.00 | T1 1.5R @ 380.97 |
| Target hit | 2025-04-21 15:20:00 | 384.90 | 381.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:15:00 | 379.50 | 381.56 | 0.00 | ORB-short ORB[382.00,386.20] vol=1.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-04-23 12:50:00 | 380.64 | 381.23 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 385.15 | 383.33 | 0.00 | ORB-long ORB[380.30,384.30] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-04-24 09:35:00 | 384.25 | 383.47 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 378.50 | 381.00 | 0.00 | ORB-short ORB[380.25,384.35] vol=2.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 376.47 | 380.36 | 0.00 | T1 1.5R @ 376.47 |
| Target hit | 2025-04-25 12:55:00 | 373.10 | 373.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 67 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 377.30 | 380.65 | 0.00 | ORB-short ORB[377.80,382.60] vol=1.9x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:15:00 | 374.71 | 378.82 | 0.00 | T1 1.5R @ 374.71 |
| Target hit | 2025-04-29 15:20:00 | 369.40 | 374.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:05:00 | 374.20 | 370.15 | 0.00 | ORB-long ORB[366.45,371.45] vol=2.2x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-04-30 11:20:00 | 372.74 | 370.70 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 359.30 | 356.69 | 0.00 | ORB-long ORB[354.05,358.00] vol=1.8x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:50:00 | 361.36 | 358.14 | 0.00 | T1 1.5R @ 361.36 |
| Stop hit — per-position SL triggered | 2025-05-05 10:05:00 | 359.30 | 358.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:15:00 | 468.60 | 2024-05-16 10:30:00 | 471.92 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-05-16 10:15:00 | 468.60 | 2024-05-16 10:50:00 | 468.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:40:00 | 467.70 | 2024-05-22 09:50:00 | 470.21 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-05-23 10:35:00 | 462.25 | 2024-05-23 11:00:00 | 463.99 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-05-24 10:00:00 | 462.10 | 2024-05-24 10:20:00 | 463.82 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-10 11:10:00 | 536.00 | 2024-06-10 11:15:00 | 533.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-11 10:55:00 | 532.50 | 2024-06-11 11:10:00 | 535.34 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-11 10:55:00 | 532.50 | 2024-06-11 12:35:00 | 532.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 10:25:00 | 539.20 | 2024-06-12 10:30:00 | 537.31 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-18 09:30:00 | 554.40 | 2024-06-18 09:50:00 | 552.40 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-21 10:45:00 | 569.85 | 2024-06-21 10:55:00 | 571.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-01 10:45:00 | 569.65 | 2024-07-01 11:10:00 | 567.82 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-03 09:35:00 | 563.45 | 2024-07-03 09:55:00 | 561.21 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-03 09:35:00 | 563.45 | 2024-07-03 10:25:00 | 563.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 11:05:00 | 569.30 | 2024-07-05 11:10:00 | 568.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-08 09:55:00 | 566.95 | 2024-07-08 10:00:00 | 568.99 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-09 11:15:00 | 574.45 | 2024-07-09 11:20:00 | 572.33 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-15 09:40:00 | 566.45 | 2024-07-15 09:45:00 | 564.62 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-16 10:05:00 | 568.30 | 2024-07-16 10:10:00 | 566.87 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-18 09:30:00 | 549.00 | 2024-07-18 09:40:00 | 551.19 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-23 09:35:00 | 551.05 | 2024-07-23 09:40:00 | 549.14 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-25 11:10:00 | 539.70 | 2024-07-25 11:15:00 | 538.45 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-26 09:45:00 | 545.00 | 2024-07-26 10:10:00 | 543.01 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-01 09:35:00 | 523.55 | 2024-08-01 09:40:00 | 524.96 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-13 09:30:00 | 503.35 | 2024-08-13 09:45:00 | 506.22 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-13 09:30:00 | 503.35 | 2024-08-13 09:55:00 | 503.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 10:45:00 | 492.40 | 2024-08-20 11:00:00 | 493.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-22 09:30:00 | 512.60 | 2024-08-22 09:35:00 | 511.36 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-27 10:15:00 | 504.55 | 2024-08-27 10:40:00 | 503.40 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-30 10:40:00 | 494.15 | 2024-08-30 12:05:00 | 492.88 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-03 10:45:00 | 489.55 | 2024-09-03 11:00:00 | 490.48 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-06 10:35:00 | 485.75 | 2024-09-06 10:40:00 | 487.51 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-11 09:35:00 | 483.70 | 2024-09-11 09:50:00 | 482.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-13 11:15:00 | 482.75 | 2024-09-13 11:20:00 | 481.61 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-18 09:35:00 | 479.30 | 2024-09-18 09:40:00 | 480.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-19 09:45:00 | 471.10 | 2024-09-19 09:50:00 | 468.93 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-19 09:45:00 | 471.10 | 2024-09-19 15:20:00 | 461.00 | TARGET_HIT | 0.50 | 2.14% |
| SELL | retest1 | 2024-09-25 11:15:00 | 474.05 | 2024-09-25 11:40:00 | 472.61 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-25 11:15:00 | 474.05 | 2024-09-25 14:50:00 | 472.85 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2024-09-27 09:50:00 | 479.35 | 2024-09-27 10:00:00 | 481.77 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-27 09:50:00 | 479.35 | 2024-09-27 15:20:00 | 498.45 | TARGET_HIT | 0.50 | 3.98% |
| SELL | retest1 | 2024-10-17 09:45:00 | 502.65 | 2024-10-17 10:05:00 | 499.39 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-17 09:45:00 | 502.65 | 2024-10-17 10:10:00 | 502.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:30:00 | 491.40 | 2024-10-21 09:45:00 | 488.92 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-21 09:30:00 | 491.40 | 2024-10-21 10:00:00 | 491.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:05:00 | 458.50 | 2024-10-29 10:10:00 | 460.46 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-11-06 10:45:00 | 450.10 | 2024-11-06 10:50:00 | 448.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-14 09:40:00 | 414.35 | 2024-11-14 09:55:00 | 416.62 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-11-27 10:30:00 | 438.90 | 2024-11-27 10:45:00 | 436.95 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-03 09:40:00 | 458.15 | 2024-12-03 10:05:00 | 456.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-04 09:35:00 | 461.00 | 2024-12-04 09:45:00 | 459.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-05 10:55:00 | 450.85 | 2024-12-05 11:40:00 | 449.00 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-05 10:55:00 | 450.85 | 2024-12-05 12:05:00 | 450.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 10:40:00 | 457.80 | 2024-12-06 11:50:00 | 460.10 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-06 10:40:00 | 457.80 | 2024-12-06 15:20:00 | 462.45 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-12-11 10:55:00 | 464.80 | 2024-12-11 11:10:00 | 465.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-16 11:00:00 | 453.85 | 2024-12-16 11:10:00 | 454.91 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-19 09:40:00 | 441.90 | 2024-12-19 10:05:00 | 439.81 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-26 10:55:00 | 415.90 | 2024-12-26 11:00:00 | 417.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-13 09:30:00 | 389.45 | 2025-01-13 09:35:00 | 387.47 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-01-20 09:35:00 | 386.70 | 2025-01-20 09:40:00 | 387.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-21 10:20:00 | 384.00 | 2025-01-21 10:45:00 | 385.24 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-23 10:20:00 | 384.60 | 2025-01-23 11:40:00 | 383.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-05 10:50:00 | 384.95 | 2025-02-05 11:05:00 | 384.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-06 09:30:00 | 380.50 | 2025-02-06 09:40:00 | 381.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-13 09:45:00 | 369.00 | 2025-02-13 11:35:00 | 367.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-20 10:55:00 | 368.90 | 2025-02-20 11:15:00 | 370.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-02-20 10:55:00 | 368.90 | 2025-02-20 11:50:00 | 368.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 10:30:00 | 349.20 | 2025-03-19 10:45:00 | 348.11 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-21 09:30:00 | 358.10 | 2025-03-21 09:45:00 | 359.94 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-03-21 09:30:00 | 358.10 | 2025-03-21 09:50:00 | 358.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-09 09:30:00 | 353.25 | 2025-04-09 09:35:00 | 354.54 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-15 10:00:00 | 376.05 | 2025-04-15 10:10:00 | 377.94 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-15 10:00:00 | 376.05 | 2025-04-15 10:45:00 | 376.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-16 09:30:00 | 377.65 | 2025-04-16 09:35:00 | 378.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-17 10:05:00 | 371.95 | 2025-04-17 10:15:00 | 369.85 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-17 10:05:00 | 371.95 | 2025-04-17 10:20:00 | 371.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:35:00 | 379.20 | 2025-04-21 09:55:00 | 380.97 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-21 09:35:00 | 379.20 | 2025-04-21 15:20:00 | 384.90 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2025-04-23 11:15:00 | 379.50 | 2025-04-23 12:50:00 | 380.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-24 09:30:00 | 385.15 | 2025-04-24 09:35:00 | 384.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-25 09:35:00 | 378.50 | 2025-04-25 09:40:00 | 376.47 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-25 09:35:00 | 378.50 | 2025-04-25 12:55:00 | 373.10 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2025-04-29 09:45:00 | 377.30 | 2025-04-29 11:15:00 | 374.71 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-04-29 09:45:00 | 377.30 | 2025-04-29 15:20:00 | 369.40 | TARGET_HIT | 0.50 | 2.09% |
| BUY | retest1 | 2025-04-30 11:05:00 | 374.20 | 2025-04-30 11:20:00 | 372.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-05 09:40:00 | 359.30 | 2025-05-05 09:50:00 | 361.36 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-05 09:40:00 | 359.30 | 2025-05-05 10:05:00 | 359.30 | STOP_HIT | 0.50 | 0.00% |
