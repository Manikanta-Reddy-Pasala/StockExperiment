# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 679.00
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
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 42
- **Target hits / Stop hits / Partials:** 8 / 42 / 23
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 11.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 16 | 42.1% | 4 | 22 | 12 | 0.17% | 6.5% |
| BUY @ 2nd Alert (retest1) | 38 | 16 | 42.1% | 4 | 22 | 12 | 0.17% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 15 | 42.9% | 4 | 20 | 11 | 0.15% | 5.1% |
| SELL @ 2nd Alert (retest1) | 35 | 15 | 42.9% | 4 | 20 | 11 | 0.15% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 73 | 31 | 42.5% | 8 | 42 | 23 | 0.16% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 11:00:00 | 445.60 | 446.81 | 0.00 | ORB-short ORB[446.95,449.80] vol=1.7x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:10:00 | 443.91 | 445.82 | 0.00 | T1 1.5R @ 443.91 |
| Stop hit — per-position SL triggered | 2024-05-14 11:45:00 | 445.60 | 445.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:45:00 | 441.60 | 442.85 | 0.00 | ORB-short ORB[443.25,445.50] vol=4.2x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 11:15:00 | 439.96 | 441.69 | 0.00 | T1 1.5R @ 439.96 |
| Stop hit — per-position SL triggered | 2024-05-16 11:35:00 | 441.60 | 441.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:55:00 | 470.70 | 467.06 | 0.00 | ORB-long ORB[464.50,469.80] vol=2.9x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:00:00 | 474.64 | 469.37 | 0.00 | T1 1.5R @ 474.64 |
| Target hit | 2024-05-22 10:30:00 | 471.55 | 472.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2024-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 09:50:00 | 460.85 | 463.36 | 0.00 | ORB-short ORB[461.30,465.00] vol=2.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-05-29 09:55:00 | 462.49 | 463.36 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 09:55:00 | 468.00 | 466.22 | 0.00 | ORB-long ORB[462.90,467.85] vol=3.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:00:00 | 469.85 | 466.50 | 0.00 | T1 1.5R @ 469.85 |
| Stop hit — per-position SL triggered | 2024-05-30 10:40:00 | 468.00 | 468.64 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:00:00 | 454.90 | 456.79 | 0.00 | ORB-short ORB[455.00,460.00] vol=2.1x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 11:30:00 | 452.60 | 456.34 | 0.00 | T1 1.5R @ 452.60 |
| Stop hit — per-position SL triggered | 2024-05-31 12:10:00 | 454.90 | 455.67 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:20:00 | 468.20 | 468.64 | 0.00 | ORB-short ORB[469.35,472.40] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 13:10:00 | 466.10 | 468.14 | 0.00 | T1 1.5R @ 466.10 |
| Stop hit — per-position SL triggered | 2024-06-12 13:15:00 | 468.20 | 468.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:45:00 | 488.05 | 489.60 | 0.00 | ORB-short ORB[489.00,492.50] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-06-19 09:55:00 | 490.15 | 489.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:40:00 | 494.40 | 492.71 | 0.00 | ORB-long ORB[487.00,494.30] vol=4.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-06-20 10:20:00 | 492.64 | 493.79 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 498.90 | 495.89 | 0.00 | ORB-long ORB[489.20,496.50] vol=4.0x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 496.65 | 496.13 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:30:00 | 475.00 | 478.69 | 0.00 | ORB-short ORB[479.00,481.90] vol=1.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-06-28 10:40:00 | 476.57 | 478.08 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 10:35:00 | 477.20 | 478.07 | 0.00 | ORB-short ORB[477.35,482.45] vol=10.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-07-01 10:40:00 | 478.48 | 478.45 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:00:00 | 468.90 | 469.75 | 0.00 | ORB-short ORB[472.90,476.10] vol=10.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:15:00 | 466.61 | 469.60 | 0.00 | T1 1.5R @ 466.61 |
| Target hit | 2024-07-02 15:20:00 | 457.90 | 463.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:45:00 | 465.10 | 461.67 | 0.00 | ORB-long ORB[454.60,461.10] vol=5.5x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:50:00 | 468.54 | 462.22 | 0.00 | T1 1.5R @ 468.54 |
| Target hit | 2024-07-03 15:20:00 | 470.40 | 466.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 476.25 | 473.11 | 0.00 | ORB-long ORB[468.05,473.90] vol=4.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 09:50:00 | 478.43 | 473.83 | 0.00 | T1 1.5R @ 478.43 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 476.25 | 475.55 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:00:00 | 465.75 | 463.86 | 0.00 | ORB-long ORB[456.55,463.45] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-07-10 10:05:00 | 464.02 | 463.77 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:25:00 | 478.50 | 476.30 | 0.00 | ORB-long ORB[469.35,471.90] vol=12.9x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:35:00 | 481.05 | 477.83 | 0.00 | T1 1.5R @ 481.05 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 478.50 | 478.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:00:00 | 486.95 | 489.31 | 0.00 | ORB-short ORB[487.55,493.00] vol=1.9x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:30:00 | 484.79 | 488.58 | 0.00 | T1 1.5R @ 484.79 |
| Stop hit — per-position SL triggered | 2024-07-18 12:20:00 | 486.95 | 487.94 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:50:00 | 483.15 | 480.49 | 0.00 | ORB-long ORB[475.00,481.80] vol=2.0x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-07-22 10:55:00 | 481.36 | 480.55 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:30:00 | 533.50 | 538.70 | 0.00 | ORB-short ORB[535.80,541.50] vol=1.7x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-08-19 10:35:00 | 536.02 | 537.46 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:00:00 | 526.60 | 523.00 | 0.00 | ORB-long ORB[520.90,526.45] vol=1.6x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:05:00 | 529.08 | 524.03 | 0.00 | T1 1.5R @ 529.08 |
| Stop hit — per-position SL triggered | 2024-08-20 11:10:00 | 526.60 | 524.11 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 549.60 | 545.95 | 0.00 | ORB-long ORB[539.60,546.50] vol=1.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-08-27 10:05:00 | 547.17 | 547.05 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:05:00 | 546.50 | 551.87 | 0.00 | ORB-short ORB[549.00,557.00] vol=1.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:20:00 | 543.28 | 550.06 | 0.00 | T1 1.5R @ 543.28 |
| Target hit | 2024-09-04 15:10:00 | 544.25 | 544.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 558.25 | 553.78 | 0.00 | ORB-long ORB[547.00,554.15] vol=3.5x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-09-05 11:25:00 | 556.82 | 554.17 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:10:00 | 556.00 | 561.22 | 0.00 | ORB-short ORB[561.15,566.10] vol=3.1x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-09-06 12:05:00 | 557.66 | 560.60 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:15:00 | 551.25 | 546.29 | 0.00 | ORB-long ORB[544.95,550.00] vol=2.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-09-09 11:20:00 | 549.44 | 546.65 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 559.70 | 554.28 | 0.00 | ORB-long ORB[547.85,556.00] vol=2.0x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 09:40:00 | 563.09 | 557.72 | 0.00 | T1 1.5R @ 563.09 |
| Stop hit — per-position SL triggered | 2024-09-10 09:50:00 | 559.70 | 558.81 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:45:00 | 616.30 | 610.04 | 0.00 | ORB-long ORB[588.80,598.00] vol=2.1x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-09-20 09:55:00 | 611.65 | 611.03 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:55:00 | 621.80 | 617.48 | 0.00 | ORB-long ORB[610.50,617.95] vol=1.9x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:10:00 | 627.07 | 620.19 | 0.00 | T1 1.5R @ 627.07 |
| Stop hit — per-position SL triggered | 2024-10-01 10:20:00 | 621.80 | 620.63 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-04 09:30:00 | 589.55 | 590.38 | 0.00 | ORB-short ORB[590.00,596.00] vol=1.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-10-04 10:00:00 | 592.17 | 590.22 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:10:00 | 588.80 | 593.90 | 0.00 | ORB-short ORB[593.05,600.75] vol=4.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-10-24 11:35:00 | 590.60 | 593.44 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:10:00 | 613.50 | 609.05 | 0.00 | ORB-long ORB[600.10,606.85] vol=1.9x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-10-30 10:20:00 | 611.13 | 610.05 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 658.95 | 653.48 | 0.00 | ORB-long ORB[646.00,654.50] vol=2.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:15:00 | 663.25 | 654.83 | 0.00 | T1 1.5R @ 663.25 |
| Stop hit — per-position SL triggered | 2024-11-29 11:40:00 | 658.95 | 655.27 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:25:00 | 701.25 | 695.68 | 0.00 | ORB-long ORB[688.00,698.10] vol=1.9x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 14:05:00 | 706.33 | 699.28 | 0.00 | T1 1.5R @ 706.33 |
| Target hit | 2024-12-06 15:20:00 | 713.35 | 705.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-12-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:10:00 | 706.05 | 714.76 | 0.00 | ORB-short ORB[713.15,721.20] vol=1.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-12-10 11:50:00 | 708.79 | 713.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:30:00 | 717.30 | 711.02 | 0.00 | ORB-long ORB[705.65,716.25] vol=1.6x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-12-11 10:40:00 | 714.59 | 711.27 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 720.95 | 715.59 | 0.00 | ORB-long ORB[710.10,717.95] vol=1.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 718.48 | 716.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 713.85 | 718.94 | 0.00 | ORB-short ORB[721.00,729.00] vol=3.9x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 716.52 | 717.99 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:35:00 | 699.60 | 703.88 | 0.00 | ORB-short ORB[707.15,717.00] vol=1.5x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:15:00 | 695.60 | 702.06 | 0.00 | T1 1.5R @ 695.60 |
| Stop hit — per-position SL triggered | 2024-12-16 15:05:00 | 699.60 | 700.04 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 682.10 | 685.66 | 0.00 | ORB-short ORB[682.15,691.95] vol=1.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:40:00 | 679.56 | 684.40 | 0.00 | T1 1.5R @ 679.56 |
| Target hit | 2024-12-26 12:30:00 | 677.00 | 675.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2024-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:45:00 | 663.00 | 669.80 | 0.00 | ORB-short ORB[667.70,675.50] vol=1.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:45:00 | 660.42 | 667.91 | 0.00 | T1 1.5R @ 660.42 |
| Stop hit — per-position SL triggered | 2024-12-27 12:15:00 | 663.00 | 666.88 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:30:00 | 695.90 | 689.71 | 0.00 | ORB-long ORB[682.10,691.00] vol=1.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:00:00 | 700.60 | 692.54 | 0.00 | T1 1.5R @ 700.60 |
| Target hit | 2024-12-30 15:20:00 | 706.00 | 715.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:45:00 | 736.05 | 730.55 | 0.00 | ORB-long ORB[717.80,727.70] vol=2.9x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 732.64 | 731.27 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:50:00 | 654.60 | 657.87 | 0.00 | ORB-short ORB[657.55,664.80] vol=1.9x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:15:00 | 651.44 | 656.71 | 0.00 | T1 1.5R @ 651.44 |
| Target hit | 2025-01-17 15:20:00 | 645.05 | 648.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-02-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 09:45:00 | 619.50 | 622.33 | 0.00 | ORB-short ORB[620.75,627.70] vol=2.9x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-02-04 11:40:00 | 622.55 | 619.89 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:40:00 | 619.05 | 623.31 | 0.00 | ORB-short ORB[626.00,631.05] vol=5.0x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-02-11 12:00:00 | 621.60 | 622.42 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 640.00 | 634.82 | 0.00 | ORB-long ORB[628.85,637.80] vol=1.6x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:45:00 | 644.04 | 636.27 | 0.00 | T1 1.5R @ 644.04 |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 640.00 | 637.94 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 640.80 | 638.24 | 0.00 | ORB-long ORB[630.65,640.10] vol=3.9x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-03-27 12:00:00 | 638.39 | 638.63 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:45:00 | 663.55 | 659.70 | 0.00 | ORB-long ORB[655.70,661.15] vol=2.3x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-04-22 11:35:00 | 661.69 | 660.88 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:55:00 | 669.05 | 670.30 | 0.00 | ORB-short ORB[670.30,679.00] vol=2.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-04-29 12:50:00 | 671.42 | 669.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 11:00:00 | 445.60 | 2024-05-14 11:10:00 | 443.91 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-05-14 11:00:00 | 445.60 | 2024-05-14 11:45:00 | 445.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 09:45:00 | 441.60 | 2024-05-16 11:15:00 | 439.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-05-16 09:45:00 | 441.60 | 2024-05-16 11:35:00 | 441.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-22 09:55:00 | 470.70 | 2024-05-22 10:00:00 | 474.64 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2024-05-22 09:55:00 | 470.70 | 2024-05-22 10:30:00 | 471.55 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-05-29 09:50:00 | 460.85 | 2024-05-29 09:55:00 | 462.49 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-30 09:55:00 | 468.00 | 2024-05-30 10:00:00 | 469.85 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-30 09:55:00 | 468.00 | 2024-05-30 10:40:00 | 468.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 11:00:00 | 454.90 | 2024-05-31 11:30:00 | 452.60 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-31 11:00:00 | 454.90 | 2024-05-31 12:10:00 | 454.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:20:00 | 468.20 | 2024-06-12 13:10:00 | 466.10 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-12 10:20:00 | 468.20 | 2024-06-12 13:15:00 | 468.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 09:45:00 | 488.05 | 2024-06-19 09:55:00 | 490.15 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-20 09:40:00 | 494.40 | 2024-06-20 10:20:00 | 492.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-21 09:35:00 | 498.90 | 2024-06-21 09:40:00 | 496.65 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-06-28 10:30:00 | 475.00 | 2024-06-28 10:40:00 | 476.57 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-01 10:35:00 | 477.20 | 2024-07-01 10:40:00 | 478.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-02 10:00:00 | 468.90 | 2024-07-02 10:15:00 | 466.61 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-02 10:00:00 | 468.90 | 2024-07-02 15:20:00 | 457.90 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2024-07-03 09:45:00 | 465.10 | 2024-07-03 09:50:00 | 468.54 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-07-03 09:45:00 | 465.10 | 2024-07-03 15:20:00 | 470.40 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2024-07-04 09:45:00 | 476.25 | 2024-07-04 09:50:00 | 478.43 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-04 09:45:00 | 476.25 | 2024-07-04 10:10:00 | 476.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-10 10:00:00 | 465.75 | 2024-07-10 10:05:00 | 464.02 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-12 10:25:00 | 478.50 | 2024-07-12 10:35:00 | 481.05 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-12 10:25:00 | 478.50 | 2024-07-12 10:50:00 | 478.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 11:00:00 | 486.95 | 2024-07-18 11:30:00 | 484.79 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-18 11:00:00 | 486.95 | 2024-07-18 12:20:00 | 486.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-22 10:50:00 | 483.15 | 2024-07-22 10:55:00 | 481.36 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-19 10:30:00 | 533.50 | 2024-08-19 10:35:00 | 536.02 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-20 11:00:00 | 526.60 | 2024-08-20 11:05:00 | 529.08 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-08-20 11:00:00 | 526.60 | 2024-08-20 11:10:00 | 526.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 09:40:00 | 549.60 | 2024-08-27 10:05:00 | 547.17 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-04 10:05:00 | 546.50 | 2024-09-04 10:20:00 | 543.28 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-04 10:05:00 | 546.50 | 2024-09-04 15:10:00 | 544.25 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-05 11:00:00 | 558.25 | 2024-09-05 11:25:00 | 556.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-06 11:10:00 | 556.00 | 2024-09-06 12:05:00 | 557.66 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-09 11:15:00 | 551.25 | 2024-09-09 11:20:00 | 549.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-10 09:30:00 | 559.70 | 2024-09-10 09:40:00 | 563.09 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-09-10 09:30:00 | 559.70 | 2024-09-10 09:50:00 | 559.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 09:45:00 | 616.30 | 2024-09-20 09:55:00 | 611.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2024-10-01 09:55:00 | 621.80 | 2024-10-01 10:10:00 | 627.07 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2024-10-01 09:55:00 | 621.80 | 2024-10-01 10:20:00 | 621.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-04 09:30:00 | 589.55 | 2024-10-04 10:00:00 | 592.17 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-24 11:10:00 | 588.80 | 2024-10-24 11:35:00 | 590.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-30 10:10:00 | 613.50 | 2024-10-30 10:20:00 | 611.13 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-11-29 10:50:00 | 658.95 | 2024-11-29 11:15:00 | 663.25 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-29 10:50:00 | 658.95 | 2024-11-29 11:40:00 | 658.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 10:25:00 | 701.25 | 2024-12-06 14:05:00 | 706.33 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-12-06 10:25:00 | 701.25 | 2024-12-06 15:20:00 | 713.35 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2024-12-10 11:10:00 | 706.05 | 2024-12-10 11:50:00 | 708.79 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-11 10:30:00 | 717.30 | 2024-12-11 10:40:00 | 714.59 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-12 09:45:00 | 720.95 | 2024-12-12 09:50:00 | 718.48 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-13 10:30:00 | 713.85 | 2024-12-13 10:55:00 | 716.52 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-16 10:35:00 | 699.60 | 2024-12-16 12:15:00 | 695.60 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-16 10:35:00 | 699.60 | 2024-12-16 15:05:00 | 699.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:30:00 | 682.10 | 2024-12-26 09:40:00 | 679.56 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-26 09:30:00 | 682.10 | 2024-12-26 12:30:00 | 677.00 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-12-27 10:45:00 | 663.00 | 2024-12-27 11:45:00 | 660.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-27 10:45:00 | 663.00 | 2024-12-27 12:15:00 | 663.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:30:00 | 695.90 | 2024-12-30 11:00:00 | 700.60 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-30 10:30:00 | 695.90 | 2024-12-30 15:20:00 | 706.00 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-01-03 09:45:00 | 736.05 | 2025-01-03 09:50:00 | 732.64 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-17 10:50:00 | 654.60 | 2025-01-17 11:15:00 | 651.44 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-17 10:50:00 | 654.60 | 2025-01-17 15:20:00 | 645.05 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-02-04 09:45:00 | 619.50 | 2025-02-04 11:40:00 | 622.55 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-02-11 10:40:00 | 619.05 | 2025-02-11 12:00:00 | 621.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-21 09:35:00 | 640.00 | 2025-03-21 09:45:00 | 644.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-21 09:35:00 | 640.00 | 2025-03-21 10:15:00 | 640.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 10:45:00 | 640.80 | 2025-03-27 12:00:00 | 638.39 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-22 10:45:00 | 663.55 | 2025-04-22 11:35:00 | 661.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-29 10:55:00 | 669.05 | 2025-04-29 12:50:00 | 671.42 | STOP_HIT | 1.00 | -0.35% |
