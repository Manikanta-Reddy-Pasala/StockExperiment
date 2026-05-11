# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 522.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 18 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 56
- **Target hits / Stop hits / Partials:** 18 / 56 / 26
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 23.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 28 | 52.8% | 13 | 25 | 15 | 0.41% | 21.7% |
| BUY @ 2nd Alert (retest1) | 53 | 28 | 52.8% | 13 | 25 | 15 | 0.41% | 21.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 16 | 34.0% | 5 | 31 | 11 | 0.03% | 1.6% |
| SELL @ 2nd Alert (retest1) | 47 | 16 | 34.0% | 5 | 31 | 11 | 0.03% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 100 | 44 | 44.0% | 18 | 56 | 26 | 0.23% | 23.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:45:00 | 368.95 | 369.75 | 0.00 | ORB-short ORB[371.10,374.80] vol=2.0x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 370.38 | 369.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:50:00 | 377.00 | 375.29 | 0.00 | ORB-long ORB[372.65,375.80] vol=3.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 12:15:00 | 378.48 | 376.26 | 0.00 | T1 1.5R @ 378.48 |
| Target hit | 2024-05-14 15:20:00 | 379.00 | 377.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:30:00 | 381.80 | 380.46 | 0.00 | ORB-long ORB[379.00,380.95] vol=1.9x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-05-17 10:50:00 | 380.97 | 380.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 375.30 | 377.67 | 0.00 | ORB-short ORB[377.60,380.05] vol=2.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 376.17 | 377.02 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:45:00 | 379.30 | 377.47 | 0.00 | ORB-long ORB[375.25,377.95] vol=3.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-05-23 13:05:00 | 378.37 | 378.22 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:40:00 | 383.35 | 380.39 | 0.00 | ORB-long ORB[377.20,380.85] vol=2.6x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-05-29 10:45:00 | 382.21 | 380.45 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 381.15 | 381.94 | 0.00 | ORB-short ORB[381.45,383.80] vol=3.2x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:05:00 | 379.79 | 381.72 | 0.00 | T1 1.5R @ 379.79 |
| Stop hit — per-position SL triggered | 2024-05-30 12:30:00 | 381.15 | 381.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 10:25:00 | 381.00 | 380.20 | 0.00 | ORB-long ORB[378.00,380.85] vol=8.7x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-05-31 10:40:00 | 379.57 | 380.21 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-03 09:55:00 | 389.15 | 385.17 | 0.00 | ORB-long ORB[382.35,387.90] vol=2.1x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-06-03 10:10:00 | 387.10 | 386.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 391.80 | 390.40 | 0.00 | ORB-long ORB[386.75,391.70] vol=4.0x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:45:00 | 393.87 | 393.28 | 0.00 | T1 1.5R @ 393.87 |
| Target hit | 2024-06-07 10:15:00 | 396.70 | 396.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 396.60 | 397.89 | 0.00 | ORB-short ORB[397.30,399.40] vol=2.0x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 397.70 | 397.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 444.40 | 440.89 | 0.00 | ORB-long ORB[437.00,443.55] vol=2.0x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-06-20 10:15:00 | 442.63 | 441.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:35:00 | 447.75 | 441.94 | 0.00 | ORB-long ORB[437.05,443.25] vol=6.4x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-06-21 10:45:00 | 445.65 | 444.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:30:00 | 441.70 | 443.41 | 0.00 | ORB-short ORB[441.75,446.25] vol=2.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-06-24 09:50:00 | 443.63 | 443.17 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:55:00 | 437.50 | 434.83 | 0.00 | ORB-long ORB[432.15,436.00] vol=2.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-07-01 10:00:00 | 436.08 | 434.95 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 433.00 | 435.13 | 0.00 | ORB-short ORB[435.00,440.05] vol=3.8x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-07-03 10:20:00 | 434.75 | 434.55 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:50:00 | 428.75 | 430.43 | 0.00 | ORB-short ORB[429.50,432.80] vol=1.6x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-07-04 10:25:00 | 429.84 | 430.00 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:45:00 | 425.00 | 427.21 | 0.00 | ORB-short ORB[425.80,429.70] vol=2.6x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 426.31 | 426.63 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 422.40 | 424.87 | 0.00 | ORB-short ORB[423.65,428.90] vol=1.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-07-08 09:35:00 | 423.59 | 424.49 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:25:00 | 430.65 | 426.78 | 0.00 | ORB-long ORB[424.05,427.20] vol=4.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-07-09 10:30:00 | 429.18 | 427.73 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:10:00 | 470.05 | 466.87 | 0.00 | ORB-long ORB[464.00,469.00] vol=2.1x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:10:00 | 472.95 | 469.66 | 0.00 | T1 1.5R @ 472.95 |
| Stop hit — per-position SL triggered | 2024-07-29 12:20:00 | 470.05 | 470.37 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:40:00 | 472.70 | 470.72 | 0.00 | ORB-long ORB[468.20,472.00] vol=1.5x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:15:00 | 475.26 | 473.05 | 0.00 | T1 1.5R @ 475.26 |
| Stop hit — per-position SL triggered | 2024-07-30 12:00:00 | 472.70 | 473.66 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:55:00 | 534.40 | 530.09 | 0.00 | ORB-long ORB[525.90,531.15] vol=1.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:15:00 | 538.01 | 532.28 | 0.00 | T1 1.5R @ 538.01 |
| Target hit | 2024-08-19 13:25:00 | 535.55 | 537.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 546.00 | 540.66 | 0.00 | ORB-long ORB[535.00,540.95] vol=2.6x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-08-20 10:25:00 | 543.52 | 541.07 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 556.45 | 559.05 | 0.00 | ORB-short ORB[557.60,562.10] vol=2.9x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-08-27 10:00:00 | 558.65 | 558.38 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 579.00 | 576.13 | 0.00 | ORB-long ORB[570.00,578.10] vol=2.1x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-08-28 09:50:00 | 576.32 | 576.19 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 585.70 | 588.02 | 0.00 | ORB-short ORB[586.60,593.80] vol=1.8x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 587.71 | 587.76 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:05:00 | 587.45 | 592.43 | 0.00 | ORB-short ORB[589.00,595.85] vol=2.2x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-09-05 15:10:00 | 589.10 | 590.21 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:55:00 | 577.00 | 572.82 | 0.00 | ORB-long ORB[568.00,574.90] vol=2.5x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-09-18 10:05:00 | 575.10 | 573.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 585.65 | 581.88 | 0.00 | ORB-long ORB[577.25,584.00] vol=2.2x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:00:00 | 589.95 | 585.10 | 0.00 | T1 1.5R @ 589.95 |
| Target hit | 2024-09-19 11:30:00 | 586.55 | 586.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 601.00 | 598.86 | 0.00 | ORB-long ORB[595.10,598.95] vol=1.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:20:00 | 603.56 | 600.29 | 0.00 | T1 1.5R @ 603.56 |
| Target hit | 2024-09-24 15:20:00 | 609.50 | 607.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2024-09-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:45:00 | 609.70 | 612.57 | 0.00 | ORB-short ORB[611.00,616.60] vol=1.6x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-09-25 11:00:00 | 611.75 | 612.43 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:45:00 | 660.65 | 657.55 | 0.00 | ORB-long ORB[642.00,651.95] vol=2.5x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 11:50:00 | 667.47 | 660.79 | 0.00 | T1 1.5R @ 667.47 |
| Target hit | 2024-10-01 15:20:00 | 682.95 | 667.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2024-10-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:05:00 | 648.20 | 657.68 | 0.00 | ORB-short ORB[662.00,668.45] vol=1.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:25:00 | 643.18 | 654.37 | 0.00 | T1 1.5R @ 643.18 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 648.20 | 647.92 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:55:00 | 660.80 | 655.42 | 0.00 | ORB-long ORB[650.60,659.45] vol=2.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-10-09 11:10:00 | 658.59 | 656.15 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 649.85 | 653.16 | 0.00 | ORB-short ORB[651.70,657.70] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-10-10 09:35:00 | 651.70 | 653.68 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:05:00 | 647.60 | 651.72 | 0.00 | ORB-short ORB[651.35,656.70] vol=3.4x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-10-11 15:20:00 | 649.70 | 648.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-10-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:55:00 | 655.55 | 653.01 | 0.00 | ORB-long ORB[648.80,654.00] vol=1.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 653.59 | 653.12 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:30:00 | 654.25 | 659.21 | 0.00 | ORB-short ORB[660.75,665.00] vol=2.7x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:35:00 | 650.92 | 657.66 | 0.00 | T1 1.5R @ 650.92 |
| Target hit | 2024-10-16 14:05:00 | 643.80 | 642.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 635.00 | 644.12 | 0.00 | ORB-short ORB[640.65,648.95] vol=2.1x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-10-21 10:05:00 | 637.87 | 640.89 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:15:00 | 604.60 | 610.12 | 0.00 | ORB-short ORB[612.00,619.40] vol=2.0x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 607.03 | 610.02 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:30:00 | 649.85 | 644.72 | 0.00 | ORB-long ORB[638.05,646.20] vol=2.2x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 653.84 | 647.54 | 0.00 | T1 1.5R @ 653.84 |
| Target hit | 2024-10-30 14:15:00 | 658.00 | 658.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2024-11-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:55:00 | 611.80 | 606.48 | 0.00 | ORB-long ORB[601.55,610.00] vol=2.1x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 609.35 | 607.63 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:05:00 | 560.40 | 565.98 | 0.00 | ORB-short ORB[561.35,569.40] vol=4.7x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:15:00 | 556.38 | 563.62 | 0.00 | T1 1.5R @ 556.38 |
| Target hit | 2024-11-28 14:20:00 | 559.65 | 558.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:45:00 | 570.35 | 566.36 | 0.00 | ORB-long ORB[560.75,566.00] vol=3.3x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 09:50:00 | 573.75 | 569.57 | 0.00 | T1 1.5R @ 573.75 |
| Target hit | 2024-11-29 15:20:00 | 585.00 | 578.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 589.55 | 582.45 | 0.00 | ORB-long ORB[576.00,582.70] vol=3.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-12-02 11:05:00 | 587.14 | 582.99 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:05:00 | 576.60 | 582.24 | 0.00 | ORB-short ORB[583.30,588.40] vol=3.5x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 12:25:00 | 573.51 | 579.54 | 0.00 | T1 1.5R @ 573.51 |
| Stop hit — per-position SL triggered | 2024-12-09 15:05:00 | 576.60 | 576.10 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:10:00 | 581.70 | 579.25 | 0.00 | ORB-long ORB[575.85,579.95] vol=2.6x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:20:00 | 584.98 | 580.79 | 0.00 | T1 1.5R @ 584.98 |
| Stop hit — per-position SL triggered | 2024-12-10 10:30:00 | 581.70 | 581.12 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:35:00 | 592.00 | 590.03 | 0.00 | ORB-long ORB[585.25,591.50] vol=2.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-12-11 09:45:00 | 589.78 | 589.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 588.80 | 587.17 | 0.00 | ORB-long ORB[584.40,588.50] vol=2.3x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 586.77 | 589.11 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 590.95 | 589.94 | 0.00 | ORB-long ORB[586.30,590.30] vol=2.4x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-12-17 10:00:00 | 589.16 | 589.90 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 521.75 | 524.95 | 0.00 | ORB-short ORB[523.40,528.60] vol=2.1x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:35:00 | 518.88 | 523.98 | 0.00 | T1 1.5R @ 518.88 |
| Target hit | 2024-12-26 12:55:00 | 520.70 | 519.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2024-12-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:00:00 | 520.50 | 522.17 | 0.00 | ORB-short ORB[520.95,524.95] vol=2.4x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 521.79 | 522.10 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:35:00 | 515.60 | 517.14 | 0.00 | ORB-short ORB[516.10,520.55] vol=2.2x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-12-30 09:40:00 | 517.33 | 517.40 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 10:40:00 | 514.55 | 518.63 | 0.00 | ORB-short ORB[515.15,520.00] vol=1.9x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-12-31 12:00:00 | 516.48 | 517.40 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:20:00 | 540.70 | 542.13 | 0.00 | ORB-short ORB[541.00,547.90] vol=2.5x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:30:00 | 538.09 | 541.97 | 0.00 | T1 1.5R @ 538.09 |
| Stop hit — per-position SL triggered | 2025-01-02 10:40:00 | 540.70 | 541.87 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 481.50 | 484.12 | 0.00 | ORB-short ORB[482.55,487.05] vol=1.6x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:35:00 | 478.93 | 482.94 | 0.00 | T1 1.5R @ 478.93 |
| Stop hit — per-position SL triggered | 2025-01-24 09:40:00 | 481.50 | 482.67 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:30:00 | 469.80 | 470.88 | 0.00 | ORB-short ORB[470.55,472.90] vol=5.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-27 10:45:00 | 471.22 | 470.78 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 09:35:00 | 470.60 | 475.51 | 0.00 | ORB-short ORB[473.35,480.00] vol=2.0x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 12:00:00 | 465.55 | 470.17 | 0.00 | T1 1.5R @ 465.55 |
| Target hit | 2025-02-03 15:20:00 | 465.00 | 467.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-02-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:50:00 | 463.55 | 468.44 | 0.00 | ORB-short ORB[467.35,472.65] vol=1.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-02-04 11:30:00 | 465.08 | 467.11 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 467.00 | 469.09 | 0.00 | ORB-short ORB[468.15,472.95] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 469.32 | 468.51 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:45:00 | 466.00 | 470.06 | 0.00 | ORB-short ORB[469.55,475.35] vol=2.2x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-02-07 11:05:00 | 467.58 | 468.57 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 463.50 | 461.93 | 0.00 | ORB-long ORB[458.20,463.40] vol=2.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-03-07 09:35:00 | 461.46 | 461.99 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:50:00 | 457.70 | 449.53 | 0.00 | ORB-long ORB[445.15,450.60] vol=2.2x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-03-11 11:10:00 | 455.46 | 450.56 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 462.00 | 458.18 | 0.00 | ORB-long ORB[453.00,459.40] vol=3.1x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:50:00 | 465.34 | 461.47 | 0.00 | T1 1.5R @ 465.34 |
| Target hit | 2025-03-12 15:20:00 | 487.85 | 476.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 09:30:00 | 535.00 | 538.77 | 0.00 | ORB-short ORB[536.00,543.00] vol=2.1x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 09:35:00 | 531.07 | 537.59 | 0.00 | T1 1.5R @ 531.07 |
| Stop hit — per-position SL triggered | 2025-03-24 09:50:00 | 535.00 | 536.74 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:10:00 | 530.05 | 526.72 | 0.00 | ORB-long ORB[522.25,529.30] vol=2.4x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:30:00 | 532.90 | 528.76 | 0.00 | T1 1.5R @ 532.90 |
| Target hit | 2025-03-27 15:05:00 | 531.45 | 535.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-03-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:10:00 | 544.90 | 540.36 | 0.00 | ORB-long ORB[534.50,540.60] vol=3.0x ATR=2.95 |
| Target hit | 2025-03-28 15:20:00 | 548.75 | 544.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-04-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:05:00 | 564.95 | 559.60 | 0.00 | ORB-long ORB[550.00,558.35] vol=6.4x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 562.05 | 560.55 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:00:00 | 514.80 | 518.97 | 0.00 | ORB-short ORB[522.55,529.70] vol=1.8x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-04-09 10:10:00 | 517.33 | 518.58 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:40:00 | 558.00 | 554.62 | 0.00 | ORB-long ORB[550.10,557.05] vol=2.1x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:50:00 | 560.39 | 556.05 | 0.00 | T1 1.5R @ 560.39 |
| Target hit | 2025-04-21 15:20:00 | 566.20 | 562.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2025-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:45:00 | 588.65 | 588.17 | 0.00 | ORB-long ORB[583.00,588.55] vol=2.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:50:00 | 591.72 | 588.64 | 0.00 | T1 1.5R @ 591.72 |
| Target hit | 2025-04-24 15:20:00 | 602.00 | 597.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2025-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:00:00 | 573.00 | 576.04 | 0.00 | ORB-short ORB[574.55,581.95] vol=2.5x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:20:00 | 569.37 | 573.02 | 0.00 | T1 1.5R @ 569.37 |
| Target hit | 2025-04-29 11:10:00 | 571.75 | 571.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 545.05 | 548.22 | 0.00 | ORB-short ORB[547.00,554.00] vol=1.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-05-06 10:40:00 | 547.03 | 548.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:45:00 | 368.95 | 2024-05-13 11:05:00 | 370.38 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-14 10:50:00 | 377.00 | 2024-05-14 12:15:00 | 378.48 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-14 10:50:00 | 377.00 | 2024-05-14 15:20:00 | 379.00 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2024-05-17 10:30:00 | 381.80 | 2024-05-17 10:50:00 | 380.97 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-22 09:40:00 | 375.30 | 2024-05-22 09:45:00 | 376.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-05-23 10:45:00 | 379.30 | 2024-05-23 13:05:00 | 378.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-29 10:40:00 | 383.35 | 2024-05-29 10:45:00 | 382.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-30 10:15:00 | 381.15 | 2024-05-30 11:05:00 | 379.79 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-30 10:15:00 | 381.15 | 2024-05-30 12:30:00 | 381.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-31 10:25:00 | 381.00 | 2024-05-31 10:40:00 | 379.57 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-03 09:55:00 | 389.15 | 2024-06-03 10:10:00 | 387.10 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-06-07 09:35:00 | 391.80 | 2024-06-07 09:45:00 | 393.87 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-07 09:35:00 | 391.80 | 2024-06-07 10:15:00 | 396.70 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2024-06-11 09:35:00 | 396.60 | 2024-06-11 09:40:00 | 397.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-20 10:10:00 | 444.40 | 2024-06-20 10:15:00 | 442.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-21 10:35:00 | 447.75 | 2024-06-21 10:45:00 | 445.65 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-06-24 09:30:00 | 441.70 | 2024-06-24 09:50:00 | 443.63 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-01 09:55:00 | 437.50 | 2024-07-01 10:00:00 | 436.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-03 09:40:00 | 433.00 | 2024-07-03 10:20:00 | 434.75 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-04 09:50:00 | 428.75 | 2024-07-04 10:25:00 | 429.84 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-05 09:45:00 | 425.00 | 2024-07-05 10:35:00 | 426.31 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-08 09:30:00 | 422.40 | 2024-07-08 09:35:00 | 423.59 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-09 10:25:00 | 430.65 | 2024-07-09 10:30:00 | 429.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-29 10:10:00 | 470.05 | 2024-07-29 11:10:00 | 472.95 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-29 10:10:00 | 470.05 | 2024-07-29 12:20:00 | 470.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 09:40:00 | 472.70 | 2024-07-30 10:15:00 | 475.26 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-30 09:40:00 | 472.70 | 2024-07-30 12:00:00 | 472.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:55:00 | 534.40 | 2024-08-19 10:15:00 | 538.01 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-08-19 09:55:00 | 534.40 | 2024-08-19 13:25:00 | 535.55 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-08-20 10:15:00 | 546.00 | 2024-08-20 10:25:00 | 543.52 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-27 09:40:00 | 556.45 | 2024-08-27 10:00:00 | 558.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-28 09:40:00 | 579.00 | 2024-08-28 09:50:00 | 576.32 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-09-03 09:30:00 | 585.70 | 2024-09-03 09:40:00 | 587.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-05 11:05:00 | 587.45 | 2024-09-05 15:10:00 | 589.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-18 09:55:00 | 577.00 | 2024-09-18 10:05:00 | 575.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-19 09:45:00 | 585.65 | 2024-09-19 11:00:00 | 589.95 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-09-19 09:45:00 | 585.65 | 2024-09-19 11:30:00 | 586.55 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-09-24 09:50:00 | 601.00 | 2024-09-24 10:20:00 | 603.56 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-24 09:50:00 | 601.00 | 2024-09-24 15:20:00 | 609.50 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2024-09-25 10:45:00 | 609.70 | 2024-09-25 11:00:00 | 611.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-01 09:45:00 | 660.65 | 2024-10-01 11:50:00 | 667.47 | PARTIAL | 0.50 | 1.03% |
| BUY | retest1 | 2024-10-01 09:45:00 | 660.65 | 2024-10-01 15:20:00 | 682.95 | TARGET_HIT | 0.50 | 3.38% |
| SELL | retest1 | 2024-10-07 10:05:00 | 648.20 | 2024-10-07 10:25:00 | 643.18 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-10-07 10:05:00 | 648.20 | 2024-10-07 11:15:00 | 648.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:55:00 | 660.80 | 2024-10-09 11:10:00 | 658.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-10 09:30:00 | 649.85 | 2024-10-10 09:35:00 | 651.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-11 11:05:00 | 647.60 | 2024-10-11 15:20:00 | 649.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-15 09:55:00 | 655.55 | 2024-10-15 10:00:00 | 653.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-16 10:30:00 | 654.25 | 2024-10-16 10:35:00 | 650.92 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-16 10:30:00 | 654.25 | 2024-10-16 14:05:00 | 643.80 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2024-10-21 09:30:00 | 635.00 | 2024-10-21 10:05:00 | 637.87 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-25 10:15:00 | 604.60 | 2024-10-25 10:20:00 | 607.03 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-30 09:30:00 | 649.85 | 2024-10-30 09:50:00 | 653.84 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-30 09:30:00 | 649.85 | 2024-10-30 14:15:00 | 658.00 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2024-11-05 09:55:00 | 611.80 | 2024-11-05 10:15:00 | 609.35 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-28 11:05:00 | 560.40 | 2024-11-28 11:15:00 | 556.38 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-11-28 11:05:00 | 560.40 | 2024-11-28 14:20:00 | 559.65 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-11-29 09:45:00 | 570.35 | 2024-11-29 09:50:00 | 573.75 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-29 09:45:00 | 570.35 | 2024-11-29 15:20:00 | 585.00 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2024-12-02 11:00:00 | 589.55 | 2024-12-02 11:05:00 | 587.14 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-09 11:05:00 | 576.60 | 2024-12-09 12:25:00 | 573.51 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-12-09 11:05:00 | 576.60 | 2024-12-09 15:05:00 | 576.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 10:10:00 | 581.70 | 2024-12-10 10:20:00 | 584.98 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-10 10:10:00 | 581.70 | 2024-12-10 10:30:00 | 581.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 09:35:00 | 592.00 | 2024-12-11 09:45:00 | 589.78 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-16 09:35:00 | 588.80 | 2024-12-16 14:15:00 | 586.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-17 09:45:00 | 590.95 | 2024-12-17 10:00:00 | 589.16 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-26 09:30:00 | 521.75 | 2024-12-26 09:35:00 | 518.88 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-26 09:30:00 | 521.75 | 2024-12-26 12:55:00 | 520.70 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-12-27 11:00:00 | 520.50 | 2024-12-27 11:10:00 | 521.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-30 09:35:00 | 515.60 | 2024-12-30 09:40:00 | 517.33 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-31 10:40:00 | 514.55 | 2024-12-31 12:00:00 | 516.48 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-02 10:20:00 | 540.70 | 2025-01-02 10:30:00 | 538.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-02 10:20:00 | 540.70 | 2025-01-02 10:40:00 | 540.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:30:00 | 481.50 | 2025-01-24 09:35:00 | 478.93 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-24 09:30:00 | 481.50 | 2025-01-24 09:40:00 | 481.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:30:00 | 469.80 | 2025-01-27 10:45:00 | 471.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-03 09:35:00 | 470.60 | 2025-02-03 12:00:00 | 465.55 | PARTIAL | 0.50 | 1.07% |
| SELL | retest1 | 2025-02-03 09:35:00 | 470.60 | 2025-02-03 15:20:00 | 465.00 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2025-02-04 10:50:00 | 463.55 | 2025-02-04 11:30:00 | 465.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-05 09:35:00 | 467.00 | 2025-02-05 09:45:00 | 469.32 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-02-07 10:45:00 | 466.00 | 2025-02-07 11:05:00 | 467.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-07 09:30:00 | 463.50 | 2025-03-07 09:35:00 | 461.46 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-03-11 10:50:00 | 457.70 | 2025-03-11 11:10:00 | 455.46 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-12 09:30:00 | 462.00 | 2025-03-12 09:50:00 | 465.34 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-03-12 09:30:00 | 462.00 | 2025-03-12 15:20:00 | 487.85 | TARGET_HIT | 0.50 | 5.60% |
| SELL | retest1 | 2025-03-24 09:30:00 | 535.00 | 2025-03-24 09:35:00 | 531.07 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-03-24 09:30:00 | 535.00 | 2025-03-24 09:50:00 | 535.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 11:10:00 | 530.05 | 2025-03-27 11:30:00 | 532.90 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-27 11:10:00 | 530.05 | 2025-03-27 15:05:00 | 531.45 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-03-28 10:10:00 | 544.90 | 2025-03-28 15:20:00 | 548.75 | TARGET_HIT | 1.00 | 0.71% |
| BUY | retest1 | 2025-04-02 10:05:00 | 564.95 | 2025-04-02 10:15:00 | 562.05 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-04-09 10:00:00 | 514.80 | 2025-04-09 10:10:00 | 517.33 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-04-21 10:40:00 | 558.00 | 2025-04-21 11:50:00 | 560.39 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-21 10:40:00 | 558.00 | 2025-04-21 15:20:00 | 566.20 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2025-04-24 10:45:00 | 588.65 | 2025-04-24 11:50:00 | 591.72 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-24 10:45:00 | 588.65 | 2025-04-24 15:20:00 | 602.00 | TARGET_HIT | 0.50 | 2.27% |
| SELL | retest1 | 2025-04-29 10:00:00 | 573.00 | 2025-04-29 10:20:00 | 569.37 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-04-29 10:00:00 | 573.00 | 2025-04-29 11:10:00 | 571.75 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-05-06 10:15:00 | 545.05 | 2025-05-06 10:40:00 | 547.03 | STOP_HIT | 1.00 | -0.36% |
