# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 743.40
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 39
- **Target hits / Stop hits / Partials:** 7 / 39 / 18
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 16.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 9 | 30.0% | 2 | 21 | 7 | 0.17% | 5.1% |
| BUY @ 2nd Alert (retest1) | 30 | 9 | 30.0% | 2 | 21 | 7 | 0.17% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 16 | 47.1% | 5 | 18 | 11 | 0.33% | 11.3% |
| SELL @ 2nd Alert (retest1) | 34 | 16 | 47.1% | 5 | 18 | 11 | 0.33% | 11.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 64 | 25 | 39.1% | 7 | 39 | 18 | 0.26% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:50:00 | 514.95 | 519.11 | 0.00 | ORB-short ORB[517.35,524.00] vol=3.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-05-14 09:55:00 | 517.26 | 518.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:55:00 | 532.10 | 535.49 | 0.00 | ORB-short ORB[535.80,539.95] vol=3.0x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 12:15:00 | 528.79 | 532.25 | 0.00 | T1 1.5R @ 528.79 |
| Target hit | 2024-05-17 15:20:00 | 526.00 | 530.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 515.80 | 521.11 | 0.00 | ORB-short ORB[519.90,525.90] vol=1.8x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 518.20 | 520.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:30:00 | 515.70 | 519.83 | 0.00 | ORB-short ORB[518.10,523.80] vol=2.3x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-05-23 10:45:00 | 517.70 | 519.04 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 11:10:00 | 508.40 | 510.72 | 0.00 | ORB-short ORB[508.90,515.45] vol=2.2x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 15:15:00 | 505.44 | 509.18 | 0.00 | T1 1.5R @ 505.44 |
| Target hit | 2024-05-27 15:20:00 | 505.05 | 509.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 514.35 | 512.04 | 0.00 | ORB-long ORB[507.00,512.95] vol=3.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-05-28 09:35:00 | 512.41 | 511.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 09:55:00 | 499.20 | 502.04 | 0.00 | ORB-short ORB[500.00,506.85] vol=3.0x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-05-29 11:00:00 | 501.45 | 500.10 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 10:05:00 | 504.50 | 506.92 | 0.00 | ORB-short ORB[505.20,510.40] vol=1.9x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:30:00 | 501.17 | 505.76 | 0.00 | T1 1.5R @ 501.17 |
| Stop hit — per-position SL triggered | 2024-06-07 10:40:00 | 504.50 | 505.68 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:40:00 | 514.50 | 508.28 | 0.00 | ORB-long ORB[502.00,509.45] vol=3.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-06-10 09:45:00 | 511.92 | 508.50 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:10:00 | 522.05 | 522.41 | 0.00 | ORB-short ORB[523.00,527.00] vol=5.4x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-06-12 11:20:00 | 523.72 | 522.45 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 517.35 | 521.29 | 0.00 | ORB-short ORB[520.00,527.00] vol=2.1x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:45:00 | 514.61 | 519.82 | 0.00 | T1 1.5R @ 514.61 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 517.35 | 519.22 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 11:05:00 | 516.00 | 517.26 | 0.00 | ORB-short ORB[516.15,521.90] vol=1.6x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:20:00 | 513.96 | 516.87 | 0.00 | T1 1.5R @ 513.96 |
| Target hit | 2024-06-14 15:20:00 | 507.00 | 512.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-06-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:40:00 | 506.10 | 507.93 | 0.00 | ORB-short ORB[506.65,513.00] vol=1.8x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:00:00 | 502.97 | 506.83 | 0.00 | T1 1.5R @ 502.97 |
| Stop hit — per-position SL triggered | 2024-06-18 10:20:00 | 506.10 | 506.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:40:00 | 526.75 | 530.24 | 0.00 | ORB-short ORB[528.10,535.75] vol=1.9x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-06-24 09:45:00 | 529.17 | 530.06 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:25:00 | 527.80 | 532.46 | 0.00 | ORB-short ORB[532.00,538.65] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:45:00 | 525.10 | 530.85 | 0.00 | T1 1.5R @ 525.10 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 527.80 | 530.41 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:45:00 | 523.05 | 521.14 | 0.00 | ORB-long ORB[518.40,522.00] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-07-01 10:45:00 | 520.89 | 521.57 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:35:00 | 528.25 | 526.47 | 0.00 | ORB-long ORB[522.00,527.35] vol=2.4x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-07-03 09:40:00 | 526.15 | 526.51 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 524.00 | 526.50 | 0.00 | ORB-short ORB[525.00,530.25] vol=5.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-07-04 10:45:00 | 525.68 | 525.68 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:40:00 | 578.00 | 584.35 | 0.00 | ORB-short ORB[583.00,589.20] vol=1.8x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-07-16 10:50:00 | 580.20 | 583.28 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:50:00 | 579.25 | 574.24 | 0.00 | ORB-long ORB[569.15,575.00] vol=2.1x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-07-24 11:10:00 | 576.04 | 574.54 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 602.35 | 607.04 | 0.00 | ORB-short ORB[606.85,612.95] vol=2.8x ATR=4.19 |
| Stop hit — per-position SL triggered | 2024-07-30 09:45:00 | 606.54 | 606.58 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:50:00 | 614.30 | 609.67 | 0.00 | ORB-long ORB[606.00,612.75] vol=2.8x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-08-01 10:10:00 | 611.36 | 611.15 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:30:00 | 620.60 | 616.17 | 0.00 | ORB-long ORB[610.45,619.00] vol=2.4x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:45:00 | 624.36 | 617.52 | 0.00 | T1 1.5R @ 624.36 |
| Target hit | 2024-08-07 15:20:00 | 667.10 | 655.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 672.20 | 667.29 | 0.00 | ORB-long ORB[662.35,668.45] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:45:00 | 676.06 | 670.74 | 0.00 | T1 1.5R @ 676.06 |
| Stop hit — per-position SL triggered | 2024-08-27 11:45:00 | 672.20 | 674.74 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:00:00 | 771.35 | 764.21 | 0.00 | ORB-long ORB[756.80,767.00] vol=3.7x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-09-23 11:40:00 | 766.22 | 768.88 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:35:00 | 759.00 | 764.84 | 0.00 | ORB-short ORB[761.00,770.25] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:55:00 | 753.92 | 761.54 | 0.00 | T1 1.5R @ 753.92 |
| Target hit | 2024-09-27 15:20:00 | 741.85 | 752.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:45:00 | 744.75 | 738.79 | 0.00 | ORB-long ORB[731.70,740.85] vol=2.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-10-10 09:50:00 | 741.47 | 738.94 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:10:00 | 725.90 | 731.37 | 0.00 | ORB-short ORB[730.00,736.25] vol=6.7x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 729.26 | 731.12 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:45:00 | 671.00 | 677.63 | 0.00 | ORB-short ORB[679.85,688.00] vol=2.1x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:55:00 | 666.75 | 676.07 | 0.00 | T1 1.5R @ 666.75 |
| Stop hit — per-position SL triggered | 2024-10-25 12:35:00 | 671.00 | 667.06 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:55:00 | 698.00 | 693.56 | 0.00 | ORB-long ORB[688.05,696.60] vol=2.5x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:35:00 | 703.19 | 696.43 | 0.00 | T1 1.5R @ 703.19 |
| Target hit | 2024-11-25 15:20:00 | 706.70 | 706.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:30:00 | 816.10 | 799.85 | 0.00 | ORB-long ORB[770.05,782.00] vol=7.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2024-12-04 10:45:00 | 810.32 | 807.25 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:50:00 | 791.40 | 778.00 | 0.00 | ORB-long ORB[768.15,776.80] vol=8.3x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-12-10 10:55:00 | 787.37 | 779.83 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:55:00 | 854.40 | 846.67 | 0.00 | ORB-long ORB[838.45,845.00] vol=2.4x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:10:00 | 860.48 | 849.74 | 0.00 | T1 1.5R @ 860.48 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 854.40 | 850.94 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:15:00 | 818.35 | 825.53 | 0.00 | ORB-short ORB[821.80,831.35] vol=1.7x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-01-01 11:20:00 | 821.70 | 825.22 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:45:00 | 852.60 | 847.09 | 0.00 | ORB-long ORB[837.70,849.45] vol=7.9x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-01-07 11:05:00 | 847.54 | 847.33 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:30:00 | 823.20 | 828.46 | 0.00 | ORB-short ORB[825.00,836.50] vol=1.9x ATR=5.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:35:00 | 815.57 | 824.49 | 0.00 | T1 1.5R @ 815.57 |
| Target hit | 2025-01-08 15:20:00 | 789.15 | 804.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-01-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:50:00 | 733.55 | 727.93 | 0.00 | ORB-long ORB[718.10,728.95] vol=3.5x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-01-16 09:55:00 | 729.83 | 728.54 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 703.55 | 709.12 | 0.00 | ORB-short ORB[706.55,716.30] vol=4.7x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:05:00 | 699.07 | 705.83 | 0.00 | T1 1.5R @ 699.07 |
| Stop hit — per-position SL triggered | 2025-01-20 11:00:00 | 703.55 | 704.14 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-02-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:00:00 | 735.95 | 729.76 | 0.00 | ORB-long ORB[718.00,726.95] vol=1.7x ATR=3.74 |
| Stop hit — per-position SL triggered | 2025-02-05 10:05:00 | 732.21 | 730.42 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:45:00 | 731.80 | 725.59 | 0.00 | ORB-long ORB[714.40,724.50] vol=1.9x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 10:55:00 | 737.62 | 726.62 | 0.00 | T1 1.5R @ 737.62 |
| Stop hit — per-position SL triggered | 2025-02-07 14:20:00 | 731.80 | 733.01 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-02-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:40:00 | 672.95 | 682.47 | 0.00 | ORB-short ORB[681.95,691.45] vol=2.3x ATR=3.90 |
| Stop hit — per-position SL triggered | 2025-02-21 10:45:00 | 676.85 | 680.91 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:50:00 | 680.75 | 676.89 | 0.00 | ORB-long ORB[668.25,677.95] vol=2.1x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:55:00 | 684.33 | 678.20 | 0.00 | T1 1.5R @ 684.33 |
| Stop hit — per-position SL triggered | 2025-03-18 10:05:00 | 680.75 | 678.47 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:30:00 | 696.00 | 692.53 | 0.00 | ORB-long ORB[685.25,694.00] vol=2.0x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-03-19 09:40:00 | 692.48 | 692.77 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:55:00 | 667.00 | 662.65 | 0.00 | ORB-long ORB[656.65,665.00] vol=2.0x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:00:00 | 670.08 | 664.52 | 0.00 | T1 1.5R @ 670.08 |
| Stop hit — per-position SL triggered | 2025-04-21 11:40:00 | 667.00 | 665.14 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:45:00 | 715.20 | 711.55 | 0.00 | ORB-long ORB[696.80,707.50] vol=5.0x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-05-06 09:50:00 | 710.95 | 711.36 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:15:00 | 700.35 | 697.76 | 0.00 | ORB-long ORB[691.30,700.00] vol=2.2x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-05-08 11:20:00 | 698.02 | 697.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:50:00 | 514.95 | 2024-05-14 09:55:00 | 517.26 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-17 09:55:00 | 532.10 | 2024-05-17 12:15:00 | 528.79 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-05-17 09:55:00 | 532.10 | 2024-05-17 15:20:00 | 526.00 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-05-22 09:40:00 | 515.80 | 2024-05-22 09:45:00 | 518.20 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-23 10:30:00 | 515.70 | 2024-05-23 10:45:00 | 517.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-27 11:10:00 | 508.40 | 2024-05-27 15:15:00 | 505.44 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-27 11:10:00 | 508.40 | 2024-05-27 15:20:00 | 505.05 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-05-28 09:30:00 | 514.35 | 2024-05-28 09:35:00 | 512.41 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-05-29 09:55:00 | 499.20 | 2024-05-29 11:00:00 | 501.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-06-07 10:05:00 | 504.50 | 2024-06-07 10:30:00 | 501.17 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-06-07 10:05:00 | 504.50 | 2024-06-07 10:40:00 | 504.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-10 09:40:00 | 514.50 | 2024-06-10 09:45:00 | 511.92 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-06-12 11:10:00 | 522.05 | 2024-06-12 11:20:00 | 523.72 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-13 09:40:00 | 517.35 | 2024-06-13 09:45:00 | 514.61 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-13 09:40:00 | 517.35 | 2024-06-13 09:55:00 | 517.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 11:05:00 | 516.00 | 2024-06-14 11:20:00 | 513.96 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-14 11:05:00 | 516.00 | 2024-06-14 15:20:00 | 507.00 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2024-06-18 09:40:00 | 506.10 | 2024-06-18 10:00:00 | 502.97 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-06-18 09:40:00 | 506.10 | 2024-06-18 10:20:00 | 506.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-24 09:40:00 | 526.75 | 2024-06-24 09:45:00 | 529.17 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-06-27 10:25:00 | 527.80 | 2024-06-27 10:45:00 | 525.10 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-27 10:25:00 | 527.80 | 2024-06-27 11:15:00 | 527.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:45:00 | 523.05 | 2024-07-01 10:45:00 | 520.89 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-03 09:35:00 | 528.25 | 2024-07-03 09:40:00 | 526.15 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-04 10:25:00 | 524.00 | 2024-07-04 10:45:00 | 525.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-16 10:40:00 | 578.00 | 2024-07-16 10:50:00 | 580.20 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-24 10:50:00 | 579.25 | 2024-07-24 11:10:00 | 576.04 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-07-30 09:35:00 | 602.35 | 2024-07-30 09:45:00 | 606.54 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2024-08-01 09:50:00 | 614.30 | 2024-08-01 10:10:00 | 611.36 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-08-07 10:30:00 | 620.60 | 2024-08-07 10:45:00 | 624.36 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-07 10:30:00 | 620.60 | 2024-08-07 15:20:00 | 667.10 | TARGET_HIT | 0.50 | 7.49% |
| BUY | retest1 | 2024-08-27 09:30:00 | 672.20 | 2024-08-27 09:45:00 | 676.06 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-27 09:30:00 | 672.20 | 2024-08-27 11:45:00 | 672.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 10:00:00 | 771.35 | 2024-09-23 11:40:00 | 766.22 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2024-09-27 09:35:00 | 759.00 | 2024-09-27 09:55:00 | 753.92 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-09-27 09:35:00 | 759.00 | 2024-09-27 15:20:00 | 741.85 | TARGET_HIT | 0.50 | 2.26% |
| BUY | retest1 | 2024-10-10 09:45:00 | 744.75 | 2024-10-10 09:50:00 | 741.47 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-15 10:10:00 | 725.90 | 2024-10-15 10:15:00 | 729.26 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-25 09:45:00 | 671.00 | 2024-10-25 09:55:00 | 666.75 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-25 09:45:00 | 671.00 | 2024-10-25 12:35:00 | 671.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 09:55:00 | 698.00 | 2024-11-25 11:35:00 | 703.19 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-11-25 09:55:00 | 698.00 | 2024-11-25 15:20:00 | 706.70 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2024-12-04 10:30:00 | 816.10 | 2024-12-04 10:45:00 | 810.32 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2024-12-10 10:50:00 | 791.40 | 2024-12-10 10:55:00 | 787.37 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-17 09:55:00 | 854.40 | 2024-12-17 10:10:00 | 860.48 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-12-17 09:55:00 | 854.40 | 2024-12-17 10:20:00 | 854.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 11:15:00 | 818.35 | 2025-01-01 11:20:00 | 821.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-07 10:45:00 | 852.60 | 2025-01-07 11:05:00 | 847.54 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-01-08 09:30:00 | 823.20 | 2025-01-08 10:35:00 | 815.57 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2025-01-08 09:30:00 | 823.20 | 2025-01-08 15:20:00 | 789.15 | TARGET_HIT | 0.50 | 4.14% |
| BUY | retest1 | 2025-01-16 09:50:00 | 733.55 | 2025-01-16 09:55:00 | 729.83 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-01-20 09:45:00 | 703.55 | 2025-01-20 10:05:00 | 699.07 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-01-20 09:45:00 | 703.55 | 2025-01-20 11:00:00 | 703.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 10:00:00 | 735.95 | 2025-02-05 10:05:00 | 732.21 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-02-07 10:45:00 | 731.80 | 2025-02-07 10:55:00 | 737.62 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-02-07 10:45:00 | 731.80 | 2025-02-07 14:20:00 | 731.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-21 10:40:00 | 672.95 | 2025-02-21 10:45:00 | 676.85 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2025-03-18 09:50:00 | 680.75 | 2025-03-18 09:55:00 | 684.33 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-18 09:50:00 | 680.75 | 2025-03-18 10:05:00 | 680.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 09:30:00 | 696.00 | 2025-03-19 09:40:00 | 692.48 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-04-21 10:55:00 | 667.00 | 2025-04-21 11:00:00 | 670.08 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-21 10:55:00 | 667.00 | 2025-04-21 11:40:00 | 667.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-06 09:45:00 | 715.20 | 2025-05-06 09:50:00 | 710.95 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-05-08 11:15:00 | 700.35 | 2025-05-08 11:20:00 | 698.02 | STOP_HIT | 1.00 | -0.33% |
