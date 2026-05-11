# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 530.45
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
| ENTRY1 | 61 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 8 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 53
- **Target hits / Stop hits / Partials:** 8 / 53 / 20
- **Avg / median % per leg:** 0.06% / -0.25%
- **Sum % (uncompounded):** 4.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 14 | 31.8% | 4 | 30 | 10 | -0.02% | -0.7% |
| BUY @ 2nd Alert (retest1) | 44 | 14 | 31.8% | 4 | 30 | 10 | -0.02% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 14 | 37.8% | 4 | 23 | 10 | 0.14% | 5.3% |
| SELL @ 2nd Alert (retest1) | 37 | 14 | 37.8% | 4 | 23 | 10 | 0.14% | 5.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 28 | 34.6% | 8 | 53 | 20 | 0.06% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 770.90 | 766.90 | 0.00 | ORB-long ORB[760.00,770.00] vol=1.8x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:45:00 | 775.07 | 769.74 | 0.00 | T1 1.5R @ 775.07 |
| Stop hit — per-position SL triggered | 2025-06-05 10:05:00 | 770.90 | 770.37 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:05:00 | 768.80 | 773.39 | 0.00 | ORB-short ORB[770.70,781.55] vol=1.7x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-06-06 12:25:00 | 771.32 | 771.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:30:00 | 772.15 | 765.54 | 0.00 | ORB-long ORB[754.30,765.65] vol=1.7x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-06-11 11:10:00 | 769.56 | 766.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 769.20 | 765.43 | 0.00 | ORB-long ORB[758.05,768.75] vol=2.1x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 765.48 | 768.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 770.50 | 766.65 | 0.00 | ORB-long ORB[759.90,769.50] vol=2.6x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-06-18 10:25:00 | 767.61 | 769.81 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 775.10 | 771.72 | 0.00 | ORB-long ORB[764.00,775.00] vol=2.2x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-06-19 09:40:00 | 772.39 | 776.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:55:00 | 751.50 | 757.81 | 0.00 | ORB-short ORB[761.05,768.60] vol=4.4x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 11:00:00 | 748.27 | 756.48 | 0.00 | T1 1.5R @ 748.27 |
| Stop hit — per-position SL triggered | 2025-06-30 13:25:00 | 751.50 | 752.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:50:00 | 743.70 | 748.43 | 0.00 | ORB-short ORB[746.50,755.60] vol=1.8x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:15:00 | 739.13 | 746.10 | 0.00 | T1 1.5R @ 739.13 |
| Target hit | 2025-07-01 15:20:00 | 722.75 | 730.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-07-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 11:10:00 | 759.10 | 753.14 | 0.00 | ORB-long ORB[746.55,755.80] vol=2.3x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:25:00 | 762.53 | 754.57 | 0.00 | T1 1.5R @ 762.53 |
| Stop hit — per-position SL triggered | 2025-07-07 13:05:00 | 759.10 | 757.05 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:55:00 | 747.10 | 754.13 | 0.00 | ORB-short ORB[753.30,762.55] vol=1.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:10:00 | 743.29 | 751.28 | 0.00 | T1 1.5R @ 743.29 |
| Stop hit — per-position SL triggered | 2025-07-08 13:55:00 | 747.10 | 746.16 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:40:00 | 757.20 | 749.27 | 0.00 | ORB-long ORB[740.00,747.45] vol=3.6x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 09:45:00 | 762.06 | 754.36 | 0.00 | T1 1.5R @ 762.06 |
| Target hit | 2025-07-09 10:35:00 | 763.65 | 765.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2025-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:00:00 | 757.85 | 766.89 | 0.00 | ORB-short ORB[765.85,775.00] vol=2.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-07-11 11:30:00 | 760.78 | 766.09 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:35:00 | 798.00 | 790.43 | 0.00 | ORB-long ORB[783.90,792.00] vol=3.7x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:40:00 | 803.67 | 796.31 | 0.00 | T1 1.5R @ 803.67 |
| Target hit | 2025-07-15 09:50:00 | 799.40 | 799.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2025-07-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:00:00 | 819.25 | 811.89 | 0.00 | ORB-long ORB[804.15,816.00] vol=3.2x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-07-16 10:05:00 | 815.59 | 812.73 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 817.85 | 825.81 | 0.00 | ORB-short ORB[823.00,830.20] vol=2.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-07-18 10:20:00 | 820.40 | 824.77 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:10:00 | 811.10 | 806.20 | 0.00 | ORB-long ORB[799.30,810.40] vol=1.7x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:30:00 | 816.35 | 807.53 | 0.00 | T1 1.5R @ 816.35 |
| Stop hit — per-position SL triggered | 2025-07-21 11:45:00 | 811.10 | 809.81 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:20:00 | 799.10 | 806.18 | 0.00 | ORB-short ORB[804.30,814.20] vol=1.5x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-07-22 10:40:00 | 801.86 | 804.38 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:55:00 | 814.00 | 808.84 | 0.00 | ORB-long ORB[803.00,811.90] vol=2.2x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 11:35:00 | 818.16 | 810.96 | 0.00 | T1 1.5R @ 818.16 |
| Stop hit — per-position SL triggered | 2025-07-30 11:50:00 | 814.00 | 811.38 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 576.60 | 571.69 | 0.00 | ORB-long ORB[566.20,574.50] vol=1.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 573.85 | 571.91 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:45:00 | 558.70 | 552.12 | 0.00 | ORB-long ORB[548.35,556.60] vol=2.7x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-09-09 10:50:00 | 556.71 | 552.47 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:05:00 | 571.60 | 567.55 | 0.00 | ORB-long ORB[562.10,569.90] vol=1.6x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-09-10 12:00:00 | 569.92 | 568.57 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:05:00 | 569.10 | 572.45 | 0.00 | ORB-short ORB[570.00,576.60] vol=2.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-09-12 11:10:00 | 571.02 | 572.40 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:30:00 | 575.00 | 573.69 | 0.00 | ORB-long ORB[569.05,574.90] vol=4.2x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-09-16 10:35:00 | 573.42 | 573.70 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:05:00 | 570.75 | 567.32 | 0.00 | ORB-long ORB[563.60,567.40] vol=3.9x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-09-19 10:10:00 | 569.01 | 567.61 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 529.05 | 532.25 | 0.00 | ORB-short ORB[530.85,538.50] vol=2.4x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-09-26 09:35:00 | 531.02 | 532.03 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:45:00 | 516.30 | 513.87 | 0.00 | ORB-long ORB[509.40,515.00] vol=2.9x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-03 10:05:00 | 514.60 | 514.68 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:30:00 | 511.00 | 512.63 | 0.00 | ORB-short ORB[511.10,518.00] vol=2.0x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 09:50:00 | 508.68 | 512.03 | 0.00 | T1 1.5R @ 508.68 |
| Stop hit — per-position SL triggered | 2025-10-06 12:00:00 | 511.00 | 508.38 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 595.85 | 589.67 | 0.00 | ORB-long ORB[587.35,594.00] vol=6.3x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-10-20 11:45:00 | 593.22 | 591.08 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:40:00 | 575.65 | 581.26 | 0.00 | ORB-short ORB[582.10,588.85] vol=3.0x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-10-23 11:30:00 | 577.59 | 580.41 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:35:00 | 580.00 | 575.92 | 0.00 | ORB-long ORB[571.05,578.25] vol=2.4x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-10-24 10:00:00 | 577.91 | 577.12 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:00:00 | 571.20 | 573.19 | 0.00 | ORB-short ORB[571.25,577.00] vol=2.1x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 10:05:00 | 569.00 | 572.93 | 0.00 | T1 1.5R @ 569.00 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 571.20 | 572.71 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:35:00 | 571.00 | 569.07 | 0.00 | ORB-long ORB[564.85,570.70] vol=1.7x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-10-29 09:55:00 | 569.07 | 569.70 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 552.35 | 562.79 | 0.00 | ORB-short ORB[565.05,572.45] vol=2.4x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-11-06 11:25:00 | 554.40 | 561.87 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:35:00 | 532.00 | 533.61 | 0.00 | ORB-short ORB[532.80,536.90] vol=2.2x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 533.76 | 533.41 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 522.65 | 528.08 | 0.00 | ORB-short ORB[529.30,536.05] vol=3.2x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:10:00 | 519.73 | 526.68 | 0.00 | T1 1.5R @ 519.73 |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 522.65 | 526.49 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:50:00 | 595.50 | 593.30 | 0.00 | ORB-long ORB[587.15,595.45] vol=1.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-12-02 11:10:00 | 594.03 | 593.77 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:30:00 | 580.85 | 574.85 | 0.00 | ORB-long ORB[569.15,573.55] vol=6.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-12-04 10:40:00 | 578.76 | 575.36 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:25:00 | 567.00 | 571.10 | 0.00 | ORB-short ORB[570.60,579.05] vol=4.3x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-12-05 10:55:00 | 569.22 | 570.75 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:10:00 | 550.50 | 552.17 | 0.00 | ORB-short ORB[551.55,557.00] vol=4.2x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-12-10 10:40:00 | 553.53 | 552.35 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:05:00 | 547.55 | 544.37 | 0.00 | ORB-long ORB[539.30,546.30] vol=3.8x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:15:00 | 551.77 | 545.08 | 0.00 | T1 1.5R @ 551.77 |
| Stop hit — per-position SL triggered | 2025-12-11 11:05:00 | 547.55 | 546.43 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 573.60 | 572.01 | 0.00 | ORB-long ORB[568.00,572.90] vol=2.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-12-19 09:40:00 | 571.63 | 572.08 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:25:00 | 584.95 | 580.91 | 0.00 | ORB-long ORB[576.10,583.35] vol=2.1x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-12-24 11:00:00 | 582.65 | 581.76 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 566.75 | 570.52 | 0.00 | ORB-short ORB[568.00,576.00] vol=1.6x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:00:00 | 563.91 | 569.59 | 0.00 | T1 1.5R @ 563.91 |
| Target hit | 2025-12-30 13:10:00 | 566.25 | 565.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 611.80 | 618.89 | 0.00 | ORB-short ORB[619.20,626.35] vol=4.2x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 613.86 | 617.68 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 600.90 | 605.77 | 0.00 | ORB-short ORB[604.00,612.50] vol=2.0x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-01-09 09:40:00 | 604.01 | 605.36 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-01-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:40:00 | 594.75 | 591.77 | 0.00 | ORB-long ORB[585.95,593.95] vol=1.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:10:00 | 598.37 | 594.21 | 0.00 | T1 1.5R @ 598.37 |
| Stop hit — per-position SL triggered | 2026-01-16 10:40:00 | 594.75 | 595.15 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 560.90 | 551.11 | 0.00 | ORB-long ORB[544.10,551.00] vol=6.0x ATR=2.30 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 558.60 | 552.77 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-02-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:10:00 | 569.20 | 570.43 | 0.00 | ORB-short ORB[571.30,579.70] vol=1.7x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:00:00 | 565.49 | 570.05 | 0.00 | T1 1.5R @ 565.49 |
| Stop hit — per-position SL triggered | 2026-02-06 13:45:00 | 569.20 | 569.15 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 606.60 | 602.66 | 0.00 | ORB-long ORB[598.30,605.60] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:20:00 | 610.39 | 604.20 | 0.00 | T1 1.5R @ 610.39 |
| Target hit | 2026-02-11 11:50:00 | 614.30 | 614.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 604.65 | 610.54 | 0.00 | ORB-short ORB[607.80,616.75] vol=1.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 607.50 | 609.06 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 624.10 | 619.31 | 0.00 | ORB-long ORB[612.05,617.85] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 621.77 | 619.91 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 618.95 | 622.54 | 0.00 | ORB-short ORB[623.25,630.60] vol=2.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 620.69 | 622.31 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 612.55 | 608.55 | 0.00 | ORB-long ORB[603.00,611.00] vol=1.5x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:10:00 | 615.84 | 609.46 | 0.00 | T1 1.5R @ 615.84 |
| Target hit | 2026-02-20 15:20:00 | 614.00 | 612.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 610.00 | 616.83 | 0.00 | ORB-short ORB[614.20,621.50] vol=1.8x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 612.02 | 615.33 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 623.20 | 616.05 | 0.00 | ORB-long ORB[611.40,619.55] vol=1.8x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-02-25 12:25:00 | 620.67 | 619.54 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 628.80 | 625.12 | 0.00 | ORB-long ORB[619.00,627.40] vol=2.2x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-02-26 10:45:00 | 626.06 | 627.17 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:40:00 | 528.60 | 520.11 | 0.00 | ORB-long ORB[511.60,519.40] vol=2.0x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-03-17 09:45:00 | 524.56 | 520.67 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:15:00 | 502.75 | 506.14 | 0.00 | ORB-short ORB[507.55,514.95] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:45:00 | 499.23 | 505.47 | 0.00 | T1 1.5R @ 499.23 |
| Target hit | 2026-03-27 15:20:00 | 487.50 | 498.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 568.00 | 563.75 | 0.00 | ORB-long ORB[559.35,565.60] vol=3.1x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 565.67 | 564.47 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 550.20 | 551.31 | 0.00 | ORB-short ORB[550.60,558.50] vol=1.5x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:55:00 | 546.35 | 550.36 | 0.00 | T1 1.5R @ 546.35 |
| Target hit | 2026-04-24 14:25:00 | 548.95 | 548.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 556.30 | 560.44 | 0.00 | ORB-short ORB[562.05,567.90] vol=1.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 558.29 | 560.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-05 09:30:00 | 770.90 | 2025-06-05 09:45:00 | 775.07 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-05 09:30:00 | 770.90 | 2025-06-05 10:05:00 | 770.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 11:05:00 | 768.80 | 2025-06-06 12:25:00 | 771.32 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-11 10:30:00 | 772.15 | 2025-06-11 11:10:00 | 769.56 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-17 09:30:00 | 769.20 | 2025-06-17 11:15:00 | 765.48 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-06-18 09:30:00 | 770.50 | 2025-06-18 10:25:00 | 767.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-19 09:30:00 | 775.10 | 2025-06-19 09:40:00 | 772.39 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-30 10:55:00 | 751.50 | 2025-06-30 11:00:00 | 748.27 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-30 10:55:00 | 751.50 | 2025-06-30 13:25:00 | 751.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 09:50:00 | 743.70 | 2025-07-01 10:15:00 | 739.13 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-07-01 09:50:00 | 743.70 | 2025-07-01 15:20:00 | 722.75 | TARGET_HIT | 0.50 | 2.82% |
| BUY | retest1 | 2025-07-07 11:10:00 | 759.10 | 2025-07-07 11:25:00 | 762.53 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-07 11:10:00 | 759.10 | 2025-07-07 13:05:00 | 759.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-08 09:55:00 | 747.10 | 2025-07-08 10:10:00 | 743.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-08 09:55:00 | 747.10 | 2025-07-08 13:55:00 | 747.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 09:40:00 | 757.20 | 2025-07-09 09:45:00 | 762.06 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-07-09 09:40:00 | 757.20 | 2025-07-09 10:35:00 | 763.65 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2025-07-11 11:00:00 | 757.85 | 2025-07-11 11:30:00 | 760.78 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-15 09:35:00 | 798.00 | 2025-07-15 09:40:00 | 803.67 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-15 09:35:00 | 798.00 | 2025-07-15 09:50:00 | 799.40 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-07-16 10:00:00 | 819.25 | 2025-07-16 10:05:00 | 815.59 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-18 10:15:00 | 817.85 | 2025-07-18 10:20:00 | 820.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-21 10:10:00 | 811.10 | 2025-07-21 10:30:00 | 816.35 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-07-21 10:10:00 | 811.10 | 2025-07-21 11:45:00 | 811.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 10:20:00 | 799.10 | 2025-07-22 10:40:00 | 801.86 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-30 10:55:00 | 814.00 | 2025-07-30 11:35:00 | 818.16 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-07-30 10:55:00 | 814.00 | 2025-07-30 11:50:00 | 814.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-25 09:50:00 | 576.60 | 2025-08-25 09:55:00 | 573.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-09-09 10:45:00 | 558.70 | 2025-09-09 10:50:00 | 556.71 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-10 11:05:00 | 571.60 | 2025-09-10 12:00:00 | 569.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-12 11:05:00 | 569.10 | 2025-09-12 11:10:00 | 571.02 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-16 10:30:00 | 575.00 | 2025-09-16 10:35:00 | 573.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-19 10:05:00 | 570.75 | 2025-09-19 10:10:00 | 569.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-26 09:30:00 | 529.05 | 2025-09-26 09:35:00 | 531.02 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-03 09:45:00 | 516.30 | 2025-10-03 10:05:00 | 514.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-06 09:30:00 | 511.00 | 2025-10-06 09:50:00 | 508.68 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-06 09:30:00 | 511.00 | 2025-10-06 12:00:00 | 511.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 11:15:00 | 595.85 | 2025-10-20 11:45:00 | 593.22 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-10-23 10:40:00 | 575.65 | 2025-10-23 11:30:00 | 577.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-24 09:35:00 | 580.00 | 2025-10-24 10:00:00 | 577.91 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-28 10:00:00 | 571.20 | 2025-10-28 10:05:00 | 569.00 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-28 10:00:00 | 571.20 | 2025-10-28 10:15:00 | 571.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 09:35:00 | 571.00 | 2025-10-29 09:55:00 | 569.07 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-06 11:15:00 | 552.35 | 2025-11-06 11:25:00 | 554.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-11 10:35:00 | 532.00 | 2025-11-11 11:00:00 | 533.76 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-12 10:00:00 | 522.65 | 2025-11-12 10:10:00 | 519.73 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-11-12 10:00:00 | 522.65 | 2025-11-12 10:15:00 | 522.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-02 10:50:00 | 595.50 | 2025-12-02 11:10:00 | 594.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-04 10:30:00 | 580.85 | 2025-12-04 10:40:00 | 578.76 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-05 10:25:00 | 567.00 | 2025-12-05 10:55:00 | 569.22 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-12-10 10:10:00 | 550.50 | 2025-12-10 10:40:00 | 553.53 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-12-11 10:05:00 | 547.55 | 2025-12-11 10:15:00 | 551.77 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-12-11 10:05:00 | 547.55 | 2025-12-11 11:05:00 | 547.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:30:00 | 573.60 | 2025-12-19 09:40:00 | 571.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-24 10:25:00 | 584.95 | 2025-12-24 11:00:00 | 582.65 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-12-30 10:50:00 | 566.75 | 2025-12-30 11:00:00 | 563.91 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-30 10:50:00 | 566.75 | 2025-12-30 13:10:00 | 566.25 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-01-08 11:15:00 | 611.80 | 2026-01-08 11:35:00 | 613.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-01-09 09:35:00 | 600.90 | 2026-01-09 09:40:00 | 604.01 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-01-16 09:40:00 | 594.75 | 2026-01-16 10:10:00 | 598.37 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-01-16 09:40:00 | 594.75 | 2026-01-16 10:40:00 | 594.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 560.90 | 2026-02-01 11:15:00 | 558.60 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-06 11:10:00 | 569.20 | 2026-02-06 12:00:00 | 565.49 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-02-06 11:10:00 | 569.20 | 2026-02-06 13:45:00 | 569.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:10:00 | 606.60 | 2026-02-11 10:20:00 | 610.39 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-11 10:10:00 | 606.60 | 2026-02-11 11:50:00 | 614.30 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2026-02-13 09:30:00 | 604.65 | 2026-02-13 09:40:00 | 607.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-17 10:35:00 | 624.10 | 2026-02-17 10:45:00 | 621.77 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-19 10:55:00 | 618.95 | 2026-02-19 11:00:00 | 620.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-20 10:50:00 | 612.55 | 2026-02-20 11:10:00 | 615.84 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-20 10:50:00 | 612.55 | 2026-02-20 15:20:00 | 614.00 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-23 10:40:00 | 610.00 | 2026-02-23 11:00:00 | 612.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 11:00:00 | 623.20 | 2026-02-25 12:25:00 | 620.67 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-26 09:35:00 | 628.80 | 2026-02-26 10:45:00 | 626.06 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-17 09:40:00 | 528.60 | 2026-03-17 09:45:00 | 524.56 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-03-27 11:15:00 | 502.75 | 2026-03-27 11:45:00 | 499.23 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-03-27 11:15:00 | 502.75 | 2026-03-27 15:20:00 | 487.50 | TARGET_HIT | 0.50 | 3.03% |
| BUY | retest1 | 2026-04-21 10:00:00 | 568.00 | 2026-04-21 10:40:00 | 565.67 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-24 10:15:00 | 550.20 | 2026-04-24 10:55:00 | 546.35 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-24 10:15:00 | 550.20 | 2026-04-24 14:25:00 | 548.95 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2026-04-29 10:15:00 | 556.30 | 2026-04-29 10:20:00 | 558.29 | STOP_HIT | 1.00 | -0.36% |
