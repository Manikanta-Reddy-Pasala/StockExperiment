# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 462.60
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 7 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 32
- **Target hits / Stop hits / Partials:** 7 / 32 / 15
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 16.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 14 | 42.4% | 4 | 19 | 10 | 0.35% | 11.6% |
| BUY @ 2nd Alert (retest1) | 33 | 14 | 42.4% | 4 | 19 | 10 | 0.35% | 11.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.22% | 4.5% |
| SELL @ 2nd Alert (retest1) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.22% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 22 | 40.7% | 7 | 32 | 15 | 0.30% | 16.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 696.05 | 700.05 | 0.00 | ORB-short ORB[699.25,706.10] vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-05-28 09:50:00 | 698.02 | 699.33 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 833.50 | 826.17 | 0.00 | ORB-long ORB[817.90,828.70] vol=2.4x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 828.45 | 828.63 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:50:00 | 794.30 | 787.36 | 0.00 | ORB-long ORB[780.90,789.50] vol=2.5x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-07-09 10:05:00 | 791.18 | 790.21 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:35:00 | 763.55 | 765.41 | 0.00 | ORB-short ORB[764.50,775.05] vol=4.2x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:35:00 | 759.09 | 763.74 | 0.00 | T1 1.5R @ 759.09 |
| Target hit | 2024-07-30 15:20:00 | 752.75 | 760.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:00:00 | 767.75 | 757.65 | 0.00 | ORB-long ORB[748.60,759.50] vol=3.2x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-08-02 11:25:00 | 764.73 | 759.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-08-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:05:00 | 721.75 | 724.28 | 0.00 | ORB-short ORB[723.15,733.00] vol=1.6x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:50:00 | 718.27 | 722.40 | 0.00 | T1 1.5R @ 718.27 |
| Target hit | 2024-08-09 15:20:00 | 701.55 | 709.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-08-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:25:00 | 690.15 | 693.64 | 0.00 | ORB-short ORB[691.45,701.10] vol=1.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-08-13 10:40:00 | 692.47 | 693.55 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 690.05 | 693.59 | 0.00 | ORB-short ORB[692.60,701.95] vol=2.2x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-08-14 12:35:00 | 692.07 | 692.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:10:00 | 708.85 | 711.55 | 0.00 | ORB-short ORB[709.05,717.80] vol=1.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-08-19 11:15:00 | 710.85 | 711.55 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 733.50 | 734.51 | 0.00 | ORB-short ORB[734.55,742.20] vol=7.0x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-08-26 09:45:00 | 736.35 | 734.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:10:00 | 736.80 | 734.80 | 0.00 | ORB-long ORB[730.00,736.00] vol=3.1x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-08-27 10:40:00 | 734.27 | 734.96 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:35:00 | 739.45 | 744.72 | 0.00 | ORB-short ORB[740.30,748.95] vol=1.5x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:20:00 | 736.17 | 743.78 | 0.00 | T1 1.5R @ 736.17 |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 739.45 | 740.70 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 766.65 | 763.92 | 0.00 | ORB-long ORB[759.80,765.00] vol=2.7x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:35:00 | 769.78 | 765.01 | 0.00 | T1 1.5R @ 769.78 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 766.65 | 766.35 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:00:00 | 749.25 | 743.38 | 0.00 | ORB-long ORB[736.00,746.00] vol=2.3x ATR=3.72 |
| Stop hit — per-position SL triggered | 2024-09-12 10:05:00 | 745.53 | 744.18 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:05:00 | 790.60 | 793.96 | 0.00 | ORB-short ORB[793.80,805.05] vol=4.4x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 794.12 | 793.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:05:00 | 821.10 | 815.93 | 0.00 | ORB-long ORB[808.85,819.90] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 816.88 | 816.28 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-10-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:05:00 | 795.05 | 789.10 | 0.00 | ORB-long ORB[780.00,787.95] vol=3.0x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:35:00 | 800.24 | 792.93 | 0.00 | T1 1.5R @ 800.24 |
| Stop hit — per-position SL triggered | 2024-10-08 10:40:00 | 795.05 | 793.00 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:05:00 | 811.00 | 809.39 | 0.00 | ORB-long ORB[804.05,810.25] vol=3.5x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 808.23 | 809.44 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:10:00 | 813.10 | 809.78 | 0.00 | ORB-long ORB[803.60,811.65] vol=1.7x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-10-10 10:20:00 | 809.65 | 809.87 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 844.60 | 837.85 | 0.00 | ORB-long ORB[826.95,838.20] vol=3.7x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:40:00 | 850.01 | 841.72 | 0.00 | T1 1.5R @ 850.01 |
| Stop hit — per-position SL triggered | 2024-10-15 09:45:00 | 844.60 | 841.91 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-11-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:50:00 | 646.25 | 644.41 | 0.00 | ORB-long ORB[640.00,644.85] vol=2.1x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 644.13 | 644.22 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:55:00 | 653.55 | 652.68 | 0.00 | ORB-long ORB[644.90,651.95] vol=7.8x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 14:05:00 | 656.35 | 653.42 | 0.00 | T1 1.5R @ 656.35 |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 653.55 | 653.51 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-12-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:20:00 | 630.70 | 633.25 | 0.00 | ORB-short ORB[630.90,639.60] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-12-04 15:20:00 | 631.10 | 630.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:35:00 | 642.05 | 636.27 | 0.00 | ORB-long ORB[628.30,635.70] vol=1.7x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:40:00 | 645.01 | 637.97 | 0.00 | T1 1.5R @ 645.01 |
| Stop hit — per-position SL triggered | 2024-12-05 09:45:00 | 642.05 | 638.36 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 645.40 | 646.51 | 0.00 | ORB-short ORB[646.00,652.70] vol=1.8x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 12:55:00 | 642.76 | 644.65 | 0.00 | T1 1.5R @ 642.76 |
| Target hit | 2024-12-06 15:20:00 | 635.35 | 642.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-12-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:05:00 | 634.10 | 638.64 | 0.00 | ORB-short ORB[637.00,643.45] vol=3.2x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-12-09 11:25:00 | 635.61 | 637.97 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:00:00 | 646.65 | 642.05 | 0.00 | ORB-long ORB[634.65,644.10] vol=3.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2024-12-12 10:05:00 | 643.81 | 642.26 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 794.00 | 783.15 | 0.00 | ORB-long ORB[776.50,785.00] vol=1.8x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:00:00 | 801.29 | 790.76 | 0.00 | T1 1.5R @ 801.29 |
| Target hit | 2025-01-02 10:55:00 | 803.05 | 803.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-01-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:55:00 | 658.10 | 662.19 | 0.00 | ORB-short ORB[660.50,666.35] vol=2.0x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-01-21 10:10:00 | 660.32 | 661.30 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:05:00 | 730.45 | 724.82 | 0.00 | ORB-long ORB[713.55,720.20] vol=1.9x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 727.60 | 725.05 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:15:00 | 759.85 | 755.91 | 0.00 | ORB-long ORB[747.50,752.55] vol=2.2x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:45:00 | 764.69 | 757.75 | 0.00 | T1 1.5R @ 764.69 |
| Target hit | 2025-01-31 15:20:00 | 797.20 | 773.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 743.90 | 737.89 | 0.00 | ORB-long ORB[730.30,739.90] vol=2.2x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-02-05 11:45:00 | 741.66 | 738.66 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:55:00 | 752.10 | 746.38 | 0.00 | ORB-long ORB[741.30,748.95] vol=4.0x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:05:00 | 756.73 | 748.65 | 0.00 | T1 1.5R @ 756.73 |
| Target hit | 2025-02-06 15:20:00 | 755.45 | 757.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 709.20 | 713.05 | 0.00 | ORB-short ORB[711.50,718.30] vol=1.9x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-03-06 10:05:00 | 712.42 | 711.13 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:50:00 | 685.05 | 692.45 | 0.00 | ORB-short ORB[690.00,699.05] vol=2.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-03-20 09:55:00 | 688.36 | 692.11 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-04-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 09:55:00 | 694.00 | 688.88 | 0.00 | ORB-long ORB[679.10,689.10] vol=2.0x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-08 10:40:00 | 701.14 | 693.38 | 0.00 | T1 1.5R @ 701.14 |
| Target hit | 2025-04-08 12:40:00 | 721.75 | 722.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2025-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:10:00 | 748.65 | 742.34 | 0.00 | ORB-long ORB[736.80,746.40] vol=1.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-04-16 10:20:00 | 745.69 | 743.77 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:05:00 | 683.70 | 686.98 | 0.00 | ORB-short ORB[685.50,695.40] vol=3.3x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:35:00 | 680.65 | 686.58 | 0.00 | T1 1.5R @ 680.65 |
| Stop hit — per-position SL triggered | 2025-05-06 12:10:00 | 683.70 | 685.41 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 699.55 | 694.86 | 0.00 | ORB-long ORB[688.75,696.75] vol=2.2x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 09:35:00 | 704.20 | 696.63 | 0.00 | T1 1.5R @ 704.20 |
| Stop hit — per-position SL triggered | 2025-05-08 10:25:00 | 699.55 | 702.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-28 09:35:00 | 696.05 | 2024-05-28 09:50:00 | 698.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-02 09:35:00 | 833.50 | 2024-07-02 09:40:00 | 828.45 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-07-09 09:50:00 | 794.30 | 2024-07-09 10:05:00 | 791.18 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-30 10:35:00 | 763.55 | 2024-07-30 12:35:00 | 759.09 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-07-30 10:35:00 | 763.55 | 2024-07-30 15:20:00 | 752.75 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-08-02 11:00:00 | 767.75 | 2024-08-02 11:25:00 | 764.73 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-09 10:05:00 | 721.75 | 2024-08-09 10:50:00 | 718.27 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-09 10:05:00 | 721.75 | 2024-08-09 15:20:00 | 701.55 | TARGET_HIT | 0.50 | 2.80% |
| SELL | retest1 | 2024-08-13 10:25:00 | 690.15 | 2024-08-13 10:40:00 | 692.47 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-14 10:50:00 | 690.05 | 2024-08-14 12:35:00 | 692.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-19 11:10:00 | 708.85 | 2024-08-19 11:15:00 | 710.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-26 09:40:00 | 733.50 | 2024-08-26 09:45:00 | 736.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-27 10:10:00 | 736.80 | 2024-08-27 10:40:00 | 734.27 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-29 10:35:00 | 739.45 | 2024-08-29 11:20:00 | 736.17 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-29 10:35:00 | 739.45 | 2024-08-29 14:15:00 | 739.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:30:00 | 766.65 | 2024-09-03 09:35:00 | 769.78 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-03 09:30:00 | 766.65 | 2024-09-03 09:50:00 | 766.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-12 10:00:00 | 749.25 | 2024-09-12 10:05:00 | 745.53 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-09-24 10:05:00 | 790.60 | 2024-09-24 10:10:00 | 794.12 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-04 10:05:00 | 821.10 | 2024-10-04 10:15:00 | 816.88 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-10-08 10:05:00 | 795.05 | 2024-10-08 10:35:00 | 800.24 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-08 10:05:00 | 795.05 | 2024-10-08 10:40:00 | 795.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:05:00 | 811.00 | 2024-10-09 10:15:00 | 808.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-10 10:10:00 | 813.10 | 2024-10-10 10:20:00 | 809.65 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-15 09:30:00 | 844.60 | 2024-10-15 09:40:00 | 850.01 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-15 09:30:00 | 844.60 | 2024-10-15 09:45:00 | 844.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:50:00 | 646.25 | 2024-11-19 11:15:00 | 644.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-28 09:55:00 | 653.55 | 2024-11-28 14:05:00 | 656.35 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-28 09:55:00 | 653.55 | 2024-11-28 14:15:00 | 653.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 10:20:00 | 630.70 | 2024-12-04 15:20:00 | 631.10 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest1 | 2024-12-05 09:35:00 | 642.05 | 2024-12-05 09:40:00 | 645.01 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-05 09:35:00 | 642.05 | 2024-12-05 09:45:00 | 642.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 11:15:00 | 645.40 | 2024-12-06 12:55:00 | 642.76 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-06 11:15:00 | 645.40 | 2024-12-06 15:20:00 | 635.35 | TARGET_HIT | 0.50 | 1.56% |
| SELL | retest1 | 2024-12-09 11:05:00 | 634.10 | 2024-12-09 11:25:00 | 635.61 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-12 10:00:00 | 646.65 | 2024-12-12 10:05:00 | 643.81 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-02 09:35:00 | 794.00 | 2025-01-02 10:00:00 | 801.29 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2025-01-02 09:35:00 | 794.00 | 2025-01-02 10:55:00 | 803.05 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-01-21 09:55:00 | 658.10 | 2025-01-21 10:10:00 | 660.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-29 11:05:00 | 730.45 | 2025-01-29 11:20:00 | 727.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-31 10:15:00 | 759.85 | 2025-01-31 11:45:00 | 764.69 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-31 10:15:00 | 759.85 | 2025-01-31 15:20:00 | 797.20 | TARGET_HIT | 0.50 | 4.92% |
| BUY | retest1 | 2025-02-05 11:15:00 | 743.90 | 2025-02-05 11:45:00 | 741.66 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-06 09:55:00 | 752.10 | 2025-02-06 10:05:00 | 756.73 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-02-06 09:55:00 | 752.10 | 2025-02-06 15:20:00 | 755.45 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-03-06 09:30:00 | 709.20 | 2025-03-06 10:05:00 | 712.42 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-03-20 09:50:00 | 685.05 | 2025-03-20 09:55:00 | 688.36 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-04-08 09:55:00 | 694.00 | 2025-04-08 10:40:00 | 701.14 | PARTIAL | 0.50 | 1.03% |
| BUY | retest1 | 2025-04-08 09:55:00 | 694.00 | 2025-04-08 12:40:00 | 721.75 | TARGET_HIT | 0.50 | 4.00% |
| BUY | retest1 | 2025-04-16 10:10:00 | 748.65 | 2025-04-16 10:20:00 | 745.69 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-06 11:05:00 | 683.70 | 2025-05-06 11:35:00 | 680.65 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-06 11:05:00 | 683.70 | 2025-05-06 12:10:00 | 683.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:30:00 | 699.55 | 2025-05-08 09:35:00 | 704.20 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-05-08 09:30:00 | 699.55 | 2025-05-08 10:25:00 | 699.55 | STOP_HIT | 0.50 | 0.00% |
