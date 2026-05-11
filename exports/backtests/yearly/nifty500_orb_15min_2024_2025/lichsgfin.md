# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 581.85
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 9 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 116 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 78
- **Target hits / Stop hits / Partials:** 9 / 78 / 29
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 10.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 19 | 30.6% | 4 | 43 | 15 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 62 | 19 | 30.6% | 4 | 43 | 15 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 19 | 35.2% | 5 | 35 | 14 | 0.20% | 10.7% |
| SELL @ 2nd Alert (retest1) | 54 | 19 | 35.2% | 5 | 35 | 14 | 0.20% | 10.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 116 | 38 | 32.8% | 9 | 78 | 29 | 0.09% | 10.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:00:00 | 619.35 | 621.87 | 0.00 | ORB-short ORB[621.10,629.25] vol=1.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:10:00 | 614.65 | 621.27 | 0.00 | T1 1.5R @ 614.65 |
| Stop hit — per-position SL triggered | 2024-05-14 10:20:00 | 619.35 | 620.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 648.40 | 653.54 | 0.00 | ORB-short ORB[652.25,657.90] vol=1.6x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:50:00 | 644.91 | 652.70 | 0.00 | T1 1.5R @ 644.91 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 648.40 | 652.33 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:40:00 | 654.95 | 658.15 | 0.00 | ORB-short ORB[656.60,663.00] vol=1.7x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 12:50:00 | 651.85 | 656.55 | 0.00 | T1 1.5R @ 651.85 |
| Stop hit — per-position SL triggered | 2024-05-23 15:00:00 | 654.95 | 654.84 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:35:00 | 649.85 | 646.74 | 0.00 | ORB-long ORB[640.20,646.15] vol=2.2x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:55:00 | 653.22 | 649.81 | 0.00 | T1 1.5R @ 653.22 |
| Stop hit — per-position SL triggered | 2024-05-29 14:05:00 | 649.85 | 650.55 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:45:00 | 629.50 | 631.29 | 0.00 | ORB-short ORB[633.00,639.00] vol=2.9x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 631.55 | 630.94 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:50:00 | 671.55 | 666.76 | 0.00 | ORB-long ORB[663.00,669.90] vol=3.6x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:00:00 | 675.55 | 668.41 | 0.00 | T1 1.5R @ 675.55 |
| Stop hit — per-position SL triggered | 2024-06-10 11:50:00 | 671.55 | 670.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 710.35 | 715.57 | 0.00 | ORB-short ORB[714.40,723.00] vol=1.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 712.93 | 714.92 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:35:00 | 753.05 | 742.52 | 0.00 | ORB-long ORB[733.65,743.90] vol=1.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:45:00 | 758.06 | 745.03 | 0.00 | T1 1.5R @ 758.06 |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 753.05 | 747.37 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:05:00 | 733.15 | 728.15 | 0.00 | ORB-long ORB[721.75,731.70] vol=1.8x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-06-24 11:45:00 | 730.42 | 730.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:05:00 | 724.75 | 729.07 | 0.00 | ORB-short ORB[731.25,736.20] vol=1.5x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 726.90 | 728.83 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 785.20 | 780.70 | 0.00 | ORB-long ORB[772.50,783.70] vol=2.8x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 781.60 | 781.71 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 789.60 | 794.15 | 0.00 | ORB-short ORB[796.40,803.50] vol=1.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:35:00 | 786.33 | 793.34 | 0.00 | T1 1.5R @ 786.33 |
| Stop hit — per-position SL triggered | 2024-07-04 10:55:00 | 789.60 | 792.63 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 786.20 | 789.95 | 0.00 | ORB-short ORB[788.50,795.25] vol=3.0x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-07-05 09:40:00 | 788.34 | 788.61 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:55:00 | 793.45 | 787.65 | 0.00 | ORB-long ORB[781.55,790.00] vol=2.4x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 790.36 | 789.42 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:50:00 | 819.00 | 810.35 | 0.00 | ORB-long ORB[801.60,812.40] vol=5.2x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:00:00 | 824.57 | 813.97 | 0.00 | T1 1.5R @ 824.57 |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 819.00 | 815.61 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:05:00 | 775.20 | 771.46 | 0.00 | ORB-long ORB[765.00,772.90] vol=3.4x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:10:00 | 782.03 | 772.58 | 0.00 | T1 1.5R @ 782.03 |
| Target hit | 2024-07-24 15:20:00 | 782.55 | 781.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-07-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:10:00 | 778.40 | 775.27 | 0.00 | ORB-long ORB[767.00,775.70] vol=4.9x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:20:00 | 782.14 | 776.46 | 0.00 | T1 1.5R @ 782.14 |
| Stop hit — per-position SL triggered | 2024-07-26 11:35:00 | 778.40 | 776.61 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:10:00 | 650.35 | 643.50 | 0.00 | ORB-long ORB[638.30,644.90] vol=2.1x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-08-08 10:20:00 | 647.49 | 643.99 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:35:00 | 633.45 | 637.38 | 0.00 | ORB-short ORB[637.00,644.20] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 636.63 | 636.72 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 654.00 | 651.43 | 0.00 | ORB-long ORB[648.10,653.50] vol=2.3x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 651.50 | 651.45 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:25:00 | 671.55 | 673.72 | 0.00 | ORB-short ORB[674.00,678.85] vol=1.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-08-23 10:35:00 | 672.98 | 673.65 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:45:00 | 680.85 | 678.30 | 0.00 | ORB-long ORB[672.20,679.60] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-08-29 10:00:00 | 679.08 | 678.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:55:00 | 677.00 | 678.46 | 0.00 | ORB-short ORB[677.15,683.65] vol=5.6x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-08-30 11:00:00 | 679.00 | 678.43 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:30:00 | 672.30 | 676.18 | 0.00 | ORB-short ORB[674.15,680.75] vol=1.9x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-09-02 10:05:00 | 674.38 | 674.79 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:15:00 | 686.45 | 682.90 | 0.00 | ORB-long ORB[678.00,685.95] vol=2.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-09-03 10:20:00 | 684.15 | 682.99 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:05:00 | 688.95 | 696.60 | 0.00 | ORB-short ORB[696.15,705.00] vol=2.1x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:40:00 | 685.41 | 695.29 | 0.00 | T1 1.5R @ 685.41 |
| Stop hit — per-position SL triggered | 2024-09-10 12:05:00 | 688.95 | 694.76 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:35:00 | 692.65 | 683.99 | 0.00 | ORB-long ORB[677.10,685.25] vol=2.1x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-09-11 10:55:00 | 689.75 | 685.61 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:00:00 | 714.85 | 708.80 | 0.00 | ORB-long ORB[704.00,713.95] vol=1.5x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:05:00 | 719.00 | 711.19 | 0.00 | T1 1.5R @ 719.00 |
| Target hit | 2024-09-13 13:20:00 | 719.90 | 720.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 671.70 | 676.21 | 0.00 | ORB-short ORB[674.55,681.80] vol=2.0x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-09-17 09:35:00 | 674.22 | 675.79 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 677.80 | 680.27 | 0.00 | ORB-short ORB[679.00,682.25] vol=2.7x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:00:00 | 674.71 | 678.50 | 0.00 | T1 1.5R @ 674.71 |
| Target hit | 2024-09-19 15:00:00 | 661.80 | 659.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 674.90 | 671.41 | 0.00 | ORB-long ORB[666.60,674.80] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 12:00:00 | 678.16 | 672.31 | 0.00 | T1 1.5R @ 678.16 |
| Target hit | 2024-09-23 15:20:00 | 680.75 | 676.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2024-09-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:35:00 | 660.50 | 662.78 | 0.00 | ORB-short ORB[661.35,667.70] vol=2.4x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 662.18 | 661.85 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:30:00 | 648.75 | 651.18 | 0.00 | ORB-short ORB[648.90,655.90] vol=1.7x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-10-03 09:35:00 | 650.61 | 651.04 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:45:00 | 625.85 | 623.24 | 0.00 | ORB-long ORB[617.10,624.80] vol=1.7x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-10-09 09:55:00 | 623.98 | 623.67 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 618.90 | 620.90 | 0.00 | ORB-short ORB[619.60,623.60] vol=2.1x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-10-14 09:35:00 | 620.25 | 620.26 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 625.80 | 623.28 | 0.00 | ORB-long ORB[617.75,624.00] vol=2.3x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 624.21 | 624.28 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 617.90 | 620.57 | 0.00 | ORB-short ORB[621.50,628.60] vol=3.3x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 619.44 | 619.82 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 613.70 | 617.36 | 0.00 | ORB-short ORB[615.80,623.85] vol=1.9x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:40:00 | 610.88 | 615.87 | 0.00 | T1 1.5R @ 610.88 |
| Stop hit — per-position SL triggered | 2024-10-21 10:05:00 | 613.70 | 613.14 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:30:00 | 597.30 | 600.01 | 0.00 | ORB-short ORB[598.50,604.25] vol=2.0x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:40:00 | 593.18 | 598.41 | 0.00 | T1 1.5R @ 593.18 |
| Stop hit — per-position SL triggered | 2024-10-28 09:50:00 | 597.30 | 596.93 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 640.00 | 637.03 | 0.00 | ORB-long ORB[633.05,638.75] vol=2.1x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-11-07 09:40:00 | 637.94 | 637.21 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 636.10 | 632.40 | 0.00 | ORB-long ORB[626.00,634.00] vol=1.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-11-12 10:00:00 | 633.84 | 634.60 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 604.50 | 610.15 | 0.00 | ORB-short ORB[611.00,619.35] vol=1.6x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 608.17 | 610.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 619.65 | 617.27 | 0.00 | ORB-long ORB[613.20,618.75] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:45:00 | 622.90 | 618.51 | 0.00 | T1 1.5R @ 622.90 |
| Target hit | 2024-11-19 10:45:00 | 620.60 | 620.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2024-11-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:55:00 | 626.50 | 623.53 | 0.00 | ORB-long ORB[619.70,625.35] vol=1.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-11-27 11:10:00 | 625.27 | 623.93 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:00:00 | 637.90 | 633.56 | 0.00 | ORB-long ORB[627.00,634.40] vol=6.4x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 636.12 | 635.54 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 641.00 | 638.17 | 0.00 | ORB-long ORB[634.10,640.00] vol=2.9x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:35:00 | 643.40 | 640.15 | 0.00 | T1 1.5R @ 643.40 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 641.00 | 640.62 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 635.05 | 637.47 | 0.00 | ORB-short ORB[637.30,639.90] vol=2.5x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-12-06 10:10:00 | 637.00 | 637.55 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:55:00 | 632.15 | 636.74 | 0.00 | ORB-short ORB[636.20,643.90] vol=2.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-12-09 12:05:00 | 633.67 | 635.56 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 630.95 | 634.03 | 0.00 | ORB-short ORB[634.20,638.05] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 632.01 | 633.95 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 634.35 | 629.99 | 0.00 | ORB-long ORB[624.40,632.00] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-12-16 09:35:00 | 632.52 | 630.67 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 618.80 | 620.78 | 0.00 | ORB-short ORB[620.10,627.45] vol=1.8x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:00:00 | 616.67 | 619.72 | 0.00 | T1 1.5R @ 616.67 |
| Target hit | 2024-12-17 15:20:00 | 604.30 | 611.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:15:00 | 596.85 | 593.63 | 0.00 | ORB-long ORB[589.50,594.75] vol=2.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:30:00 | 599.74 | 594.26 | 0.00 | T1 1.5R @ 599.74 |
| Stop hit — per-position SL triggered | 2024-12-20 12:25:00 | 596.85 | 596.19 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 590.95 | 588.18 | 0.00 | ORB-long ORB[583.35,590.00] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-12-24 10:55:00 | 589.13 | 589.44 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 583.65 | 585.41 | 0.00 | ORB-short ORB[583.75,589.85] vol=1.7x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 585.27 | 584.63 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:20:00 | 602.20 | 595.50 | 0.00 | ORB-long ORB[588.85,594.60] vol=2.1x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-12-27 10:30:00 | 600.50 | 596.96 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:35:00 | 602.95 | 599.33 | 0.00 | ORB-long ORB[595.65,600.00] vol=2.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-01-01 09:40:00 | 601.19 | 600.02 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:40:00 | 602.05 | 604.16 | 0.00 | ORB-short ORB[604.00,611.65] vol=7.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:50:00 | 599.16 | 602.93 | 0.00 | T1 1.5R @ 599.16 |
| Target hit | 2025-01-06 15:20:00 | 591.65 | 596.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-01-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:35:00 | 584.95 | 589.18 | 0.00 | ORB-short ORB[592.25,598.25] vol=3.7x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-01-07 10:45:00 | 587.19 | 588.44 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 580.20 | 582.20 | 0.00 | ORB-short ORB[580.95,587.40] vol=2.6x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:00:00 | 577.25 | 581.16 | 0.00 | T1 1.5R @ 577.25 |
| Stop hit — per-position SL triggered | 2025-01-09 10:25:00 | 580.20 | 580.79 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 11:00:00 | 557.55 | 562.91 | 0.00 | ORB-short ORB[560.55,565.40] vol=2.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-01-17 11:05:00 | 558.96 | 562.71 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 562.15 | 568.75 | 0.00 | ORB-short ORB[568.15,573.90] vol=1.9x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 563.69 | 566.53 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 554.30 | 557.91 | 0.00 | ORB-short ORB[555.85,562.75] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-01-22 10:05:00 | 556.24 | 557.45 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:50:00 | 563.25 | 559.25 | 0.00 | ORB-long ORB[551.70,556.70] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-01-23 11:05:00 | 561.77 | 559.89 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 583.00 | 578.89 | 0.00 | ORB-long ORB[573.10,581.20] vol=1.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-01-29 09:50:00 | 580.70 | 579.74 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 09:55:00 | 576.90 | 582.20 | 0.00 | ORB-short ORB[579.75,586.50] vol=1.9x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:00:00 | 572.08 | 580.33 | 0.00 | T1 1.5R @ 572.08 |
| Stop hit — per-position SL triggered | 2025-01-31 10:20:00 | 576.90 | 579.19 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 572.20 | 576.36 | 0.00 | ORB-short ORB[574.25,581.00] vol=2.3x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-02-06 09:45:00 | 574.23 | 575.46 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:35:00 | 544.75 | 541.16 | 0.00 | ORB-long ORB[532.55,540.05] vol=2.1x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:00:00 | 547.40 | 542.32 | 0.00 | T1 1.5R @ 547.40 |
| Stop hit — per-position SL triggered | 2025-02-20 12:00:00 | 544.75 | 543.04 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:30:00 | 531.10 | 534.22 | 0.00 | ORB-short ORB[532.05,539.10] vol=1.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 11:50:00 | 528.66 | 532.92 | 0.00 | T1 1.5R @ 528.66 |
| Target hit | 2025-02-25 15:20:00 | 520.90 | 526.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:00:00 | 515.55 | 520.32 | 0.00 | ORB-short ORB[517.40,524.70] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 517.63 | 519.07 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 532.65 | 529.92 | 0.00 | ORB-long ORB[525.95,532.25] vol=2.0x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-03-06 10:00:00 | 530.32 | 530.83 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:10:00 | 535.05 | 531.45 | 0.00 | ORB-long ORB[526.35,533.75] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-03-07 11:45:00 | 533.72 | 532.02 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 528.70 | 524.60 | 0.00 | ORB-long ORB[518.00,525.85] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-03-11 11:25:00 | 526.61 | 525.63 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:05:00 | 533.05 | 528.02 | 0.00 | ORB-long ORB[522.25,528.50] vol=1.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-03-18 10:20:00 | 531.53 | 528.76 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:00:00 | 550.10 | 545.65 | 0.00 | ORB-long ORB[538.65,544.95] vol=4.3x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:20:00 | 552.31 | 547.65 | 0.00 | T1 1.5R @ 552.31 |
| Stop hit — per-position SL triggered | 2025-03-19 13:10:00 | 550.10 | 548.64 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:25:00 | 559.60 | 555.21 | 0.00 | ORB-long ORB[553.00,559.00] vol=2.8x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-03-20 10:45:00 | 557.65 | 556.11 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 563.60 | 560.46 | 0.00 | ORB-long ORB[556.10,562.65] vol=3.0x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 561.94 | 561.02 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 577.85 | 574.41 | 0.00 | ORB-long ORB[571.70,577.40] vol=3.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-03-24 11:10:00 | 576.09 | 574.91 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:10:00 | 565.00 | 571.25 | 0.00 | ORB-short ORB[571.00,577.80] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-03-28 11:35:00 | 566.64 | 570.61 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:20:00 | 564.55 | 561.41 | 0.00 | ORB-long ORB[555.35,562.95] vol=2.4x ATR=2.92 |
| Stop hit — per-position SL triggered | 2025-04-08 10:30:00 | 561.63 | 561.56 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 596.40 | 593.63 | 0.00 | ORB-long ORB[588.00,595.90] vol=1.7x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-04-16 10:00:00 | 594.40 | 594.06 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:55:00 | 614.15 | 610.55 | 0.00 | ORB-long ORB[605.00,612.50] vol=1.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:00:00 | 617.39 | 611.39 | 0.00 | T1 1.5R @ 617.39 |
| Stop hit — per-position SL triggered | 2025-04-21 10:10:00 | 614.15 | 611.87 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:10:00 | 613.70 | 610.37 | 0.00 | ORB-long ORB[603.00,610.70] vol=3.3x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-04-22 11:15:00 | 612.09 | 610.42 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:20:00 | 597.55 | 603.92 | 0.00 | ORB-short ORB[608.50,613.75] vol=2.3x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 599.69 | 602.27 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 601.85 | 597.98 | 0.00 | ORB-long ORB[593.50,600.75] vol=1.8x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-04-28 11:00:00 | 598.76 | 599.39 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 619.80 | 617.43 | 0.00 | ORB-long ORB[612.00,619.40] vol=1.9x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:35:00 | 623.16 | 620.45 | 0.00 | T1 1.5R @ 623.16 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 619.80 | 620.32 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:30:00 | 607.35 | 605.14 | 0.00 | ORB-long ORB[597.85,605.45] vol=2.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-04-30 13:20:00 | 605.33 | 606.06 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 10:50:00 | 594.90 | 598.82 | 0.00 | ORB-short ORB[599.25,605.45] vol=2.1x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:35:00 | 592.40 | 597.43 | 0.00 | T1 1.5R @ 592.40 |
| Target hit | 2025-05-08 15:20:00 | 575.65 | 586.84 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:00:00 | 619.35 | 2024-05-14 10:10:00 | 614.65 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-05-14 10:00:00 | 619.35 | 2024-05-14 10:20:00 | 619.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:45:00 | 648.40 | 2024-05-22 09:50:00 | 644.91 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-05-22 09:45:00 | 648.40 | 2024-05-22 09:55:00 | 648.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:40:00 | 654.95 | 2024-05-23 12:50:00 | 651.85 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-23 10:40:00 | 654.95 | 2024-05-23 15:00:00 | 654.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 09:35:00 | 649.85 | 2024-05-29 11:55:00 | 653.22 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-29 09:35:00 | 649.85 | 2024-05-29 14:05:00 | 649.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 10:45:00 | 629.50 | 2024-05-31 11:15:00 | 631.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-10 10:50:00 | 671.55 | 2024-06-10 11:00:00 | 675.55 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-10 10:50:00 | 671.55 | 2024-06-10 11:50:00 | 671.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:40:00 | 710.35 | 2024-06-13 09:55:00 | 712.93 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-18 10:35:00 | 753.05 | 2024-06-18 10:45:00 | 758.06 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-06-18 10:35:00 | 753.05 | 2024-06-18 11:15:00 | 753.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 10:05:00 | 733.15 | 2024-06-24 11:45:00 | 730.42 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-25 11:05:00 | 724.75 | 2024-06-25 11:20:00 | 726.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-27 09:45:00 | 785.20 | 2024-06-27 10:05:00 | 781.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-04 10:25:00 | 789.60 | 2024-07-04 10:35:00 | 786.33 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-04 10:25:00 | 789.60 | 2024-07-04 10:55:00 | 789.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:30:00 | 786.20 | 2024-07-05 09:40:00 | 788.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-11 09:55:00 | 793.45 | 2024-07-11 10:15:00 | 790.36 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-16 10:50:00 | 819.00 | 2024-07-16 11:00:00 | 824.57 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-07-16 10:50:00 | 819.00 | 2024-07-16 11:15:00 | 819.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:05:00 | 775.20 | 2024-07-24 10:10:00 | 782.03 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2024-07-24 10:05:00 | 775.20 | 2024-07-24 15:20:00 | 782.55 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2024-07-26 11:10:00 | 778.40 | 2024-07-26 11:20:00 | 782.14 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-26 11:10:00 | 778.40 | 2024-07-26 11:35:00 | 778.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:10:00 | 650.35 | 2024-08-08 10:20:00 | 647.49 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-08-14 09:35:00 | 633.45 | 2024-08-14 09:45:00 | 636.63 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-08-16 09:35:00 | 654.00 | 2024-08-16 09:40:00 | 651.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-23 10:25:00 | 671.55 | 2024-08-23 10:35:00 | 672.98 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-29 09:45:00 | 680.85 | 2024-08-29 10:00:00 | 679.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-30 10:55:00 | 677.00 | 2024-08-30 11:00:00 | 679.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-02 09:30:00 | 672.30 | 2024-09-02 10:05:00 | 674.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-03 10:15:00 | 686.45 | 2024-09-03 10:20:00 | 684.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-10 11:05:00 | 688.95 | 2024-09-10 11:40:00 | 685.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-10 11:05:00 | 688.95 | 2024-09-10 12:05:00 | 688.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:35:00 | 692.65 | 2024-09-11 10:55:00 | 689.75 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-13 10:00:00 | 714.85 | 2024-09-13 10:05:00 | 719.00 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-13 10:00:00 | 714.85 | 2024-09-13 13:20:00 | 719.90 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-09-17 09:30:00 | 671.70 | 2024-09-17 09:35:00 | 674.22 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-19 09:30:00 | 677.80 | 2024-09-19 10:00:00 | 674.71 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-19 09:30:00 | 677.80 | 2024-09-19 15:00:00 | 661.80 | TARGET_HIT | 0.50 | 2.36% |
| BUY | retest1 | 2024-09-23 11:10:00 | 674.90 | 2024-09-23 12:00:00 | 678.16 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-23 11:10:00 | 674.90 | 2024-09-23 15:20:00 | 680.75 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-09-30 09:35:00 | 660.50 | 2024-09-30 09:55:00 | 662.18 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-03 09:30:00 | 648.75 | 2024-10-03 09:35:00 | 650.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-09 09:45:00 | 625.85 | 2024-10-09 09:55:00 | 623.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-14 09:30:00 | 618.90 | 2024-10-14 09:35:00 | 620.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-10-15 09:35:00 | 625.80 | 2024-10-15 09:50:00 | 624.21 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-17 09:55:00 | 617.90 | 2024-10-17 10:15:00 | 619.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-21 09:30:00 | 613.70 | 2024-10-21 09:40:00 | 610.88 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-21 09:30:00 | 613.70 | 2024-10-21 10:05:00 | 613.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-28 09:30:00 | 597.30 | 2024-10-28 09:40:00 | 593.18 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-10-28 09:30:00 | 597.30 | 2024-10-28 09:50:00 | 597.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-07 09:35:00 | 640.00 | 2024-11-07 09:40:00 | 637.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-12 09:30:00 | 636.10 | 2024-11-12 10:00:00 | 633.84 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-13 09:45:00 | 604.50 | 2024-11-13 09:50:00 | 608.17 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-11-19 09:30:00 | 619.65 | 2024-11-19 09:45:00 | 622.90 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-11-19 09:30:00 | 619.65 | 2024-11-19 10:45:00 | 620.60 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-11-27 10:55:00 | 626.50 | 2024-11-27 11:10:00 | 625.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-11-28 10:00:00 | 637.90 | 2024-11-28 10:35:00 | 636.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-04 09:30:00 | 641.00 | 2024-12-04 09:35:00 | 643.40 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-04 09:30:00 | 641.00 | 2024-12-04 09:45:00 | 641.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 10:05:00 | 635.05 | 2024-12-06 10:10:00 | 637.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-09 10:55:00 | 632.15 | 2024-12-09 12:05:00 | 633.67 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-12 09:45:00 | 630.95 | 2024-12-12 09:50:00 | 632.01 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-12-16 09:30:00 | 634.35 | 2024-12-16 09:35:00 | 632.52 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-17 09:45:00 | 618.80 | 2024-12-17 10:00:00 | 616.67 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-17 09:45:00 | 618.80 | 2024-12-17 15:20:00 | 604.30 | TARGET_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2024-12-20 10:15:00 | 596.85 | 2024-12-20 10:30:00 | 599.74 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-20 10:15:00 | 596.85 | 2024-12-20 12:25:00 | 596.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:50:00 | 590.95 | 2024-12-24 10:55:00 | 589.13 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-26 10:30:00 | 583.65 | 2024-12-26 11:00:00 | 585.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-27 10:20:00 | 602.20 | 2024-12-27 10:30:00 | 600.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-01 09:35:00 | 602.95 | 2025-01-01 09:40:00 | 601.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-06 10:40:00 | 602.05 | 2025-01-06 10:50:00 | 599.16 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-06 10:40:00 | 602.05 | 2025-01-06 15:20:00 | 591.65 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2025-01-07 10:35:00 | 584.95 | 2025-01-07 10:45:00 | 587.19 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-09 09:30:00 | 580.20 | 2025-01-09 10:00:00 | 577.25 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-09 09:30:00 | 580.20 | 2025-01-09 10:25:00 | 580.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 11:00:00 | 557.55 | 2025-01-17 11:05:00 | 558.96 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-21 10:20:00 | 562.15 | 2025-01-21 11:40:00 | 563.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-22 09:50:00 | 554.30 | 2025-01-22 10:05:00 | 556.24 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-23 10:50:00 | 563.25 | 2025-01-23 11:05:00 | 561.77 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-29 09:30:00 | 583.00 | 2025-01-29 09:50:00 | 580.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-31 09:55:00 | 576.90 | 2025-01-31 10:00:00 | 572.08 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2025-01-31 09:55:00 | 576.90 | 2025-01-31 10:20:00 | 576.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 09:30:00 | 572.20 | 2025-02-06 09:45:00 | 574.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-20 10:35:00 | 544.75 | 2025-02-20 11:00:00 | 547.40 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-20 10:35:00 | 544.75 | 2025-02-20 12:00:00 | 544.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-25 10:30:00 | 531.10 | 2025-02-25 11:50:00 | 528.66 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-25 10:30:00 | 531.10 | 2025-02-25 15:20:00 | 520.90 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2025-02-27 10:00:00 | 515.55 | 2025-02-27 10:15:00 | 517.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-06 09:35:00 | 532.65 | 2025-03-06 10:00:00 | 530.32 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-03-07 11:10:00 | 535.05 | 2025-03-07 11:45:00 | 533.72 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-11 11:00:00 | 528.70 | 2025-03-11 11:25:00 | 526.61 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-18 10:05:00 | 533.05 | 2025-03-18 10:20:00 | 531.53 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-19 11:00:00 | 550.10 | 2025-03-19 12:20:00 | 552.31 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-19 11:00:00 | 550.10 | 2025-03-19 13:10:00 | 550.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 10:25:00 | 559.60 | 2025-03-20 10:45:00 | 557.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-21 09:35:00 | 563.60 | 2025-03-21 09:50:00 | 561.94 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-24 10:40:00 | 577.85 | 2025-03-24 11:10:00 | 576.09 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-03-28 11:10:00 | 565.00 | 2025-03-28 11:35:00 | 566.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-08 10:20:00 | 564.55 | 2025-04-08 10:30:00 | 561.63 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-04-16 09:45:00 | 596.40 | 2025-04-16 10:00:00 | 594.40 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-21 09:55:00 | 614.15 | 2025-04-21 10:00:00 | 617.39 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-21 09:55:00 | 614.15 | 2025-04-21 10:10:00 | 614.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 11:10:00 | 613.70 | 2025-04-22 11:15:00 | 612.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-23 10:20:00 | 597.55 | 2025-04-23 11:00:00 | 599.69 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-28 09:30:00 | 601.85 | 2025-04-28 11:00:00 | 598.76 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-04-29 09:30:00 | 619.80 | 2025-04-29 09:35:00 | 623.16 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-29 09:30:00 | 619.80 | 2025-04-29 09:40:00 | 619.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 10:30:00 | 607.35 | 2025-04-30 13:20:00 | 605.33 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-08 10:50:00 | 594.90 | 2025-05-08 11:35:00 | 592.40 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-05-08 10:50:00 | 594.90 | 2025-05-08 15:20:00 | 575.65 | TARGET_HIT | 0.50 | 3.24% |
