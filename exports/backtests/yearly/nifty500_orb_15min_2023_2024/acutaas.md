# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-08-28 15:25:00 (42667 bars)
- **Last close:** 1403.00
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 16 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 68
- **Target hits / Stop hits / Partials:** 16 / 68 / 34
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 12.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 17 | 37.8% | 7 | 28 | 10 | 0.03% | 1.1% |
| BUY @ 2nd Alert (retest1) | 45 | 17 | 37.8% | 7 | 28 | 10 | 0.03% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 33 | 45.2% | 9 | 40 | 24 | 0.16% | 11.5% |
| SELL @ 2nd Alert (retest1) | 73 | 33 | 45.2% | 9 | 40 | 24 | 0.16% | 11.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 118 | 50 | 42.4% | 16 | 68 | 34 | 0.11% | 12.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 586.55 | 590.31 | 0.00 | ORB-short ORB[587.50,595.50] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-05-25 09:45:00 | 588.38 | 589.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:20:00 | 621.95 | 614.08 | 0.00 | ORB-long ORB[606.00,614.95] vol=5.5x ATR=3.85 |
| Stop hit — per-position SL triggered | 2023-05-26 10:30:00 | 618.10 | 615.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-06-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 10:30:00 | 604.23 | 609.14 | 0.00 | ORB-short ORB[606.67,615.00] vol=1.8x ATR=2.22 |
| Stop hit — per-position SL triggered | 2023-06-05 10:50:00 | 606.45 | 608.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 10:00:00 | 600.00 | 604.97 | 0.00 | ORB-short ORB[603.00,609.90] vol=2.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-06-08 10:10:00 | 602.55 | 604.37 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 10:35:00 | 628.80 | 622.15 | 0.00 | ORB-long ORB[618.63,626.03] vol=3.9x ATR=3.14 |
| Stop hit — per-position SL triggered | 2023-06-09 10:50:00 | 625.66 | 623.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 09:30:00 | 633.50 | 637.58 | 0.00 | ORB-short ORB[636.50,641.00] vol=2.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-06-14 09:35:00 | 635.49 | 637.26 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 11:00:00 | 645.58 | 651.05 | 0.00 | ORB-short ORB[650.08,654.50] vol=1.9x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 11:20:00 | 643.13 | 650.31 | 0.00 | T1 1.5R @ 643.13 |
| Stop hit — per-position SL triggered | 2023-06-15 11:25:00 | 645.58 | 649.90 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:35:00 | 658.93 | 653.37 | 0.00 | ORB-long ORB[647.45,652.48] vol=5.4x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-06-16 09:40:00 | 656.36 | 655.96 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:25:00 | 636.78 | 644.28 | 0.00 | ORB-short ORB[641.80,651.00] vol=2.5x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:35:00 | 631.63 | 642.47 | 0.00 | T1 1.5R @ 631.63 |
| Stop hit — per-position SL triggered | 2023-06-20 11:40:00 | 636.78 | 640.54 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:35:00 | 642.48 | 641.43 | 0.00 | ORB-long ORB[636.88,640.48] vol=11.2x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 10:15:00 | 646.14 | 641.80 | 0.00 | T1 1.5R @ 646.14 |
| Target hit | 2023-06-22 11:30:00 | 643.90 | 645.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2023-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:45:00 | 639.00 | 636.00 | 0.00 | ORB-long ORB[630.03,638.17] vol=3.7x ATR=2.52 |
| Stop hit — per-position SL triggered | 2023-06-26 09:55:00 | 636.48 | 636.15 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:40:00 | 637.48 | 640.84 | 0.00 | ORB-short ORB[641.00,647.95] vol=7.8x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-06-30 11:40:00 | 639.84 | 640.24 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 639.03 | 640.73 | 0.00 | ORB-short ORB[639.30,643.00] vol=3.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 10:00:00 | 636.86 | 639.62 | 0.00 | T1 1.5R @ 636.86 |
| Stop hit — per-position SL triggered | 2023-07-04 10:25:00 | 639.03 | 638.91 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 11:05:00 | 621.40 | 617.67 | 0.00 | ORB-long ORB[610.50,617.98] vol=1.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-07-12 11:10:00 | 619.57 | 617.80 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:45:00 | 594.00 | 592.19 | 0.00 | ORB-long ORB[585.67,592.88] vol=1.6x ATR=2.76 |
| Stop hit — per-position SL triggered | 2023-07-17 12:50:00 | 591.24 | 594.05 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 593.50 | 591.83 | 0.00 | ORB-long ORB[586.28,592.42] vol=2.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-07-18 09:40:00 | 591.14 | 591.84 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:55:00 | 580.05 | 582.13 | 0.00 | ORB-short ORB[580.67,587.20] vol=2.5x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:05:00 | 576.83 | 581.42 | 0.00 | T1 1.5R @ 576.83 |
| Stop hit — per-position SL triggered | 2023-07-19 10:25:00 | 580.05 | 580.85 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:45:00 | 579.58 | 585.95 | 0.00 | ORB-short ORB[584.40,591.92] vol=1.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 12:00:00 | 575.86 | 579.76 | 0.00 | T1 1.5R @ 575.86 |
| Stop hit — per-position SL triggered | 2023-07-20 13:30:00 | 579.58 | 577.68 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:00:00 | 581.88 | 578.29 | 0.00 | ORB-long ORB[573.50,579.50] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2023-07-25 10:25:00 | 579.74 | 579.31 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:05:00 | 566.80 | 568.68 | 0.00 | ORB-short ORB[567.05,571.90] vol=1.6x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:10:00 | 564.20 | 568.34 | 0.00 | T1 1.5R @ 564.20 |
| Stop hit — per-position SL triggered | 2023-07-28 11:40:00 | 566.80 | 567.44 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 09:35:00 | 563.08 | 566.23 | 0.00 | ORB-short ORB[564.00,570.00] vol=4.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-07-31 09:40:00 | 565.10 | 566.00 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:30:00 | 575.50 | 573.53 | 0.00 | ORB-long ORB[570.30,575.00] vol=6.7x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-08-01 09:40:00 | 573.64 | 573.59 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 11:00:00 | 666.50 | 660.30 | 0.00 | ORB-long ORB[655.78,664.90] vol=4.2x ATR=2.07 |
| Stop hit — per-position SL triggered | 2023-08-22 11:05:00 | 664.43 | 660.61 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 10:35:00 | 675.65 | 671.92 | 0.00 | ORB-long ORB[666.78,674.50] vol=3.7x ATR=2.44 |
| Stop hit — per-position SL triggered | 2023-08-28 11:20:00 | 673.21 | 672.37 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 669.23 | 667.35 | 0.00 | ORB-long ORB[662.18,668.53] vol=3.3x ATR=2.53 |
| Stop hit — per-position SL triggered | 2023-09-04 09:40:00 | 666.70 | 667.74 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 665.23 | 668.12 | 0.00 | ORB-short ORB[667.53,671.85] vol=1.5x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 09:55:00 | 662.36 | 666.37 | 0.00 | T1 1.5R @ 662.36 |
| Target hit | 2023-09-05 15:20:00 | 659.75 | 661.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2023-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 09:30:00 | 658.58 | 661.10 | 0.00 | ORB-short ORB[659.50,664.53] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 09:50:00 | 656.24 | 659.89 | 0.00 | T1 1.5R @ 656.24 |
| Stop hit — per-position SL triggered | 2023-09-06 10:20:00 | 658.58 | 659.34 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:55:00 | 666.28 | 660.66 | 0.00 | ORB-long ORB[655.00,659.85] vol=1.6x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:00:00 | 669.70 | 662.49 | 0.00 | T1 1.5R @ 669.70 |
| Stop hit — per-position SL triggered | 2023-09-07 10:05:00 | 666.28 | 662.81 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:30:00 | 657.15 | 661.32 | 0.00 | ORB-short ORB[660.40,665.45] vol=3.7x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-09-08 09:45:00 | 659.64 | 659.65 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:40:00 | 642.53 | 645.21 | 0.00 | ORB-short ORB[645.00,651.85] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2023-09-14 10:55:00 | 645.57 | 644.97 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 10:50:00 | 642.20 | 649.60 | 0.00 | ORB-short ORB[648.00,656.28] vol=2.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:55:00 | 638.73 | 647.68 | 0.00 | T1 1.5R @ 638.73 |
| Stop hit — per-position SL triggered | 2023-09-20 11:25:00 | 642.20 | 646.86 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:30:00 | 633.50 | 636.04 | 0.00 | ORB-short ORB[633.80,642.48] vol=1.6x ATR=2.77 |
| Stop hit — per-position SL triggered | 2023-09-22 10:40:00 | 636.27 | 636.03 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:25:00 | 634.98 | 636.62 | 0.00 | ORB-short ORB[636.30,644.00] vol=5.9x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-09-25 11:45:00 | 637.18 | 636.37 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:20:00 | 628.55 | 629.95 | 0.00 | ORB-short ORB[629.03,636.70] vol=3.1x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 10:30:00 | 626.32 | 628.17 | 0.00 | T1 1.5R @ 626.32 |
| Target hit | 2023-09-26 11:25:00 | 627.50 | 626.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — BUY (started 2023-09-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:45:00 | 633.42 | 631.66 | 0.00 | ORB-long ORB[625.05,633.40] vol=1.7x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-09-28 10:20:00 | 631.22 | 632.19 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:50:00 | 592.50 | 593.55 | 0.00 | ORB-short ORB[592.98,597.95] vol=3.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-10-06 11:10:00 | 593.64 | 593.41 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:40:00 | 617.38 | 611.19 | 0.00 | ORB-long ORB[602.98,610.98] vol=8.0x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 09:45:00 | 622.10 | 613.81 | 0.00 | T1 1.5R @ 622.10 |
| Target hit | 2023-10-11 15:00:00 | 622.92 | 623.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 620.60 | 622.09 | 0.00 | ORB-short ORB[620.90,626.48] vol=5.1x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:55:00 | 618.77 | 621.72 | 0.00 | T1 1.5R @ 618.77 |
| Target hit | 2023-10-18 15:20:00 | 607.80 | 613.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-10-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 11:05:00 | 603.38 | 605.39 | 0.00 | ORB-short ORB[604.53,607.50] vol=5.1x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 11:10:00 | 601.48 | 605.12 | 0.00 | T1 1.5R @ 601.48 |
| Stop hit — per-position SL triggered | 2023-10-19 11:15:00 | 603.38 | 605.12 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:35:00 | 594.00 | 588.86 | 0.00 | ORB-long ORB[582.28,589.70] vol=7.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-10-31 09:40:00 | 591.58 | 589.67 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 10:50:00 | 589.17 | 585.04 | 0.00 | ORB-long ORB[579.00,587.38] vol=1.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-11-01 11:10:00 | 587.08 | 585.46 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:45:00 | 545.65 | 540.19 | 0.00 | ORB-long ORB[535.45,540.00] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-11-16 10:00:00 | 543.72 | 541.35 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:30:00 | 551.00 | 547.19 | 0.00 | ORB-long ORB[542.50,547.40] vol=2.4x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 09:40:00 | 554.08 | 549.70 | 0.00 | T1 1.5R @ 554.08 |
| Target hit | 2023-11-17 10:15:00 | 555.23 | 555.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2023-11-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:30:00 | 554.00 | 548.79 | 0.00 | ORB-long ORB[544.25,549.98] vol=2.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2023-11-23 09:35:00 | 551.16 | 549.84 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 11:00:00 | 554.75 | 557.81 | 0.00 | ORB-short ORB[555.40,563.48] vol=4.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 12:30:00 | 552.25 | 556.54 | 0.00 | T1 1.5R @ 552.25 |
| Target hit | 2023-12-01 15:20:00 | 551.15 | 554.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2023-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:55:00 | 558.00 | 556.68 | 0.00 | ORB-long ORB[553.42,557.45] vol=1.9x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:00:00 | 559.39 | 556.97 | 0.00 | T1 1.5R @ 559.39 |
| Stop hit — per-position SL triggered | 2023-12-05 11:05:00 | 558.00 | 556.99 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:50:00 | 550.73 | 550.24 | 0.00 | ORB-long ORB[545.17,548.38] vol=2.3x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:05:00 | 553.77 | 550.54 | 0.00 | T1 1.5R @ 553.77 |
| Target hit | 2023-12-06 13:20:00 | 555.78 | 556.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2023-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 09:45:00 | 550.00 | 553.18 | 0.00 | ORB-short ORB[555.00,558.25] vol=9.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 10:15:00 | 547.11 | 551.75 | 0.00 | T1 1.5R @ 547.11 |
| Target hit | 2023-12-11 15:20:00 | 539.50 | 544.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2023-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:35:00 | 547.25 | 544.73 | 0.00 | ORB-long ORB[542.80,546.00] vol=2.5x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-12-12 09:40:00 | 545.31 | 544.83 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 10:05:00 | 548.65 | 552.98 | 0.00 | ORB-short ORB[549.00,556.00] vol=1.8x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-12-14 10:30:00 | 550.73 | 551.92 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:15:00 | 570.00 | 574.06 | 0.00 | ORB-short ORB[571.15,576.38] vol=3.2x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:30:00 | 566.20 | 572.97 | 0.00 | T1 1.5R @ 566.20 |
| Target hit | 2024-01-02 13:30:00 | 569.15 | 569.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:15:00 | 584.73 | 581.78 | 0.00 | ORB-long ORB[578.03,584.45] vol=1.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-01-04 11:40:00 | 582.36 | 583.37 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:40:00 | 577.83 | 581.13 | 0.00 | ORB-short ORB[580.65,586.00] vol=2.2x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:55:00 | 575.58 | 580.57 | 0.00 | T1 1.5R @ 575.58 |
| Stop hit — per-position SL triggered | 2024-01-05 11:10:00 | 577.83 | 580.27 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:50:00 | 574.88 | 578.11 | 0.00 | ORB-short ORB[576.00,581.95] vol=2.0x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-01-15 10:15:00 | 577.84 | 577.43 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-01-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 11:00:00 | 568.98 | 574.29 | 0.00 | ORB-short ORB[573.13,579.88] vol=2.4x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-01-16 11:45:00 | 570.66 | 572.69 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:40:00 | 557.00 | 561.27 | 0.00 | ORB-short ORB[561.80,566.95] vol=2.4x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 553.40 | 559.60 | 0.00 | T1 1.5R @ 553.40 |
| Stop hit — per-position SL triggered | 2024-01-18 10:20:00 | 557.00 | 555.08 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:45:00 | 565.83 | 568.85 | 0.00 | ORB-short ORB[568.25,571.30] vol=1.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-01-20 10:35:00 | 567.61 | 567.83 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 556.40 | 558.78 | 0.00 | ORB-short ORB[556.58,563.98] vol=4.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-01-23 10:10:00 | 558.50 | 558.58 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 540.15 | 543.15 | 0.00 | ORB-short ORB[542.03,549.98] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-01-25 11:05:00 | 541.64 | 543.08 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-02-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:20:00 | 573.30 | 578.68 | 0.00 | ORB-short ORB[578.05,583.38] vol=2.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-02-01 10:40:00 | 575.04 | 577.87 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:30:00 | 571.98 | 567.79 | 0.00 | ORB-long ORB[563.50,568.92] vol=2.2x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:40:00 | 575.49 | 569.45 | 0.00 | T1 1.5R @ 575.49 |
| Target hit | 2024-02-02 11:50:00 | 573.75 | 573.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — SELL (started 2024-02-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:40:00 | 564.55 | 567.63 | 0.00 | ORB-short ORB[565.65,573.58] vol=1.8x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-02-05 11:30:00 | 566.94 | 567.10 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:15:00 | 559.80 | 563.72 | 0.00 | ORB-short ORB[561.67,568.98] vol=3.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 12:00:00 | 557.09 | 562.43 | 0.00 | T1 1.5R @ 557.09 |
| Target hit | 2024-02-07 15:20:00 | 552.25 | 558.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-02-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:35:00 | 549.25 | 550.63 | 0.00 | ORB-short ORB[549.33,553.53] vol=1.5x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 10:20:00 | 545.65 | 549.41 | 0.00 | T1 1.5R @ 545.65 |
| Stop hit — per-position SL triggered | 2024-02-15 11:30:00 | 549.25 | 546.39 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:55:00 | 547.35 | 549.50 | 0.00 | ORB-short ORB[549.45,554.83] vol=1.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-02-16 11:25:00 | 548.88 | 549.36 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 10:45:00 | 557.05 | 552.80 | 0.00 | ORB-long ORB[550.00,554.95] vol=2.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 10:50:00 | 559.63 | 553.85 | 0.00 | T1 1.5R @ 559.63 |
| Stop hit — per-position SL triggered | 2024-02-19 11:00:00 | 557.05 | 554.33 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:35:00 | 555.67 | 558.19 | 0.00 | ORB-short ORB[557.42,562.45] vol=2.5x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 11:00:00 | 552.18 | 556.15 | 0.00 | T1 1.5R @ 552.18 |
| Target hit | 2024-02-20 11:30:00 | 554.45 | 554.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — BUY (started 2024-02-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:10:00 | 556.50 | 554.90 | 0.00 | ORB-long ORB[551.50,556.08] vol=2.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-02-21 10:20:00 | 554.83 | 555.02 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:30:00 | 546.30 | 544.55 | 0.00 | ORB-long ORB[540.00,545.70] vol=1.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-02-23 09:55:00 | 544.35 | 544.70 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:35:00 | 556.03 | 561.03 | 0.00 | ORB-short ORB[558.13,565.35] vol=1.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-02-29 12:25:00 | 558.56 | 559.53 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 09:40:00 | 571.78 | 567.08 | 0.00 | ORB-long ORB[562.50,567.50] vol=6.6x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:50:00 | 576.90 | 569.61 | 0.00 | T1 1.5R @ 576.90 |
| Target hit | 2024-03-04 10:50:00 | 580.55 | 581.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2024-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 09:30:00 | 545.00 | 541.76 | 0.00 | ORB-long ORB[536.10,543.23] vol=2.8x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 09:35:00 | 549.35 | 545.25 | 0.00 | T1 1.5R @ 549.35 |
| Target hit | 2024-03-18 09:50:00 | 546.63 | 547.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2024-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 09:30:00 | 563.95 | 556.59 | 0.00 | ORB-long ORB[548.60,555.00] vol=4.2x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-03-19 09:35:00 | 561.25 | 559.11 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-03-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 10:30:00 | 563.35 | 564.83 | 0.00 | ORB-short ORB[563.48,568.80] vol=1.8x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 11:35:00 | 560.22 | 564.17 | 0.00 | T1 1.5R @ 560.22 |
| Target hit | 2024-03-21 15:20:00 | 557.50 | 558.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 09:30:00 | 560.85 | 562.09 | 0.00 | ORB-short ORB[561.00,567.50] vol=1.6x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 09:35:00 | 557.83 | 560.10 | 0.00 | T1 1.5R @ 557.83 |
| Stop hit — per-position SL triggered | 2024-03-26 09:45:00 | 560.85 | 560.01 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-03-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:40:00 | 557.28 | 560.94 | 0.00 | ORB-short ORB[561.05,565.00] vol=1.9x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-03-27 14:20:00 | 559.28 | 558.84 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:00:00 | 569.42 | 563.79 | 0.00 | ORB-long ORB[557.00,563.70] vol=3.1x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-04-02 10:05:00 | 567.08 | 564.02 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-04-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:40:00 | 625.00 | 633.58 | 0.00 | ORB-short ORB[635.00,643.42] vol=1.9x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 10:55:00 | 620.57 | 630.58 | 0.00 | T1 1.5R @ 620.57 |
| Stop hit — per-position SL triggered | 2024-04-12 11:00:00 | 625.00 | 630.26 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:10:00 | 597.03 | 592.78 | 0.00 | ORB-long ORB[588.00,595.17] vol=2.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 594.74 | 592.91 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:10:00 | 619.38 | 615.63 | 0.00 | ORB-long ORB[612.00,618.63] vol=1.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-04-30 10:25:00 | 617.44 | 616.38 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-05-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:05:00 | 605.58 | 610.57 | 0.00 | ORB-short ORB[607.63,613.48] vol=2.1x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-05-03 10:15:00 | 607.95 | 610.24 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:40:00 | 591.67 | 595.55 | 0.00 | ORB-short ORB[592.50,599.13] vol=2.5x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:50:00 | 587.60 | 592.85 | 0.00 | T1 1.5R @ 587.60 |
| Stop hit — per-position SL triggered | 2024-05-06 10:00:00 | 591.67 | 592.40 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-05-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:35:00 | 601.00 | 603.27 | 0.00 | ORB-short ORB[602.03,608.30] vol=1.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-05-09 09:40:00 | 603.14 | 603.14 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-05-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 09:35:00 | 586.10 | 592.45 | 0.00 | ORB-short ORB[591.42,599.25] vol=2.4x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-05-10 09:40:00 | 589.74 | 592.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-25 09:30:00 | 586.55 | 2023-05-25 09:45:00 | 588.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-05-26 10:20:00 | 621.95 | 2023-05-26 10:30:00 | 618.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2023-06-05 10:30:00 | 604.23 | 2023-06-05 10:50:00 | 606.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-06-08 10:00:00 | 600.00 | 2023-06-08 10:10:00 | 602.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-06-09 10:35:00 | 628.80 | 2023-06-09 10:50:00 | 625.66 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-06-14 09:30:00 | 633.50 | 2023-06-14 09:35:00 | 635.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-06-15 11:00:00 | 645.58 | 2023-06-15 11:20:00 | 643.13 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-15 11:00:00 | 645.58 | 2023-06-15 11:25:00 | 645.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 09:35:00 | 658.93 | 2023-06-16 09:40:00 | 656.36 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-06-20 10:25:00 | 636.78 | 2023-06-20 10:35:00 | 631.63 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2023-06-20 10:25:00 | 636.78 | 2023-06-20 11:40:00 | 636.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-22 09:35:00 | 642.48 | 2023-06-22 10:15:00 | 646.14 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-06-22 09:35:00 | 642.48 | 2023-06-22 11:30:00 | 643.90 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2023-06-26 09:45:00 | 639.00 | 2023-06-26 09:55:00 | 636.48 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-06-30 10:40:00 | 637.48 | 2023-06-30 11:40:00 | 639.84 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-07-04 09:40:00 | 639.03 | 2023-07-04 10:00:00 | 636.86 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-07-04 09:40:00 | 639.03 | 2023-07-04 10:25:00 | 639.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-12 11:05:00 | 621.40 | 2023-07-12 11:10:00 | 619.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-17 09:45:00 | 594.00 | 2023-07-17 12:50:00 | 591.24 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-07-18 09:35:00 | 593.50 | 2023-07-18 09:40:00 | 591.14 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-07-19 09:55:00 | 580.05 | 2023-07-19 10:05:00 | 576.83 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-07-19 09:55:00 | 580.05 | 2023-07-19 10:25:00 | 580.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-20 09:45:00 | 579.58 | 2023-07-20 12:00:00 | 575.86 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2023-07-20 09:45:00 | 579.58 | 2023-07-20 13:30:00 | 579.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-25 10:00:00 | 581.88 | 2023-07-25 10:25:00 | 579.74 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-07-28 10:05:00 | 566.80 | 2023-07-28 10:10:00 | 564.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-07-28 10:05:00 | 566.80 | 2023-07-28 11:40:00 | 566.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-31 09:35:00 | 563.08 | 2023-07-31 09:40:00 | 565.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-01 09:30:00 | 575.50 | 2023-08-01 09:40:00 | 573.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-22 11:00:00 | 666.50 | 2023-08-22 11:05:00 | 664.43 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-28 10:35:00 | 675.65 | 2023-08-28 11:20:00 | 673.21 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-09-04 09:30:00 | 669.23 | 2023-09-04 09:40:00 | 666.70 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-05 09:40:00 | 665.23 | 2023-09-05 09:55:00 | 662.36 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-09-05 09:40:00 | 665.23 | 2023-09-05 15:20:00 | 659.75 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2023-09-06 09:30:00 | 658.58 | 2023-09-06 09:50:00 | 656.24 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-09-06 09:30:00 | 658.58 | 2023-09-06 10:20:00 | 658.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 09:55:00 | 666.28 | 2023-09-07 10:00:00 | 669.70 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-09-07 09:55:00 | 666.28 | 2023-09-07 10:05:00 | 666.28 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 09:30:00 | 657.15 | 2023-09-08 09:45:00 | 659.64 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-14 10:40:00 | 642.53 | 2023-09-14 10:55:00 | 645.57 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-09-20 10:50:00 | 642.20 | 2023-09-20 10:55:00 | 638.73 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-09-20 10:50:00 | 642.20 | 2023-09-20 11:25:00 | 642.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-22 10:30:00 | 633.50 | 2023-09-22 10:40:00 | 636.27 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-09-25 10:25:00 | 634.98 | 2023-09-25 11:45:00 | 637.18 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-09-26 10:20:00 | 628.55 | 2023-09-26 10:30:00 | 626.32 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-26 10:20:00 | 628.55 | 2023-09-26 11:25:00 | 627.50 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2023-09-28 09:45:00 | 633.42 | 2023-09-28 10:20:00 | 631.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-10-06 10:50:00 | 592.50 | 2023-10-06 11:10:00 | 593.64 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-11 09:40:00 | 617.38 | 2023-10-11 09:45:00 | 622.10 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2023-10-11 09:40:00 | 617.38 | 2023-10-11 15:00:00 | 622.92 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2023-10-18 10:45:00 | 620.60 | 2023-10-18 10:55:00 | 618.77 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-10-18 10:45:00 | 620.60 | 2023-10-18 15:20:00 | 607.80 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2023-10-19 11:05:00 | 603.38 | 2023-10-19 11:10:00 | 601.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-10-19 11:05:00 | 603.38 | 2023-10-19 11:15:00 | 603.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 09:35:00 | 594.00 | 2023-10-31 09:40:00 | 591.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-11-01 10:50:00 | 589.17 | 2023-11-01 11:10:00 | 587.08 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-16 09:45:00 | 545.65 | 2023-11-16 10:00:00 | 543.72 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-17 09:30:00 | 551.00 | 2023-11-17 09:40:00 | 554.08 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-11-17 09:30:00 | 551.00 | 2023-11-17 10:15:00 | 555.23 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2023-11-23 09:30:00 | 554.00 | 2023-11-23 09:35:00 | 551.16 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2023-12-01 11:00:00 | 554.75 | 2023-12-01 12:30:00 | 552.25 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-12-01 11:00:00 | 554.75 | 2023-12-01 15:20:00 | 551.15 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2023-12-05 10:55:00 | 558.00 | 2023-12-05 11:00:00 | 559.39 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-12-05 10:55:00 | 558.00 | 2023-12-05 11:05:00 | 558.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-06 10:50:00 | 550.73 | 2023-12-06 11:05:00 | 553.77 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-12-06 10:50:00 | 550.73 | 2023-12-06 13:20:00 | 555.78 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2023-12-11 09:45:00 | 550.00 | 2023-12-11 10:15:00 | 547.11 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-12-11 09:45:00 | 550.00 | 2023-12-11 15:20:00 | 539.50 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2023-12-12 09:35:00 | 547.25 | 2023-12-12 09:40:00 | 545.31 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-12-14 10:05:00 | 548.65 | 2023-12-14 10:30:00 | 550.73 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-01-02 10:15:00 | 570.00 | 2024-01-02 10:30:00 | 566.20 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-01-02 10:15:00 | 570.00 | 2024-01-02 13:30:00 | 569.15 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-01-04 10:15:00 | 584.73 | 2024-01-04 11:40:00 | 582.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-01-05 10:40:00 | 577.83 | 2024-01-05 10:55:00 | 575.58 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-01-05 10:40:00 | 577.83 | 2024-01-05 11:10:00 | 577.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-15 09:50:00 | 574.88 | 2024-01-15 10:15:00 | 577.84 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-01-16 11:00:00 | 568.98 | 2024-01-16 11:45:00 | 570.66 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-18 09:40:00 | 557.00 | 2024-01-18 09:45:00 | 553.40 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-01-18 09:40:00 | 557.00 | 2024-01-18 10:20:00 | 557.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 09:45:00 | 565.83 | 2024-01-20 10:35:00 | 567.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-23 09:55:00 | 556.40 | 2024-01-23 10:10:00 | 558.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-01-25 10:55:00 | 540.15 | 2024-01-25 11:05:00 | 541.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-01 10:20:00 | 573.30 | 2024-02-01 10:40:00 | 575.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-02 09:30:00 | 571.98 | 2024-02-02 09:40:00 | 575.49 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-02-02 09:30:00 | 571.98 | 2024-02-02 11:50:00 | 573.75 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-02-05 10:40:00 | 564.55 | 2024-02-05 11:30:00 | 566.94 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-02-07 11:15:00 | 559.80 | 2024-02-07 12:00:00 | 557.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-07 11:15:00 | 559.80 | 2024-02-07 15:20:00 | 552.25 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2024-02-15 09:35:00 | 549.25 | 2024-02-15 10:20:00 | 545.65 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-02-15 09:35:00 | 549.25 | 2024-02-15 11:30:00 | 549.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-16 10:55:00 | 547.35 | 2024-02-16 11:25:00 | 548.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-02-19 10:45:00 | 557.05 | 2024-02-19 10:50:00 | 559.63 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-02-19 10:45:00 | 557.05 | 2024-02-19 11:00:00 | 557.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-20 09:35:00 | 555.67 | 2024-02-20 11:00:00 | 552.18 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-02-20 09:35:00 | 555.67 | 2024-02-20 11:30:00 | 554.45 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-02-21 10:10:00 | 556.50 | 2024-02-21 10:20:00 | 554.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-23 09:30:00 | 546.30 | 2024-02-23 09:55:00 | 544.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-29 10:35:00 | 556.03 | 2024-02-29 12:25:00 | 558.56 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-03-04 09:40:00 | 571.78 | 2024-03-04 09:50:00 | 576.90 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-03-04 09:40:00 | 571.78 | 2024-03-04 10:50:00 | 580.55 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2024-03-18 09:30:00 | 545.00 | 2024-03-18 09:35:00 | 549.35 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-03-18 09:30:00 | 545.00 | 2024-03-18 09:50:00 | 546.63 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-03-19 09:30:00 | 563.95 | 2024-03-19 09:35:00 | 561.25 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-03-21 10:30:00 | 563.35 | 2024-03-21 11:35:00 | 560.22 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-03-21 10:30:00 | 563.35 | 2024-03-21 15:20:00 | 557.50 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2024-03-26 09:30:00 | 560.85 | 2024-03-26 09:35:00 | 557.83 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-03-26 09:30:00 | 560.85 | 2024-03-26 09:45:00 | 560.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-27 10:40:00 | 557.28 | 2024-03-27 14:20:00 | 559.28 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-04-02 10:00:00 | 569.42 | 2024-04-02 10:05:00 | 567.08 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-04-12 10:40:00 | 625.00 | 2024-04-12 10:55:00 | 620.57 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-04-12 10:40:00 | 625.00 | 2024-04-12 11:00:00 | 625.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 10:10:00 | 597.03 | 2024-04-23 10:15:00 | 594.74 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-04-30 10:10:00 | 619.38 | 2024-04-30 10:25:00 | 617.44 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-03 10:05:00 | 605.58 | 2024-05-03 10:15:00 | 607.95 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-06 09:40:00 | 591.67 | 2024-05-06 09:50:00 | 587.60 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-05-06 09:40:00 | 591.67 | 2024-05-06 10:00:00 | 591.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 09:35:00 | 601.00 | 2024-05-09 09:40:00 | 603.14 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-10 09:35:00 | 586.10 | 2024-05-10 09:40:00 | 589.74 | STOP_HIT | 1.00 | -0.62% |
