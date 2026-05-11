# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 459.50
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
| ENTRY1 | 94 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 13 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 81
- **Target hits / Stop hits / Partials:** 13 / 81 / 36
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 11.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 28 | 38.9% | 6 | 44 | 22 | 0.11% | 7.9% |
| BUY @ 2nd Alert (retest1) | 72 | 28 | 38.9% | 6 | 44 | 22 | 0.11% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 21 | 36.2% | 7 | 37 | 14 | 0.07% | 3.9% |
| SELL @ 2nd Alert (retest1) | 58 | 21 | 36.2% | 7 | 37 | 14 | 0.07% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 49 | 37.7% | 13 | 81 | 36 | 0.09% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 11:00:00 | 642.80 | 639.72 | 0.00 | ORB-long ORB[633.30,641.00] vol=4.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-05-16 11:05:00 | 641.47 | 640.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 648.35 | 649.96 | 0.00 | ORB-short ORB[650.05,652.45] vol=2.0x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 13:25:00 | 646.87 | 648.96 | 0.00 | T1 1.5R @ 646.87 |
| Target hit | 2025-05-29 15:05:00 | 648.15 | 647.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2025-06-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:05:00 | 645.30 | 648.38 | 0.00 | ORB-short ORB[645.40,654.40] vol=2.2x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 643.65 | 648.10 | 0.00 | T1 1.5R @ 643.65 |
| Target hit | 2025-06-03 15:20:00 | 639.25 | 644.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:55:00 | 643.20 | 645.80 | 0.00 | ORB-short ORB[645.90,649.55] vol=4.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-06-06 11:30:00 | 644.38 | 645.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:10:00 | 651.65 | 649.56 | 0.00 | ORB-long ORB[646.05,650.90] vol=2.9x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:20:00 | 653.63 | 650.04 | 0.00 | T1 1.5R @ 653.63 |
| Stop hit — per-position SL triggered | 2025-06-09 10:40:00 | 651.65 | 650.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:10:00 | 664.00 | 661.53 | 0.00 | ORB-long ORB[657.10,663.00] vol=2.9x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 10:25:00 | 666.65 | 662.57 | 0.00 | T1 1.5R @ 666.65 |
| Stop hit — per-position SL triggered | 2025-06-10 10:50:00 | 664.00 | 662.87 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 11:10:00 | 662.90 | 659.47 | 0.00 | ORB-long ORB[657.35,660.45] vol=2.4x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-06-11 11:25:00 | 661.65 | 659.68 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:40:00 | 668.00 | 664.32 | 0.00 | ORB-long ORB[662.75,667.60] vol=1.8x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:50:00 | 670.28 | 664.94 | 0.00 | T1 1.5R @ 670.28 |
| Stop hit — per-position SL triggered | 2025-06-12 10:55:00 | 668.00 | 667.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:00:00 | 640.50 | 642.82 | 0.00 | ORB-short ORB[642.55,648.00] vol=1.6x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:30:00 | 638.70 | 642.48 | 0.00 | T1 1.5R @ 638.70 |
| Target hit | 2025-06-19 15:20:00 | 625.65 | 633.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:15:00 | 651.60 | 648.12 | 0.00 | ORB-long ORB[643.60,649.60] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-06-27 10:35:00 | 649.63 | 648.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 640.10 | 644.12 | 0.00 | ORB-short ORB[645.85,649.45] vol=2.1x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 12:35:00 | 637.99 | 642.54 | 0.00 | T1 1.5R @ 637.99 |
| Target hit | 2025-06-30 15:20:00 | 639.10 | 639.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 635.10 | 637.79 | 0.00 | ORB-short ORB[638.90,643.70] vol=2.0x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-07-01 10:25:00 | 636.45 | 637.55 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 10:50:00 | 635.25 | 638.89 | 0.00 | ORB-short ORB[637.05,642.45] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-07-03 10:55:00 | 636.65 | 638.80 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:30:00 | 634.20 | 635.31 | 0.00 | ORB-short ORB[634.45,637.00] vol=2.2x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:35:00 | 632.43 | 634.45 | 0.00 | T1 1.5R @ 632.43 |
| Target hit | 2025-07-08 12:05:00 | 631.80 | 631.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2025-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:50:00 | 630.60 | 634.47 | 0.00 | ORB-short ORB[636.10,641.00] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-07-10 11:05:00 | 631.70 | 634.12 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 629.90 | 625.39 | 0.00 | ORB-long ORB[620.70,627.70] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-07-14 10:05:00 | 628.04 | 627.00 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:50:00 | 655.50 | 651.84 | 0.00 | ORB-long ORB[642.20,647.95] vol=2.6x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:55:00 | 659.51 | 654.10 | 0.00 | T1 1.5R @ 659.51 |
| Stop hit — per-position SL triggered | 2025-07-15 12:10:00 | 655.50 | 654.23 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:50:00 | 669.50 | 664.54 | 0.00 | ORB-long ORB[655.35,665.35] vol=1.8x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-07-16 10:10:00 | 667.09 | 666.09 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:05:00 | 661.50 | 663.84 | 0.00 | ORB-short ORB[665.95,672.10] vol=11.2x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:10:00 | 659.17 | 663.70 | 0.00 | T1 1.5R @ 659.17 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 661.50 | 663.68 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:55:00 | 664.75 | 659.26 | 0.00 | ORB-long ORB[654.50,659.10] vol=3.3x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-07-21 11:05:00 | 663.19 | 659.94 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:50:00 | 698.75 | 692.87 | 0.00 | ORB-long ORB[678.15,688.50] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-07-28 11:00:00 | 696.31 | 693.32 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:35:00 | 700.55 | 704.59 | 0.00 | ORB-short ORB[702.10,709.50] vol=2.2x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 09:50:00 | 697.92 | 704.00 | 0.00 | T1 1.5R @ 697.92 |
| Stop hit — per-position SL triggered | 2025-08-05 10:40:00 | 700.55 | 700.56 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:35:00 | 666.25 | 672.08 | 0.00 | ORB-short ORB[670.05,676.65] vol=2.4x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-08-07 11:30:00 | 668.50 | 670.52 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:35:00 | 646.45 | 650.43 | 0.00 | ORB-short ORB[650.00,655.55] vol=2.3x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:45:00 | 643.31 | 649.02 | 0.00 | T1 1.5R @ 643.31 |
| Stop hit — per-position SL triggered | 2025-08-11 09:55:00 | 646.45 | 647.49 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:05:00 | 640.35 | 642.35 | 0.00 | ORB-short ORB[640.85,645.65] vol=4.3x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-08-12 10:35:00 | 642.18 | 641.40 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:30:00 | 677.85 | 675.06 | 0.00 | ORB-long ORB[668.55,673.60] vol=1.9x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 11:00:00 | 680.30 | 676.23 | 0.00 | T1 1.5R @ 680.30 |
| Stop hit — per-position SL triggered | 2025-08-18 11:10:00 | 677.85 | 676.25 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:40:00 | 674.50 | 668.32 | 0.00 | ORB-long ORB[662.30,668.90] vol=2.0x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-08-19 11:10:00 | 672.61 | 668.80 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:45:00 | 674.35 | 669.08 | 0.00 | ORB-long ORB[666.80,672.95] vol=1.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:05:00 | 676.48 | 670.76 | 0.00 | T1 1.5R @ 676.48 |
| Stop hit — per-position SL triggered | 2025-08-21 12:20:00 | 674.35 | 671.39 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 10:10:00 | 640.75 | 641.85 | 0.00 | ORB-short ORB[642.30,651.40] vol=7.1x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 12:20:00 | 638.07 | 641.36 | 0.00 | T1 1.5R @ 638.07 |
| Target hit | 2025-08-28 15:20:00 | 630.00 | 637.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-09-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:40:00 | 630.65 | 629.67 | 0.00 | ORB-long ORB[622.10,629.10] vol=8.9x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:05:00 | 632.92 | 630.02 | 0.00 | T1 1.5R @ 632.92 |
| Stop hit — per-position SL triggered | 2025-09-01 11:55:00 | 630.65 | 630.63 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 640.40 | 637.81 | 0.00 | ORB-long ORB[634.10,638.95] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-09-03 09:35:00 | 638.96 | 638.00 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:50:00 | 657.45 | 654.53 | 0.00 | ORB-long ORB[649.85,656.40] vol=2.7x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:00:00 | 660.45 | 655.73 | 0.00 | T1 1.5R @ 660.45 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 657.45 | 656.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:05:00 | 652.50 | 649.12 | 0.00 | ORB-long ORB[642.35,651.80] vol=2.3x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:15:00 | 655.34 | 650.04 | 0.00 | T1 1.5R @ 655.34 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 652.50 | 650.43 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:00:00 | 659.40 | 656.73 | 0.00 | ORB-long ORB[651.75,658.05] vol=1.6x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-09-11 10:10:00 | 657.85 | 657.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:10:00 | 660.30 | 662.27 | 0.00 | ORB-short ORB[661.10,665.60] vol=3.3x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-09-12 10:30:00 | 661.63 | 662.12 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 10:00:00 | 667.65 | 668.13 | 0.00 | ORB-short ORB[669.50,673.90] vol=7.9x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 12:05:00 | 665.29 | 667.83 | 0.00 | T1 1.5R @ 665.29 |
| Target hit | 2025-09-16 15:20:00 | 657.90 | 659.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:15:00 | 653.10 | 657.21 | 0.00 | ORB-short ORB[655.45,661.85] vol=8.2x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-09-17 11:30:00 | 654.63 | 657.16 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:35:00 | 662.00 | 659.78 | 0.00 | ORB-long ORB[656.30,661.40] vol=2.0x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 660.32 | 661.06 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:45:00 | 647.05 | 652.22 | 0.00 | ORB-short ORB[654.60,659.10] vol=1.5x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:10:00 | 644.97 | 650.26 | 0.00 | T1 1.5R @ 644.97 |
| Stop hit — per-position SL triggered | 2025-09-23 11:50:00 | 647.05 | 649.37 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:15:00 | 636.35 | 638.37 | 0.00 | ORB-short ORB[638.35,642.15] vol=2.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-09-25 11:20:00 | 637.49 | 638.48 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:40:00 | 617.00 | 616.91 | 0.00 | ORB-long ORB[607.05,614.80] vol=2.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-09-29 12:05:00 | 615.29 | 616.92 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:35:00 | 621.50 | 620.22 | 0.00 | ORB-long ORB[613.35,620.95] vol=2.1x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-09-30 11:50:00 | 619.73 | 620.80 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:40:00 | 639.00 | 636.04 | 0.00 | ORB-long ORB[630.05,635.70] vol=1.9x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-03 09:45:00 | 637.34 | 636.17 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 640.25 | 636.61 | 0.00 | ORB-long ORB[632.25,636.90] vol=1.8x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-10-07 09:35:00 | 638.22 | 637.23 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 630.55 | 631.98 | 0.00 | ORB-short ORB[630.75,637.70] vol=3.2x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 632.39 | 631.77 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:55:00 | 624.50 | 629.42 | 0.00 | ORB-short ORB[628.50,635.30] vol=1.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:05:00 | 622.16 | 629.26 | 0.00 | T1 1.5R @ 622.16 |
| Stop hit — per-position SL triggered | 2025-10-14 13:10:00 | 624.50 | 626.76 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:15:00 | 633.95 | 633.39 | 0.00 | ORB-long ORB[628.10,633.65] vol=1.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:25:00 | 635.97 | 634.05 | 0.00 | T1 1.5R @ 635.97 |
| Target hit | 2025-10-16 14:55:00 | 640.00 | 640.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 652.35 | 649.32 | 0.00 | ORB-long ORB[642.00,651.35] vol=2.3x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:35:00 | 655.20 | 651.56 | 0.00 | T1 1.5R @ 655.20 |
| Stop hit — per-position SL triggered | 2025-10-23 10:50:00 | 652.35 | 651.75 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 662.25 | 659.77 | 0.00 | ORB-long ORB[657.05,661.70] vol=5.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-10-29 11:05:00 | 661.12 | 659.94 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 654.50 | 655.94 | 0.00 | ORB-short ORB[654.80,660.05] vol=3.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-10-30 09:45:00 | 655.69 | 655.05 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 649.45 | 652.79 | 0.00 | ORB-short ORB[650.65,656.50] vol=1.7x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 651.38 | 652.74 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:10:00 | 657.85 | 656.77 | 0.00 | ORB-long ORB[652.35,656.75] vol=1.7x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:20:00 | 660.20 | 657.17 | 0.00 | T1 1.5R @ 660.20 |
| Stop hit — per-position SL triggered | 2025-11-03 10:30:00 | 657.85 | 657.45 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 617.75 | 617.03 | 0.00 | ORB-long ORB[608.60,617.50] vol=9.0x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 614.89 | 617.70 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 619.00 | 613.86 | 0.00 | ORB-long ORB[609.40,615.55] vol=2.0x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:40:00 | 622.38 | 615.50 | 0.00 | T1 1.5R @ 622.38 |
| Target hit | 2025-11-10 15:20:00 | 630.30 | 624.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:40:00 | 632.25 | 633.40 | 0.00 | ORB-short ORB[632.30,636.20] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-11-20 09:45:00 | 633.64 | 633.36 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 10:40:00 | 635.00 | 632.70 | 0.00 | ORB-long ORB[630.05,634.50] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:55:00 | 637.10 | 633.50 | 0.00 | T1 1.5R @ 637.10 |
| Stop hit — per-position SL triggered | 2025-11-21 14:10:00 | 635.00 | 635.66 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 638.60 | 635.53 | 0.00 | ORB-long ORB[630.25,637.35] vol=1.7x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:00:00 | 641.70 | 638.73 | 0.00 | T1 1.5R @ 641.70 |
| Target hit | 2025-11-24 11:15:00 | 640.10 | 640.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2025-11-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:40:00 | 641.00 | 644.96 | 0.00 | ORB-short ORB[643.95,650.70] vol=2.2x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-11-27 10:45:00 | 642.48 | 644.69 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 642.85 | 639.79 | 0.00 | ORB-long ORB[637.05,641.60] vol=1.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 641.34 | 640.34 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:55:00 | 639.80 | 643.59 | 0.00 | ORB-short ORB[644.50,652.10] vol=2.1x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 641.18 | 643.43 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:55:00 | 632.85 | 637.96 | 0.00 | ORB-short ORB[635.00,643.80] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 634.43 | 637.65 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:35:00 | 643.70 | 639.31 | 0.00 | ORB-long ORB[631.00,639.00] vol=2.1x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-12-04 11:45:00 | 641.92 | 640.28 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:25:00 | 634.50 | 635.35 | 0.00 | ORB-short ORB[634.90,641.75] vol=5.2x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 636.49 | 635.20 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:30:00 | 633.25 | 634.37 | 0.00 | ORB-short ORB[635.00,639.80] vol=5.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-12-08 10:55:00 | 634.86 | 634.31 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 633.50 | 631.42 | 0.00 | ORB-long ORB[627.20,630.90] vol=2.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-12-10 10:55:00 | 632.11 | 631.65 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 634.75 | 630.03 | 0.00 | ORB-long ORB[625.15,630.70] vol=2.1x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:25:00 | 636.97 | 631.09 | 0.00 | T1 1.5R @ 636.97 |
| Stop hit — per-position SL triggered | 2025-12-11 11:30:00 | 634.75 | 631.23 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 648.80 | 645.50 | 0.00 | ORB-long ORB[639.90,647.55] vol=3.2x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-12-12 09:40:00 | 647.02 | 645.57 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:55:00 | 654.30 | 651.72 | 0.00 | ORB-long ORB[646.95,653.85] vol=2.1x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 11:35:00 | 656.45 | 653.07 | 0.00 | T1 1.5R @ 656.45 |
| Target hit | 2025-12-15 15:20:00 | 660.80 | 655.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 652.20 | 654.58 | 0.00 | ORB-short ORB[653.10,661.20] vol=2.4x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-12-16 11:05:00 | 653.38 | 654.50 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:10:00 | 649.20 | 645.98 | 0.00 | ORB-long ORB[641.25,648.90] vol=1.8x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:30:00 | 651.82 | 646.23 | 0.00 | T1 1.5R @ 651.82 |
| Target hit | 2025-12-18 15:20:00 | 655.80 | 651.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2025-12-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:50:00 | 651.50 | 659.20 | 0.00 | ORB-short ORB[658.00,665.30] vol=2.9x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:45:00 | 648.33 | 656.32 | 0.00 | T1 1.5R @ 648.33 |
| Stop hit — per-position SL triggered | 2025-12-19 14:50:00 | 651.50 | 652.95 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:35:00 | 658.60 | 656.18 | 0.00 | ORB-long ORB[651.65,657.90] vol=2.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-12-29 09:45:00 | 657.07 | 656.64 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 09:40:00 | 644.35 | 645.24 | 0.00 | ORB-short ORB[647.75,652.15] vol=3.2x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-01-01 11:20:00 | 646.12 | 644.93 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 648.00 | 649.20 | 0.00 | ORB-short ORB[648.45,652.40] vol=1.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-01-02 09:45:00 | 649.36 | 648.95 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:20:00 | 662.35 | 660.55 | 0.00 | ORB-long ORB[655.80,661.05] vol=2.1x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:50:00 | 664.43 | 661.54 | 0.00 | T1 1.5R @ 664.43 |
| Stop hit — per-position SL triggered | 2026-01-05 11:00:00 | 662.35 | 661.83 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:55:00 | 658.50 | 655.57 | 0.00 | ORB-long ORB[652.50,655.75] vol=2.5x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-01-06 10:05:00 | 656.81 | 655.86 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 624.55 | 628.72 | 0.00 | ORB-short ORB[627.90,636.35] vol=1.7x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-01-13 09:50:00 | 626.82 | 628.07 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:15:00 | 628.05 | 629.71 | 0.00 | ORB-short ORB[628.45,633.05] vol=1.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-01-14 10:25:00 | 629.99 | 629.67 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:10:00 | 633.30 | 630.96 | 0.00 | ORB-long ORB[625.00,631.85] vol=1.6x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 12:15:00 | 635.65 | 632.63 | 0.00 | T1 1.5R @ 635.65 |
| Stop hit — per-position SL triggered | 2026-01-16 12:25:00 | 633.30 | 632.68 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-01-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:40:00 | 596.60 | 603.47 | 0.00 | ORB-short ORB[602.20,610.55] vol=1.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-01-21 10:45:00 | 599.35 | 603.13 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:30:00 | 469.80 | 475.21 | 0.00 | ORB-short ORB[477.35,483.85] vol=2.1x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-01-29 11:05:00 | 471.57 | 474.69 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 478.00 | 472.64 | 0.00 | ORB-long ORB[467.70,474.30] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-01-30 10:05:00 | 475.77 | 475.06 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:50:00 | 479.05 | 474.54 | 0.00 | ORB-long ORB[471.55,475.95] vol=3.4x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-01 11:00:00 | 477.63 | 475.86 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 461.40 | 456.88 | 0.00 | ORB-long ORB[454.05,459.00] vol=3.4x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-02-10 11:05:00 | 460.05 | 456.99 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 431.50 | 432.57 | 0.00 | ORB-short ORB[433.00,436.65] vol=7.4x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:15:00 | 429.80 | 431.97 | 0.00 | T1 1.5R @ 429.80 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 431.50 | 431.80 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 437.20 | 435.12 | 0.00 | ORB-long ORB[431.75,436.50] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 435.31 | 435.70 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 439.80 | 442.34 | 0.00 | ORB-short ORB[442.20,446.90] vol=1.5x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 441.15 | 442.25 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 436.15 | 437.74 | 0.00 | ORB-short ORB[437.00,439.85] vol=3.1x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-25 12:20:00 | 437.43 | 437.15 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 439.20 | 437.41 | 0.00 | ORB-long ORB[434.00,437.70] vol=2.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 437.72 | 437.69 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:45:00 | 398.55 | 400.56 | 0.00 | ORB-short ORB[401.00,402.75] vol=1.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-03-05 10:05:00 | 399.66 | 399.85 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 409.30 | 413.50 | 0.00 | ORB-short ORB[413.75,419.80] vol=2.4x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-03-19 11:45:00 | 410.56 | 413.00 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:15:00 | 402.75 | 407.83 | 0.00 | ORB-short ORB[408.75,414.70] vol=2.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-03-23 11:20:00 | 404.15 | 407.71 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 409.85 | 408.17 | 0.00 | ORB-long ORB[405.00,409.30] vol=2.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:10:00 | 412.09 | 409.41 | 0.00 | T1 1.5R @ 412.09 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 409.85 | 409.45 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 431.10 | 426.60 | 0.00 | ORB-long ORB[421.25,427.50] vol=2.5x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 433.76 | 428.05 | 0.00 | T1 1.5R @ 433.76 |
| Target hit | 2026-04-27 15:20:00 | 439.20 | 435.22 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 11:00:00 | 642.80 | 2025-05-16 11:05:00 | 641.47 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-29 11:00:00 | 648.35 | 2025-05-29 13:25:00 | 646.87 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-05-29 11:00:00 | 648.35 | 2025-05-29 15:05:00 | 648.15 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2025-06-03 11:05:00 | 645.30 | 2025-06-03 11:15:00 | 643.65 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-03 11:05:00 | 645.30 | 2025-06-03 15:20:00 | 639.25 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-06-06 10:55:00 | 643.20 | 2025-06-06 11:30:00 | 644.38 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-09 10:10:00 | 651.65 | 2025-06-09 10:20:00 | 653.63 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-09 10:10:00 | 651.65 | 2025-06-09 10:40:00 | 651.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:10:00 | 664.00 | 2025-06-10 10:25:00 | 666.65 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-06-10 10:10:00 | 664.00 | 2025-06-10 10:50:00 | 664.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 11:10:00 | 662.90 | 2025-06-11 11:25:00 | 661.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-12 10:40:00 | 668.00 | 2025-06-12 10:50:00 | 670.28 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-06-12 10:40:00 | 668.00 | 2025-06-12 10:55:00 | 668.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 11:00:00 | 640.50 | 2025-06-19 11:30:00 | 638.70 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-19 11:00:00 | 640.50 | 2025-06-19 15:20:00 | 625.65 | TARGET_HIT | 0.50 | 2.32% |
| BUY | retest1 | 2025-06-27 10:15:00 | 651.60 | 2025-06-27 10:35:00 | 649.63 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-30 11:15:00 | 640.10 | 2025-06-30 12:35:00 | 637.99 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-30 11:15:00 | 640.10 | 2025-06-30 15:20:00 | 639.10 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-07-01 10:15:00 | 635.10 | 2025-07-01 10:25:00 | 636.45 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-03 10:50:00 | 635.25 | 2025-07-03 10:55:00 | 636.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-08 10:30:00 | 634.20 | 2025-07-08 10:35:00 | 632.43 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-08 10:30:00 | 634.20 | 2025-07-08 12:05:00 | 631.80 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-10 10:50:00 | 630.60 | 2025-07-10 11:05:00 | 631.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-14 09:40:00 | 629.90 | 2025-07-14 10:05:00 | 628.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-15 09:50:00 | 655.50 | 2025-07-15 11:55:00 | 659.51 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-07-15 09:50:00 | 655.50 | 2025-07-15 12:10:00 | 655.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:50:00 | 669.50 | 2025-07-16 10:10:00 | 667.09 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-18 11:05:00 | 661.50 | 2025-07-18 11:10:00 | 659.17 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-18 11:05:00 | 661.50 | 2025-07-18 11:15:00 | 661.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 10:55:00 | 664.75 | 2025-07-21 11:05:00 | 663.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-28 10:50:00 | 698.75 | 2025-07-28 11:00:00 | 696.31 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-05 09:35:00 | 700.55 | 2025-08-05 09:50:00 | 697.92 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-05 09:35:00 | 700.55 | 2025-08-05 10:40:00 | 700.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 10:35:00 | 666.25 | 2025-08-07 11:30:00 | 668.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-11 09:35:00 | 646.45 | 2025-08-11 09:45:00 | 643.31 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-08-11 09:35:00 | 646.45 | 2025-08-11 09:55:00 | 646.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-12 10:05:00 | 640.35 | 2025-08-12 10:35:00 | 642.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-18 10:30:00 | 677.85 | 2025-08-18 11:00:00 | 680.30 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-08-18 10:30:00 | 677.85 | 2025-08-18 11:10:00 | 677.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:40:00 | 674.50 | 2025-08-19 11:10:00 | 672.61 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-21 10:45:00 | 674.35 | 2025-08-21 12:05:00 | 676.48 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-21 10:45:00 | 674.35 | 2025-08-21 12:20:00 | 674.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-28 10:10:00 | 640.75 | 2025-08-28 12:20:00 | 638.07 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-28 10:10:00 | 640.75 | 2025-08-28 15:20:00 | 630.00 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-09-01 10:40:00 | 630.65 | 2025-09-01 11:05:00 | 632.92 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-01 10:40:00 | 630.65 | 2025-09-01 11:55:00 | 630.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:30:00 | 640.40 | 2025-09-03 09:35:00 | 638.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-05 09:50:00 | 657.45 | 2025-09-05 10:00:00 | 660.45 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-09-05 09:50:00 | 657.45 | 2025-09-05 10:10:00 | 657.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 10:05:00 | 652.50 | 2025-09-08 10:15:00 | 655.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-08 10:05:00 | 652.50 | 2025-09-08 10:30:00 | 652.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 10:00:00 | 659.40 | 2025-09-11 10:10:00 | 657.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-12 10:10:00 | 660.30 | 2025-09-12 10:30:00 | 661.63 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-16 10:00:00 | 667.65 | 2025-09-16 12:05:00 | 665.29 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-16 10:00:00 | 667.65 | 2025-09-16 15:20:00 | 657.90 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-09-17 11:15:00 | 653.10 | 2025-09-17 11:30:00 | 654.63 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-19 09:35:00 | 662.00 | 2025-09-19 10:00:00 | 660.32 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-23 10:45:00 | 647.05 | 2025-09-23 11:10:00 | 644.97 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-23 10:45:00 | 647.05 | 2025-09-23 11:50:00 | 647.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 11:15:00 | 636.35 | 2025-09-25 11:20:00 | 637.49 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-29 10:40:00 | 617.00 | 2025-09-29 12:05:00 | 615.29 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-30 10:35:00 | 621.50 | 2025-09-30 11:50:00 | 619.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-03 09:40:00 | 639.00 | 2025-10-03 09:45:00 | 637.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-07 09:30:00 | 640.25 | 2025-10-07 09:35:00 | 638.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-08 11:05:00 | 630.55 | 2025-10-08 11:15:00 | 632.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-14 10:55:00 | 624.50 | 2025-10-14 11:05:00 | 622.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-14 10:55:00 | 624.50 | 2025-10-14 13:10:00 | 624.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 10:15:00 | 633.95 | 2025-10-16 10:25:00 | 635.97 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-16 10:15:00 | 633.95 | 2025-10-16 14:55:00 | 640.00 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2025-10-23 09:40:00 | 652.35 | 2025-10-23 10:35:00 | 655.20 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-23 09:40:00 | 652.35 | 2025-10-23 10:50:00 | 652.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:55:00 | 662.25 | 2025-10-29 11:05:00 | 661.12 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-30 09:35:00 | 654.50 | 2025-10-30 09:45:00 | 655.69 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-31 09:50:00 | 649.45 | 2025-10-31 10:45:00 | 651.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-03 10:10:00 | 657.85 | 2025-11-03 10:20:00 | 660.20 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-03 10:10:00 | 657.85 | 2025-11-03 10:30:00 | 657.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 09:35:00 | 617.75 | 2025-11-07 10:55:00 | 614.89 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-11-10 09:30:00 | 619.00 | 2025-11-10 09:40:00 | 622.38 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-10 09:30:00 | 619.00 | 2025-11-10 15:20:00 | 630.30 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2025-11-20 09:40:00 | 632.25 | 2025-11-20 09:45:00 | 633.64 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-21 10:40:00 | 635.00 | 2025-11-21 10:55:00 | 637.10 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-21 10:40:00 | 635.00 | 2025-11-21 14:10:00 | 635.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 09:30:00 | 638.60 | 2025-11-24 10:00:00 | 641.70 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-11-24 09:30:00 | 638.60 | 2025-11-24 11:15:00 | 640.10 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-11-27 10:40:00 | 641.00 | 2025-11-27 10:45:00 | 642.48 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-28 09:30:00 | 642.85 | 2025-11-28 09:35:00 | 641.34 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-01 10:55:00 | 639.80 | 2025-12-01 11:15:00 | 641.18 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-03 10:55:00 | 632.85 | 2025-12-03 11:05:00 | 634.43 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-04 10:35:00 | 643.70 | 2025-12-04 11:45:00 | 641.92 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-05 10:25:00 | 634.50 | 2025-12-05 11:15:00 | 636.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-08 10:30:00 | 633.25 | 2025-12-08 10:55:00 | 634.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-10 10:45:00 | 633.50 | 2025-12-10 10:55:00 | 632.11 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-11 10:55:00 | 634.75 | 2025-12-11 11:25:00 | 636.97 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-11 10:55:00 | 634.75 | 2025-12-11 11:30:00 | 634.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 09:35:00 | 648.80 | 2025-12-12 09:40:00 | 647.02 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-15 10:55:00 | 654.30 | 2025-12-15 11:35:00 | 656.45 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-15 10:55:00 | 654.30 | 2025-12-15 15:20:00 | 660.80 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2025-12-16 11:00:00 | 652.20 | 2025-12-16 11:05:00 | 653.38 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-18 10:10:00 | 649.20 | 2025-12-18 10:30:00 | 651.82 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-18 10:10:00 | 649.20 | 2025-12-18 15:20:00 | 655.80 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2025-12-19 10:50:00 | 651.50 | 2025-12-19 11:45:00 | 648.33 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-19 10:50:00 | 651.50 | 2025-12-19 14:50:00 | 651.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-29 09:35:00 | 658.60 | 2025-12-29 09:45:00 | 657.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-01 09:40:00 | 644.35 | 2026-01-01 11:20:00 | 646.12 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-02 09:35:00 | 648.00 | 2026-01-02 09:45:00 | 649.36 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-05 10:20:00 | 662.35 | 2026-01-05 10:50:00 | 664.43 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-05 10:20:00 | 662.35 | 2026-01-05 11:00:00 | 662.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 09:55:00 | 658.50 | 2026-01-06 10:05:00 | 656.81 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-13 09:35:00 | 624.55 | 2026-01-13 09:50:00 | 626.82 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-14 10:15:00 | 628.05 | 2026-01-14 10:25:00 | 629.99 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-16 10:10:00 | 633.30 | 2026-01-16 12:15:00 | 635.65 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-16 10:10:00 | 633.30 | 2026-01-16 12:25:00 | 633.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 10:40:00 | 596.60 | 2026-01-21 10:45:00 | 599.35 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-01-29 10:30:00 | 469.80 | 2026-01-29 11:05:00 | 471.57 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-01-30 09:30:00 | 478.00 | 2026-01-30 10:05:00 | 475.77 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-01 10:50:00 | 479.05 | 2026-02-01 11:00:00 | 477.63 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-10 11:00:00 | 461.40 | 2026-02-10 11:05:00 | 460.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-18 09:55:00 | 431.50 | 2026-02-18 10:15:00 | 429.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-18 09:55:00 | 431.50 | 2026-02-18 10:35:00 | 431.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:30:00 | 437.20 | 2026-02-19 09:40:00 | 435.31 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-23 10:45:00 | 439.80 | 2026-02-23 10:55:00 | 441.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-25 10:40:00 | 436.15 | 2026-02-25 12:20:00 | 437.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-26 09:45:00 | 439.20 | 2026-02-26 10:10:00 | 437.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-05 09:45:00 | 398.55 | 2026-03-05 10:05:00 | 399.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-19 11:15:00 | 409.30 | 2026-03-19 11:45:00 | 410.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-23 11:15:00 | 402.75 | 2026-03-23 11:20:00 | 404.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-15 09:30:00 | 409.85 | 2026-04-15 10:10:00 | 412.09 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-15 09:30:00 | 409.85 | 2026-04-15 10:15:00 | 409.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 431.10 | 2026-04-27 09:50:00 | 433.76 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-27 09:40:00 | 431.10 | 2026-04-27 15:20:00 | 439.20 | TARGET_HIT | 0.50 | 1.88% |
