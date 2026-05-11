# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 632.00
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 11 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 42
- **Target hits / Stop hits / Partials:** 11 / 42 / 15
- **Avg / median % per leg:** 0.03% / -0.20%
- **Sum % (uncompounded):** 2.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 7 | 33.3% | 2 | 14 | 5 | -0.09% | -1.8% |
| BUY @ 2nd Alert (retest1) | 21 | 7 | 33.3% | 2 | 14 | 5 | -0.09% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 19 | 40.4% | 9 | 28 | 10 | 0.08% | 3.9% |
| SELL @ 2nd Alert (retest1) | 47 | 19 | 40.4% | 9 | 28 | 10 | 0.08% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 68 | 26 | 38.2% | 11 | 42 | 15 | 0.03% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:35:00 | 667.40 | 671.98 | 0.00 | ORB-short ORB[672.05,675.00] vol=1.5x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-06-09 09:40:00 | 669.13 | 671.54 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-06-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:50:00 | 663.40 | 667.18 | 0.00 | ORB-short ORB[666.00,673.00] vol=1.9x ATR=1.35 |
| Stop hit — per-position SL triggered | 2023-06-13 10:10:00 | 664.75 | 666.27 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-06-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:10:00 | 660.30 | 657.70 | 0.00 | ORB-long ORB[655.40,659.95] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2023-06-22 10:15:00 | 659.00 | 658.23 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:25:00 | 642.90 | 647.60 | 0.00 | ORB-short ORB[645.55,652.00] vol=1.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-06-27 10:30:00 | 644.49 | 647.25 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 652.05 | 655.40 | 0.00 | ORB-short ORB[653.30,662.00] vol=1.8x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-07-04 09:50:00 | 653.51 | 655.11 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 646.85 | 648.67 | 0.00 | ORB-short ORB[647.40,652.00] vol=2.1x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-07-05 10:10:00 | 648.36 | 648.28 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-07-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 10:35:00 | 655.90 | 649.72 | 0.00 | ORB-long ORB[643.30,652.85] vol=6.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-07-18 10:40:00 | 653.35 | 650.79 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-07-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:25:00 | 645.45 | 647.43 | 0.00 | ORB-short ORB[646.30,649.70] vol=1.5x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 12:40:00 | 643.65 | 646.37 | 0.00 | T1 1.5R @ 643.65 |
| Target hit | 2023-07-19 15:20:00 | 641.20 | 644.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2023-07-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 09:55:00 | 633.90 | 636.14 | 0.00 | ORB-short ORB[634.50,642.00] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-07-24 10:00:00 | 635.32 | 636.09 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:00:00 | 636.80 | 632.06 | 0.00 | ORB-long ORB[628.00,636.70] vol=2.3x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 10:10:00 | 640.05 | 633.48 | 0.00 | T1 1.5R @ 640.05 |
| Target hit | 2023-07-25 10:45:00 | 637.00 | 637.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 659.75 | 655.99 | 0.00 | ORB-long ORB[650.05,658.70] vol=1.8x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:10:00 | 663.96 | 657.67 | 0.00 | T1 1.5R @ 663.96 |
| Stop hit — per-position SL triggered | 2023-08-08 10:35:00 | 659.75 | 658.13 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-08-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:35:00 | 646.40 | 649.49 | 0.00 | ORB-short ORB[647.50,652.80] vol=2.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-08-10 10:50:00 | 648.05 | 649.33 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-08-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:55:00 | 633.85 | 636.40 | 0.00 | ORB-short ORB[635.55,639.90] vol=2.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-08-17 14:10:00 | 635.37 | 635.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:35:00 | 640.25 | 637.76 | 0.00 | ORB-long ORB[632.50,638.00] vol=3.9x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 09:45:00 | 642.62 | 639.42 | 0.00 | T1 1.5R @ 642.62 |
| Target hit | 2023-08-18 10:10:00 | 641.20 | 641.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2023-08-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 09:55:00 | 657.85 | 661.24 | 0.00 | ORB-short ORB[658.00,665.80] vol=1.8x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:10:00 | 654.50 | 659.81 | 0.00 | T1 1.5R @ 654.50 |
| Target hit | 2023-08-25 15:20:00 | 653.60 | 655.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 09:35:00 | 654.50 | 656.41 | 0.00 | ORB-short ORB[655.05,662.50] vol=1.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2023-08-30 11:45:00 | 656.28 | 654.97 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 634.80 | 636.95 | 0.00 | ORB-short ORB[635.00,640.90] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2023-09-04 09:50:00 | 636.70 | 636.55 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 633.50 | 635.61 | 0.00 | ORB-short ORB[633.60,639.45] vol=1.5x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-09-05 09:50:00 | 634.98 | 635.19 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-09-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:55:00 | 653.00 | 650.26 | 0.00 | ORB-long ORB[647.30,651.25] vol=2.0x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-09-08 10:10:00 | 651.14 | 650.53 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-09-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:40:00 | 646.75 | 643.06 | 0.00 | ORB-long ORB[638.00,646.65] vol=1.5x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-09-21 10:05:00 | 644.93 | 644.44 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-09-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:55:00 | 636.40 | 640.59 | 0.00 | ORB-short ORB[639.60,644.80] vol=1.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-09-22 10:05:00 | 638.05 | 640.27 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-09-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:00:00 | 630.50 | 632.31 | 0.00 | ORB-short ORB[630.60,635.90] vol=2.0x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-09-27 10:20:00 | 631.76 | 631.71 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-09-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-29 09:40:00 | 619.00 | 621.83 | 0.00 | ORB-short ORB[621.05,627.50] vol=2.1x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 09:50:00 | 616.89 | 620.13 | 0.00 | T1 1.5R @ 616.89 |
| Target hit | 2023-09-29 11:00:00 | 617.80 | 616.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2023-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 09:30:00 | 614.60 | 616.95 | 0.00 | ORB-short ORB[615.00,619.85] vol=1.5x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 09:40:00 | 611.80 | 615.18 | 0.00 | T1 1.5R @ 611.80 |
| Target hit | 2023-10-03 15:20:00 | 610.80 | 612.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2023-10-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 09:30:00 | 609.00 | 611.21 | 0.00 | ORB-short ORB[609.25,615.70] vol=2.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-10-04 09:45:00 | 610.56 | 610.89 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 612.45 | 613.75 | 0.00 | ORB-short ORB[613.85,616.90] vol=3.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2023-10-05 10:10:00 | 614.43 | 613.36 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 09:45:00 | 602.55 | 604.62 | 0.00 | ORB-short ORB[603.55,607.95] vol=2.4x ATR=1.38 |
| Stop hit — per-position SL triggered | 2023-10-13 09:55:00 | 603.93 | 604.46 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 10:00:00 | 596.00 | 598.95 | 0.00 | ORB-short ORB[598.35,604.50] vol=1.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-10-16 10:15:00 | 597.52 | 598.48 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:30:00 | 599.65 | 598.44 | 0.00 | ORB-long ORB[596.00,599.40] vol=3.3x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 09:50:00 | 601.26 | 599.13 | 0.00 | T1 1.5R @ 601.26 |
| Stop hit — per-position SL triggered | 2023-10-17 10:05:00 | 599.65 | 599.23 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 594.55 | 596.03 | 0.00 | ORB-short ORB[595.10,598.80] vol=2.0x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 595.56 | 596.01 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:05:00 | 593.35 | 595.48 | 0.00 | ORB-short ORB[593.60,599.80] vol=1.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 12:55:00 | 590.60 | 592.71 | 0.00 | T1 1.5R @ 590.60 |
| Stop hit — per-position SL triggered | 2023-10-20 15:10:00 | 593.35 | 592.12 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 10:40:00 | 552.80 | 554.85 | 0.00 | ORB-short ORB[554.80,559.80] vol=1.6x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 15:15:00 | 550.92 | 553.33 | 0.00 | T1 1.5R @ 550.92 |
| Target hit | 2023-11-06 15:20:00 | 550.00 | 553.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2023-11-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:10:00 | 539.30 | 540.75 | 0.00 | ORB-short ORB[539.85,544.20] vol=1.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 10:30:00 | 537.41 | 540.37 | 0.00 | T1 1.5R @ 537.41 |
| Target hit | 2023-11-09 15:20:00 | 534.85 | 537.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 540.50 | 542.79 | 0.00 | ORB-short ORB[542.15,549.80] vol=2.2x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-11-13 13:25:00 | 541.89 | 541.90 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:40:00 | 529.50 | 531.70 | 0.00 | ORB-short ORB[530.45,537.00] vol=1.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-11-20 09:55:00 | 530.64 | 531.32 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:55:00 | 529.55 | 531.23 | 0.00 | ORB-short ORB[530.40,534.00] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2023-11-24 11:15:00 | 530.61 | 530.68 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 1058.25 | 1040.78 | 0.00 | ORB-long ORB[1025.00,1039.80] vol=3.4x ATR=6.22 |
| Stop hit — per-position SL triggered | 2023-12-20 09:35:00 | 1052.03 | 1043.31 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:35:00 | 998.70 | 990.23 | 0.00 | ORB-long ORB[987.00,992.95] vol=4.3x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 10:40:00 | 1004.25 | 993.03 | 0.00 | T1 1.5R @ 1004.25 |
| Stop hit — per-position SL triggered | 2024-01-01 11:15:00 | 998.70 | 998.80 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 993.00 | 999.97 | 0.00 | ORB-short ORB[995.05,1005.05] vol=2.2x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 996.67 | 999.21 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:00:00 | 1081.55 | 1072.71 | 0.00 | ORB-long ORB[1067.05,1079.80] vol=3.5x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-01-09 10:05:00 | 1076.68 | 1073.21 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-01-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 09:50:00 | 1071.05 | 1073.75 | 0.00 | ORB-short ORB[1072.25,1080.00] vol=1.7x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:30:00 | 1066.09 | 1072.19 | 0.00 | T1 1.5R @ 1066.09 |
| Target hit | 2024-01-11 15:20:00 | 1060.10 | 1068.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:30:00 | 1052.10 | 1056.74 | 0.00 | ORB-short ORB[1055.20,1064.65] vol=2.8x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 09:45:00 | 1047.66 | 1054.12 | 0.00 | T1 1.5R @ 1047.66 |
| Target hit | 2024-01-15 15:20:00 | 1039.20 | 1044.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:40:00 | 1002.00 | 1010.01 | 0.00 | ORB-short ORB[1008.00,1021.25] vol=3.2x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-02-01 10:45:00 | 1005.30 | 1009.41 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-02-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:50:00 | 1036.70 | 1021.07 | 0.00 | ORB-long ORB[1001.05,1014.80] vol=8.8x ATR=6.42 |
| Stop hit — per-position SL triggered | 2024-02-07 09:55:00 | 1030.28 | 1027.66 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-02-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 09:30:00 | 1024.10 | 1017.32 | 0.00 | ORB-long ORB[1007.00,1018.75] vol=2.8x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-02-15 09:35:00 | 1018.83 | 1017.68 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:50:00 | 1008.50 | 1012.20 | 0.00 | ORB-short ORB[1008.80,1017.90] vol=2.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-02-16 12:00:00 | 1011.88 | 1011.72 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:40:00 | 1030.45 | 1025.06 | 0.00 | ORB-long ORB[1016.50,1022.70] vol=2.4x ATR=3.46 |
| Stop hit — per-position SL triggered | 2024-02-23 09:45:00 | 1026.99 | 1025.75 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 1004.55 | 1013.34 | 0.00 | ORB-short ORB[1015.00,1023.10] vol=1.8x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:00:00 | 998.56 | 1009.99 | 0.00 | T1 1.5R @ 998.56 |
| Stop hit — per-position SL triggered | 2024-03-06 10:05:00 | 1004.55 | 1009.55 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 10:40:00 | 1002.50 | 1009.87 | 0.00 | ORB-short ORB[1008.45,1017.85] vol=2.2x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-03-07 11:00:00 | 1005.37 | 1008.36 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-04-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:35:00 | 987.00 | 979.61 | 0.00 | ORB-long ORB[975.50,983.95] vol=1.8x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-04-05 10:45:00 | 983.45 | 979.87 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:40:00 | 976.80 | 980.35 | 0.00 | ORB-short ORB[977.05,986.95] vol=1.5x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-04-08 11:25:00 | 979.35 | 980.36 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 922.80 | 925.08 | 0.00 | ORB-short ORB[923.00,929.90] vol=2.4x ATR=1.87 |
| Target hit | 2024-04-24 15:20:00 | 917.00 | 921.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2024-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:35:00 | 947.30 | 931.11 | 0.00 | ORB-long ORB[921.30,926.95] vol=6.2x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-04-30 09:40:00 | 943.97 | 934.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-06-09 09:35:00 | 667.40 | 2023-06-09 09:40:00 | 669.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-06-13 09:50:00 | 663.40 | 2023-06-13 10:10:00 | 664.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-22 10:10:00 | 660.30 | 2023-06-22 10:15:00 | 659.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-27 10:25:00 | 642.90 | 2023-06-27 10:30:00 | 644.49 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-04 09:40:00 | 652.05 | 2023-07-04 09:50:00 | 653.51 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-05 09:30:00 | 646.85 | 2023-07-05 10:10:00 | 648.36 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-18 10:35:00 | 655.90 | 2023-07-18 10:40:00 | 653.35 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-07-19 10:25:00 | 645.45 | 2023-07-19 12:40:00 | 643.65 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-07-19 10:25:00 | 645.45 | 2023-07-19 15:20:00 | 641.20 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2023-07-24 09:55:00 | 633.90 | 2023-07-24 10:00:00 | 635.32 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-25 10:00:00 | 636.80 | 2023-07-25 10:10:00 | 640.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-07-25 10:00:00 | 636.80 | 2023-07-25 10:45:00 | 637.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2023-08-08 09:45:00 | 659.75 | 2023-08-08 10:10:00 | 663.96 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-08-08 09:45:00 | 659.75 | 2023-08-08 10:35:00 | 659.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-10 10:35:00 | 646.40 | 2023-08-10 10:50:00 | 648.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-17 10:55:00 | 633.85 | 2023-08-17 14:10:00 | 635.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-18 09:35:00 | 640.25 | 2023-08-18 09:45:00 | 642.62 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-08-18 09:35:00 | 640.25 | 2023-08-18 10:10:00 | 641.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2023-08-25 09:55:00 | 657.85 | 2023-08-25 10:10:00 | 654.50 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-08-25 09:55:00 | 657.85 | 2023-08-25 15:20:00 | 653.60 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2023-08-30 09:35:00 | 654.50 | 2023-08-30 11:45:00 | 656.28 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-04 09:30:00 | 634.80 | 2023-09-04 09:50:00 | 636.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-05 09:35:00 | 633.50 | 2023-09-05 09:50:00 | 634.98 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-08 09:55:00 | 653.00 | 2023-09-08 10:10:00 | 651.14 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-21 09:40:00 | 646.75 | 2023-09-21 10:05:00 | 644.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-09-22 09:55:00 | 636.40 | 2023-09-22 10:05:00 | 638.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-27 10:00:00 | 630.50 | 2023-09-27 10:20:00 | 631.76 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-29 09:40:00 | 619.00 | 2023-09-29 09:50:00 | 616.89 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-09-29 09:40:00 | 619.00 | 2023-09-29 11:00:00 | 617.80 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-10-03 09:30:00 | 614.60 | 2023-10-03 09:40:00 | 611.80 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-10-03 09:30:00 | 614.60 | 2023-10-03 15:20:00 | 610.80 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2023-10-04 09:30:00 | 609.00 | 2023-10-04 09:45:00 | 610.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-05 09:40:00 | 612.45 | 2023-10-05 10:10:00 | 614.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-10-13 09:45:00 | 602.55 | 2023-10-13 09:55:00 | 603.93 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-16 10:00:00 | 596.00 | 2023-10-16 10:15:00 | 597.52 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-17 09:30:00 | 599.65 | 2023-10-17 09:50:00 | 601.26 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-10-17 09:30:00 | 599.65 | 2023-10-17 10:05:00 | 599.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 11:10:00 | 594.55 | 2023-10-18 11:15:00 | 595.56 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-20 10:05:00 | 593.35 | 2023-10-20 12:55:00 | 590.60 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-10-20 10:05:00 | 593.35 | 2023-10-20 15:10:00 | 593.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-06 10:40:00 | 552.80 | 2023-11-06 15:15:00 | 550.92 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-11-06 10:40:00 | 552.80 | 2023-11-06 15:20:00 | 550.00 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2023-11-09 10:10:00 | 539.30 | 2023-11-09 10:30:00 | 537.41 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-11-09 10:10:00 | 539.30 | 2023-11-09 15:20:00 | 534.85 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2023-11-13 10:20:00 | 540.50 | 2023-11-13 13:25:00 | 541.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-20 09:40:00 | 529.50 | 2023-11-20 09:55:00 | 530.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-24 09:55:00 | 529.55 | 2023-11-24 11:15:00 | 530.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-20 09:30:00 | 1058.25 | 2023-12-20 09:35:00 | 1052.03 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-01-01 10:35:00 | 998.70 | 2024-01-01 10:40:00 | 1004.25 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-01 10:35:00 | 998.70 | 2024-01-01 11:15:00 | 998.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 09:55:00 | 993.00 | 2024-01-02 10:05:00 | 996.67 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-01-09 10:00:00 | 1081.55 | 2024-01-09 10:05:00 | 1076.68 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-01-11 09:50:00 | 1071.05 | 2024-01-11 10:30:00 | 1066.09 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-01-11 09:50:00 | 1071.05 | 2024-01-11 15:20:00 | 1060.10 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-01-15 09:30:00 | 1052.10 | 2024-01-15 09:45:00 | 1047.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-15 09:30:00 | 1052.10 | 2024-01-15 15:20:00 | 1039.20 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2024-02-01 10:40:00 | 1002.00 | 2024-02-01 10:45:00 | 1005.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-02-07 09:50:00 | 1036.70 | 2024-02-07 09:55:00 | 1030.28 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-02-15 09:30:00 | 1024.10 | 2024-02-15 09:35:00 | 1018.83 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-02-16 10:50:00 | 1008.50 | 2024-02-16 12:00:00 | 1011.88 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-23 09:40:00 | 1030.45 | 2024-02-23 09:45:00 | 1026.99 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-06 09:55:00 | 1004.55 | 2024-03-06 10:00:00 | 998.56 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-03-06 09:55:00 | 1004.55 | 2024-03-06 10:05:00 | 1004.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-07 10:40:00 | 1002.50 | 2024-03-07 11:00:00 | 1005.37 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-05 10:35:00 | 987.00 | 2024-04-05 10:45:00 | 983.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-08 10:40:00 | 976.80 | 2024-04-08 11:25:00 | 979.35 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-24 09:30:00 | 922.80 | 2024-04-24 15:20:00 | 917.00 | TARGET_HIT | 1.00 | 0.63% |
| BUY | retest1 | 2024-04-30 09:35:00 | 947.30 | 2024-04-30 09:40:00 | 943.97 | STOP_HIT | 1.00 | -0.35% |
