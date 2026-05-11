# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36856 bars)
- **Last close:** 409.05
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
| ENTRY1 | 76 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 12 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 64
- **Target hits / Stop hits / Partials:** 12 / 64 / 28
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 10.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 14 | 30.4% | 4 | 32 | 10 | 0.01% | 0.5% |
| BUY @ 2nd Alert (retest1) | 46 | 14 | 30.4% | 4 | 32 | 10 | 0.01% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 26 | 44.8% | 8 | 32 | 18 | 0.18% | 10.5% |
| SELL @ 2nd Alert (retest1) | 58 | 26 | 44.8% | 8 | 32 | 18 | 0.18% | 10.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 40 | 38.5% | 12 | 64 | 28 | 0.11% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 611.80 | 615.94 | 0.00 | ORB-short ORB[617.80,625.65] vol=1.8x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-05-13 11:20:00 | 614.96 | 615.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:30:00 | 654.70 | 649.92 | 0.00 | ORB-long ORB[645.60,650.70] vol=4.5x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:35:00 | 658.64 | 651.39 | 0.00 | T1 1.5R @ 658.64 |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 654.70 | 657.52 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:35:00 | 660.95 | 658.17 | 0.00 | ORB-long ORB[651.60,659.90] vol=6.4x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-05-21 09:45:00 | 657.35 | 658.59 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:30:00 | 669.25 | 665.85 | 0.00 | ORB-long ORB[661.70,669.00] vol=3.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-05-22 11:10:00 | 666.61 | 666.51 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:40:00 | 669.90 | 666.21 | 0.00 | ORB-long ORB[660.80,667.20] vol=4.0x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:00:00 | 673.27 | 670.15 | 0.00 | T1 1.5R @ 673.27 |
| Target hit | 2024-05-24 14:25:00 | 671.35 | 673.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:35:00 | 679.80 | 678.45 | 0.00 | ORB-long ORB[674.95,679.40] vol=1.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-06-12 10:20:00 | 677.39 | 679.03 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:40:00 | 692.80 | 697.27 | 0.00 | ORB-short ORB[695.15,702.90] vol=1.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:15:00 | 688.89 | 695.72 | 0.00 | T1 1.5R @ 688.89 |
| Target hit | 2024-06-18 15:20:00 | 688.55 | 690.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:45:00 | 658.95 | 655.14 | 0.00 | ORB-long ORB[646.00,655.00] vol=3.3x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-06-26 09:50:00 | 655.76 | 654.77 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:45:00 | 631.90 | 636.08 | 0.00 | ORB-short ORB[635.60,641.00] vol=3.2x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-06-27 11:50:00 | 634.02 | 635.74 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:25:00 | 627.45 | 632.92 | 0.00 | ORB-short ORB[634.10,638.70] vol=1.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-06-28 10:30:00 | 629.55 | 632.65 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 656.00 | 648.76 | 0.00 | ORB-long ORB[642.65,650.15] vol=2.5x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-07-02 10:25:00 | 653.39 | 649.39 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:45:00 | 634.10 | 637.57 | 0.00 | ORB-short ORB[635.95,643.20] vol=2.2x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-07-04 15:20:00 | 634.95 | 635.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 630.85 | 632.47 | 0.00 | ORB-short ORB[631.20,637.45] vol=2.1x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:50:00 | 628.96 | 630.69 | 0.00 | T1 1.5R @ 628.96 |
| Stop hit — per-position SL triggered | 2024-07-05 10:25:00 | 630.85 | 630.31 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 603.90 | 606.87 | 0.00 | ORB-short ORB[604.80,612.15] vol=1.9x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:15:00 | 600.86 | 604.63 | 0.00 | T1 1.5R @ 600.86 |
| Target hit | 2024-07-11 15:20:00 | 599.50 | 600.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:40:00 | 584.90 | 586.61 | 0.00 | ORB-short ORB[585.20,591.00] vol=1.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:05:00 | 581.43 | 585.04 | 0.00 | T1 1.5R @ 581.43 |
| Stop hit — per-position SL triggered | 2024-07-19 11:30:00 | 584.90 | 580.95 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:15:00 | 601.00 | 596.02 | 0.00 | ORB-long ORB[592.00,596.80] vol=2.7x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:35:00 | 604.51 | 603.27 | 0.00 | T1 1.5R @ 604.51 |
| Target hit | 2024-07-26 15:00:00 | 609.30 | 612.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:50:00 | 614.05 | 607.13 | 0.00 | ORB-long ORB[605.50,612.10] vol=1.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-07-30 10:55:00 | 611.52 | 607.22 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:05:00 | 602.75 | 603.90 | 0.00 | ORB-short ORB[603.05,610.25] vol=1.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:05:00 | 599.79 | 603.21 | 0.00 | T1 1.5R @ 599.79 |
| Stop hit — per-position SL triggered | 2024-07-31 14:05:00 | 602.75 | 601.93 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 606.70 | 605.28 | 0.00 | ORB-long ORB[601.35,606.00] vol=1.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 09:40:00 | 609.67 | 606.01 | 0.00 | T1 1.5R @ 609.67 |
| Stop hit — per-position SL triggered | 2024-08-01 09:45:00 | 606.70 | 606.02 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 591.05 | 581.53 | 0.00 | ORB-long ORB[575.10,580.95] vol=2.0x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-08-07 10:55:00 | 588.88 | 582.60 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 583.10 | 589.42 | 0.00 | ORB-short ORB[586.55,593.95] vol=3.9x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-08-08 11:00:00 | 584.72 | 589.13 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 626.50 | 629.12 | 0.00 | ORB-short ORB[629.10,631.75] vol=1.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 628.21 | 627.45 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 618.05 | 620.48 | 0.00 | ORB-short ORB[618.80,625.00] vol=2.2x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:05:00 | 615.10 | 619.20 | 0.00 | T1 1.5R @ 615.10 |
| Target hit | 2024-08-29 15:20:00 | 609.35 | 611.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 613.05 | 610.73 | 0.00 | ORB-long ORB[603.10,612.10] vol=2.3x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-09-04 09:50:00 | 610.71 | 611.11 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:50:00 | 599.85 | 602.34 | 0.00 | ORB-short ORB[601.60,606.95] vol=4.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-09-05 11:25:00 | 601.36 | 601.72 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:10:00 | 591.00 | 587.38 | 0.00 | ORB-long ORB[583.70,588.35] vol=4.4x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 589.53 | 587.68 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:55:00 | 583.55 | 586.21 | 0.00 | ORB-short ORB[585.75,590.45] vol=2.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-09-12 10:00:00 | 585.22 | 586.09 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:25:00 | 580.60 | 576.33 | 0.00 | ORB-long ORB[574.00,579.25] vol=2.8x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:35:00 | 583.82 | 577.44 | 0.00 | T1 1.5R @ 583.82 |
| Stop hit — per-position SL triggered | 2024-09-16 10:50:00 | 580.60 | 578.20 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:30:00 | 578.15 | 579.60 | 0.00 | ORB-short ORB[579.50,583.95] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 580.07 | 579.79 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 11:15:00 | 596.15 | 592.92 | 0.00 | ORB-long ORB[591.05,595.20] vol=3.5x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-09-18 11:25:00 | 594.49 | 593.08 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:55:00 | 583.20 | 585.96 | 0.00 | ORB-short ORB[586.55,591.35] vol=3.0x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:10:00 | 581.09 | 585.63 | 0.00 | T1 1.5R @ 581.09 |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 583.20 | 585.46 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:05:00 | 597.85 | 594.79 | 0.00 | ORB-long ORB[591.15,596.45] vol=5.1x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-09-20 11:30:00 | 596.44 | 595.42 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 603.00 | 597.81 | 0.00 | ORB-long ORB[590.55,597.00] vol=3.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:05:00 | 607.64 | 600.97 | 0.00 | T1 1.5R @ 607.64 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 603.00 | 602.08 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 602.20 | 604.19 | 0.00 | ORB-short ORB[602.40,609.95] vol=2.1x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-10-01 09:40:00 | 604.60 | 604.13 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:40:00 | 620.45 | 616.24 | 0.00 | ORB-long ORB[610.00,619.05] vol=1.7x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 617.31 | 617.98 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 09:40:00 | 618.00 | 615.99 | 0.00 | ORB-long ORB[610.05,616.90] vol=2.1x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-10-07 09:45:00 | 614.76 | 615.49 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 582.60 | 584.55 | 0.00 | ORB-short ORB[583.00,588.25] vol=2.7x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:35:00 | 578.39 | 583.49 | 0.00 | T1 1.5R @ 578.39 |
| Stop hit — per-position SL triggered | 2024-10-08 09:45:00 | 582.60 | 582.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:40:00 | 600.00 | 593.79 | 0.00 | ORB-long ORB[587.95,594.45] vol=3.1x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-10-10 09:45:00 | 597.84 | 594.01 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 590.00 | 588.23 | 0.00 | ORB-long ORB[585.45,589.75] vol=2.7x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:55:00 | 593.03 | 588.60 | 0.00 | T1 1.5R @ 593.03 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 590.00 | 589.45 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:05:00 | 596.20 | 598.38 | 0.00 | ORB-short ORB[598.10,602.60] vol=2.3x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 597.63 | 598.98 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:10:00 | 620.55 | 622.88 | 0.00 | ORB-short ORB[621.30,628.95] vol=2.2x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:25:00 | 616.09 | 621.55 | 0.00 | T1 1.5R @ 616.09 |
| Stop hit — per-position SL triggered | 2024-10-16 11:55:00 | 620.55 | 619.98 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 620.25 | 621.16 | 0.00 | ORB-short ORB[621.00,628.70] vol=2.3x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 617.54 | 620.81 | 0.00 | T1 1.5R @ 617.54 |
| Target hit | 2024-10-17 15:20:00 | 609.70 | 616.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 602.05 | 604.90 | 0.00 | ORB-short ORB[603.00,611.30] vol=1.7x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:40:00 | 598.74 | 602.76 | 0.00 | T1 1.5R @ 598.74 |
| Target hit | 2024-10-21 10:15:00 | 599.65 | 598.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2024-10-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:35:00 | 584.95 | 587.76 | 0.00 | ORB-short ORB[586.15,592.10] vol=1.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-10-22 09:45:00 | 587.19 | 587.62 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:30:00 | 562.80 | 567.17 | 0.00 | ORB-short ORB[565.65,573.95] vol=3.1x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:45:00 | 559.07 | 564.90 | 0.00 | T1 1.5R @ 559.07 |
| Stop hit — per-position SL triggered | 2024-10-28 09:50:00 | 562.80 | 564.89 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:55:00 | 578.40 | 574.04 | 0.00 | ORB-long ORB[568.85,572.80] vol=2.9x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:00:00 | 581.44 | 576.02 | 0.00 | T1 1.5R @ 581.44 |
| Target hit | 2024-10-31 12:40:00 | 587.95 | 588.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2024-11-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:30:00 | 591.40 | 585.90 | 0.00 | ORB-long ORB[582.05,587.55] vol=3.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-11-06 10:35:00 | 589.31 | 586.38 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:40:00 | 517.15 | 519.55 | 0.00 | ORB-short ORB[523.80,529.90] vol=10.1x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 519.25 | 519.43 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:40:00 | 498.45 | 494.97 | 0.00 | ORB-long ORB[490.20,496.50] vol=1.9x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:45:00 | 502.16 | 495.82 | 0.00 | T1 1.5R @ 502.16 |
| Target hit | 2024-11-22 15:20:00 | 509.60 | 506.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-11-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:55:00 | 515.15 | 517.99 | 0.00 | ORB-short ORB[515.30,522.05] vol=1.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-11-26 10:00:00 | 516.82 | 517.94 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 531.90 | 530.48 | 0.00 | ORB-long ORB[526.00,531.35] vol=2.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-11-28 09:55:00 | 530.22 | 530.49 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:15:00 | 561.00 | 556.64 | 0.00 | ORB-long ORB[550.00,555.45] vol=2.7x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-12-04 11:25:00 | 558.58 | 556.71 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:15:00 | 575.80 | 571.28 | 0.00 | ORB-long ORB[566.35,572.75] vol=5.5x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-12-11 10:20:00 | 573.63 | 571.41 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:45:00 | 551.80 | 554.07 | 0.00 | ORB-short ORB[553.00,559.35] vol=4.5x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:50:00 | 548.40 | 552.58 | 0.00 | T1 1.5R @ 548.40 |
| Stop hit — per-position SL triggered | 2024-12-13 10:05:00 | 551.80 | 551.83 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:05:00 | 556.30 | 559.76 | 0.00 | ORB-short ORB[557.85,564.45] vol=7.8x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:10:00 | 553.69 | 558.56 | 0.00 | T1 1.5R @ 553.69 |
| Stop hit — per-position SL triggered | 2024-12-16 12:20:00 | 556.30 | 558.29 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:05:00 | 548.85 | 549.96 | 0.00 | ORB-short ORB[549.00,555.50] vol=1.8x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:25:00 | 546.06 | 549.64 | 0.00 | T1 1.5R @ 546.06 |
| Stop hit — per-position SL triggered | 2024-12-17 12:20:00 | 548.85 | 548.45 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:45:00 | 572.00 | 561.37 | 0.00 | ORB-long ORB[546.60,554.05] vol=3.6x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-12-18 09:50:00 | 568.75 | 565.68 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:45:00 | 523.40 | 528.55 | 0.00 | ORB-short ORB[528.60,534.75] vol=1.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:40:00 | 520.43 | 526.11 | 0.00 | T1 1.5R @ 520.43 |
| Target hit | 2024-12-20 15:20:00 | 510.10 | 519.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2024-12-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 09:35:00 | 509.65 | 512.82 | 0.00 | ORB-short ORB[513.60,518.50] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-12-23 11:00:00 | 512.66 | 511.53 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:55:00 | 502.00 | 503.53 | 0.00 | ORB-short ORB[503.05,507.50] vol=4.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-12-26 10:30:00 | 504.02 | 502.82 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 510.50 | 507.83 | 0.00 | ORB-long ORB[503.10,507.90] vol=2.1x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 12:30:00 | 512.81 | 509.82 | 0.00 | T1 1.5R @ 512.81 |
| Stop hit — per-position SL triggered | 2024-12-27 13:55:00 | 510.50 | 510.62 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 504.90 | 505.86 | 0.00 | ORB-short ORB[506.55,512.35] vol=5.9x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-12-30 09:50:00 | 506.60 | 505.92 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 501.85 | 503.89 | 0.00 | ORB-short ORB[503.90,509.50] vol=6.8x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:35:00 | 498.97 | 501.88 | 0.00 | T1 1.5R @ 498.97 |
| Target hit | 2025-01-03 12:30:00 | 496.75 | 496.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2025-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 11:00:00 | 499.85 | 497.33 | 0.00 | ORB-long ORB[492.35,498.60] vol=6.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-01-06 11:05:00 | 497.74 | 497.35 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:30:00 | 490.45 | 494.04 | 0.00 | ORB-short ORB[492.50,499.45] vol=2.7x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 10:45:00 | 487.63 | 493.07 | 0.00 | T1 1.5R @ 487.63 |
| Target hit | 2025-01-07 15:15:00 | 489.70 | 489.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2025-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:55:00 | 496.05 | 492.60 | 0.00 | ORB-long ORB[488.35,494.05] vol=2.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-01-08 11:45:00 | 494.40 | 492.79 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:50:00 | 483.45 | 486.94 | 0.00 | ORB-short ORB[486.95,493.00] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-01-13 10:00:00 | 485.49 | 486.67 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:30:00 | 492.80 | 489.72 | 0.00 | ORB-long ORB[483.20,490.55] vol=1.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-01-16 09:40:00 | 490.98 | 490.50 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:00:00 | 477.00 | 479.38 | 0.00 | ORB-short ORB[477.40,481.70] vol=4.7x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 479.30 | 478.70 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:05:00 | 470.60 | 472.71 | 0.00 | ORB-short ORB[472.35,475.20] vol=3.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 471.71 | 472.70 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-02-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:55:00 | 478.50 | 476.18 | 0.00 | ORB-long ORB[473.75,478.05] vol=2.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 476.41 | 476.31 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:40:00 | 475.60 | 472.47 | 0.00 | ORB-long ORB[468.20,475.00] vol=2.2x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-02-05 10:00:00 | 473.67 | 473.71 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:10:00 | 393.50 | 397.79 | 0.00 | ORB-short ORB[395.70,401.50] vol=1.9x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-03-12 10:35:00 | 395.20 | 397.05 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 11:00:00 | 385.70 | 388.64 | 0.00 | ORB-short ORB[389.25,394.35] vol=2.9x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-03-17 11:55:00 | 387.08 | 387.87 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 09:35:00 | 486.00 | 475.87 | 0.00 | ORB-long ORB[460.00,466.60] vol=9.1x ATR=3.99 |
| Stop hit — per-position SL triggered | 2025-04-09 09:40:00 | 482.01 | 479.24 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:55:00 | 452.25 | 449.65 | 0.00 | ORB-long ORB[443.80,450.55] vol=2.0x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-05-08 13:10:00 | 450.30 | 451.06 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 611.80 | 2024-05-13 11:20:00 | 614.96 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-05-17 10:30:00 | 654.70 | 2024-05-17 10:35:00 | 658.64 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-05-17 10:30:00 | 654.70 | 2024-05-17 14:15:00 | 654.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 09:35:00 | 660.95 | 2024-05-21 09:45:00 | 657.35 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-05-22 10:30:00 | 669.25 | 2024-05-22 11:10:00 | 666.61 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-24 09:40:00 | 669.90 | 2024-05-24 11:00:00 | 673.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-05-24 09:40:00 | 669.90 | 2024-05-24 14:25:00 | 671.35 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-06-12 09:35:00 | 679.80 | 2024-06-12 10:20:00 | 677.39 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-06-18 09:40:00 | 692.80 | 2024-06-18 10:15:00 | 688.89 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-06-18 09:40:00 | 692.80 | 2024-06-18 15:20:00 | 688.55 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-06-26 09:45:00 | 658.95 | 2024-06-26 09:50:00 | 655.76 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-06-27 10:45:00 | 631.90 | 2024-06-27 11:50:00 | 634.02 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-28 10:25:00 | 627.45 | 2024-06-28 10:30:00 | 629.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-02 10:20:00 | 656.00 | 2024-07-02 10:25:00 | 653.39 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-04 10:45:00 | 634.10 | 2024-07-04 15:20:00 | 634.95 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2024-07-05 09:30:00 | 630.85 | 2024-07-05 09:50:00 | 628.96 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-07-05 09:30:00 | 630.85 | 2024-07-05 10:25:00 | 630.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 09:30:00 | 603.90 | 2024-07-11 10:15:00 | 600.86 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-07-11 09:30:00 | 603.90 | 2024-07-11 15:20:00 | 599.50 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-19 09:40:00 | 584.90 | 2024-07-19 10:05:00 | 581.43 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-07-19 09:40:00 | 584.90 | 2024-07-19 11:30:00 | 584.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:15:00 | 601.00 | 2024-07-26 10:35:00 | 604.51 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-26 10:15:00 | 601.00 | 2024-07-26 15:00:00 | 609.30 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2024-07-30 10:50:00 | 614.05 | 2024-07-30 10:55:00 | 611.52 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-31 10:05:00 | 602.75 | 2024-07-31 11:05:00 | 599.79 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-31 10:05:00 | 602.75 | 2024-07-31 14:05:00 | 602.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 09:35:00 | 606.70 | 2024-08-01 09:40:00 | 609.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-08-01 09:35:00 | 606.70 | 2024-08-01 09:45:00 | 606.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-07 10:45:00 | 591.05 | 2024-08-07 10:55:00 | 588.88 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-08 10:55:00 | 583.10 | 2024-08-08 11:00:00 | 584.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-27 09:30:00 | 626.50 | 2024-08-27 10:15:00 | 628.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-29 09:30:00 | 618.05 | 2024-08-29 10:05:00 | 615.10 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-29 09:30:00 | 618.05 | 2024-08-29 15:20:00 | 609.35 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-09-04 09:30:00 | 613.05 | 2024-09-04 09:50:00 | 610.71 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-05 10:50:00 | 599.85 | 2024-09-05 11:25:00 | 601.36 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-10 11:10:00 | 591.00 | 2024-09-10 11:15:00 | 589.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-12 09:55:00 | 583.55 | 2024-09-12 10:00:00 | 585.22 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-16 10:25:00 | 580.60 | 2024-09-16 10:35:00 | 583.82 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-16 10:25:00 | 580.60 | 2024-09-16 10:50:00 | 580.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:30:00 | 578.15 | 2024-09-17 10:35:00 | 580.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-18 11:15:00 | 596.15 | 2024-09-18 11:25:00 | 594.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-19 10:55:00 | 583.20 | 2024-09-19 11:10:00 | 581.09 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-19 10:55:00 | 583.20 | 2024-09-19 11:15:00 | 583.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 11:05:00 | 597.85 | 2024-09-20 11:30:00 | 596.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-24 09:55:00 | 603.00 | 2024-09-24 10:05:00 | 607.64 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-09-24 09:55:00 | 603.00 | 2024-09-24 10:10:00 | 603.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 09:30:00 | 602.20 | 2024-10-01 09:40:00 | 604.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-03 09:40:00 | 620.45 | 2024-10-03 10:15:00 | 617.31 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-10-07 09:40:00 | 618.00 | 2024-10-07 09:45:00 | 614.76 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-10-08 09:30:00 | 582.60 | 2024-10-08 09:35:00 | 578.39 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-10-08 09:30:00 | 582.60 | 2024-10-08 09:45:00 | 582.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 09:40:00 | 600.00 | 2024-10-10 09:45:00 | 597.84 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-11 09:40:00 | 590.00 | 2024-10-11 09:55:00 | 593.03 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-11 09:40:00 | 590.00 | 2024-10-11 10:55:00 | 590.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 10:05:00 | 596.20 | 2024-10-15 10:15:00 | 597.63 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-16 10:10:00 | 620.55 | 2024-10-16 10:25:00 | 616.09 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-10-16 10:10:00 | 620.55 | 2024-10-16 11:55:00 | 620.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 11:05:00 | 620.25 | 2024-10-17 11:25:00 | 617.54 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-17 11:05:00 | 620.25 | 2024-10-17 15:20:00 | 609.70 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2024-10-21 09:30:00 | 602.05 | 2024-10-21 09:40:00 | 598.74 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-21 09:30:00 | 602.05 | 2024-10-21 10:15:00 | 599.65 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-22 09:35:00 | 584.95 | 2024-10-22 09:45:00 | 587.19 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-28 09:30:00 | 562.80 | 2024-10-28 09:45:00 | 559.07 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-28 09:30:00 | 562.80 | 2024-10-28 09:50:00 | 562.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:55:00 | 578.40 | 2024-10-31 10:00:00 | 581.44 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-31 09:55:00 | 578.40 | 2024-10-31 12:40:00 | 587.95 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2024-11-06 10:30:00 | 591.40 | 2024-11-06 10:35:00 | 589.31 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-18 09:40:00 | 517.15 | 2024-11-18 09:45:00 | 519.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-22 09:40:00 | 498.45 | 2024-11-22 09:45:00 | 502.16 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-11-22 09:40:00 | 498.45 | 2024-11-22 15:20:00 | 509.60 | TARGET_HIT | 0.50 | 2.24% |
| SELL | retest1 | 2024-11-26 09:55:00 | 515.15 | 2024-11-26 10:00:00 | 516.82 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-28 09:45:00 | 531.90 | 2024-11-28 09:55:00 | 530.22 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-04 11:15:00 | 561.00 | 2024-12-04 11:25:00 | 558.58 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-12-11 10:15:00 | 575.80 | 2024-12-11 10:20:00 | 573.63 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-13 09:45:00 | 551.80 | 2024-12-13 09:50:00 | 548.40 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-13 09:45:00 | 551.80 | 2024-12-13 10:05:00 | 551.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:05:00 | 556.30 | 2024-12-16 12:10:00 | 553.69 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-16 11:05:00 | 556.30 | 2024-12-16 12:20:00 | 556.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:05:00 | 548.85 | 2024-12-17 10:25:00 | 546.06 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-12-17 10:05:00 | 548.85 | 2024-12-17 12:20:00 | 548.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 09:45:00 | 572.00 | 2024-12-18 09:50:00 | 568.75 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-12-20 10:45:00 | 523.40 | 2024-12-20 11:40:00 | 520.43 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-20 10:45:00 | 523.40 | 2024-12-20 15:20:00 | 510.10 | TARGET_HIT | 0.50 | 2.54% |
| SELL | retest1 | 2024-12-23 09:35:00 | 509.65 | 2024-12-23 11:00:00 | 512.66 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-12-26 09:55:00 | 502.00 | 2024-12-26 10:30:00 | 504.02 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-27 09:55:00 | 510.50 | 2024-12-27 12:30:00 | 512.81 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-27 09:55:00 | 510.50 | 2024-12-27 13:55:00 | 510.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 09:40:00 | 504.90 | 2024-12-30 09:50:00 | 506.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-03 09:30:00 | 501.85 | 2025-01-03 09:35:00 | 498.97 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-03 09:30:00 | 501.85 | 2025-01-03 12:30:00 | 496.75 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-01-06 11:00:00 | 499.85 | 2025-01-06 11:05:00 | 497.74 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-07 10:30:00 | 490.45 | 2025-01-07 10:45:00 | 487.63 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-07 10:30:00 | 490.45 | 2025-01-07 15:15:00 | 489.70 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-01-08 10:55:00 | 496.05 | 2025-01-08 11:45:00 | 494.40 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-13 09:50:00 | 483.45 | 2025-01-13 10:00:00 | 485.49 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-01-16 09:30:00 | 492.80 | 2025-01-16 09:40:00 | 490.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-24 10:00:00 | 477.00 | 2025-01-24 11:15:00 | 479.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-01 11:05:00 | 470.60 | 2025-02-01 11:15:00 | 471.71 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-02-04 09:55:00 | 478.50 | 2025-02-04 10:15:00 | 476.41 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-02-05 09:40:00 | 475.60 | 2025-02-05 10:00:00 | 473.67 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-03-12 10:10:00 | 393.50 | 2025-03-12 10:35:00 | 395.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-03-17 11:00:00 | 385.70 | 2025-03-17 11:55:00 | 387.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-09 09:35:00 | 486.00 | 2025-04-09 09:40:00 | 482.01 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2025-05-08 09:55:00 | 452.25 | 2025-05-08 13:10:00 | 450.30 | STOP_HIT | 1.00 | -0.43% |
