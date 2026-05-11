# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 362.50
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 13 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 66
- **Target hits / Stop hits / Partials:** 13 / 66 / 27
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 11.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 17 | 29.3% | 5 | 41 | 12 | -0.02% | -1.1% |
| BUY @ 2nd Alert (retest1) | 58 | 17 | 29.3% | 5 | 41 | 12 | -0.02% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 23 | 47.9% | 8 | 25 | 15 | 0.25% | 12.1% |
| SELL @ 2nd Alert (retest1) | 48 | 23 | 47.9% | 8 | 25 | 15 | 0.25% | 12.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 40 | 37.7% | 13 | 66 | 27 | 0.10% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 609.50 | 616.87 | 0.00 | ORB-short ORB[616.50,623.80] vol=2.3x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-05-16 12:25:00 | 611.90 | 615.30 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 610.25 | 613.98 | 0.00 | ORB-short ORB[613.55,618.00] vol=1.5x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-05-17 10:55:00 | 612.06 | 613.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 610.20 | 606.10 | 0.00 | ORB-long ORB[601.30,608.05] vol=2.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 608.02 | 606.57 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 629.50 | 627.00 | 0.00 | ORB-long ORB[622.20,629.25] vol=3.0x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 627.05 | 627.05 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:55:00 | 608.90 | 610.27 | 0.00 | ORB-short ORB[611.00,618.50] vol=9.0x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:10:00 | 604.67 | 609.86 | 0.00 | T1 1.5R @ 604.67 |
| Stop hit — per-position SL triggered | 2024-05-31 11:25:00 | 608.90 | 608.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 684.55 | 681.10 | 0.00 | ORB-long ORB[676.05,682.70] vol=2.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:40:00 | 688.37 | 683.45 | 0.00 | T1 1.5R @ 688.37 |
| Stop hit — per-position SL triggered | 2024-06-13 09:45:00 | 684.55 | 683.70 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:25:00 | 673.30 | 676.30 | 0.00 | ORB-short ORB[675.75,682.00] vol=2.0x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-06-18 10:30:00 | 675.49 | 676.02 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:45:00 | 677.75 | 681.64 | 0.00 | ORB-short ORB[679.00,687.90] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-06-27 11:05:00 | 679.83 | 681.36 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:15:00 | 695.80 | 691.85 | 0.00 | ORB-long ORB[685.05,693.00] vol=2.8x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-06-28 10:45:00 | 692.76 | 692.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 703.45 | 698.84 | 0.00 | ORB-long ORB[691.10,699.50] vol=4.0x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-07-01 09:35:00 | 700.74 | 700.87 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:55:00 | 714.70 | 718.11 | 0.00 | ORB-short ORB[717.40,721.95] vol=1.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 716.69 | 717.18 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 714.95 | 716.64 | 0.00 | ORB-short ORB[715.05,724.00] vol=2.1x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:10:00 | 711.02 | 716.16 | 0.00 | T1 1.5R @ 711.02 |
| Target hit | 2024-07-08 15:20:00 | 706.75 | 710.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 697.00 | 700.04 | 0.00 | ORB-short ORB[697.50,706.60] vol=2.3x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:30:00 | 692.94 | 697.50 | 0.00 | T1 1.5R @ 692.94 |
| Target hit | 2024-07-10 13:05:00 | 692.75 | 692.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2024-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:35:00 | 712.00 | 706.18 | 0.00 | ORB-long ORB[702.20,709.85] vol=1.9x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-07-12 09:45:00 | 709.20 | 707.09 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 11:05:00 | 736.15 | 730.61 | 0.00 | ORB-long ORB[725.05,734.60] vol=2.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-07-18 12:25:00 | 732.91 | 731.53 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 703.80 | 707.90 | 0.00 | ORB-short ORB[704.30,712.45] vol=3.3x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-07-23 10:40:00 | 706.79 | 705.82 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 09:35:00 | 720.55 | 724.36 | 0.00 | ORB-short ORB[721.00,728.75] vol=1.9x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:55:00 | 714.48 | 722.38 | 0.00 | T1 1.5R @ 714.48 |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 720.55 | 721.73 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 717.30 | 712.65 | 0.00 | ORB-long ORB[707.50,716.00] vol=2.0x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:40:00 | 720.94 | 716.11 | 0.00 | T1 1.5R @ 720.94 |
| Target hit | 2024-07-26 15:20:00 | 727.95 | 723.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-07-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:40:00 | 714.30 | 718.45 | 0.00 | ORB-short ORB[716.95,721.40] vol=1.9x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:55:00 | 711.14 | 716.71 | 0.00 | T1 1.5R @ 711.14 |
| Target hit | 2024-07-30 15:20:00 | 709.20 | 713.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 703.45 | 706.57 | 0.00 | ORB-short ORB[705.10,715.55] vol=1.7x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-07-31 10:00:00 | 706.13 | 706.46 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:00:00 | 603.00 | 595.36 | 0.00 | ORB-long ORB[591.10,600.00] vol=2.3x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-08-06 10:35:00 | 597.82 | 597.64 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:15:00 | 578.60 | 586.14 | 0.00 | ORB-short ORB[586.15,593.10] vol=1.9x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-08-07 11:35:00 | 581.18 | 585.40 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:05:00 | 582.00 | 573.99 | 0.00 | ORB-long ORB[570.40,577.50] vol=2.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 579.91 | 575.30 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:45:00 | 635.50 | 627.89 | 0.00 | ORB-long ORB[621.00,628.40] vol=2.0x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:05:00 | 639.36 | 631.00 | 0.00 | T1 1.5R @ 639.36 |
| Stop hit — per-position SL triggered | 2024-08-27 10:55:00 | 635.50 | 634.12 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 637.45 | 643.17 | 0.00 | ORB-short ORB[640.80,649.90] vol=1.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 639.71 | 642.13 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 669.70 | 666.57 | 0.00 | ORB-long ORB[662.50,668.00] vol=2.3x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-09-03 09:35:00 | 668.01 | 666.76 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:30:00 | 651.95 | 645.74 | 0.00 | ORB-long ORB[639.95,643.75] vol=3.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:45:00 | 655.20 | 649.57 | 0.00 | T1 1.5R @ 655.20 |
| Target hit | 2024-09-13 15:20:00 | 661.05 | 657.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 626.90 | 629.74 | 0.00 | ORB-short ORB[627.30,635.80] vol=2.2x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-09-17 09:40:00 | 629.44 | 629.16 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 631.80 | 627.98 | 0.00 | ORB-long ORB[625.50,631.40] vol=2.3x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-09-25 09:50:00 | 629.45 | 628.18 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:25:00 | 577.60 | 585.08 | 0.00 | ORB-short ORB[582.30,590.25] vol=1.7x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:45:00 | 572.92 | 582.48 | 0.00 | T1 1.5R @ 572.92 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 577.60 | 578.75 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 587.20 | 591.21 | 0.00 | ORB-short ORB[587.60,594.95] vol=2.2x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:45:00 | 584.22 | 590.37 | 0.00 | T1 1.5R @ 584.22 |
| Stop hit — per-position SL triggered | 2024-10-16 11:55:00 | 587.20 | 590.24 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:40:00 | 590.95 | 593.49 | 0.00 | ORB-short ORB[592.15,599.45] vol=1.7x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-10-22 09:55:00 | 593.47 | 593.21 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:30:00 | 566.60 | 570.53 | 0.00 | ORB-short ORB[569.60,576.45] vol=3.4x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:00:00 | 563.36 | 568.87 | 0.00 | T1 1.5R @ 563.36 |
| Stop hit — per-position SL triggered | 2024-10-29 11:20:00 | 566.60 | 568.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 569.35 | 564.00 | 0.00 | ORB-long ORB[559.00,566.00] vol=2.3x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:50:00 | 573.13 | 566.81 | 0.00 | T1 1.5R @ 573.13 |
| Stop hit — per-position SL triggered | 2024-11-06 10:30:00 | 569.35 | 568.68 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 580.40 | 576.56 | 0.00 | ORB-long ORB[572.00,578.45] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 578.57 | 576.87 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:45:00 | 570.80 | 564.61 | 0.00 | ORB-long ORB[560.15,565.75] vol=1.5x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-11-11 09:55:00 | 568.67 | 565.41 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 559.35 | 562.42 | 0.00 | ORB-short ORB[560.65,565.45] vol=2.4x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 556.18 | 561.05 | 0.00 | T1 1.5R @ 556.18 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 559.35 | 559.97 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:10:00 | 560.15 | 555.85 | 0.00 | ORB-long ORB[548.00,553.95] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-11-19 12:20:00 | 558.34 | 558.30 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:30:00 | 554.70 | 550.43 | 0.00 | ORB-long ORB[546.10,551.45] vol=3.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:35:00 | 557.28 | 552.43 | 0.00 | T1 1.5R @ 557.28 |
| Stop hit — per-position SL triggered | 2024-11-22 09:40:00 | 554.70 | 553.27 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 603.10 | 598.97 | 0.00 | ORB-long ORB[594.00,601.95] vol=2.3x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-11-27 10:05:00 | 600.46 | 599.45 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:55:00 | 600.80 | 598.84 | 0.00 | ORB-long ORB[596.50,600.65] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 599.16 | 599.27 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 594.20 | 591.47 | 0.00 | ORB-long ORB[587.70,592.95] vol=1.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-11-29 11:35:00 | 592.30 | 591.61 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 592.70 | 588.84 | 0.00 | ORB-long ORB[585.10,592.00] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:40:00 | 595.41 | 590.57 | 0.00 | T1 1.5R @ 595.41 |
| Target hit | 2024-12-02 12:00:00 | 596.00 | 597.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 600.05 | 597.80 | 0.00 | ORB-long ORB[595.00,600.00] vol=2.3x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:35:00 | 602.36 | 599.12 | 0.00 | T1 1.5R @ 602.36 |
| Target hit | 2024-12-03 10:20:00 | 603.75 | 603.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 606.55 | 604.44 | 0.00 | ORB-long ORB[601.25,605.50] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 605.19 | 605.48 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 609.50 | 607.50 | 0.00 | ORB-long ORB[604.90,609.45] vol=1.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-12-06 11:10:00 | 608.06 | 607.54 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:45:00 | 612.55 | 609.47 | 0.00 | ORB-long ORB[604.50,611.60] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-12-10 11:05:00 | 610.85 | 611.19 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 619.20 | 617.10 | 0.00 | ORB-long ORB[615.00,618.80] vol=2.5x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-12-12 09:45:00 | 617.39 | 617.18 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:15:00 | 596.45 | 599.89 | 0.00 | ORB-short ORB[603.30,608.80] vol=4.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-12-13 11:55:00 | 598.41 | 599.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:50:00 | 609.00 | 606.20 | 0.00 | ORB-long ORB[601.80,607.00] vol=3.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-12-16 11:10:00 | 607.48 | 606.85 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:35:00 | 608.10 | 607.32 | 0.00 | ORB-long ORB[602.35,607.20] vol=2.5x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:40:00 | 610.01 | 608.15 | 0.00 | T1 1.5R @ 610.01 |
| Stop hit — per-position SL triggered | 2024-12-17 11:00:00 | 608.10 | 608.43 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:35:00 | 594.55 | 589.30 | 0.00 | ORB-long ORB[584.75,592.75] vol=2.4x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 591.81 | 591.22 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:50:00 | 593.85 | 597.55 | 0.00 | ORB-short ORB[595.55,602.55] vol=1.8x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:05:00 | 589.77 | 594.42 | 0.00 | T1 1.5R @ 589.77 |
| Target hit | 2024-12-20 15:20:00 | 575.10 | 583.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2024-12-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:45:00 | 582.05 | 577.04 | 0.00 | ORB-long ORB[570.25,578.75] vol=2.0x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:20:00 | 585.51 | 579.86 | 0.00 | T1 1.5R @ 585.51 |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 582.05 | 580.91 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 572.90 | 575.08 | 0.00 | ORB-short ORB[575.85,584.40] vol=3.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:20:00 | 570.91 | 574.44 | 0.00 | T1 1.5R @ 570.91 |
| Target hit | 2024-12-26 15:20:00 | 569.65 | 571.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:15:00 | 574.25 | 572.34 | 0.00 | ORB-long ORB[568.10,573.25] vol=1.8x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-12-27 11:30:00 | 572.75 | 572.49 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 563.80 | 559.67 | 0.00 | ORB-long ORB[558.15,563.00] vol=2.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-01-01 10:55:00 | 562.31 | 559.88 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:45:00 | 569.00 | 562.90 | 0.00 | ORB-long ORB[558.55,566.00] vol=1.5x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-01-02 09:50:00 | 567.09 | 562.93 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 559.35 | 563.48 | 0.00 | ORB-short ORB[565.10,568.80] vol=2.0x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:10:00 | 556.74 | 561.69 | 0.00 | T1 1.5R @ 556.74 |
| Target hit | 2025-01-03 11:35:00 | 558.50 | 558.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 556.65 | 554.27 | 0.00 | ORB-long ORB[550.55,555.95] vol=2.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-01-06 09:35:00 | 554.97 | 554.84 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:30:00 | 547.40 | 544.72 | 0.00 | ORB-long ORB[541.05,546.90] vol=2.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-01-09 10:40:00 | 545.31 | 545.02 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 529.60 | 533.01 | 0.00 | ORB-short ORB[532.00,537.50] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-01-15 10:00:00 | 531.39 | 531.37 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:10:00 | 535.00 | 532.15 | 0.00 | ORB-long ORB[527.25,534.95] vol=1.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:25:00 | 537.53 | 533.42 | 0.00 | T1 1.5R @ 537.53 |
| Stop hit — per-position SL triggered | 2025-01-17 10:50:00 | 535.00 | 534.76 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 532.50 | 534.57 | 0.00 | ORB-short ORB[533.20,539.90] vol=2.2x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-01-27 09:35:00 | 534.91 | 534.45 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:40:00 | 517.00 | 520.21 | 0.00 | ORB-short ORB[517.95,525.00] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-01-28 10:50:00 | 519.65 | 518.14 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 530.60 | 527.77 | 0.00 | ORB-long ORB[521.80,528.50] vol=2.4x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-01-29 10:10:00 | 528.19 | 529.44 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:35:00 | 541.50 | 538.55 | 0.00 | ORB-long ORB[536.00,540.00] vol=2.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-02-01 09:45:00 | 540.07 | 539.02 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 09:30:00 | 532.90 | 535.32 | 0.00 | ORB-short ORB[533.95,538.90] vol=2.0x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 09:35:00 | 529.95 | 533.92 | 0.00 | T1 1.5R @ 529.95 |
| Stop hit — per-position SL triggered | 2025-02-04 09:50:00 | 532.90 | 532.93 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-02-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:55:00 | 538.95 | 535.40 | 0.00 | ORB-long ORB[528.15,535.95] vol=2.4x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-02-05 11:10:00 | 537.12 | 535.75 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 534.70 | 537.86 | 0.00 | ORB-short ORB[536.05,541.40] vol=1.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 536.50 | 537.56 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:15:00 | 470.40 | 474.83 | 0.00 | ORB-short ORB[475.60,482.45] vol=2.3x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:25:00 | 466.95 | 472.88 | 0.00 | T1 1.5R @ 466.95 |
| Target hit | 2025-02-18 14:10:00 | 469.75 | 469.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2025-02-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:30:00 | 437.35 | 441.21 | 0.00 | ORB-short ORB[439.70,445.90] vol=1.7x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 11:00:00 | 433.54 | 437.92 | 0.00 | T1 1.5R @ 433.54 |
| Target hit | 2025-02-28 15:20:00 | 423.70 | 430.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 443.15 | 439.70 | 0.00 | ORB-long ORB[436.05,441.30] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-03-06 09:45:00 | 441.23 | 439.96 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 393.25 | 388.97 | 0.00 | ORB-long ORB[386.15,389.40] vol=1.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 391.81 | 389.26 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 395.70 | 401.01 | 0.00 | ORB-short ORB[401.25,404.95] vol=1.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-03-26 09:45:00 | 397.36 | 400.45 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 369.00 | 366.55 | 0.00 | ORB-long ORB[362.40,367.35] vol=1.8x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-04-15 09:50:00 | 367.39 | 366.83 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 11:10:00 | 395.45 | 392.55 | 0.00 | ORB-long ORB[389.00,394.80] vol=1.9x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 11:15:00 | 398.22 | 392.94 | 0.00 | T1 1.5R @ 398.22 |
| Target hit | 2025-04-23 15:20:00 | 399.10 | 395.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2025-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:50:00 | 404.00 | 399.28 | 0.00 | ORB-long ORB[395.35,399.50] vol=2.4x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-05-05 09:55:00 | 402.32 | 399.94 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 390.80 | 391.65 | 0.00 | ORB-short ORB[391.00,393.90] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-05-08 09:35:00 | 392.10 | 391.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 609.50 | 2024-05-16 12:25:00 | 611.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-17 10:50:00 | 610.25 | 2024-05-17 10:55:00 | 612.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-23 09:40:00 | 610.20 | 2024-05-23 09:50:00 | 608.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-27 09:35:00 | 629.50 | 2024-05-27 09:40:00 | 627.05 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-31 09:55:00 | 608.90 | 2024-05-31 10:10:00 | 604.67 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-05-31 09:55:00 | 608.90 | 2024-05-31 11:25:00 | 608.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 09:35:00 | 684.55 | 2024-06-13 09:40:00 | 688.37 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-06-13 09:35:00 | 684.55 | 2024-06-13 09:45:00 | 684.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 10:25:00 | 673.30 | 2024-06-18 10:30:00 | 675.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-27 10:45:00 | 677.75 | 2024-06-27 11:05:00 | 679.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-28 10:15:00 | 695.80 | 2024-06-28 10:45:00 | 692.76 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-01 09:30:00 | 703.45 | 2024-07-01 09:35:00 | 700.74 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-05 09:55:00 | 714.70 | 2024-07-05 10:15:00 | 716.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-08 10:50:00 | 714.95 | 2024-07-08 11:10:00 | 711.02 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-08 10:50:00 | 714.95 | 2024-07-08 15:20:00 | 706.75 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-07-10 10:10:00 | 697.00 | 2024-07-10 10:30:00 | 692.94 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-07-10 10:10:00 | 697.00 | 2024-07-10 13:05:00 | 692.75 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-12 09:35:00 | 712.00 | 2024-07-12 09:45:00 | 709.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-18 11:05:00 | 736.15 | 2024-07-18 12:25:00 | 732.91 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-23 09:30:00 | 703.80 | 2024-07-23 10:40:00 | 706.79 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-24 09:35:00 | 720.55 | 2024-07-24 09:55:00 | 714.48 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-07-24 09:35:00 | 720.55 | 2024-07-24 10:15:00 | 720.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:30:00 | 717.30 | 2024-07-26 09:40:00 | 720.94 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-26 09:30:00 | 717.30 | 2024-07-26 15:20:00 | 727.95 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2024-07-30 10:40:00 | 714.30 | 2024-07-30 11:55:00 | 711.14 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-30 10:40:00 | 714.30 | 2024-07-30 15:20:00 | 709.20 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-07-31 09:50:00 | 703.45 | 2024-07-31 10:00:00 | 706.13 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-06 10:00:00 | 603.00 | 2024-08-06 10:35:00 | 597.82 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest1 | 2024-08-07 11:15:00 | 578.60 | 2024-08-07 11:35:00 | 581.18 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-12 11:05:00 | 582.00 | 2024-08-12 11:15:00 | 579.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-27 09:45:00 | 635.50 | 2024-08-27 10:05:00 | 639.36 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-27 09:45:00 | 635.50 | 2024-08-27 10:55:00 | 635.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 637.45 | 2024-08-28 09:40:00 | 639.71 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-03 09:30:00 | 669.70 | 2024-09-03 09:35:00 | 668.01 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-13 10:30:00 | 651.95 | 2024-09-13 10:45:00 | 655.20 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-13 10:30:00 | 651.95 | 2024-09-13 15:20:00 | 661.05 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-09-17 09:30:00 | 626.90 | 2024-09-17 09:40:00 | 629.44 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-25 09:45:00 | 631.80 | 2024-09-25 09:50:00 | 629.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-07 10:25:00 | 577.60 | 2024-10-07 10:45:00 | 572.92 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2024-10-07 10:25:00 | 577.60 | 2024-10-07 11:25:00 | 577.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 11:15:00 | 587.20 | 2024-10-16 11:45:00 | 584.22 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-16 11:15:00 | 587.20 | 2024-10-16 11:55:00 | 587.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 09:40:00 | 590.95 | 2024-10-22 09:55:00 | 593.47 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-29 10:30:00 | 566.60 | 2024-10-29 11:00:00 | 563.36 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-10-29 10:30:00 | 566.60 | 2024-10-29 11:20:00 | 566.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 09:40:00 | 569.35 | 2024-11-06 09:50:00 | 573.13 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-06 09:40:00 | 569.35 | 2024-11-06 10:30:00 | 569.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 09:45:00 | 580.40 | 2024-11-08 09:50:00 | 578.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-11 09:45:00 | 570.80 | 2024-11-11 09:55:00 | 568.67 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-13 09:30:00 | 559.35 | 2024-11-13 09:40:00 | 556.18 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-11-13 09:30:00 | 559.35 | 2024-11-13 09:50:00 | 559.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:10:00 | 560.15 | 2024-11-19 12:20:00 | 558.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-22 09:30:00 | 554.70 | 2024-11-22 09:35:00 | 557.28 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-11-22 09:30:00 | 554.70 | 2024-11-22 09:40:00 | 554.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:50:00 | 603.10 | 2024-11-27 10:05:00 | 600.46 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-28 09:55:00 | 600.80 | 2024-11-28 10:35:00 | 599.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-29 10:55:00 | 594.20 | 2024-11-29 11:35:00 | 592.30 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-02 09:30:00 | 592.70 | 2024-12-02 09:40:00 | 595.41 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-02 09:30:00 | 592.70 | 2024-12-02 12:00:00 | 596.00 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-03 09:30:00 | 600.05 | 2024-12-03 09:35:00 | 602.36 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-03 09:30:00 | 600.05 | 2024-12-03 10:20:00 | 603.75 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2024-12-04 09:30:00 | 606.55 | 2024-12-04 09:45:00 | 605.19 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-06 11:00:00 | 609.50 | 2024-12-06 11:10:00 | 608.06 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-10 09:45:00 | 612.55 | 2024-12-10 11:05:00 | 610.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-12 09:40:00 | 619.20 | 2024-12-12 09:45:00 | 617.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-13 11:15:00 | 596.45 | 2024-12-13 11:55:00 | 598.41 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-16 10:50:00 | 609.00 | 2024-12-16 11:10:00 | 607.48 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-17 10:35:00 | 608.10 | 2024-12-17 10:40:00 | 610.01 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-12-17 10:35:00 | 608.10 | 2024-12-17 11:00:00 | 608.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-19 09:35:00 | 594.55 | 2024-12-19 10:00:00 | 591.81 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-20 09:50:00 | 593.85 | 2024-12-20 12:05:00 | 589.77 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-20 09:50:00 | 593.85 | 2024-12-20 15:20:00 | 575.10 | TARGET_HIT | 0.50 | 3.16% |
| BUY | retest1 | 2024-12-24 09:45:00 | 582.05 | 2024-12-24 10:20:00 | 585.51 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-12-24 09:45:00 | 582.05 | 2024-12-24 11:15:00 | 582.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 11:05:00 | 572.90 | 2024-12-26 12:20:00 | 570.91 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-26 11:05:00 | 572.90 | 2024-12-26 15:20:00 | 569.65 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-12-27 11:15:00 | 574.25 | 2024-12-27 11:30:00 | 572.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-01 10:50:00 | 563.80 | 2025-01-01 10:55:00 | 562.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-02 09:45:00 | 569.00 | 2025-01-02 09:50:00 | 567.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-03 10:00:00 | 559.35 | 2025-01-03 10:10:00 | 556.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-03 10:00:00 | 559.35 | 2025-01-03 11:35:00 | 558.50 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-01-06 09:30:00 | 556.65 | 2025-01-06 09:35:00 | 554.97 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-09 10:30:00 | 547.40 | 2025-01-09 10:40:00 | 545.31 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-15 09:30:00 | 529.60 | 2025-01-15 10:00:00 | 531.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-17 10:10:00 | 535.00 | 2025-01-17 10:25:00 | 537.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-17 10:10:00 | 535.00 | 2025-01-17 10:50:00 | 535.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 09:30:00 | 532.50 | 2025-01-27 09:35:00 | 534.91 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-01-28 09:40:00 | 517.00 | 2025-01-28 10:50:00 | 519.65 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-29 09:30:00 | 530.60 | 2025-01-29 10:10:00 | 528.19 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-02-01 09:35:00 | 541.50 | 2025-02-01 09:45:00 | 540.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-04 09:30:00 | 532.90 | 2025-02-04 09:35:00 | 529.95 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-02-04 09:30:00 | 532.90 | 2025-02-04 09:50:00 | 532.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 10:55:00 | 538.95 | 2025-02-05 11:10:00 | 537.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-06 09:30:00 | 534.70 | 2025-02-06 09:40:00 | 536.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-18 10:15:00 | 470.40 | 2025-02-18 11:25:00 | 466.95 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-02-18 10:15:00 | 470.40 | 2025-02-18 14:10:00 | 469.75 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-02-28 09:30:00 | 437.35 | 2025-02-28 11:00:00 | 433.54 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2025-02-28 09:30:00 | 437.35 | 2025-02-28 15:20:00 | 423.70 | TARGET_HIT | 0.50 | 3.12% |
| BUY | retest1 | 2025-03-06 09:35:00 | 443.15 | 2025-03-06 09:45:00 | 441.23 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-17 09:30:00 | 393.25 | 2025-03-17 09:35:00 | 391.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-26 09:40:00 | 395.70 | 2025-03-26 09:45:00 | 397.36 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-15 09:40:00 | 369.00 | 2025-04-15 09:50:00 | 367.39 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-23 11:10:00 | 395.45 | 2025-04-23 11:15:00 | 398.22 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-04-23 11:10:00 | 395.45 | 2025-04-23 15:20:00 | 399.10 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2025-05-05 09:50:00 | 404.00 | 2025-05-05 09:55:00 | 402.32 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-08 09:30:00 | 390.80 | 2025-05-08 09:35:00 | 392.10 | STOP_HIT | 1.00 | -0.33% |
