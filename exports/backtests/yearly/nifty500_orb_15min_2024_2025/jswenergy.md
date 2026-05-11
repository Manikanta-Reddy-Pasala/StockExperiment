# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-07-04 15:25:00 (21408 bars)
- **Last close:** 512.25
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
| ENTRY1 | 51 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 14 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 37
- **Target hits / Stop hits / Partials:** 14 / 37 / 26
- **Avg / median % per leg:** 0.31% / 0.16%
- **Sum % (uncompounded):** 23.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 12 | 41.4% | 5 | 17 | 7 | 0.23% | 6.7% |
| BUY @ 2nd Alert (retest1) | 29 | 12 | 41.4% | 5 | 17 | 7 | 0.23% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 28 | 58.3% | 9 | 20 | 19 | 0.36% | 17.3% |
| SELL @ 2nd Alert (retest1) | 48 | 28 | 58.3% | 9 | 20 | 19 | 0.36% | 17.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 40 | 51.9% | 14 | 37 | 26 | 0.31% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:30:00 | 583.50 | 579.07 | 0.00 | ORB-long ORB[573.05,580.50] vol=3.2x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 09:55:00 | 587.71 | 582.51 | 0.00 | T1 1.5R @ 587.71 |
| Target hit | 2024-05-14 13:55:00 | 589.40 | 589.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 641.85 | 638.47 | 0.00 | ORB-long ORB[635.15,640.05] vol=2.1x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-06-12 09:35:00 | 639.30 | 638.74 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 666.80 | 662.39 | 0.00 | ORB-long ORB[657.05,664.50] vol=3.5x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:35:00 | 670.46 | 665.31 | 0.00 | T1 1.5R @ 670.46 |
| Target hit | 2024-06-14 15:20:00 | 686.45 | 677.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 710.30 | 706.60 | 0.00 | ORB-long ORB[700.00,709.65] vol=3.0x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-06-21 09:45:00 | 707.44 | 706.77 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:50:00 | 740.50 | 744.34 | 0.00 | ORB-short ORB[743.45,750.80] vol=2.4x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:00:00 | 736.24 | 743.67 | 0.00 | T1 1.5R @ 736.24 |
| Target hit | 2024-07-02 15:20:00 | 732.00 | 734.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-07-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:25:00 | 726.30 | 731.63 | 0.00 | ORB-short ORB[729.25,739.60] vol=1.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:40:00 | 722.91 | 730.37 | 0.00 | T1 1.5R @ 722.91 |
| Stop hit — per-position SL triggered | 2024-07-03 11:25:00 | 726.30 | 728.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 735.65 | 740.54 | 0.00 | ORB-short ORB[737.05,746.85] vol=1.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-07-08 11:45:00 | 738.10 | 740.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:05:00 | 718.80 | 713.79 | 0.00 | ORB-long ORB[709.10,718.65] vol=3.2x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-07-12 11:25:00 | 715.19 | 714.08 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 719.30 | 717.70 | 0.00 | ORB-long ORB[713.30,718.95] vol=2.3x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-07-16 09:45:00 | 716.11 | 717.73 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 727.60 | 723.42 | 0.00 | ORB-long ORB[716.80,727.35] vol=1.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 724.82 | 723.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:40:00 | 738.90 | 735.63 | 0.00 | ORB-long ORB[728.50,737.95] vol=2.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-08-01 10:30:00 | 736.22 | 736.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:50:00 | 711.00 | 704.02 | 0.00 | ORB-long ORB[701.20,707.00] vol=1.9x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 13:40:00 | 715.31 | 708.60 | 0.00 | T1 1.5R @ 715.31 |
| Stop hit — per-position SL triggered | 2024-08-09 14:00:00 | 711.00 | 708.88 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:25:00 | 678.60 | 673.59 | 0.00 | ORB-long ORB[667.90,674.50] vol=2.4x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-08-20 10:30:00 | 676.31 | 673.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:55:00 | 703.85 | 708.29 | 0.00 | ORB-short ORB[707.50,714.75] vol=1.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-08-26 11:30:00 | 705.81 | 707.87 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 724.30 | 730.61 | 0.00 | ORB-short ORB[728.10,736.05] vol=1.8x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 727.70 | 729.72 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 699.05 | 704.05 | 0.00 | ORB-short ORB[703.40,709.05] vol=3.1x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:55:00 | 695.05 | 701.57 | 0.00 | T1 1.5R @ 695.05 |
| Stop hit — per-position SL triggered | 2024-09-03 12:10:00 | 699.05 | 698.67 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:05:00 | 687.00 | 682.60 | 0.00 | ORB-long ORB[676.25,686.40] vol=1.6x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-09-04 10:10:00 | 684.49 | 682.82 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:30:00 | 764.50 | 767.75 | 0.00 | ORB-short ORB[766.40,776.80] vol=1.7x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-09-17 10:50:00 | 767.24 | 767.42 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:40:00 | 801.00 | 792.94 | 0.00 | ORB-long ORB[786.00,796.15] vol=5.1x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-09-24 10:45:00 | 797.67 | 793.40 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:00:00 | 725.40 | 718.36 | 0.00 | ORB-long ORB[714.50,723.70] vol=2.5x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:25:00 | 729.89 | 720.71 | 0.00 | T1 1.5R @ 729.89 |
| Target hit | 2024-10-09 15:20:00 | 730.70 | 727.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:50:00 | 737.75 | 733.88 | 0.00 | ORB-long ORB[727.90,735.00] vol=1.5x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-10-10 10:25:00 | 734.92 | 735.44 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:15:00 | 709.45 | 711.00 | 0.00 | ORB-short ORB[710.40,718.40] vol=3.3x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:35:00 | 705.75 | 710.66 | 0.00 | T1 1.5R @ 705.75 |
| Target hit | 2024-10-14 15:20:00 | 704.05 | 707.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:15:00 | 677.80 | 681.78 | 0.00 | ORB-short ORB[679.10,689.05] vol=2.0x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-10-22 10:55:00 | 680.86 | 680.49 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:15:00 | 655.00 | 661.68 | 0.00 | ORB-short ORB[664.25,670.40] vol=6.4x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:20:00 | 651.87 | 660.92 | 0.00 | T1 1.5R @ 651.87 |
| Stop hit — per-position SL triggered | 2024-11-29 11:30:00 | 655.00 | 660.56 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 644.35 | 648.56 | 0.00 | ORB-short ORB[646.80,654.00] vol=1.8x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 646.45 | 648.22 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 655.95 | 653.68 | 0.00 | ORB-long ORB[646.50,655.85] vol=2.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-12-05 09:35:00 | 654.11 | 653.64 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:05:00 | 675.40 | 669.82 | 0.00 | ORB-long ORB[666.30,671.45] vol=2.0x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-12-06 11:10:00 | 673.19 | 669.94 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 680.95 | 684.52 | 0.00 | ORB-short ORB[681.50,688.00] vol=2.0x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:05:00 | 676.24 | 682.90 | 0.00 | T1 1.5R @ 676.24 |
| Stop hit — per-position SL triggered | 2024-12-09 10:35:00 | 680.95 | 682.41 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:35:00 | 667.60 | 671.44 | 0.00 | ORB-short ORB[670.10,677.25] vol=2.1x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:55:00 | 664.43 | 670.08 | 0.00 | T1 1.5R @ 664.43 |
| Target hit | 2024-12-10 14:25:00 | 667.20 | 667.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 674.90 | 679.33 | 0.00 | ORB-short ORB[677.25,683.90] vol=1.7x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:50:00 | 671.02 | 677.01 | 0.00 | T1 1.5R @ 671.02 |
| Stop hit — per-position SL triggered | 2024-12-13 11:30:00 | 674.90 | 673.08 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:20:00 | 676.80 | 679.40 | 0.00 | ORB-short ORB[680.50,684.95] vol=1.5x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-12-20 10:25:00 | 679.55 | 679.39 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 10:45:00 | 669.80 | 672.73 | 0.00 | ORB-short ORB[670.10,679.25] vol=1.6x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-12-23 11:30:00 | 672.39 | 671.75 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 660.85 | 655.13 | 0.00 | ORB-long ORB[651.20,660.45] vol=1.6x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-12-24 10:10:00 | 658.21 | 655.99 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:15:00 | 642.60 | 645.65 | 0.00 | ORB-short ORB[645.35,652.25] vol=3.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:50:00 | 640.31 | 644.63 | 0.00 | T1 1.5R @ 640.31 |
| Target hit | 2024-12-26 15:05:00 | 640.15 | 638.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 633.65 | 636.13 | 0.00 | ORB-short ORB[634.35,641.00] vol=1.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:45:00 | 631.34 | 635.21 | 0.00 | T1 1.5R @ 631.34 |
| Target hit | 2024-12-27 15:20:00 | 624.30 | 630.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-01-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:20:00 | 640.05 | 643.53 | 0.00 | ORB-short ORB[642.35,646.90] vol=2.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:15:00 | 637.29 | 641.95 | 0.00 | T1 1.5R @ 637.29 |
| Target hit | 2025-01-03 15:20:00 | 634.70 | 637.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:05:00 | 564.50 | 570.78 | 0.00 | ORB-short ORB[573.75,581.40] vol=1.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:45:00 | 561.74 | 569.26 | 0.00 | T1 1.5R @ 561.74 |
| Stop hit — per-position SL triggered | 2025-01-09 12:25:00 | 564.50 | 568.32 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:30:00 | 553.25 | 555.63 | 0.00 | ORB-short ORB[553.90,561.50] vol=1.6x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:40:00 | 549.23 | 554.80 | 0.00 | T1 1.5R @ 549.23 |
| Target hit | 2025-01-22 14:35:00 | 548.45 | 548.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 474.40 | 471.53 | 0.00 | ORB-long ORB[468.00,473.60] vol=1.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 11:45:00 | 477.65 | 474.13 | 0.00 | T1 1.5R @ 477.65 |
| Target hit | 2025-02-13 12:40:00 | 475.15 | 475.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:25:00 | 492.70 | 489.78 | 0.00 | ORB-long ORB[483.00,489.90] vol=1.6x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:35:00 | 496.15 | 490.86 | 0.00 | T1 1.5R @ 496.15 |
| Target hit | 2025-03-05 15:20:00 | 509.00 | 501.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:10:00 | 518.35 | 510.97 | 0.00 | ORB-long ORB[498.10,505.85] vol=3.3x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-03-11 11:30:00 | 515.83 | 511.67 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:15:00 | 510.95 | 518.65 | 0.00 | ORB-short ORB[516.25,523.85] vol=2.2x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:40:00 | 507.65 | 517.60 | 0.00 | T1 1.5R @ 507.65 |
| Stop hit — per-position SL triggered | 2025-03-12 12:50:00 | 510.95 | 516.41 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:40:00 | 525.70 | 521.19 | 0.00 | ORB-long ORB[516.65,522.80] vol=2.2x ATR=2.47 |
| Stop hit — per-position SL triggered | 2025-03-17 10:00:00 | 523.23 | 522.61 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 528.30 | 530.39 | 0.00 | ORB-short ORB[528.75,533.95] vol=2.3x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:50:00 | 525.49 | 529.76 | 0.00 | T1 1.5R @ 525.49 |
| Stop hit — per-position SL triggered | 2025-03-18 09:55:00 | 528.30 | 529.67 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 11:10:00 | 546.00 | 549.74 | 0.00 | ORB-short ORB[546.55,554.70] vol=1.6x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 548.05 | 548.76 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-04-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:55:00 | 510.90 | 514.76 | 0.00 | ORB-short ORB[512.25,515.65] vol=2.4x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 12:05:00 | 508.21 | 513.03 | 0.00 | T1 1.5R @ 508.21 |
| Stop hit — per-position SL triggered | 2025-04-16 15:05:00 | 510.90 | 510.56 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 11:10:00 | 507.75 | 509.85 | 0.00 | ORB-short ORB[508.15,512.60] vol=1.9x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:25:00 | 505.94 | 509.47 | 0.00 | T1 1.5R @ 505.94 |
| Stop hit — per-position SL triggered | 2025-04-24 11:45:00 | 507.75 | 509.30 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 501.55 | 504.90 | 0.00 | ORB-short ORB[502.75,508.70] vol=1.9x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 498.71 | 503.32 | 0.00 | T1 1.5R @ 498.71 |
| Target hit | 2025-04-25 15:20:00 | 481.95 | 488.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 478.35 | 481.83 | 0.00 | ORB-short ORB[479.60,485.00] vol=1.8x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 480.40 | 481.35 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:55:00 | 478.00 | 471.20 | 0.00 | ORB-long ORB[463.05,469.60] vol=2.3x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:00:00 | 481.02 | 472.22 | 0.00 | T1 1.5R @ 481.02 |
| Stop hit — per-position SL triggered | 2025-04-30 11:05:00 | 478.00 | 472.79 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-05-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:55:00 | 477.10 | 480.32 | 0.00 | ORB-short ORB[477.50,483.80] vol=2.3x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:20:00 | 474.33 | 479.79 | 0.00 | T1 1.5R @ 474.33 |
| Target hit | 2025-05-02 15:20:00 | 468.90 | 475.26 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:30:00 | 583.50 | 2024-05-14 09:55:00 | 587.71 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-05-14 09:30:00 | 583.50 | 2024-05-14 13:55:00 | 589.40 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-06-12 09:30:00 | 641.85 | 2024-06-12 09:35:00 | 639.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-14 09:30:00 | 666.80 | 2024-06-14 09:35:00 | 670.46 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-14 09:30:00 | 666.80 | 2024-06-14 15:20:00 | 686.45 | TARGET_HIT | 0.50 | 2.95% |
| BUY | retest1 | 2024-06-21 09:40:00 | 710.30 | 2024-06-21 09:45:00 | 707.44 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-02 10:50:00 | 740.50 | 2024-07-02 11:00:00 | 736.24 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-07-02 10:50:00 | 740.50 | 2024-07-02 15:20:00 | 732.00 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-07-03 10:25:00 | 726.30 | 2024-07-03 10:40:00 | 722.91 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-03 10:25:00 | 726.30 | 2024-07-03 11:25:00 | 726.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:10:00 | 735.65 | 2024-07-08 11:45:00 | 738.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-12 11:05:00 | 718.80 | 2024-07-12 11:25:00 | 715.19 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-07-16 09:30:00 | 719.30 | 2024-07-16 09:45:00 | 716.11 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-31 09:30:00 | 727.60 | 2024-07-31 09:35:00 | 724.82 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-01 09:40:00 | 738.90 | 2024-08-01 10:30:00 | 736.22 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-09 10:50:00 | 711.00 | 2024-08-09 13:40:00 | 715.31 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-09 10:50:00 | 711.00 | 2024-08-09 14:00:00 | 711.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:25:00 | 678.60 | 2024-08-20 10:30:00 | 676.31 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-26 10:55:00 | 703.85 | 2024-08-26 11:30:00 | 705.81 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-28 09:30:00 | 724.30 | 2024-08-28 09:40:00 | 727.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-09-03 09:35:00 | 699.05 | 2024-09-03 09:55:00 | 695.05 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-03 09:35:00 | 699.05 | 2024-09-03 12:10:00 | 699.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 10:05:00 | 687.00 | 2024-09-04 10:10:00 | 684.49 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-17 10:30:00 | 764.50 | 2024-09-17 10:50:00 | 767.24 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-24 10:40:00 | 801.00 | 2024-09-24 10:45:00 | 797.67 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-09 11:00:00 | 725.40 | 2024-10-09 11:25:00 | 729.89 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-09 11:00:00 | 725.40 | 2024-10-09 15:20:00 | 730.70 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2024-10-10 09:50:00 | 737.75 | 2024-10-10 10:25:00 | 734.92 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-14 11:15:00 | 709.45 | 2024-10-14 11:35:00 | 705.75 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-14 11:15:00 | 709.45 | 2024-10-14 15:20:00 | 704.05 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2024-10-22 10:15:00 | 677.80 | 2024-10-22 10:55:00 | 680.86 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-11-29 11:15:00 | 655.00 | 2024-11-29 11:20:00 | 651.87 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-29 11:15:00 | 655.00 | 2024-11-29 11:30:00 | 655.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 09:45:00 | 644.35 | 2024-12-04 09:55:00 | 646.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-05 09:30:00 | 655.95 | 2024-12-05 09:35:00 | 654.11 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-06 11:05:00 | 675.40 | 2024-12-06 11:10:00 | 673.19 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-09 09:30:00 | 680.95 | 2024-12-09 10:05:00 | 676.24 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-09 09:30:00 | 680.95 | 2024-12-09 10:35:00 | 680.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 10:35:00 | 667.60 | 2024-12-10 10:55:00 | 664.43 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-10 10:35:00 | 667.60 | 2024-12-10 14:25:00 | 667.20 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-12-13 09:35:00 | 674.90 | 2024-12-13 09:50:00 | 671.02 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-13 09:35:00 | 674.90 | 2024-12-13 11:30:00 | 674.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:20:00 | 676.80 | 2024-12-20 10:25:00 | 679.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-23 10:45:00 | 669.80 | 2024-12-23 11:30:00 | 672.39 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-24 09:50:00 | 660.85 | 2024-12-24 10:10:00 | 658.21 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-26 11:15:00 | 642.60 | 2024-12-26 11:50:00 | 640.31 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-26 11:15:00 | 642.60 | 2024-12-26 15:05:00 | 640.15 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-27 10:55:00 | 633.65 | 2024-12-27 11:45:00 | 631.34 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-27 10:55:00 | 633.65 | 2024-12-27 15:20:00 | 624.30 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2025-01-03 10:20:00 | 640.05 | 2025-01-03 11:15:00 | 637.29 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-03 10:20:00 | 640.05 | 2025-01-03 15:20:00 | 634.70 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2025-01-09 11:05:00 | 564.50 | 2025-01-09 11:45:00 | 561.74 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-09 11:05:00 | 564.50 | 2025-01-09 12:25:00 | 564.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 09:30:00 | 553.25 | 2025-01-22 09:40:00 | 549.23 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-01-22 09:30:00 | 553.25 | 2025-01-22 14:35:00 | 548.45 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2025-02-13 09:30:00 | 474.40 | 2025-02-13 11:45:00 | 477.65 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-02-13 09:30:00 | 474.40 | 2025-02-13 12:40:00 | 475.15 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-03-05 10:25:00 | 492.70 | 2025-03-05 10:35:00 | 496.15 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-03-05 10:25:00 | 492.70 | 2025-03-05 15:20:00 | 509.00 | TARGET_HIT | 0.50 | 3.31% |
| BUY | retest1 | 2025-03-11 11:10:00 | 518.35 | 2025-03-11 11:30:00 | 515.83 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-03-12 11:15:00 | 510.95 | 2025-03-12 11:40:00 | 507.65 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-03-12 11:15:00 | 510.95 | 2025-03-12 12:50:00 | 510.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-17 09:40:00 | 525.70 | 2025-03-17 10:00:00 | 523.23 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-18 09:35:00 | 528.30 | 2025-03-18 09:50:00 | 525.49 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-03-18 09:35:00 | 528.30 | 2025-03-18 09:55:00 | 528.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-27 11:10:00 | 546.00 | 2025-03-27 12:15:00 | 548.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-04-16 10:55:00 | 510.90 | 2025-04-16 12:05:00 | 508.21 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-16 10:55:00 | 510.90 | 2025-04-16 15:05:00 | 510.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-24 11:10:00 | 507.75 | 2025-04-24 11:25:00 | 505.94 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-04-24 11:10:00 | 507.75 | 2025-04-24 11:45:00 | 507.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:35:00 | 501.55 | 2025-04-25 09:45:00 | 498.71 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-25 09:35:00 | 501.55 | 2025-04-25 15:20:00 | 481.95 | TARGET_HIT | 0.50 | 3.91% |
| SELL | retest1 | 2025-04-29 09:45:00 | 478.35 | 2025-04-29 09:55:00 | 480.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-04-30 10:55:00 | 478.00 | 2025-04-30 11:00:00 | 481.02 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-04-30 10:55:00 | 478.00 | 2025-04-30 11:05:00 | 478.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-02 10:55:00 | 477.10 | 2025-05-02 11:20:00 | 474.33 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-05-02 10:55:00 | 477.10 | 2025-05-02 15:20:00 | 468.90 | TARGET_HIT | 0.50 | 1.72% |
