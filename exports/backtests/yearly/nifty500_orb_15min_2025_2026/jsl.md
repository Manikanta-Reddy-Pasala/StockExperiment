# Jindal Stainless Ltd. (JSL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 753.00
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 13 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 47
- **Target hits / Stop hits / Partials:** 13 / 47 / 25
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 24.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 26 | 48.1% | 9 | 28 | 17 | 0.28% | 15.3% |
| BUY @ 2nd Alert (retest1) | 54 | 26 | 48.1% | 9 | 28 | 17 | 0.28% | 15.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 31 | 12 | 38.7% | 4 | 19 | 8 | 0.29% | 9.0% |
| SELL @ 2nd Alert (retest1) | 31 | 12 | 38.7% | 4 | 19 | 8 | 0.29% | 9.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 85 | 38 | 44.7% | 13 | 47 | 25 | 0.29% | 24.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:20:00 | 626.40 | 618.59 | 0.00 | ORB-long ORB[610.30,617.50] vol=1.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-05-13 10:25:00 | 623.80 | 618.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:40:00 | 666.80 | 662.88 | 0.00 | ORB-long ORB[657.35,666.70] vol=1.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-05-20 09:45:00 | 663.87 | 662.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 644.90 | 646.19 | 0.00 | ORB-short ORB[645.00,654.30] vol=3.2x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-05-28 11:20:00 | 646.23 | 646.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:50:00 | 654.95 | 651.18 | 0.00 | ORB-long ORB[644.85,653.00] vol=2.1x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:10:00 | 658.35 | 652.50 | 0.00 | T1 1.5R @ 658.35 |
| Target hit | 2025-06-03 15:20:00 | 661.30 | 660.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:25:00 | 654.10 | 654.68 | 0.00 | ORB-short ORB[657.50,665.00] vol=2.1x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-06-04 10:35:00 | 656.73 | 654.87 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:05:00 | 663.20 | 659.16 | 0.00 | ORB-long ORB[650.50,658.85] vol=2.3x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 10:10:00 | 666.59 | 660.32 | 0.00 | T1 1.5R @ 666.59 |
| Target hit | 2025-06-05 15:20:00 | 672.00 | 670.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 687.50 | 683.78 | 0.00 | ORB-long ORB[676.50,686.70] vol=6.1x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:25:00 | 690.75 | 684.81 | 0.00 | T1 1.5R @ 690.75 |
| Target hit | 2025-06-06 15:20:00 | 694.95 | 688.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 708.30 | 702.06 | 0.00 | ORB-long ORB[695.20,703.90] vol=3.3x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:35:00 | 711.82 | 704.53 | 0.00 | T1 1.5R @ 711.82 |
| Stop hit — per-position SL triggered | 2025-06-09 09:40:00 | 708.30 | 705.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:45:00 | 726.90 | 722.70 | 0.00 | ORB-long ORB[715.05,725.70] vol=1.9x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-06-10 09:50:00 | 723.87 | 723.27 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:45:00 | 700.05 | 705.35 | 0.00 | ORB-short ORB[705.95,714.85] vol=1.8x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 702.63 | 705.13 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:15:00 | 696.70 | 700.13 | 0.00 | ORB-short ORB[697.00,706.75] vol=2.0x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 11:50:00 | 693.18 | 699.36 | 0.00 | T1 1.5R @ 693.18 |
| Target hit | 2025-06-17 15:20:00 | 687.50 | 693.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:00:00 | 704.60 | 699.86 | 0.00 | ORB-long ORB[691.55,701.00] vol=2.1x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-06-30 11:25:00 | 702.25 | 700.17 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:45:00 | 680.85 | 678.05 | 0.00 | ORB-long ORB[673.55,678.90] vol=1.9x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-07-08 11:50:00 | 678.69 | 679.71 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:40:00 | 675.00 | 677.84 | 0.00 | ORB-short ORB[675.55,684.90] vol=1.8x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:05:00 | 671.97 | 676.60 | 0.00 | T1 1.5R @ 671.97 |
| Stop hit — per-position SL triggered | 2025-07-09 10:45:00 | 675.00 | 676.07 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 683.60 | 686.35 | 0.00 | ORB-short ORB[685.50,691.80] vol=1.9x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:40:00 | 681.07 | 685.37 | 0.00 | T1 1.5R @ 681.07 |
| Stop hit — per-position SL triggered | 2025-07-23 09:50:00 | 683.60 | 684.94 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 658.30 | 660.94 | 0.00 | ORB-short ORB[660.40,666.30] vol=2.0x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-07-25 09:55:00 | 660.23 | 660.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 664.55 | 658.22 | 0.00 | ORB-long ORB[653.00,659.75] vol=1.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-07-28 09:35:00 | 662.41 | 658.88 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 11:00:00 | 725.90 | 716.78 | 0.00 | ORB-long ORB[706.40,712.80] vol=2.2x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-08-04 11:10:00 | 723.24 | 717.16 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 735.80 | 732.46 | 0.00 | ORB-long ORB[726.55,734.50] vol=2.0x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-08-05 09:35:00 | 733.25 | 732.36 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:00:00 | 686.95 | 682.81 | 0.00 | ORB-long ORB[675.25,684.90] vol=2.1x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:20:00 | 690.39 | 685.86 | 0.00 | T1 1.5R @ 690.39 |
| Target hit | 2025-08-12 15:20:00 | 713.80 | 701.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-08-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:55:00 | 756.95 | 752.48 | 0.00 | ORB-long ORB[747.00,753.50] vol=1.6x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:50:00 | 760.58 | 755.36 | 0.00 | T1 1.5R @ 760.58 |
| Stop hit — per-position SL triggered | 2025-08-19 12:55:00 | 756.95 | 756.62 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:55:00 | 767.45 | 761.95 | 0.00 | ORB-long ORB[756.00,764.50] vol=1.8x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:45:00 | 770.65 | 765.14 | 0.00 | T1 1.5R @ 770.65 |
| Stop hit — per-position SL triggered | 2025-08-20 14:10:00 | 767.45 | 768.93 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:55:00 | 779.10 | 776.01 | 0.00 | ORB-long ORB[769.00,776.90] vol=2.0x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-08-22 13:00:00 | 776.44 | 776.98 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 11:00:00 | 755.45 | 750.98 | 0.00 | ORB-long ORB[741.70,753.00] vol=3.9x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:25:00 | 758.80 | 751.77 | 0.00 | T1 1.5R @ 758.80 |
| Target hit | 2025-09-09 15:20:00 | 768.95 | 760.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:55:00 | 765.45 | 773.94 | 0.00 | ORB-short ORB[772.75,783.50] vol=4.3x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-09-11 11:10:00 | 768.34 | 773.37 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:50:00 | 744.50 | 739.82 | 0.00 | ORB-long ORB[733.10,743.15] vol=2.0x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 12:35:00 | 748.47 | 742.86 | 0.00 | T1 1.5R @ 748.47 |
| Stop hit — per-position SL triggered | 2025-09-15 13:55:00 | 744.50 | 743.35 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:40:00 | 770.85 | 768.28 | 0.00 | ORB-long ORB[762.80,769.00] vol=4.2x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-09-19 09:50:00 | 768.65 | 768.61 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:10:00 | 802.30 | 808.10 | 0.00 | ORB-short ORB[802.45,812.05] vol=3.1x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-09-24 11:35:00 | 804.62 | 807.66 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 751.00 | 750.08 | 0.00 | ORB-long ORB[741.60,749.20] vol=1.5x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:55:00 | 754.25 | 750.55 | 0.00 | T1 1.5R @ 754.25 |
| Stop hit — per-position SL triggered | 2025-10-01 12:50:00 | 751.00 | 752.27 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:40:00 | 773.30 | 768.81 | 0.00 | ORB-long ORB[760.95,769.90] vol=2.3x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 09:50:00 | 778.75 | 770.67 | 0.00 | T1 1.5R @ 778.75 |
| Target hit | 2025-10-03 10:40:00 | 777.55 | 778.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 755.25 | 762.43 | 0.00 | ORB-short ORB[759.75,768.00] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-10-07 11:35:00 | 757.08 | 761.34 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 744.70 | 750.15 | 0.00 | ORB-short ORB[750.30,759.40] vol=1.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 746.46 | 749.87 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:00:00 | 767.60 | 762.87 | 0.00 | ORB-long ORB[751.50,762.85] vol=2.8x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-10-09 11:20:00 | 764.93 | 763.25 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:45:00 | 777.45 | 771.86 | 0.00 | ORB-long ORB[763.75,772.70] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-10-13 10:50:00 | 775.13 | 772.06 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:10:00 | 785.90 | 782.02 | 0.00 | ORB-long ORB[775.55,781.20] vol=2.3x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 789.49 | 784.00 | 0.00 | T1 1.5R @ 789.49 |
| Target hit | 2025-10-15 13:30:00 | 787.15 | 787.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2025-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:35:00 | 797.50 | 789.53 | 0.00 | ORB-long ORB[780.50,788.00] vol=1.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:40:00 | 801.02 | 792.47 | 0.00 | T1 1.5R @ 801.02 |
| Stop hit — per-position SL triggered | 2025-10-16 10:50:00 | 797.50 | 793.65 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 811.00 | 808.33 | 0.00 | ORB-long ORB[800.30,808.80] vol=3.7x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-10-24 09:40:00 | 808.36 | 809.87 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 811.55 | 807.55 | 0.00 | ORB-long ORB[799.15,808.90] vol=1.6x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:40:00 | 814.85 | 811.57 | 0.00 | T1 1.5R @ 814.85 |
| Target hit | 2025-10-29 09:55:00 | 814.85 | 815.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2025-11-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:05:00 | 747.70 | 751.16 | 0.00 | ORB-short ORB[749.05,755.60] vol=1.7x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:25:00 | 745.27 | 750.72 | 0.00 | T1 1.5R @ 745.27 |
| Target hit | 2025-11-04 15:20:00 | 738.70 | 744.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 717.00 | 719.77 | 0.00 | ORB-short ORB[717.75,726.45] vol=1.8x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 719.49 | 719.59 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:55:00 | 752.35 | 749.53 | 0.00 | ORB-long ORB[742.35,751.95] vol=2.0x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-11-24 11:50:00 | 749.78 | 750.80 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:55:00 | 756.80 | 758.62 | 0.00 | ORB-short ORB[757.50,762.20] vol=1.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:15:00 | 753.57 | 757.71 | 0.00 | T1 1.5R @ 753.57 |
| Stop hit — per-position SL triggered | 2025-12-03 11:55:00 | 756.80 | 756.18 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 762.70 | 759.61 | 0.00 | ORB-long ORB[755.25,761.70] vol=2.4x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-12-08 11:30:00 | 759.99 | 759.80 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:50:00 | 784.00 | 779.91 | 0.00 | ORB-long ORB[770.95,781.00] vol=2.4x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-12-10 09:55:00 | 781.38 | 780.09 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:45:00 | 795.10 | 790.55 | 0.00 | ORB-long ORB[785.70,791.75] vol=3.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-12-18 11:30:00 | 792.85 | 791.47 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 792.25 | 794.64 | 0.00 | ORB-short ORB[793.00,803.05] vol=1.6x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-12-24 09:55:00 | 794.77 | 794.55 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 783.00 | 787.47 | 0.00 | ORB-short ORB[784.00,794.00] vol=1.5x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:20:00 | 779.63 | 786.66 | 0.00 | T1 1.5R @ 779.63 |
| Stop hit — per-position SL triggered | 2025-12-29 11:30:00 | 783.00 | 786.04 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 805.05 | 797.54 | 0.00 | ORB-long ORB[786.95,795.60] vol=2.6x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:45:00 | 810.23 | 803.17 | 0.00 | T1 1.5R @ 810.23 |
| Stop hit — per-position SL triggered | 2025-12-30 09:50:00 | 805.05 | 803.67 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:15:00 | 799.40 | 801.43 | 0.00 | ORB-short ORB[800.25,811.70] vol=2.5x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:40:00 | 795.19 | 800.87 | 0.00 | T1 1.5R @ 795.19 |
| Target hit | 2026-01-20 15:20:00 | 757.60 | 777.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:05:00 | 798.10 | 804.70 | 0.00 | ORB-short ORB[802.30,813.40] vol=2.4x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 12:15:00 | 794.08 | 803.19 | 0.00 | T1 1.5R @ 794.08 |
| Target hit | 2026-02-04 15:20:00 | 782.00 | 788.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 757.35 | 749.26 | 0.00 | ORB-long ORB[738.60,748.70] vol=1.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 754.87 | 750.75 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 768.60 | 765.56 | 0.00 | ORB-long ORB[757.65,765.95] vol=1.7x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:30:00 | 771.62 | 765.92 | 0.00 | T1 1.5R @ 771.62 |
| Target hit | 2026-02-23 15:20:00 | 797.00 | 787.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 793.65 | 797.82 | 0.00 | ORB-short ORB[797.25,807.00] vol=2.0x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 795.15 | 797.48 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 757.70 | 763.47 | 0.00 | ORB-short ORB[762.20,767.95] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 759.86 | 762.51 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-04-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:40:00 | 729.25 | 722.13 | 0.00 | ORB-long ORB[713.35,723.15] vol=3.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-04-06 09:45:00 | 725.94 | 722.66 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 800.35 | 795.61 | 0.00 | ORB-long ORB[788.65,793.00] vol=2.3x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 797.50 | 796.31 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 783.00 | 777.20 | 0.00 | ORB-long ORB[771.05,779.65] vol=1.8x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:30:00 | 787.23 | 779.80 | 0.00 | T1 1.5R @ 787.23 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 783.00 | 780.27 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 761.25 | 769.46 | 0.00 | ORB-short ORB[776.00,787.40] vol=1.9x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 763.69 | 769.04 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 765.35 | 768.71 | 0.00 | ORB-short ORB[767.15,773.75] vol=2.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-05-07 11:10:00 | 767.17 | 768.59 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 763.60 | 765.66 | 0.00 | ORB-short ORB[766.40,773.80] vol=2.4x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 765.08 | 765.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:20:00 | 626.40 | 2025-05-13 10:25:00 | 623.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-20 09:40:00 | 666.80 | 2025-05-20 09:45:00 | 663.87 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-05-28 11:15:00 | 644.90 | 2025-05-28 11:20:00 | 646.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-03 09:50:00 | 654.95 | 2025-06-03 10:10:00 | 658.35 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-06-03 09:50:00 | 654.95 | 2025-06-03 15:20:00 | 661.30 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2025-06-04 10:25:00 | 654.10 | 2025-06-04 10:35:00 | 656.73 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-05 10:05:00 | 663.20 | 2025-06-05 10:10:00 | 666.59 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-05 10:05:00 | 663.20 | 2025-06-05 15:20:00 | 672.00 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-06-06 11:15:00 | 687.50 | 2025-06-06 11:25:00 | 690.75 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-06 11:15:00 | 687.50 | 2025-06-06 15:20:00 | 694.95 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-06-09 09:30:00 | 708.30 | 2025-06-09 09:35:00 | 711.82 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-06-09 09:30:00 | 708.30 | 2025-06-09 09:40:00 | 708.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 09:45:00 | 726.90 | 2025-06-10 09:50:00 | 723.87 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-06-16 09:45:00 | 700.05 | 2025-06-16 09:50:00 | 702.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-17 11:15:00 | 696.70 | 2025-06-17 11:50:00 | 693.18 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-06-17 11:15:00 | 696.70 | 2025-06-17 15:20:00 | 687.50 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-06-30 11:00:00 | 704.60 | 2025-06-30 11:25:00 | 702.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-08 09:45:00 | 680.85 | 2025-07-08 11:50:00 | 678.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-09 09:40:00 | 675.00 | 2025-07-09 10:05:00 | 671.97 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-09 09:40:00 | 675.00 | 2025-07-09 10:45:00 | 675.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:30:00 | 683.60 | 2025-07-23 09:40:00 | 681.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-23 09:30:00 | 683.60 | 2025-07-23 09:50:00 | 683.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 09:40:00 | 658.30 | 2025-07-25 09:55:00 | 660.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-28 09:30:00 | 664.55 | 2025-07-28 09:35:00 | 662.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-04 11:00:00 | 725.90 | 2025-08-04 11:10:00 | 723.24 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-05 09:30:00 | 735.80 | 2025-08-05 09:35:00 | 733.25 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-12 10:00:00 | 686.95 | 2025-08-12 10:20:00 | 690.39 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-08-12 10:00:00 | 686.95 | 2025-08-12 15:20:00 | 713.80 | TARGET_HIT | 0.50 | 3.91% |
| BUY | retest1 | 2025-08-19 09:55:00 | 756.95 | 2025-08-19 11:50:00 | 760.58 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-08-19 09:55:00 | 756.95 | 2025-08-19 12:55:00 | 756.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 09:55:00 | 767.45 | 2025-08-20 10:45:00 | 770.65 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-20 09:55:00 | 767.45 | 2025-08-20 14:10:00 | 767.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 10:55:00 | 779.10 | 2025-08-22 13:00:00 | 776.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-09 11:00:00 | 755.45 | 2025-09-09 11:25:00 | 758.80 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-09 11:00:00 | 755.45 | 2025-09-09 15:20:00 | 768.95 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2025-09-11 10:55:00 | 765.45 | 2025-09-11 11:10:00 | 768.34 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-15 09:50:00 | 744.50 | 2025-09-15 12:35:00 | 748.47 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-15 09:50:00 | 744.50 | 2025-09-15 13:55:00 | 744.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 09:40:00 | 770.85 | 2025-09-19 09:50:00 | 768.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-24 11:10:00 | 802.30 | 2025-09-24 11:35:00 | 804.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-01 10:45:00 | 751.00 | 2025-10-01 11:55:00 | 754.25 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-01 10:45:00 | 751.00 | 2025-10-01 12:50:00 | 751.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 09:40:00 | 773.30 | 2025-10-03 09:50:00 | 778.75 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-10-03 09:40:00 | 773.30 | 2025-10-03 10:40:00 | 777.55 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-10-07 10:55:00 | 755.25 | 2025-10-07 11:35:00 | 757.08 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-08 11:05:00 | 744.70 | 2025-10-08 11:25:00 | 746.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-09 11:00:00 | 767.60 | 2025-10-09 11:20:00 | 764.93 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-13 10:45:00 | 777.45 | 2025-10-13 10:50:00 | 775.13 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-15 10:10:00 | 785.90 | 2025-10-15 10:15:00 | 789.49 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-15 10:10:00 | 785.90 | 2025-10-15 13:30:00 | 787.15 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-10-16 10:35:00 | 797.50 | 2025-10-16 10:40:00 | 801.02 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-16 10:35:00 | 797.50 | 2025-10-16 10:50:00 | 797.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:30:00 | 811.00 | 2025-10-24 09:40:00 | 808.36 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-29 09:30:00 | 811.55 | 2025-10-29 09:40:00 | 814.85 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-10-29 09:30:00 | 811.55 | 2025-10-29 09:55:00 | 814.85 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-04 11:05:00 | 747.70 | 2025-11-04 11:25:00 | 745.27 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-04 11:05:00 | 747.70 | 2025-11-04 15:20:00 | 738.70 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2025-11-07 09:30:00 | 717.00 | 2025-11-07 09:40:00 | 719.49 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-24 09:55:00 | 752.35 | 2025-11-24 11:50:00 | 749.78 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-03 09:55:00 | 756.80 | 2025-12-03 10:15:00 | 753.57 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-03 09:55:00 | 756.80 | 2025-12-03 11:55:00 | 756.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-08 11:00:00 | 762.70 | 2025-12-08 11:30:00 | 759.99 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-10 09:50:00 | 784.00 | 2025-12-10 09:55:00 | 781.38 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-18 10:45:00 | 795.10 | 2025-12-18 11:30:00 | 792.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-24 09:50:00 | 792.25 | 2025-12-24 09:55:00 | 794.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-29 11:05:00 | 783.00 | 2025-12-29 11:20:00 | 779.63 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-29 11:05:00 | 783.00 | 2025-12-29 11:30:00 | 783.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 09:40:00 | 805.05 | 2025-12-30 09:45:00 | 810.23 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-12-30 09:40:00 | 805.05 | 2025-12-30 09:50:00 | 805.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 11:15:00 | 799.40 | 2026-01-20 11:40:00 | 795.19 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-01-20 11:15:00 | 799.40 | 2026-01-20 15:20:00 | 757.60 | TARGET_HIT | 0.50 | 5.23% |
| SELL | retest1 | 2026-02-04 11:05:00 | 798.10 | 2026-02-04 12:15:00 | 794.08 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-04 11:05:00 | 798.10 | 2026-02-04 15:20:00 | 782.00 | TARGET_HIT | 0.50 | 2.02% |
| BUY | retest1 | 2026-02-20 10:40:00 | 757.35 | 2026-02-20 10:55:00 | 754.87 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-23 11:15:00 | 768.60 | 2026-02-23 11:30:00 | 771.62 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-23 11:15:00 | 768.60 | 2026-02-23 15:20:00 | 797.00 | TARGET_HIT | 0.50 | 3.70% |
| SELL | retest1 | 2026-02-26 11:00:00 | 793.65 | 2026-02-26 11:25:00 | 795.15 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-03-06 10:45:00 | 757.70 | 2026-03-06 11:35:00 | 759.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-06 09:40:00 | 729.25 | 2026-04-06 09:45:00 | 725.94 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-21 09:35:00 | 800.35 | 2026-04-21 09:45:00 | 797.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-28 10:10:00 | 783.00 | 2026-04-28 10:30:00 | 787.23 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-28 10:10:00 | 783.00 | 2026-04-28 10:40:00 | 783.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:10:00 | 761.25 | 2026-05-06 11:25:00 | 763.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-07 11:05:00 | 765.35 | 2026-05-07 11:10:00 | 767.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 11:00:00 | 763.60 | 2026-05-08 11:40:00 | 765.08 | STOP_HIT | 1.00 | -0.19% |
