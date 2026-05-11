# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 834.95
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
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 13 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 39
- **Target hits / Stop hits / Partials:** 13 / 39 / 29
- **Avg / median % per leg:** 0.27% / 0.18%
- **Sum % (uncompounded):** 21.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 22 | 51.2% | 8 | 21 | 14 | 0.25% | 10.7% |
| BUY @ 2nd Alert (retest1) | 43 | 22 | 51.2% | 8 | 21 | 14 | 0.25% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 20 | 52.6% | 5 | 18 | 15 | 0.29% | 11.1% |
| SELL @ 2nd Alert (retest1) | 38 | 20 | 52.6% | 5 | 18 | 15 | 0.29% | 11.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 42 | 51.9% | 13 | 39 | 29 | 0.27% | 21.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 621.20 | 620.31 | 0.00 | ORB-long ORB[611.90,621.05] vol=3.2x ATR=2.66 |
| Stop hit — per-position SL triggered | 2024-05-14 09:45:00 | 618.54 | 620.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 626.95 | 625.15 | 0.00 | ORB-long ORB[620.70,626.50] vol=1.5x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-05-15 10:05:00 | 624.32 | 625.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:50:00 | 626.00 | 623.05 | 0.00 | ORB-long ORB[620.55,623.55] vol=2.4x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:10:00 | 629.15 | 624.97 | 0.00 | T1 1.5R @ 629.15 |
| Stop hit — per-position SL triggered | 2024-05-16 10:50:00 | 626.00 | 625.33 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:20:00 | 631.75 | 633.81 | 0.00 | ORB-short ORB[633.00,639.15] vol=1.6x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:45:00 | 628.25 | 632.73 | 0.00 | T1 1.5R @ 628.25 |
| Stop hit — per-position SL triggered | 2024-05-28 15:00:00 | 631.75 | 631.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:20:00 | 707.70 | 710.43 | 0.00 | ORB-short ORB[710.05,717.35] vol=3.1x ATR=2.36 |
| Target hit | 2024-06-12 15:20:00 | 705.00 | 709.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 705.75 | 711.23 | 0.00 | ORB-short ORB[707.95,716.95] vol=1.9x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:45:00 | 702.07 | 710.31 | 0.00 | T1 1.5R @ 702.07 |
| Stop hit — per-position SL triggered | 2024-06-13 10:30:00 | 705.75 | 709.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:35:00 | 757.80 | 761.73 | 0.00 | ORB-short ORB[760.25,767.00] vol=3.0x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:50:00 | 753.79 | 760.88 | 0.00 | T1 1.5R @ 753.79 |
| Stop hit — per-position SL triggered | 2024-06-27 10:55:00 | 757.80 | 760.72 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:00:00 | 766.35 | 764.30 | 0.00 | ORB-long ORB[758.15,765.00] vol=2.4x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 13:55:00 | 770.79 | 766.71 | 0.00 | T1 1.5R @ 770.79 |
| Target hit | 2024-07-01 15:15:00 | 767.05 | 767.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:05:00 | 756.75 | 760.72 | 0.00 | ORB-short ORB[761.45,767.50] vol=1.6x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-07-03 11:50:00 | 759.08 | 760.17 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 760.00 | 762.69 | 0.00 | ORB-short ORB[762.00,767.50] vol=2.7x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:45:00 | 756.53 | 761.24 | 0.00 | T1 1.5R @ 756.53 |
| Stop hit — per-position SL triggered | 2024-07-05 11:55:00 | 760.00 | 761.15 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 765.90 | 761.54 | 0.00 | ORB-long ORB[755.15,763.60] vol=3.6x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:45:00 | 771.20 | 763.74 | 0.00 | T1 1.5R @ 771.20 |
| Target hit | 2024-07-08 11:35:00 | 779.30 | 779.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 756.45 | 762.52 | 0.00 | ORB-short ORB[773.00,782.00] vol=7.7x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 750.37 | 761.46 | 0.00 | T1 1.5R @ 750.37 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 756.45 | 759.49 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 798.10 | 792.38 | 0.00 | ORB-long ORB[785.10,795.20] vol=3.2x ATR=3.57 |
| Stop hit — per-position SL triggered | 2024-07-12 09:40:00 | 794.53 | 794.62 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 788.70 | 792.56 | 0.00 | ORB-short ORB[788.95,797.50] vol=2.4x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 791.84 | 792.18 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:40:00 | 782.40 | 772.74 | 0.00 | ORB-long ORB[764.65,776.25] vol=1.5x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:50:00 | 786.38 | 773.79 | 0.00 | T1 1.5R @ 786.38 |
| Stop hit — per-position SL triggered | 2024-07-26 11:20:00 | 782.40 | 775.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:05:00 | 819.70 | 822.88 | 0.00 | ORB-short ORB[821.55,828.90] vol=1.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:20:00 | 815.05 | 821.35 | 0.00 | T1 1.5R @ 815.05 |
| Target hit | 2024-08-01 15:20:00 | 799.00 | 805.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:00:00 | 779.40 | 768.62 | 0.00 | ORB-long ORB[758.25,769.70] vol=1.8x ATR=4.35 |
| Stop hit — per-position SL triggered | 2024-08-06 10:05:00 | 775.05 | 769.39 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 767.70 | 764.00 | 0.00 | ORB-long ORB[759.00,766.60] vol=1.8x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 09:45:00 | 772.79 | 767.44 | 0.00 | T1 1.5R @ 772.79 |
| Stop hit — per-position SL triggered | 2024-08-08 10:00:00 | 767.70 | 767.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 779.00 | 772.49 | 0.00 | ORB-long ORB[767.10,774.20] vol=2.2x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-08-09 10:00:00 | 774.85 | 773.45 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:15:00 | 763.90 | 771.46 | 0.00 | ORB-short ORB[773.50,783.80] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2024-08-14 12:55:00 | 766.95 | 768.77 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:45:00 | 774.75 | 767.71 | 0.00 | ORB-long ORB[759.40,770.40] vol=3.3x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 09:55:00 | 780.08 | 771.73 | 0.00 | T1 1.5R @ 780.08 |
| Target hit | 2024-08-19 10:45:00 | 780.95 | 781.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 820.00 | 814.08 | 0.00 | ORB-long ORB[808.50,814.70] vol=2.7x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:55:00 | 824.47 | 819.18 | 0.00 | T1 1.5R @ 824.47 |
| Stop hit — per-position SL triggered | 2024-08-27 12:30:00 | 820.00 | 820.02 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:40:00 | 818.40 | 813.43 | 0.00 | ORB-long ORB[809.95,818.00] vol=3.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-08-29 10:55:00 | 815.12 | 815.32 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 825.45 | 829.39 | 0.00 | ORB-short ORB[826.20,833.00] vol=2.0x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-09-03 10:40:00 | 828.57 | 827.94 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:40:00 | 833.90 | 837.21 | 0.00 | ORB-short ORB[834.25,846.45] vol=2.0x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:45:00 | 830.22 | 835.64 | 0.00 | T1 1.5R @ 830.22 |
| Target hit | 2024-09-11 15:20:00 | 818.90 | 825.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-09-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:35:00 | 813.55 | 809.82 | 0.00 | ORB-long ORB[805.00,811.00] vol=1.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-09-13 10:40:00 | 810.94 | 810.24 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 811.25 | 816.99 | 0.00 | ORB-short ORB[818.05,826.45] vol=6.4x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:15:00 | 806.99 | 815.39 | 0.00 | T1 1.5R @ 806.99 |
| Stop hit — per-position SL triggered | 2024-09-18 11:45:00 | 811.25 | 813.84 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:15:00 | 801.00 | 817.48 | 0.00 | ORB-short ORB[818.50,828.00] vol=1.6x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:20:00 | 796.32 | 816.08 | 0.00 | T1 1.5R @ 796.32 |
| Stop hit — per-position SL triggered | 2024-09-19 12:35:00 | 801.00 | 808.76 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 831.90 | 827.09 | 0.00 | ORB-long ORB[819.05,829.00] vol=1.6x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-09-25 09:55:00 | 828.30 | 827.30 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:25:00 | 823.60 | 825.28 | 0.00 | ORB-short ORB[825.30,834.30] vol=1.6x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-09-26 11:35:00 | 825.75 | 824.52 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 863.15 | 859.17 | 0.00 | ORB-long ORB[852.55,862.85] vol=2.8x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:35:00 | 869.74 | 864.46 | 0.00 | T1 1.5R @ 869.74 |
| Target hit | 2024-10-01 10:15:00 | 865.85 | 869.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:55:00 | 834.50 | 828.54 | 0.00 | ORB-long ORB[824.45,832.00] vol=2.1x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-10-09 11:35:00 | 831.23 | 829.22 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 812.70 | 821.13 | 0.00 | ORB-short ORB[818.20,828.00] vol=2.4x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:40:00 | 808.51 | 815.08 | 0.00 | T1 1.5R @ 808.51 |
| Target hit | 2024-10-10 12:30:00 | 810.75 | 810.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2024-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:30:00 | 809.00 | 812.12 | 0.00 | ORB-short ORB[809.70,817.00] vol=1.5x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-10-11 09:40:00 | 811.21 | 811.99 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:40:00 | 817.70 | 813.67 | 0.00 | ORB-long ORB[812.00,817.40] vol=2.0x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 814.82 | 814.19 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:55:00 | 819.15 | 821.96 | 0.00 | ORB-short ORB[822.90,830.00] vol=3.0x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:30:00 | 815.08 | 820.09 | 0.00 | T1 1.5R @ 815.08 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 819.15 | 820.02 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:45:00 | 779.40 | 773.87 | 0.00 | ORB-long ORB[769.05,775.85] vol=4.3x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 11:00:00 | 783.66 | 775.71 | 0.00 | T1 1.5R @ 783.66 |
| Stop hit — per-position SL triggered | 2024-10-24 11:05:00 | 779.40 | 775.86 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 765.85 | 770.45 | 0.00 | ORB-short ORB[769.00,779.85] vol=4.4x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:45:00 | 760.65 | 768.40 | 0.00 | T1 1.5R @ 760.65 |
| Target hit | 2024-10-25 11:05:00 | 763.60 | 761.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2024-11-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:55:00 | 835.90 | 829.32 | 0.00 | ORB-long ORB[823.65,833.10] vol=2.0x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-11-22 10:45:00 | 832.38 | 831.68 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:20:00 | 921.50 | 914.31 | 0.00 | ORB-long ORB[905.00,915.00] vol=2.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:30:00 | 927.00 | 920.04 | 0.00 | T1 1.5R @ 927.00 |
| Target hit | 2024-12-12 15:20:00 | 945.95 | 930.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 893.85 | 899.01 | 0.00 | ORB-short ORB[903.50,910.25] vol=1.8x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:20:00 | 889.31 | 897.76 | 0.00 | T1 1.5R @ 889.31 |
| Stop hit — per-position SL triggered | 2024-12-26 14:05:00 | 893.85 | 895.00 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:40:00 | 876.00 | 878.66 | 0.00 | ORB-short ORB[876.40,885.95] vol=2.8x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:50:00 | 869.64 | 877.52 | 0.00 | T1 1.5R @ 869.64 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 876.00 | 876.34 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:25:00 | 907.85 | 902.05 | 0.00 | ORB-long ORB[893.00,904.60] vol=1.7x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:30:00 | 913.36 | 904.77 | 0.00 | T1 1.5R @ 913.36 |
| Target hit | 2025-01-01 15:20:00 | 918.35 | 912.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:15:00 | 860.00 | 867.60 | 0.00 | ORB-short ORB[860.60,872.60] vol=2.1x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:30:00 | 856.04 | 866.31 | 0.00 | T1 1.5R @ 856.04 |
| Stop hit — per-position SL triggered | 2025-01-09 11:40:00 | 860.00 | 865.67 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:45:00 | 834.35 | 825.37 | 0.00 | ORB-long ORB[815.50,825.20] vol=1.6x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:10:00 | 840.67 | 827.87 | 0.00 | T1 1.5R @ 840.67 |
| Stop hit — per-position SL triggered | 2025-01-23 11:50:00 | 834.35 | 834.36 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 820.95 | 817.49 | 0.00 | ORB-long ORB[809.00,820.80] vol=2.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-01-31 11:20:00 | 818.22 | 817.72 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 701.25 | 698.28 | 0.00 | ORB-long ORB[691.60,698.30] vol=3.1x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:35:00 | 705.08 | 702.08 | 0.00 | T1 1.5R @ 705.08 |
| Target hit | 2025-03-17 12:30:00 | 713.00 | 713.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-03-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:55:00 | 736.00 | 728.56 | 0.00 | ORB-long ORB[719.05,729.50] vol=1.7x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-03-18 11:00:00 | 732.22 | 729.16 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 827.30 | 832.92 | 0.00 | ORB-short ORB[833.00,842.00] vol=2.0x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-04-16 10:00:00 | 831.06 | 831.96 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:35:00 | 858.20 | 851.80 | 0.00 | ORB-long ORB[844.90,857.20] vol=2.0x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-04-22 12:40:00 | 854.50 | 854.25 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:30:00 | 870.25 | 853.82 | 0.00 | ORB-long ORB[842.00,852.25] vol=3.9x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-04-24 10:35:00 | 866.56 | 855.37 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:50:00 | 854.50 | 849.80 | 0.00 | ORB-long ORB[844.15,854.45] vol=1.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:55:00 | 858.33 | 853.02 | 0.00 | T1 1.5R @ 858.33 |
| Target hit | 2025-05-08 12:15:00 | 856.05 | 856.68 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:35:00 | 621.20 | 2024-05-14 09:45:00 | 618.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-15 09:30:00 | 626.95 | 2024-05-15 10:05:00 | 624.32 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-05-16 09:50:00 | 626.00 | 2024-05-16 10:10:00 | 629.15 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-05-16 09:50:00 | 626.00 | 2024-05-16 10:50:00 | 626.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 10:20:00 | 631.75 | 2024-05-28 11:45:00 | 628.25 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-28 10:20:00 | 631.75 | 2024-05-28 15:00:00 | 631.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:20:00 | 707.70 | 2024-06-12 15:20:00 | 705.00 | TARGET_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2024-06-13 09:30:00 | 705.75 | 2024-06-13 09:45:00 | 702.07 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-13 09:30:00 | 705.75 | 2024-06-13 10:30:00 | 705.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:35:00 | 757.80 | 2024-06-27 10:50:00 | 753.79 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-27 10:35:00 | 757.80 | 2024-06-27 10:55:00 | 757.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 10:00:00 | 766.35 | 2024-07-01 13:55:00 | 770.79 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-01 10:00:00 | 766.35 | 2024-07-01 15:15:00 | 767.05 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2024-07-03 11:05:00 | 756.75 | 2024-07-03 11:50:00 | 759.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-05 10:45:00 | 760.00 | 2024-07-05 11:45:00 | 756.53 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-05 10:45:00 | 760.00 | 2024-07-05 11:55:00 | 760.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 09:40:00 | 765.90 | 2024-07-08 09:45:00 | 771.20 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-08 09:40:00 | 765.90 | 2024-07-08 11:35:00 | 779.30 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2024-07-10 10:15:00 | 756.45 | 2024-07-10 10:20:00 | 750.37 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-07-10 10:15:00 | 756.45 | 2024-07-10 10:45:00 | 756.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 09:30:00 | 798.10 | 2024-07-12 09:40:00 | 794.53 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-18 09:35:00 | 788.70 | 2024-07-18 09:40:00 | 791.84 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-26 10:40:00 | 782.40 | 2024-07-26 10:50:00 | 786.38 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-26 10:40:00 | 782.40 | 2024-07-26 11:20:00 | 782.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-01 10:05:00 | 819.70 | 2024-08-01 10:20:00 | 815.05 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-01 10:05:00 | 819.70 | 2024-08-01 15:20:00 | 799.00 | TARGET_HIT | 0.50 | 2.53% |
| BUY | retest1 | 2024-08-06 10:00:00 | 779.40 | 2024-08-06 10:05:00 | 775.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-08-08 09:30:00 | 767.70 | 2024-08-08 09:45:00 | 772.79 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-08-08 09:30:00 | 767.70 | 2024-08-08 10:00:00 | 767.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 09:50:00 | 779.00 | 2024-08-09 10:00:00 | 774.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-08-14 11:15:00 | 763.90 | 2024-08-14 12:55:00 | 766.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-19 09:45:00 | 774.75 | 2024-08-19 09:55:00 | 780.08 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-08-19 09:45:00 | 774.75 | 2024-08-19 10:45:00 | 780.95 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2024-08-27 09:40:00 | 820.00 | 2024-08-27 10:55:00 | 824.47 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-27 09:40:00 | 820.00 | 2024-08-27 12:30:00 | 820.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 10:40:00 | 818.40 | 2024-08-29 10:55:00 | 815.12 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-03 09:45:00 | 825.45 | 2024-09-03 10:40:00 | 828.57 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-11 10:40:00 | 833.90 | 2024-09-11 11:45:00 | 830.22 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-11 10:40:00 | 833.90 | 2024-09-11 15:20:00 | 818.90 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2024-09-13 10:35:00 | 813.55 | 2024-09-13 10:40:00 | 810.94 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-18 10:50:00 | 811.25 | 2024-09-18 11:15:00 | 806.99 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-18 10:50:00 | 811.25 | 2024-09-18 11:45:00 | 811.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:15:00 | 801.00 | 2024-09-19 11:20:00 | 796.32 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-19 11:15:00 | 801.00 | 2024-09-19 12:35:00 | 801.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 09:45:00 | 831.90 | 2024-09-25 09:55:00 | 828.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-09-26 10:25:00 | 823.60 | 2024-09-26 11:35:00 | 825.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-01 09:30:00 | 863.15 | 2024-10-01 09:35:00 | 869.74 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-10-01 09:30:00 | 863.15 | 2024-10-01 10:15:00 | 865.85 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-10-09 10:55:00 | 834.50 | 2024-10-09 11:35:00 | 831.23 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-10 11:10:00 | 812.70 | 2024-10-10 11:40:00 | 808.51 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-10 11:10:00 | 812.70 | 2024-10-10 12:30:00 | 810.75 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-10-11 09:30:00 | 809.00 | 2024-10-11 09:40:00 | 811.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-15 09:40:00 | 817.70 | 2024-10-15 09:50:00 | 814.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-16 09:55:00 | 819.15 | 2024-10-16 10:30:00 | 815.08 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-16 09:55:00 | 819.15 | 2024-10-16 10:35:00 | 819.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-24 10:45:00 | 779.40 | 2024-10-24 11:00:00 | 783.66 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-24 10:45:00 | 779.40 | 2024-10-24 11:05:00 | 779.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 09:35:00 | 765.85 | 2024-10-25 09:45:00 | 760.65 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-10-25 09:35:00 | 765.85 | 2024-10-25 11:05:00 | 763.60 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-11-22 09:55:00 | 835.90 | 2024-11-22 10:45:00 | 832.38 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-12 10:20:00 | 921.50 | 2024-12-12 11:30:00 | 927.00 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-12-12 10:20:00 | 921.50 | 2024-12-12 15:20:00 | 945.95 | TARGET_HIT | 0.50 | 2.65% |
| SELL | retest1 | 2024-12-26 11:00:00 | 893.85 | 2024-12-26 12:20:00 | 889.31 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-12-26 11:00:00 | 893.85 | 2024-12-26 14:05:00 | 893.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-31 09:40:00 | 876.00 | 2024-12-31 09:50:00 | 869.64 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-12-31 09:40:00 | 876.00 | 2024-12-31 10:15:00 | 876.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:25:00 | 907.85 | 2025-01-01 10:30:00 | 913.36 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-01-01 10:25:00 | 907.85 | 2025-01-01 15:20:00 | 918.35 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-01-09 11:15:00 | 860.00 | 2025-01-09 11:30:00 | 856.04 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-09 11:15:00 | 860.00 | 2025-01-09 11:40:00 | 860.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 09:45:00 | 834.35 | 2025-01-23 10:10:00 | 840.67 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-01-23 09:45:00 | 834.35 | 2025-01-23 11:50:00 | 834.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 11:00:00 | 820.95 | 2025-01-31 11:20:00 | 818.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-17 09:30:00 | 701.25 | 2025-03-17 09:35:00 | 705.08 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-17 09:30:00 | 701.25 | 2025-03-17 12:30:00 | 713.00 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-03-18 10:55:00 | 736.00 | 2025-03-18 11:00:00 | 732.22 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-04-16 09:55:00 | 827.30 | 2025-04-16 10:00:00 | 831.06 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-22 10:35:00 | 858.20 | 2025-04-22 12:40:00 | 854.50 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-04-24 10:30:00 | 870.25 | 2025-04-24 10:35:00 | 866.56 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-08 10:50:00 | 854.50 | 2025-05-08 10:55:00 | 858.33 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-08 10:50:00 | 854.50 | 2025-05-08 12:15:00 | 856.05 | TARGET_HIT | 0.50 | 0.18% |
