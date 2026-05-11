# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36870 bars)
- **Last close:** 738.40
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
| PARTIAL | 22 |
| TARGET_HIT | 14 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 39
- **Target hits / Stop hits / Partials:** 14 / 39 / 22
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 18.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 12 | 46.2% | 4 | 14 | 8 | 0.28% | 7.3% |
| BUY @ 2nd Alert (retest1) | 26 | 12 | 46.2% | 4 | 14 | 8 | 0.28% | 7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 24 | 49.0% | 10 | 25 | 14 | 0.23% | 11.2% |
| SELL @ 2nd Alert (retest1) | 49 | 24 | 49.0% | 10 | 25 | 14 | 0.23% | 11.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 36 | 48.0% | 14 | 39 | 22 | 0.25% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 689.60 | 694.34 | 0.00 | ORB-short ORB[696.55,702.00] vol=3.2x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:00:00 | 685.45 | 691.79 | 0.00 | T1 1.5R @ 685.45 |
| Target hit | 2024-05-15 12:15:00 | 687.90 | 687.64 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 685.55 | 688.41 | 0.00 | ORB-short ORB[687.30,695.45] vol=3.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 688.82 | 688.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:45:00 | 684.45 | 687.55 | 0.00 | ORB-short ORB[686.90,691.00] vol=1.9x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:00:00 | 681.67 | 685.62 | 0.00 | T1 1.5R @ 681.67 |
| Target hit | 2024-05-23 11:50:00 | 684.35 | 683.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:40:00 | 681.95 | 687.04 | 0.00 | ORB-short ORB[685.55,692.70] vol=4.6x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 11:05:00 | 679.00 | 684.21 | 0.00 | T1 1.5R @ 679.00 |
| Target hit | 2024-05-27 15:20:00 | 676.70 | 679.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 671.50 | 674.16 | 0.00 | ORB-short ORB[675.40,681.70] vol=1.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-05-28 10:30:00 | 674.25 | 673.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 656.40 | 660.95 | 0.00 | ORB-short ORB[659.45,664.45] vol=1.7x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-05-30 09:45:00 | 658.67 | 659.50 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:35:00 | 650.15 | 652.27 | 0.00 | ORB-short ORB[653.35,656.80] vol=7.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-05-31 10:05:00 | 652.11 | 651.57 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 739.10 | 735.98 | 0.00 | ORB-long ORB[732.60,737.70] vol=1.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 735.98 | 737.15 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:40:00 | 727.85 | 729.45 | 0.00 | ORB-short ORB[728.00,734.00] vol=3.3x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:45:00 | 724.11 | 726.30 | 0.00 | T1 1.5R @ 724.11 |
| Target hit | 2024-06-19 11:00:00 | 724.95 | 724.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 855.60 | 859.04 | 0.00 | ORB-short ORB[858.90,867.00] vol=2.5x ATR=2.95 |
| Stop hit — per-position SL triggered | 2024-07-15 09:40:00 | 858.55 | 858.63 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 886.90 | 883.47 | 0.00 | ORB-long ORB[878.10,884.35] vol=2.7x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:25:00 | 891.20 | 886.14 | 0.00 | T1 1.5R @ 891.20 |
| Stop hit — per-position SL triggered | 2024-07-26 11:00:00 | 886.90 | 887.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 810.00 | 806.73 | 0.00 | ORB-long ORB[795.70,807.05] vol=1.9x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:45:00 | 814.58 | 813.44 | 0.00 | T1 1.5R @ 814.58 |
| Stop hit — per-position SL triggered | 2024-08-07 12:30:00 | 810.00 | 813.52 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:05:00 | 823.80 | 829.04 | 0.00 | ORB-short ORB[828.10,839.00] vol=3.4x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:15:00 | 819.42 | 827.68 | 0.00 | T1 1.5R @ 819.42 |
| Target hit | 2024-08-08 15:20:00 | 811.45 | 817.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-08-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:10:00 | 815.00 | 817.73 | 0.00 | ORB-short ORB[815.75,822.35] vol=2.4x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:25:00 | 811.28 | 816.31 | 0.00 | T1 1.5R @ 811.28 |
| Target hit | 2024-08-09 10:35:00 | 811.85 | 811.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-08-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:00:00 | 872.90 | 868.61 | 0.00 | ORB-long ORB[862.40,872.55] vol=5.7x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:15:00 | 878.61 | 871.57 | 0.00 | T1 1.5R @ 878.61 |
| Target hit | 2024-08-19 15:20:00 | 895.05 | 885.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 878.45 | 882.54 | 0.00 | ORB-short ORB[878.75,890.05] vol=1.8x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 13:25:00 | 873.76 | 879.60 | 0.00 | T1 1.5R @ 873.76 |
| Target hit | 2024-08-23 15:20:00 | 869.55 | 876.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 860.15 | 866.20 | 0.00 | ORB-short ORB[861.00,873.00] vol=2.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-08-29 11:00:00 | 862.88 | 865.56 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:20:00 | 860.20 | 863.65 | 0.00 | ORB-short ORB[865.80,875.50] vol=3.9x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-08-30 10:35:00 | 863.42 | 863.60 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:10:00 | 878.90 | 870.00 | 0.00 | ORB-long ORB[863.70,875.75] vol=3.0x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 874.98 | 870.32 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 873.10 | 870.38 | 0.00 | ORB-long ORB[865.00,870.90] vol=4.0x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:35:00 | 877.63 | 872.27 | 0.00 | T1 1.5R @ 877.63 |
| Target hit | 2024-09-06 11:50:00 | 878.70 | 878.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2024-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:05:00 | 856.00 | 861.74 | 0.00 | ORB-short ORB[858.65,868.95] vol=2.3x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 858.40 | 861.69 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:15:00 | 863.60 | 857.42 | 0.00 | ORB-long ORB[851.00,857.05] vol=5.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:20:00 | 867.30 | 857.71 | 0.00 | T1 1.5R @ 867.30 |
| Stop hit — per-position SL triggered | 2024-09-11 10:25:00 | 863.60 | 857.89 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 870.20 | 874.06 | 0.00 | ORB-short ORB[874.20,883.00] vol=1.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-09-18 09:50:00 | 872.68 | 873.79 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:15:00 | 872.35 | 880.55 | 0.00 | ORB-short ORB[883.50,896.45] vol=3.3x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-09-19 12:10:00 | 876.15 | 878.60 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:45:00 | 875.60 | 875.98 | 0.00 | ORB-short ORB[877.70,890.45] vol=1.7x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-09-20 12:40:00 | 878.95 | 875.80 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 868.00 | 873.11 | 0.00 | ORB-short ORB[870.00,880.00] vol=2.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-09-25 09:45:00 | 870.83 | 871.74 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:45:00 | 866.10 | 872.78 | 0.00 | ORB-short ORB[874.00,885.10] vol=5.7x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-09-30 10:55:00 | 868.59 | 871.32 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:35:00 | 817.55 | 823.86 | 0.00 | ORB-short ORB[820.10,830.80] vol=2.4x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:40:00 | 812.02 | 822.09 | 0.00 | T1 1.5R @ 812.02 |
| Target hit | 2024-10-07 15:20:00 | 780.00 | 798.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 788.60 | 790.84 | 0.00 | ORB-short ORB[789.15,795.65] vol=2.1x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:00:00 | 784.57 | 789.05 | 0.00 | T1 1.5R @ 784.57 |
| Target hit | 2024-10-14 11:20:00 | 785.60 | 784.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:35:00 | 784.05 | 780.06 | 0.00 | ORB-long ORB[776.10,781.80] vol=2.0x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-10-15 10:40:00 | 782.46 | 780.13 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 776.05 | 778.55 | 0.00 | ORB-short ORB[776.30,785.50] vol=3.4x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:20:00 | 773.15 | 777.14 | 0.00 | T1 1.5R @ 773.15 |
| Stop hit — per-position SL triggered | 2024-10-16 13:05:00 | 776.05 | 775.33 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 781.00 | 789.33 | 0.00 | ORB-short ORB[794.95,803.25] vol=2.8x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 776.69 | 786.24 | 0.00 | T1 1.5R @ 776.69 |
| Stop hit — per-position SL triggered | 2024-10-17 13:30:00 | 781.00 | 784.01 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:30:00 | 768.25 | 775.42 | 0.00 | ORB-short ORB[771.00,781.85] vol=3.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-11-07 10:35:00 | 770.76 | 772.93 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:35:00 | 698.10 | 696.70 | 0.00 | ORB-long ORB[688.35,697.00] vol=3.4x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:45:00 | 702.32 | 698.93 | 0.00 | T1 1.5R @ 702.32 |
| Stop hit — per-position SL triggered | 2024-11-19 09:50:00 | 698.10 | 698.93 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 11:10:00 | 773.65 | 766.91 | 0.00 | ORB-long ORB[760.00,767.75] vol=3.3x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-12-05 11:15:00 | 769.95 | 767.24 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:25:00 | 792.05 | 788.67 | 0.00 | ORB-long ORB[775.25,785.05] vol=1.8x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-12-10 10:30:00 | 789.23 | 788.69 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 723.95 | 727.47 | 0.00 | ORB-short ORB[726.15,734.45] vol=7.0x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:35:00 | 720.44 | 726.64 | 0.00 | T1 1.5R @ 720.44 |
| Target hit | 2024-12-17 15:20:00 | 709.40 | 717.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 679.60 | 682.47 | 0.00 | ORB-short ORB[681.80,691.55] vol=6.8x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 09:40:00 | 676.54 | 681.63 | 0.00 | T1 1.5R @ 676.54 |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 679.60 | 679.47 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:15:00 | 663.00 | 657.71 | 0.00 | ORB-long ORB[653.00,661.50] vol=1.5x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-01-08 10:20:00 | 660.15 | 657.74 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 09:30:00 | 640.10 | 635.67 | 0.00 | ORB-long ORB[630.00,637.00] vol=5.2x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-01-13 09:35:00 | 636.84 | 636.07 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 589.10 | 585.44 | 0.00 | ORB-long ORB[578.70,584.90] vol=1.9x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:35:00 | 593.97 | 586.44 | 0.00 | T1 1.5R @ 593.97 |
| Target hit | 2025-01-29 10:35:00 | 607.85 | 608.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 621.05 | 617.95 | 0.00 | ORB-long ORB[607.00,612.65] vol=3.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-02-01 11:05:00 | 619.52 | 617.86 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-02-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-03 09:45:00 | 623.20 | 618.56 | 0.00 | ORB-long ORB[611.45,620.55] vol=3.8x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-02-03 09:55:00 | 620.15 | 618.62 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:10:00 | 608.05 | 611.65 | 0.00 | ORB-short ORB[612.25,616.45] vol=1.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-02-04 12:55:00 | 610.27 | 608.30 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 10:45:00 | 601.30 | 604.54 | 0.00 | ORB-short ORB[602.25,608.85] vol=1.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-03-19 12:25:00 | 603.58 | 601.64 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 616.25 | 625.20 | 0.00 | ORB-short ORB[622.05,630.45] vol=1.8x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:35:00 | 612.42 | 623.78 | 0.00 | T1 1.5R @ 612.42 |
| Stop hit — per-position SL triggered | 2025-04-01 12:20:00 | 616.25 | 622.96 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 10:05:00 | 593.90 | 598.59 | 0.00 | ORB-short ORB[596.95,603.15] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-04-11 11:05:00 | 596.29 | 597.70 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-04-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:05:00 | 627.00 | 628.16 | 0.00 | ORB-short ORB[627.85,634.75] vol=6.5x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-04-16 10:30:00 | 628.77 | 628.14 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:00:00 | 637.00 | 633.08 | 0.00 | ORB-long ORB[627.05,635.20] vol=1.8x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:05:00 | 640.18 | 636.12 | 0.00 | T1 1.5R @ 640.18 |
| Target hit | 2025-04-17 13:10:00 | 638.75 | 641.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-21 09:45:00 | 628.80 | 630.49 | 0.00 | ORB-short ORB[629.60,634.60] vol=3.1x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-04-21 09:50:00 | 630.59 | 630.47 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 603.50 | 607.66 | 0.00 | ORB-short ORB[605.75,613.60] vol=2.3x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 605.12 | 606.55 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:00:00 | 650.15 | 647.90 | 0.00 | ORB-long ORB[638.85,647.00] vol=2.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-05-05 11:20:00 | 647.80 | 648.80 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:50:00 | 646.00 | 649.83 | 0.00 | ORB-short ORB[647.00,656.70] vol=1.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-05-06 09:55:00 | 648.36 | 649.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:05:00 | 689.60 | 2024-05-15 11:00:00 | 685.45 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-15 10:05:00 | 689.60 | 2024-05-15 12:15:00 | 687.90 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-05-22 09:40:00 | 685.55 | 2024-05-22 09:50:00 | 688.82 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-05-23 09:45:00 | 684.45 | 2024-05-23 10:00:00 | 681.67 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-23 09:45:00 | 684.45 | 2024-05-23 11:50:00 | 684.35 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-05-27 10:40:00 | 681.95 | 2024-05-27 11:05:00 | 679.00 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-27 10:40:00 | 681.95 | 2024-05-27 15:20:00 | 676.70 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2024-05-28 09:30:00 | 671.50 | 2024-05-28 10:30:00 | 674.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-30 09:30:00 | 656.40 | 2024-05-30 09:45:00 | 658.67 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-31 09:35:00 | 650.15 | 2024-05-31 10:05:00 | 652.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-13 09:35:00 | 739.10 | 2024-06-13 11:05:00 | 735.98 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-19 09:40:00 | 727.85 | 2024-06-19 09:45:00 | 724.11 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-19 09:40:00 | 727.85 | 2024-06-19 11:00:00 | 724.95 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-15 09:30:00 | 855.60 | 2024-07-15 09:40:00 | 858.55 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-26 09:50:00 | 886.90 | 2024-07-26 10:25:00 | 891.20 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-26 09:50:00 | 886.90 | 2024-07-26 11:00:00 | 886.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-07 10:45:00 | 810.00 | 2024-08-07 11:45:00 | 814.58 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-07 10:45:00 | 810.00 | 2024-08-07 12:30:00 | 810.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 11:05:00 | 823.80 | 2024-08-08 11:15:00 | 819.42 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-08 11:05:00 | 823.80 | 2024-08-08 15:20:00 | 811.45 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-08-09 10:10:00 | 815.00 | 2024-08-09 10:25:00 | 811.28 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-09 10:10:00 | 815.00 | 2024-08-09 10:35:00 | 811.85 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-19 10:00:00 | 872.90 | 2024-08-19 10:15:00 | 878.61 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-08-19 10:00:00 | 872.90 | 2024-08-19 15:20:00 | 895.05 | TARGET_HIT | 0.50 | 2.54% |
| SELL | retest1 | 2024-08-23 09:50:00 | 878.45 | 2024-08-23 13:25:00 | 873.76 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-23 09:50:00 | 878.45 | 2024-08-23 15:20:00 | 869.55 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2024-08-29 10:50:00 | 860.15 | 2024-08-29 11:00:00 | 862.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-30 10:20:00 | 860.20 | 2024-08-30 10:35:00 | 863.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-02 10:10:00 | 878.90 | 2024-09-02 10:15:00 | 874.98 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-06 09:35:00 | 873.10 | 2024-09-06 10:35:00 | 877.63 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-06 09:35:00 | 873.10 | 2024-09-06 11:50:00 | 878.70 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-09-10 11:05:00 | 856.00 | 2024-09-10 11:15:00 | 858.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-11 10:15:00 | 863.60 | 2024-09-11 10:20:00 | 867.30 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-11 10:15:00 | 863.60 | 2024-09-11 10:25:00 | 863.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 09:45:00 | 870.20 | 2024-09-18 09:50:00 | 872.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-19 11:15:00 | 872.35 | 2024-09-19 12:10:00 | 876.15 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-20 10:45:00 | 875.60 | 2024-09-20 12:40:00 | 878.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-25 09:30:00 | 868.00 | 2024-09-25 09:45:00 | 870.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-30 10:45:00 | 866.10 | 2024-09-30 10:55:00 | 868.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-07 09:35:00 | 817.55 | 2024-10-07 09:40:00 | 812.02 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-10-07 09:35:00 | 817.55 | 2024-10-07 15:20:00 | 780.00 | TARGET_HIT | 0.50 | 4.59% |
| SELL | retest1 | 2024-10-14 09:30:00 | 788.60 | 2024-10-14 10:00:00 | 784.57 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-14 09:30:00 | 788.60 | 2024-10-14 11:20:00 | 785.60 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2024-10-15 10:35:00 | 784.05 | 2024-10-15 10:40:00 | 782.46 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-16 10:00:00 | 776.05 | 2024-10-16 10:20:00 | 773.15 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-16 10:00:00 | 776.05 | 2024-10-16 13:05:00 | 776.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 11:00:00 | 781.00 | 2024-10-17 11:25:00 | 776.69 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-17 11:00:00 | 781.00 | 2024-10-17 13:30:00 | 781.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:30:00 | 768.25 | 2024-11-07 10:35:00 | 770.76 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-19 09:35:00 | 698.10 | 2024-11-19 09:45:00 | 702.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-11-19 09:35:00 | 698.10 | 2024-11-19 09:50:00 | 698.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-05 11:10:00 | 773.65 | 2024-12-05 11:15:00 | 769.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-12-10 10:25:00 | 792.05 | 2024-12-10 10:30:00 | 789.23 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-17 09:30:00 | 723.95 | 2024-12-17 09:35:00 | 720.44 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-17 09:30:00 | 723.95 | 2024-12-17 15:20:00 | 709.40 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2024-12-27 09:30:00 | 679.60 | 2024-12-27 09:40:00 | 676.54 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-27 09:30:00 | 679.60 | 2024-12-27 10:15:00 | 679.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-08 10:15:00 | 663.00 | 2025-01-08 10:20:00 | 660.15 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-13 09:30:00 | 640.10 | 2025-01-13 09:35:00 | 636.84 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-29 09:30:00 | 589.10 | 2025-01-29 09:35:00 | 593.97 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-01-29 09:30:00 | 589.10 | 2025-01-29 10:35:00 | 607.85 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2025-02-01 11:00:00 | 621.05 | 2025-02-01 11:05:00 | 619.52 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-02-03 09:45:00 | 623.20 | 2025-02-03 09:55:00 | 620.15 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-02-04 10:10:00 | 608.05 | 2025-02-04 12:55:00 | 610.27 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-03-19 10:45:00 | 601.30 | 2025-03-19 12:25:00 | 603.58 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-04-01 10:55:00 | 616.25 | 2025-04-01 11:35:00 | 612.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-01 10:55:00 | 616.25 | 2025-04-01 12:20:00 | 616.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-11 10:05:00 | 593.90 | 2025-04-11 11:05:00 | 596.29 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-16 10:05:00 | 627.00 | 2025-04-16 10:30:00 | 628.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-17 10:00:00 | 637.00 | 2025-04-17 11:05:00 | 640.18 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-17 10:00:00 | 637.00 | 2025-04-17 13:10:00 | 638.75 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-04-21 09:45:00 | 628.80 | 2025-04-21 09:50:00 | 630.59 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-23 09:30:00 | 603.50 | 2025-04-23 09:40:00 | 605.12 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-05 10:00:00 | 650.15 | 2025-05-05 11:20:00 | 647.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-05-06 09:50:00 | 646.00 | 2025-05-06 09:55:00 | 648.36 | STOP_HIT | 1.00 | -0.37% |
