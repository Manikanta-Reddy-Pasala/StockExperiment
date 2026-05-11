# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 878.10
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 11 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 79
- **Target hits / Stop hits / Partials:** 11 / 79 / 30
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 6.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 15 | 28.8% | 3 | 37 | 12 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 52 | 15 | 28.8% | 3 | 37 | 12 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 26 | 38.2% | 8 | 42 | 18 | 0.10% | 6.9% |
| SELL @ 2nd Alert (retest1) | 68 | 26 | 38.2% | 8 | 42 | 18 | 0.10% | 6.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 41 | 34.2% | 11 | 79 | 30 | 0.06% | 6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:35:00 | 724.55 | 728.61 | 0.00 | ORB-short ORB[730.00,739.75] vol=2.4x ATR=4.84 |
| Stop hit — per-position SL triggered | 2024-05-13 11:20:00 | 729.39 | 727.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:30:00 | 748.05 | 743.01 | 0.00 | ORB-long ORB[737.85,745.40] vol=2.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-05-14 10:50:00 | 745.64 | 743.54 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 755.95 | 753.10 | 0.00 | ORB-long ORB[748.20,755.00] vol=3.2x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-05-15 10:00:00 | 753.89 | 753.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 755.60 | 758.81 | 0.00 | ORB-short ORB[759.00,765.95] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-05-17 11:00:00 | 757.04 | 758.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 752.25 | 754.92 | 0.00 | ORB-short ORB[752.55,759.95] vol=1.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-05-21 09:35:00 | 754.62 | 754.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 738.90 | 743.08 | 0.00 | ORB-short ORB[744.90,752.60] vol=1.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 741.12 | 741.64 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:40:00 | 735.55 | 739.16 | 0.00 | ORB-short ORB[736.10,744.30] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-05-23 10:50:00 | 737.34 | 739.03 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 740.20 | 742.72 | 0.00 | ORB-short ORB[743.50,747.20] vol=2.7x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 14:30:00 | 737.70 | 740.86 | 0.00 | T1 1.5R @ 737.70 |
| Target hit | 2024-05-24 15:20:00 | 738.15 | 740.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 751.10 | 748.06 | 0.00 | ORB-long ORB[740.20,748.00] vol=4.0x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:45:00 | 754.71 | 750.66 | 0.00 | T1 1.5R @ 754.71 |
| Stop hit — per-position SL triggered | 2024-05-28 09:55:00 | 751.10 | 751.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:15:00 | 736.50 | 739.87 | 0.00 | ORB-short ORB[738.35,743.90] vol=1.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:30:00 | 733.83 | 737.58 | 0.00 | T1 1.5R @ 733.83 |
| Stop hit — per-position SL triggered | 2024-05-29 10:40:00 | 736.50 | 736.91 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 11:00:00 | 727.00 | 721.71 | 0.00 | ORB-long ORB[718.10,726.90] vol=2.2x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-05-30 11:20:00 | 724.99 | 722.56 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:55:00 | 785.25 | 779.04 | 0.00 | ORB-long ORB[774.30,780.90] vol=7.6x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-06-10 11:00:00 | 782.38 | 779.41 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 776.00 | 780.90 | 0.00 | ORB-short ORB[778.25,784.70] vol=2.2x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 778.93 | 780.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:05:00 | 804.65 | 804.74 | 0.00 | ORB-short ORB[804.80,815.00] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-06-13 11:10:00 | 806.54 | 804.94 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 825.00 | 821.71 | 0.00 | ORB-long ORB[816.50,824.40] vol=2.0x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-06-14 10:45:00 | 822.43 | 823.13 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 848.45 | 842.05 | 0.00 | ORB-long ORB[834.30,844.80] vol=1.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-06-18 09:45:00 | 845.21 | 843.98 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:05:00 | 871.10 | 860.21 | 0.00 | ORB-long ORB[850.05,862.50] vol=2.4x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-06-21 10:35:00 | 867.88 | 864.13 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 875.50 | 881.38 | 0.00 | ORB-short ORB[877.30,887.00] vol=2.5x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 871.62 | 880.33 | 0.00 | T1 1.5R @ 871.62 |
| Stop hit — per-position SL triggered | 2024-06-25 14:20:00 | 875.50 | 875.35 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:45:00 | 900.95 | 893.05 | 0.00 | ORB-long ORB[886.90,897.00] vol=2.9x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:55:00 | 906.91 | 897.34 | 0.00 | T1 1.5R @ 906.91 |
| Stop hit — per-position SL triggered | 2024-06-26 11:35:00 | 900.95 | 899.09 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 912.30 | 903.80 | 0.00 | ORB-long ORB[897.00,907.85] vol=3.8x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:50:00 | 918.71 | 909.90 | 0.00 | T1 1.5R @ 918.71 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 912.30 | 912.48 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-06-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:50:00 | 921.40 | 916.21 | 0.00 | ORB-long ORB[910.20,918.00] vol=1.5x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 918.03 | 916.46 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 933.05 | 925.93 | 0.00 | ORB-long ORB[918.90,928.80] vol=3.1x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-07-01 09:40:00 | 929.12 | 930.54 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 889.15 | 891.76 | 0.00 | ORB-short ORB[890.30,895.40] vol=1.8x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-07-05 09:35:00 | 891.43 | 891.57 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:50:00 | 895.60 | 890.24 | 0.00 | ORB-long ORB[884.50,891.50] vol=3.1x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-07-08 09:55:00 | 892.34 | 890.87 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 883.60 | 894.75 | 0.00 | ORB-short ORB[890.10,902.00] vol=2.1x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:50:00 | 877.75 | 892.49 | 0.00 | T1 1.5R @ 877.75 |
| Stop hit — per-position SL triggered | 2024-07-09 10:05:00 | 883.60 | 891.54 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 867.50 | 873.85 | 0.00 | ORB-short ORB[880.00,884.10] vol=2.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 870.88 | 873.46 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:40:00 | 856.50 | 860.23 | 0.00 | ORB-short ORB[856.90,869.00] vol=2.8x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-07-15 09:45:00 | 859.59 | 860.16 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:35:00 | 891.60 | 888.64 | 0.00 | ORB-long ORB[878.50,889.95] vol=1.8x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-07-16 09:45:00 | 888.39 | 888.98 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:00:00 | 860.25 | 850.16 | 0.00 | ORB-long ORB[847.20,858.45] vol=1.9x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 856.54 | 851.25 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-07-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:55:00 | 848.95 | 851.19 | 0.00 | ORB-short ORB[851.00,857.85] vol=3.5x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-07-30 10:20:00 | 851.21 | 850.51 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:30:00 | 845.05 | 848.53 | 0.00 | ORB-short ORB[848.50,853.45] vol=1.7x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:45:00 | 842.26 | 846.39 | 0.00 | T1 1.5R @ 842.26 |
| Target hit | 2024-08-01 15:20:00 | 834.80 | 837.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:10:00 | 789.95 | 784.65 | 0.00 | ORB-long ORB[781.00,787.90] vol=3.2x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-08-08 10:30:00 | 787.19 | 786.50 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 802.25 | 797.46 | 0.00 | ORB-long ORB[790.05,798.40] vol=2.1x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 799.14 | 800.70 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:00:00 | 806.45 | 809.33 | 0.00 | ORB-short ORB[808.55,818.20] vol=1.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 808.75 | 809.30 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:15:00 | 821.00 | 814.40 | 0.00 | ORB-long ORB[805.40,814.70] vol=2.0x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-08-16 11:10:00 | 818.43 | 817.52 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 849.75 | 846.95 | 0.00 | ORB-long ORB[840.65,849.50] vol=1.9x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-08-20 09:55:00 | 847.50 | 847.13 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 856.90 | 853.58 | 0.00 | ORB-long ORB[844.10,856.00] vol=1.7x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 854.09 | 854.02 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-08-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 11:05:00 | 851.00 | 848.32 | 0.00 | ORB-long ORB[844.05,849.90] vol=3.4x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:15:00 | 853.48 | 850.59 | 0.00 | T1 1.5R @ 853.48 |
| Stop hit — per-position SL triggered | 2024-08-23 11:25:00 | 851.00 | 850.62 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-08-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:45:00 | 845.65 | 849.82 | 0.00 | ORB-short ORB[847.55,856.95] vol=2.4x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-08-26 12:25:00 | 847.63 | 848.62 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:15:00 | 871.75 | 864.69 | 0.00 | ORB-long ORB[861.30,867.35] vol=3.1x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-08-28 10:55:00 | 868.66 | 869.34 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:20:00 | 883.20 | 887.75 | 0.00 | ORB-short ORB[886.85,899.50] vol=2.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-08-30 10:25:00 | 886.68 | 887.58 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-09-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:55:00 | 928.35 | 918.53 | 0.00 | ORB-long ORB[909.35,918.35] vol=1.9x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:00:00 | 935.45 | 921.63 | 0.00 | T1 1.5R @ 935.45 |
| Target hit | 2024-09-13 12:05:00 | 933.50 | 934.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2024-09-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:35:00 | 931.80 | 937.94 | 0.00 | ORB-short ORB[935.10,942.70] vol=2.1x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 936.69 | 937.81 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 876.55 | 883.85 | 0.00 | ORB-short ORB[880.00,889.20] vol=1.5x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:50:00 | 871.41 | 879.27 | 0.00 | T1 1.5R @ 871.41 |
| Target hit | 2024-09-19 15:20:00 | 852.95 | 854.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 871.85 | 880.02 | 0.00 | ORB-short ORB[877.15,888.65] vol=1.7x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:00:00 | 867.26 | 874.97 | 0.00 | T1 1.5R @ 867.26 |
| Target hit | 2024-09-24 15:20:00 | 861.60 | 867.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-09-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:30:00 | 865.05 | 862.62 | 0.00 | ORB-long ORB[857.65,865.00] vol=2.5x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:50:00 | 869.37 | 863.27 | 0.00 | T1 1.5R @ 869.37 |
| Stop hit — per-position SL triggered | 2024-09-25 11:50:00 | 865.05 | 864.45 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-09-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:25:00 | 875.05 | 876.81 | 0.00 | ORB-short ORB[875.35,885.00] vol=2.2x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-09-30 10:40:00 | 877.57 | 876.15 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:40:00 | 882.70 | 887.84 | 0.00 | ORB-short ORB[885.30,892.90] vol=2.5x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:00:00 | 878.10 | 886.56 | 0.00 | T1 1.5R @ 878.10 |
| Target hit | 2024-10-03 15:00:00 | 879.30 | 878.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 835.80 | 842.99 | 0.00 | ORB-short ORB[850.50,858.30] vol=1.5x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 839.48 | 842.51 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 885.70 | 879.60 | 0.00 | ORB-long ORB[874.00,880.20] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:40:00 | 889.60 | 885.16 | 0.00 | T1 1.5R @ 889.60 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 885.70 | 886.10 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:00:00 | 879.30 | 881.31 | 0.00 | ORB-short ORB[880.85,889.95] vol=2.8x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-10-16 11:05:00 | 881.82 | 881.29 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 866.05 | 871.71 | 0.00 | ORB-short ORB[872.00,883.00] vol=2.8x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-10-17 09:55:00 | 868.79 | 870.28 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:40:00 | 850.30 | 854.28 | 0.00 | ORB-short ORB[854.00,865.45] vol=1.7x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-10-21 09:50:00 | 853.06 | 853.74 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:10:00 | 884.90 | 874.82 | 0.00 | ORB-long ORB[861.90,871.95] vol=3.1x ATR=4.57 |
| Stop hit — per-position SL triggered | 2024-10-24 10:35:00 | 880.33 | 878.90 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-10-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:10:00 | 842.95 | 846.01 | 0.00 | ORB-short ORB[844.35,855.45] vol=2.1x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:30:00 | 838.13 | 845.33 | 0.00 | T1 1.5R @ 838.13 |
| Stop hit — per-position SL triggered | 2024-10-29 11:35:00 | 842.95 | 842.38 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:05:00 | 856.15 | 867.06 | 0.00 | ORB-short ORB[868.25,879.00] vol=3.1x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:00:00 | 850.27 | 862.92 | 0.00 | T1 1.5R @ 850.27 |
| Stop hit — per-position SL triggered | 2024-11-04 11:25:00 | 856.15 | 862.15 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:50:00 | 864.30 | 868.24 | 0.00 | ORB-short ORB[866.10,874.45] vol=1.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:00:00 | 860.47 | 867.28 | 0.00 | T1 1.5R @ 860.47 |
| Stop hit — per-position SL triggered | 2024-11-07 12:30:00 | 864.30 | 863.47 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 839.70 | 834.31 | 0.00 | ORB-long ORB[828.60,836.35] vol=1.5x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:40:00 | 844.19 | 838.91 | 0.00 | T1 1.5R @ 844.19 |
| Target hit | 2024-11-19 13:40:00 | 845.85 | 846.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 866.00 | 861.27 | 0.00 | ORB-long ORB[855.00,863.70] vol=1.8x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 09:40:00 | 871.41 | 864.39 | 0.00 | T1 1.5R @ 871.41 |
| Stop hit — per-position SL triggered | 2024-11-25 09:55:00 | 866.00 | 865.23 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 819.30 | 823.88 | 0.00 | ORB-short ORB[823.90,829.10] vol=3.8x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-11-29 10:30:00 | 821.59 | 823.08 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 845.20 | 836.80 | 0.00 | ORB-long ORB[828.30,840.90] vol=2.3x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 842.03 | 837.27 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:15:00 | 799.25 | 803.65 | 0.00 | ORB-short ORB[799.75,807.60] vol=1.6x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:20:00 | 796.37 | 803.11 | 0.00 | T1 1.5R @ 796.37 |
| Stop hit — per-position SL triggered | 2024-12-09 11:40:00 | 799.25 | 802.34 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:05:00 | 799.90 | 804.15 | 0.00 | ORB-short ORB[804.35,811.75] vol=2.4x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:20:00 | 796.37 | 802.83 | 0.00 | T1 1.5R @ 796.37 |
| Target hit | 2024-12-13 11:10:00 | 796.35 | 796.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2024-12-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:00:00 | 791.90 | 796.12 | 0.00 | ORB-short ORB[795.95,804.55] vol=1.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:20:00 | 789.02 | 794.16 | 0.00 | T1 1.5R @ 789.02 |
| Stop hit — per-position SL triggered | 2024-12-17 10:55:00 | 791.90 | 792.00 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 763.00 | 766.82 | 0.00 | ORB-short ORB[765.00,772.85] vol=2.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-12-20 09:35:00 | 765.37 | 766.65 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 09:40:00 | 747.10 | 749.85 | 0.00 | ORB-short ORB[747.25,756.00] vol=1.7x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-12-23 10:05:00 | 750.62 | 748.53 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:15:00 | 718.65 | 720.92 | 0.00 | ORB-short ORB[720.30,729.70] vol=1.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-12-26 12:40:00 | 720.34 | 720.51 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 756.45 | 751.33 | 0.00 | ORB-long ORB[746.55,751.25] vol=2.9x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-12-30 11:05:00 | 754.23 | 752.20 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:00:00 | 748.95 | 753.26 | 0.00 | ORB-short ORB[750.40,755.95] vol=1.5x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:25:00 | 745.61 | 749.27 | 0.00 | T1 1.5R @ 745.61 |
| Target hit | 2025-01-01 15:20:00 | 730.00 | 739.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:10:00 | 725.00 | 730.75 | 0.00 | ORB-short ORB[727.10,736.00] vol=1.9x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-01-02 10:25:00 | 727.37 | 729.87 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:25:00 | 719.85 | 726.23 | 0.00 | ORB-short ORB[728.55,733.15] vol=2.2x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-01-03 10:55:00 | 722.18 | 723.27 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:55:00 | 719.55 | 719.60 | 0.00 | ORB-short ORB[720.45,728.65] vol=2.1x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-01-06 10:00:00 | 722.28 | 719.70 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 698.50 | 702.46 | 0.00 | ORB-short ORB[702.50,708.35] vol=3.0x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:45:00 | 695.75 | 699.79 | 0.00 | T1 1.5R @ 695.75 |
| Target hit | 2025-01-09 15:20:00 | 693.00 | 697.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2025-01-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:35:00 | 677.65 | 682.48 | 0.00 | ORB-short ORB[680.20,687.90] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-01-17 09:55:00 | 680.54 | 680.33 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:15:00 | 693.75 | 687.65 | 0.00 | ORB-long ORB[681.10,684.75] vol=2.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-01-23 10:40:00 | 691.90 | 689.32 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-01-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:30:00 | 686.65 | 688.70 | 0.00 | ORB-short ORB[689.00,693.75] vol=1.8x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-01-24 10:35:00 | 688.40 | 688.67 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:05:00 | 676.55 | 669.24 | 0.00 | ORB-long ORB[658.80,665.00] vol=1.8x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-01-29 10:45:00 | 673.93 | 670.66 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:45:00 | 662.65 | 667.44 | 0.00 | ORB-short ORB[667.15,674.80] vol=4.3x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:55:00 | 659.73 | 666.24 | 0.00 | T1 1.5R @ 659.73 |
| Stop hit — per-position SL triggered | 2025-01-30 11:10:00 | 662.65 | 665.69 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-01-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:10:00 | 667.50 | 658.42 | 0.00 | ORB-long ORB[648.85,657.60] vol=2.0x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:50:00 | 670.70 | 660.67 | 0.00 | T1 1.5R @ 670.70 |
| Stop hit — per-position SL triggered | 2025-01-31 12:30:00 | 667.50 | 661.88 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:50:00 | 693.35 | 682.12 | 0.00 | ORB-long ORB[669.00,677.00] vol=10.7x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 690.69 | 684.08 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-02-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 10:20:00 | 660.00 | 657.02 | 0.00 | ORB-long ORB[649.30,658.55] vol=2.3x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-02-04 10:30:00 | 657.68 | 657.38 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-02-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:55:00 | 678.50 | 674.44 | 0.00 | ORB-long ORB[665.05,675.00] vol=2.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-02-05 11:55:00 | 676.04 | 677.04 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-02-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:30:00 | 671.50 | 673.59 | 0.00 | ORB-short ORB[671.85,677.20] vol=1.6x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:05:00 | 668.72 | 673.09 | 0.00 | T1 1.5R @ 668.72 |
| Stop hit — per-position SL triggered | 2025-02-06 11:35:00 | 671.50 | 672.69 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 667.20 | 664.88 | 0.00 | ORB-long ORB[658.60,666.95] vol=3.2x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 664.91 | 664.55 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 605.40 | 601.43 | 0.00 | ORB-long ORB[595.20,602.10] vol=7.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 603.09 | 602.58 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:40:00 | 630.55 | 627.20 | 0.00 | ORB-long ORB[622.55,629.95] vol=4.4x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:45:00 | 633.61 | 627.65 | 0.00 | T1 1.5R @ 633.61 |
| Target hit | 2025-03-18 15:20:00 | 644.05 | 635.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 736.05 | 740.83 | 0.00 | ORB-short ORB[739.80,746.40] vol=1.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 738.65 | 740.55 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 716.15 | 724.71 | 0.00 | ORB-short ORB[723.40,730.50] vol=2.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-04-29 09:50:00 | 719.02 | 723.15 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 10:20:00 | 712.80 | 717.32 | 0.00 | ORB-short ORB[713.80,724.30] vol=1.5x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 715.93 | 715.25 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 725.35 | 723.10 | 0.00 | ORB-long ORB[718.10,723.85] vol=4.4x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:10:00 | 728.95 | 725.72 | 0.00 | T1 1.5R @ 728.95 |
| Stop hit — per-position SL triggered | 2025-05-08 10:25:00 | 725.35 | 726.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:35:00 | 724.55 | 2024-05-13 11:20:00 | 729.39 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2024-05-14 10:30:00 | 748.05 | 2024-05-14 10:50:00 | 745.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-15 09:55:00 | 755.95 | 2024-05-15 10:00:00 | 753.89 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-17 10:55:00 | 755.60 | 2024-05-17 11:00:00 | 757.04 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-21 09:30:00 | 752.25 | 2024-05-21 09:35:00 | 754.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-22 09:40:00 | 738.90 | 2024-05-22 09:55:00 | 741.12 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-23 10:40:00 | 735.55 | 2024-05-23 10:50:00 | 737.34 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-24 11:05:00 | 740.20 | 2024-05-24 14:30:00 | 737.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-05-24 11:05:00 | 740.20 | 2024-05-24 15:20:00 | 738.15 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-05-28 09:35:00 | 751.10 | 2024-05-28 09:45:00 | 754.71 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-28 09:35:00 | 751.10 | 2024-05-28 09:55:00 | 751.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-29 10:15:00 | 736.50 | 2024-05-29 10:30:00 | 733.83 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-29 10:15:00 | 736.50 | 2024-05-29 10:40:00 | 736.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-30 11:00:00 | 727.00 | 2024-05-30 11:20:00 | 724.99 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-10 10:55:00 | 785.25 | 2024-06-10 11:00:00 | 782.38 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-11 09:35:00 | 776.00 | 2024-06-11 09:40:00 | 778.93 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-13 11:05:00 | 804.65 | 2024-06-13 11:10:00 | 806.54 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-14 09:55:00 | 825.00 | 2024-06-14 10:45:00 | 822.43 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-18 09:35:00 | 848.45 | 2024-06-18 09:45:00 | 845.21 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-21 10:05:00 | 871.10 | 2024-06-21 10:35:00 | 867.88 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-25 11:00:00 | 875.50 | 2024-06-25 11:15:00 | 871.62 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-06-25 11:00:00 | 875.50 | 2024-06-25 14:20:00 | 875.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 10:45:00 | 900.95 | 2024-06-26 10:55:00 | 906.91 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-26 10:45:00 | 900.95 | 2024-06-26 11:35:00 | 900.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 09:45:00 | 912.30 | 2024-06-27 09:50:00 | 918.71 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-06-27 09:45:00 | 912.30 | 2024-06-27 10:05:00 | 912.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 09:50:00 | 921.40 | 2024-06-28 10:00:00 | 918.03 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-01 09:30:00 | 933.05 | 2024-07-01 09:40:00 | 929.12 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-05 09:30:00 | 889.15 | 2024-07-05 09:35:00 | 891.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-08 09:50:00 | 895.60 | 2024-07-08 09:55:00 | 892.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-09 09:40:00 | 883.60 | 2024-07-09 09:50:00 | 877.75 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-09 09:40:00 | 883.60 | 2024-07-09 10:05:00 | 883.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:10:00 | 867.50 | 2024-07-10 10:15:00 | 870.88 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-15 09:40:00 | 856.50 | 2024-07-15 09:45:00 | 859.59 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-16 09:35:00 | 891.60 | 2024-07-16 09:45:00 | 888.39 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-23 11:00:00 | 860.25 | 2024-07-23 11:15:00 | 856.54 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-30 09:55:00 | 848.95 | 2024-07-30 10:20:00 | 851.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-01 10:30:00 | 845.05 | 2024-08-01 10:45:00 | 842.26 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-01 10:30:00 | 845.05 | 2024-08-01 15:20:00 | 834.80 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2024-08-08 10:10:00 | 789.95 | 2024-08-08 10:30:00 | 787.19 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-09 09:30:00 | 802.25 | 2024-08-09 10:15:00 | 799.14 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-13 11:00:00 | 806.45 | 2024-08-13 11:15:00 | 808.75 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-16 10:15:00 | 821.00 | 2024-08-16 11:10:00 | 818.43 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-20 09:45:00 | 849.75 | 2024-08-20 09:55:00 | 847.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-21 09:45:00 | 856.90 | 2024-08-21 10:15:00 | 854.09 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-23 11:05:00 | 851.00 | 2024-08-23 11:15:00 | 853.48 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-08-23 11:05:00 | 851.00 | 2024-08-23 11:25:00 | 851.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 10:45:00 | 845.65 | 2024-08-26 12:25:00 | 847.63 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-28 10:15:00 | 871.75 | 2024-08-28 10:55:00 | 868.66 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-30 10:20:00 | 883.20 | 2024-08-30 10:25:00 | 886.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-13 09:55:00 | 928.35 | 2024-09-13 10:00:00 | 935.45 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-09-13 09:55:00 | 928.35 | 2024-09-13 12:05:00 | 933.50 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-16 09:35:00 | 931.80 | 2024-09-16 09:45:00 | 936.69 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-09-19 09:35:00 | 876.55 | 2024-09-19 09:50:00 | 871.41 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-19 09:35:00 | 876.55 | 2024-09-19 15:20:00 | 852.95 | TARGET_HIT | 0.50 | 2.69% |
| SELL | retest1 | 2024-09-24 09:30:00 | 871.85 | 2024-09-24 11:00:00 | 867.26 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-24 09:30:00 | 871.85 | 2024-09-24 15:20:00 | 861.60 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2024-09-25 10:30:00 | 865.05 | 2024-09-25 10:50:00 | 869.37 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-25 10:30:00 | 865.05 | 2024-09-25 11:50:00 | 865.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 10:25:00 | 875.05 | 2024-09-30 10:40:00 | 877.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-03 10:40:00 | 882.70 | 2024-10-03 11:00:00 | 878.10 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-03 10:40:00 | 882.70 | 2024-10-03 15:00:00 | 879.30 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-10-07 11:05:00 | 835.80 | 2024-10-07 11:15:00 | 839.48 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-15 09:35:00 | 885.70 | 2024-10-15 09:40:00 | 889.60 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-10-15 09:35:00 | 885.70 | 2024-10-15 10:00:00 | 885.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 11:00:00 | 879.30 | 2024-10-16 11:05:00 | 881.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-17 09:40:00 | 866.05 | 2024-10-17 09:55:00 | 868.79 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-21 09:40:00 | 850.30 | 2024-10-21 09:50:00 | 853.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-24 10:10:00 | 884.90 | 2024-10-24 10:35:00 | 880.33 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-10-29 10:10:00 | 842.95 | 2024-10-29 10:30:00 | 838.13 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-10-29 10:10:00 | 842.95 | 2024-10-29 11:35:00 | 842.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:05:00 | 856.15 | 2024-11-04 11:00:00 | 850.27 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-11-04 10:05:00 | 856.15 | 2024-11-04 11:25:00 | 856.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 09:50:00 | 864.30 | 2024-11-07 10:00:00 | 860.47 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-07 09:50:00 | 864.30 | 2024-11-07 12:30:00 | 864.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:30:00 | 839.70 | 2024-11-19 09:40:00 | 844.19 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-11-19 09:30:00 | 839.70 | 2024-11-19 13:40:00 | 845.85 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2024-11-25 09:30:00 | 866.00 | 2024-11-25 09:40:00 | 871.41 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-25 09:30:00 | 866.00 | 2024-11-25 09:55:00 | 866.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 10:10:00 | 819.30 | 2024-11-29 10:30:00 | 821.59 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-04 09:40:00 | 845.20 | 2024-12-04 09:45:00 | 842.03 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-09 11:15:00 | 799.25 | 2024-12-09 11:20:00 | 796.37 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-09 11:15:00 | 799.25 | 2024-12-09 11:40:00 | 799.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:05:00 | 799.90 | 2024-12-13 10:20:00 | 796.37 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-13 10:05:00 | 799.90 | 2024-12-13 11:10:00 | 796.35 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-17 10:00:00 | 791.90 | 2024-12-17 10:20:00 | 789.02 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-17 10:00:00 | 791.90 | 2024-12-17 10:55:00 | 791.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:30:00 | 763.00 | 2024-12-20 09:35:00 | 765.37 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-23 09:40:00 | 747.10 | 2024-12-23 10:05:00 | 750.62 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-26 11:15:00 | 718.65 | 2024-12-26 12:40:00 | 720.34 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-30 11:00:00 | 756.45 | 2024-12-30 11:05:00 | 754.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-01 10:00:00 | 748.95 | 2025-01-01 11:25:00 | 745.61 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-01 10:00:00 | 748.95 | 2025-01-01 15:20:00 | 730.00 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2025-01-02 10:10:00 | 725.00 | 2025-01-02 10:25:00 | 727.37 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-03 10:25:00 | 719.85 | 2025-01-03 10:55:00 | 722.18 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-06 09:55:00 | 719.55 | 2025-01-06 10:00:00 | 722.28 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-09 10:45:00 | 698.50 | 2025-01-09 11:45:00 | 695.75 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-09 10:45:00 | 698.50 | 2025-01-09 15:20:00 | 693.00 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-01-17 09:35:00 | 677.65 | 2025-01-17 09:55:00 | 680.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-23 10:15:00 | 693.75 | 2025-01-23 10:40:00 | 691.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-24 10:30:00 | 686.65 | 2025-01-24 10:35:00 | 688.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-29 10:05:00 | 676.55 | 2025-01-29 10:45:00 | 673.93 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-30 10:45:00 | 662.65 | 2025-01-30 10:55:00 | 659.73 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-30 10:45:00 | 662.65 | 2025-01-30 11:10:00 | 662.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 11:10:00 | 667.50 | 2025-01-31 11:50:00 | 670.70 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-31 11:10:00 | 667.50 | 2025-01-31 12:30:00 | 667.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 10:50:00 | 693.35 | 2025-02-01 11:00:00 | 690.69 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-04 10:20:00 | 660.00 | 2025-02-04 10:30:00 | 657.68 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-05 09:55:00 | 678.50 | 2025-02-05 11:55:00 | 676.04 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-06 10:30:00 | 671.50 | 2025-02-06 11:05:00 | 668.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-06 10:30:00 | 671.50 | 2025-02-06 11:35:00 | 671.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 10:10:00 | 667.20 | 2025-02-07 10:15:00 | 664.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-20 09:35:00 | 605.40 | 2025-02-20 09:45:00 | 603.09 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-18 10:40:00 | 630.55 | 2025-03-18 10:45:00 | 633.61 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-18 10:40:00 | 630.55 | 2025-03-18 15:20:00 | 644.05 | TARGET_HIT | 0.50 | 2.14% |
| SELL | retest1 | 2025-04-23 09:35:00 | 736.05 | 2025-04-23 09:40:00 | 738.65 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-29 09:40:00 | 716.15 | 2025-04-29 09:50:00 | 719.02 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-05 10:20:00 | 712.80 | 2025-05-05 11:15:00 | 715.93 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-08 09:35:00 | 725.35 | 2025-05-08 10:10:00 | 728.95 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-05-08 09:35:00 | 725.35 | 2025-05-08 10:25:00 | 725.35 | STOP_HIT | 0.50 | 0.00% |
