# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 14 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 103 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 57
- **Target hits / Stop hits / Partials:** 14 / 57 / 32
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 24.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 26 | 40.0% | 6 | 39 | 20 | 0.23% | 14.6% |
| BUY @ 2nd Alert (retest1) | 65 | 26 | 40.0% | 6 | 39 | 20 | 0.23% | 14.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 20 | 52.6% | 8 | 18 | 12 | 0.27% | 10.2% |
| SELL @ 2nd Alert (retest1) | 38 | 20 | 52.6% | 8 | 18 | 12 | 0.27% | 10.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 103 | 46 | 44.7% | 14 | 57 | 32 | 0.24% | 24.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:50:00 | 876.55 | 867.27 | 0.00 | ORB-long ORB[863.00,871.60] vol=2.3x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:15:00 | 881.13 | 870.49 | 0.00 | T1 1.5R @ 881.13 |
| Target hit | 2024-05-14 15:20:00 | 909.65 | 907.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 903.50 | 910.81 | 0.00 | ORB-short ORB[912.00,924.15] vol=1.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-16 11:30:00 | 905.97 | 910.45 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:10:00 | 912.95 | 908.59 | 0.00 | ORB-long ORB[902.10,909.70] vol=1.8x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:20:00 | 916.77 | 909.67 | 0.00 | T1 1.5R @ 916.77 |
| Stop hit — per-position SL triggered | 2024-05-17 10:50:00 | 912.95 | 910.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 921.95 | 911.97 | 0.00 | ORB-long ORB[905.15,915.00] vol=2.1x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-05-21 10:05:00 | 918.39 | 915.54 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:15:00 | 952.00 | 936.07 | 0.00 | ORB-long ORB[929.80,941.55] vol=8.4x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 948.17 | 936.93 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 968.75 | 972.86 | 0.00 | ORB-short ORB[969.70,983.65] vol=1.6x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:55:00 | 963.11 | 971.70 | 0.00 | T1 1.5R @ 963.11 |
| Stop hit — per-position SL triggered | 2024-05-28 10:20:00 | 968.75 | 969.82 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 951.95 | 955.34 | 0.00 | ORB-short ORB[952.25,960.00] vol=1.5x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 954.49 | 954.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 956.05 | 949.30 | 0.00 | ORB-long ORB[942.20,951.05] vol=3.8x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-06-14 09:55:00 | 952.70 | 949.73 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 956.05 | 951.99 | 0.00 | ORB-long ORB[948.20,954.80] vol=2.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-06-18 09:40:00 | 953.42 | 952.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:05:00 | 936.80 | 931.31 | 0.00 | ORB-long ORB[926.20,934.85] vol=1.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:15:00 | 941.37 | 932.60 | 0.00 | T1 1.5R @ 941.37 |
| Stop hit — per-position SL triggered | 2024-06-20 10:30:00 | 936.80 | 933.00 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:10:00 | 911.10 | 914.82 | 0.00 | ORB-short ORB[911.75,920.00] vol=2.4x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 908.80 | 914.45 | 0.00 | T1 1.5R @ 908.80 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 911.10 | 914.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 11:05:00 | 895.00 | 897.60 | 0.00 | ORB-short ORB[897.30,904.70] vol=2.8x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-06-27 11:25:00 | 897.15 | 897.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:25:00 | 900.50 | 894.96 | 0.00 | ORB-long ORB[891.10,898.40] vol=2.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-07-12 10:30:00 | 897.65 | 895.30 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:15:00 | 898.95 | 892.34 | 0.00 | ORB-long ORB[888.30,896.20] vol=2.2x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:25:00 | 903.31 | 894.72 | 0.00 | T1 1.5R @ 903.31 |
| Stop hit — per-position SL triggered | 2024-07-15 10:30:00 | 898.95 | 894.89 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:15:00 | 888.35 | 885.77 | 0.00 | ORB-long ORB[880.50,887.00] vol=2.4x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 11:05:00 | 892.80 | 888.63 | 0.00 | T1 1.5R @ 892.80 |
| Target hit | 2024-07-24 14:40:00 | 890.35 | 891.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:10:00 | 893.20 | 888.20 | 0.00 | ORB-long ORB[883.00,889.00] vol=1.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:15:00 | 897.49 | 891.04 | 0.00 | T1 1.5R @ 897.49 |
| Stop hit — per-position SL triggered | 2024-07-25 10:20:00 | 893.20 | 891.22 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 908.50 | 902.71 | 0.00 | ORB-long ORB[898.40,906.40] vol=3.1x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-07-31 10:05:00 | 905.62 | 902.91 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:10:00 | 883.00 | 878.96 | 0.00 | ORB-long ORB[874.20,881.60] vol=2.9x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 880.65 | 879.39 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:20:00 | 877.25 | 880.46 | 0.00 | ORB-short ORB[879.05,886.70] vol=1.7x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 11:05:00 | 873.55 | 879.29 | 0.00 | T1 1.5R @ 873.55 |
| Target hit | 2024-08-09 12:20:00 | 875.75 | 874.66 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2024-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 11:00:00 | 855.00 | 851.31 | 0.00 | ORB-long ORB[846.05,853.45] vol=3.9x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:10:00 | 858.46 | 853.33 | 0.00 | T1 1.5R @ 858.46 |
| Stop hit — per-position SL triggered | 2024-08-19 11:15:00 | 855.00 | 853.56 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:05:00 | 863.30 | 860.00 | 0.00 | ORB-long ORB[856.00,859.90] vol=1.7x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:10:00 | 866.38 | 861.25 | 0.00 | T1 1.5R @ 866.38 |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 863.30 | 861.50 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:50:00 | 865.70 | 862.05 | 0.00 | ORB-long ORB[858.00,862.95] vol=4.2x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-08-22 11:10:00 | 863.73 | 862.81 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:35:00 | 865.30 | 864.15 | 0.00 | ORB-long ORB[858.15,863.00] vol=1.9x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-08-23 10:40:00 | 863.58 | 864.16 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:00:00 | 856.50 | 858.06 | 0.00 | ORB-short ORB[856.55,865.00] vol=2.2x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-08-26 11:25:00 | 858.50 | 858.00 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 11:00:00 | 855.00 | 856.92 | 0.00 | ORB-short ORB[855.45,859.00] vol=4.3x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-08-27 11:10:00 | 857.01 | 856.79 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 835.50 | 836.98 | 0.00 | ORB-short ORB[837.30,843.80] vol=6.5x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 838.33 | 837.00 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:00:00 | 851.95 | 850.08 | 0.00 | ORB-long ORB[842.85,850.65] vol=2.0x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-09-03 11:05:00 | 849.56 | 850.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:40:00 | 849.65 | 847.04 | 0.00 | ORB-long ORB[837.60,847.95] vol=1.9x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:50:00 | 853.36 | 848.08 | 0.00 | T1 1.5R @ 853.36 |
| Stop hit — per-position SL triggered | 2024-09-04 11:00:00 | 849.65 | 848.60 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 837.50 | 841.67 | 0.00 | ORB-short ORB[840.00,846.75] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 834.49 | 839.12 | 0.00 | T1 1.5R @ 834.49 |
| Target hit | 2024-09-06 15:20:00 | 826.80 | 833.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-09-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:40:00 | 803.00 | 805.83 | 0.00 | ORB-short ORB[804.20,813.90] vol=1.6x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:50:00 | 799.04 | 804.60 | 0.00 | T1 1.5R @ 799.04 |
| Target hit | 2024-09-12 13:55:00 | 802.10 | 801.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 813.00 | 808.16 | 0.00 | ORB-long ORB[802.00,809.95] vol=2.0x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 09:40:00 | 816.53 | 812.10 | 0.00 | T1 1.5R @ 816.53 |
| Target hit | 2024-09-16 10:15:00 | 814.95 | 815.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2024-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:40:00 | 800.45 | 804.65 | 0.00 | ORB-short ORB[804.65,810.75] vol=2.0x ATR=1.81 |
| Target hit | 2024-09-17 15:20:00 | 798.75 | 802.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2024-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:45:00 | 796.05 | 800.18 | 0.00 | ORB-short ORB[799.50,804.75] vol=1.8x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-09-18 12:10:00 | 797.58 | 798.77 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 790.75 | 794.14 | 0.00 | ORB-short ORB[792.00,800.00] vol=1.9x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:55:00 | 787.45 | 792.47 | 0.00 | T1 1.5R @ 787.45 |
| Target hit | 2024-09-19 15:20:00 | 776.95 | 782.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:15:00 | 785.95 | 781.71 | 0.00 | ORB-long ORB[775.60,784.95] vol=1.5x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:30:00 | 788.64 | 782.14 | 0.00 | T1 1.5R @ 788.64 |
| Target hit | 2024-09-20 15:20:00 | 787.55 | 787.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 793.30 | 791.10 | 0.00 | ORB-long ORB[784.05,793.00] vol=2.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-10-01 10:00:00 | 790.99 | 791.61 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 754.60 | 762.76 | 0.00 | ORB-short ORB[758.55,769.70] vol=2.0x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:30:00 | 750.52 | 759.30 | 0.00 | T1 1.5R @ 750.52 |
| Target hit | 2024-10-07 15:20:00 | 743.80 | 749.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:10:00 | 771.10 | 765.56 | 0.00 | ORB-long ORB[759.15,767.60] vol=2.1x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-10-09 11:30:00 | 769.32 | 765.86 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:15:00 | 764.55 | 759.83 | 0.00 | ORB-long ORB[754.00,760.70] vol=3.1x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:45:00 | 767.86 | 762.51 | 0.00 | T1 1.5R @ 767.86 |
| Stop hit — per-position SL triggered | 2024-10-11 12:00:00 | 764.55 | 766.51 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:05:00 | 740.50 | 743.47 | 0.00 | ORB-short ORB[742.30,749.95] vol=1.5x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 742.27 | 743.26 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:45:00 | 724.30 | 715.05 | 0.00 | ORB-long ORB[705.00,715.40] vol=2.2x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-10-30 10:05:00 | 721.32 | 718.38 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 11:10:00 | 710.95 | 714.75 | 0.00 | ORB-short ORB[713.10,720.05] vol=2.1x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-11-04 11:20:00 | 713.05 | 714.62 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:10:00 | 729.90 | 725.69 | 0.00 | ORB-long ORB[720.05,729.50] vol=1.9x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-11-06 10:25:00 | 727.11 | 726.30 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 691.95 | 699.87 | 0.00 | ORB-short ORB[698.10,706.90] vol=2.1x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-11-13 10:35:00 | 694.83 | 697.14 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 729.55 | 719.71 | 0.00 | ORB-long ORB[715.10,722.30] vol=4.5x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 726.27 | 721.82 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 716.90 | 711.38 | 0.00 | ORB-long ORB[705.10,713.90] vol=2.8x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 714.55 | 711.96 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:35:00 | 677.40 | 671.40 | 0.00 | ORB-long ORB[666.50,673.95] vol=2.2x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-12-24 10:50:00 | 674.83 | 671.89 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 676.60 | 672.76 | 0.00 | ORB-long ORB[668.60,675.40] vol=2.0x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:00:00 | 680.08 | 673.48 | 0.00 | T1 1.5R @ 680.08 |
| Stop hit — per-position SL triggered | 2024-12-26 11:25:00 | 676.60 | 674.14 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 680.15 | 676.42 | 0.00 | ORB-long ORB[670.00,679.00] vol=1.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-12-27 11:00:00 | 678.08 | 676.54 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 738.80 | 743.47 | 0.00 | ORB-short ORB[740.50,749.45] vol=1.9x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:00:00 | 735.02 | 742.88 | 0.00 | T1 1.5R @ 735.02 |
| Stop hit — per-position SL triggered | 2025-01-02 11:20:00 | 738.80 | 742.40 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 747.00 | 742.25 | 0.00 | ORB-long ORB[734.00,743.55] vol=2.8x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 743.78 | 742.59 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 667.50 | 673.93 | 0.00 | ORB-short ORB[674.15,678.30] vol=1.7x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-01-21 11:00:00 | 669.71 | 671.65 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 659.50 | 655.06 | 0.00 | ORB-long ORB[651.65,657.75] vol=1.7x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-01-23 09:40:00 | 656.61 | 655.53 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 653.50 | 658.80 | 0.00 | ORB-short ORB[656.80,663.70] vol=1.8x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:45:00 | 649.37 | 656.26 | 0.00 | T1 1.5R @ 649.37 |
| Target hit | 2025-01-24 15:20:00 | 641.25 | 648.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:45:00 | 625.75 | 619.74 | 0.00 | ORB-long ORB[613.05,622.10] vol=2.0x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-01-29 10:20:00 | 622.16 | 621.72 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 568.80 | 574.45 | 0.00 | ORB-short ORB[573.15,579.00] vol=1.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-02-18 09:35:00 | 571.33 | 574.00 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:55:00 | 603.75 | 599.02 | 0.00 | ORB-long ORB[592.80,600.00] vol=1.7x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 11:05:00 | 608.06 | 600.38 | 0.00 | T1 1.5R @ 608.06 |
| Stop hit — per-position SL triggered | 2025-03-11 11:20:00 | 603.75 | 601.10 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:00:00 | 590.00 | 598.15 | 0.00 | ORB-short ORB[598.60,605.80] vol=2.0x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 586.81 | 596.59 | 0.00 | T1 1.5R @ 586.81 |
| Stop hit — per-position SL triggered | 2025-03-12 13:20:00 | 590.00 | 592.56 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 611.80 | 604.59 | 0.00 | ORB-long ORB[598.20,607.00] vol=2.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-03-13 09:40:00 | 607.96 | 605.51 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:45:00 | 607.65 | 604.07 | 0.00 | ORB-long ORB[598.20,606.60] vol=2.1x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-03-17 10:00:00 | 604.76 | 604.41 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:30:00 | 628.20 | 623.63 | 0.00 | ORB-long ORB[619.00,625.00] vol=2.0x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:50:00 | 631.64 | 625.70 | 0.00 | T1 1.5R @ 631.64 |
| Stop hit — per-position SL triggered | 2025-03-21 11:40:00 | 628.20 | 628.18 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 630.15 | 635.47 | 0.00 | ORB-short ORB[634.70,641.90] vol=1.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-03-24 11:20:00 | 631.93 | 635.26 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 11:10:00 | 616.35 | 623.78 | 0.00 | ORB-short ORB[622.00,631.15] vol=1.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:40:00 | 613.29 | 621.67 | 0.00 | T1 1.5R @ 613.29 |
| Stop hit — per-position SL triggered | 2025-03-25 11:55:00 | 616.35 | 621.20 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:55:00 | 608.05 | 602.07 | 0.00 | ORB-long ORB[596.55,603.00] vol=3.6x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:00:00 | 610.95 | 603.78 | 0.00 | T1 1.5R @ 610.95 |
| Stop hit — per-position SL triggered | 2025-04-15 11:15:00 | 608.05 | 604.99 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:25:00 | 614.10 | 608.93 | 0.00 | ORB-long ORB[604.80,613.95] vol=1.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:35:00 | 617.80 | 612.51 | 0.00 | T1 1.5R @ 617.80 |
| Stop hit — per-position SL triggered | 2025-04-17 10:50:00 | 614.10 | 613.55 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 616.00 | 612.17 | 0.00 | ORB-long ORB[609.00,615.50] vol=1.9x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:25:00 | 619.23 | 614.30 | 0.00 | T1 1.5R @ 619.23 |
| Stop hit — per-position SL triggered | 2025-04-21 10:55:00 | 616.00 | 614.79 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:30:00 | 633.90 | 628.22 | 0.00 | ORB-long ORB[622.10,630.00] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-04-22 10:35:00 | 631.74 | 628.45 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:55:00 | 629.75 | 626.07 | 0.00 | ORB-long ORB[621.95,628.00] vol=1.7x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-04-23 10:00:00 | 627.54 | 626.44 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 609.40 | 604.98 | 0.00 | ORB-long ORB[600.00,607.55] vol=3.0x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 10:05:00 | 614.25 | 607.24 | 0.00 | T1 1.5R @ 614.25 |
| Target hit | 2025-04-28 15:20:00 | 617.40 | 614.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 616.65 | 619.76 | 0.00 | ORB-short ORB[618.25,625.00] vol=1.7x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:40:00 | 613.34 | 618.56 | 0.00 | T1 1.5R @ 613.34 |
| Target hit | 2025-04-29 15:20:00 | 608.35 | 613.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:35:00 | 613.40 | 608.07 | 0.00 | ORB-long ORB[603.45,608.55] vol=3.5x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:40:00 | 617.27 | 610.45 | 0.00 | T1 1.5R @ 617.27 |
| Target hit | 2025-05-05 13:20:00 | 661.30 | 662.22 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:50:00 | 876.55 | 2024-05-14 11:15:00 | 881.13 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-14 10:50:00 | 876.55 | 2024-05-14 15:20:00 | 909.65 | TARGET_HIT | 0.50 | 3.78% |
| SELL | retest1 | 2024-05-16 11:15:00 | 903.50 | 2024-05-16 11:30:00 | 905.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-17 10:10:00 | 912.95 | 2024-05-17 10:20:00 | 916.77 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-05-17 10:10:00 | 912.95 | 2024-05-17 10:50:00 | 912.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 10:00:00 | 921.95 | 2024-05-21 10:05:00 | 918.39 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-23 11:15:00 | 952.00 | 2024-05-23 11:20:00 | 948.17 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-28 09:45:00 | 968.75 | 2024-05-28 09:55:00 | 963.11 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-28 09:45:00 | 968.75 | 2024-05-28 10:20:00 | 968.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:35:00 | 951.95 | 2024-06-13 09:55:00 | 954.49 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-14 09:50:00 | 956.05 | 2024-06-14 09:55:00 | 952.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-18 09:35:00 | 956.05 | 2024-06-18 09:40:00 | 953.42 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-20 10:05:00 | 936.80 | 2024-06-20 10:15:00 | 941.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-06-20 10:05:00 | 936.80 | 2024-06-20 10:30:00 | 936.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:10:00 | 911.10 | 2024-06-25 11:15:00 | 908.80 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-06-25 11:10:00 | 911.10 | 2024-06-25 11:20:00 | 911.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 11:05:00 | 895.00 | 2024-06-27 11:25:00 | 897.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-12 10:25:00 | 900.50 | 2024-07-12 10:30:00 | 897.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-15 10:15:00 | 898.95 | 2024-07-15 10:25:00 | 903.31 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-15 10:15:00 | 898.95 | 2024-07-15 10:30:00 | 898.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:15:00 | 888.35 | 2024-07-24 11:05:00 | 892.80 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-24 10:15:00 | 888.35 | 2024-07-24 14:40:00 | 890.35 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-07-25 10:10:00 | 893.20 | 2024-07-25 10:15:00 | 897.49 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-25 10:10:00 | 893.20 | 2024-07-25 10:20:00 | 893.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:00:00 | 908.50 | 2024-07-31 10:05:00 | 905.62 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-08 11:10:00 | 883.00 | 2024-08-08 11:20:00 | 880.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-09 10:20:00 | 877.25 | 2024-08-09 11:05:00 | 873.55 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-09 10:20:00 | 877.25 | 2024-08-09 12:20:00 | 875.75 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-08-19 11:00:00 | 855.00 | 2024-08-19 11:10:00 | 858.46 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-19 11:00:00 | 855.00 | 2024-08-19 11:15:00 | 855.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:05:00 | 863.30 | 2024-08-21 10:10:00 | 866.38 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-21 10:05:00 | 863.30 | 2024-08-21 10:15:00 | 863.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:50:00 | 865.70 | 2024-08-22 11:10:00 | 863.73 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-23 10:35:00 | 865.30 | 2024-08-23 10:40:00 | 863.58 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-26 11:00:00 | 856.50 | 2024-08-26 11:25:00 | 858.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-27 11:00:00 | 855.00 | 2024-08-27 11:10:00 | 857.01 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-30 09:35:00 | 835.50 | 2024-08-30 09:40:00 | 838.33 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-03 11:00:00 | 851.95 | 2024-09-03 11:05:00 | 849.56 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-04 10:40:00 | 849.65 | 2024-09-04 10:50:00 | 853.36 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-04 10:40:00 | 849.65 | 2024-09-04 11:00:00 | 849.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:45:00 | 837.50 | 2024-09-06 10:05:00 | 834.49 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-06 09:45:00 | 837.50 | 2024-09-06 15:20:00 | 826.80 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2024-09-12 09:40:00 | 803.00 | 2024-09-12 09:50:00 | 799.04 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-12 09:40:00 | 803.00 | 2024-09-12 13:55:00 | 802.10 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-09-16 09:30:00 | 813.00 | 2024-09-16 09:40:00 | 816.53 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-09-16 09:30:00 | 813.00 | 2024-09-16 10:15:00 | 814.95 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-09-17 10:40:00 | 800.45 | 2024-09-17 15:20:00 | 798.75 | TARGET_HIT | 1.00 | 0.21% |
| SELL | retest1 | 2024-09-18 10:45:00 | 796.05 | 2024-09-18 12:10:00 | 797.58 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-19 09:40:00 | 790.75 | 2024-09-19 09:55:00 | 787.45 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-19 09:40:00 | 790.75 | 2024-09-19 15:20:00 | 776.95 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2024-09-20 11:15:00 | 785.95 | 2024-09-20 11:30:00 | 788.64 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-20 11:15:00 | 785.95 | 2024-09-20 15:20:00 | 787.55 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-10-01 09:30:00 | 793.30 | 2024-10-01 10:00:00 | 790.99 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-07 09:55:00 | 754.60 | 2024-10-07 10:30:00 | 750.52 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-07 09:55:00 | 754.60 | 2024-10-07 15:20:00 | 743.80 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-10-09 11:10:00 | 771.10 | 2024-10-09 11:30:00 | 769.32 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-11 10:15:00 | 764.55 | 2024-10-11 10:45:00 | 767.86 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-10-11 10:15:00 | 764.55 | 2024-10-11 12:00:00 | 764.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:05:00 | 740.50 | 2024-10-17 10:15:00 | 742.27 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-30 09:45:00 | 724.30 | 2024-10-30 10:05:00 | 721.32 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-11-04 11:10:00 | 710.95 | 2024-11-04 11:20:00 | 713.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-06 10:10:00 | 729.90 | 2024-11-06 10:25:00 | 727.11 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-13 09:45:00 | 691.95 | 2024-11-13 10:35:00 | 694.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-12 10:30:00 | 729.55 | 2024-12-12 10:35:00 | 726.27 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-12-17 09:30:00 | 716.90 | 2024-12-17 09:35:00 | 714.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-24 10:35:00 | 677.40 | 2024-12-24 10:50:00 | 674.83 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-26 10:55:00 | 676.60 | 2024-12-26 11:00:00 | 680.08 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-26 10:55:00 | 676.60 | 2024-12-26 11:25:00 | 676.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 10:55:00 | 680.15 | 2024-12-27 11:00:00 | 678.08 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-02 10:45:00 | 738.80 | 2025-01-02 11:00:00 | 735.02 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-02 10:45:00 | 738.80 | 2025-01-02 11:20:00 | 738.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 09:35:00 | 747.00 | 2025-01-03 09:40:00 | 743.78 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-21 10:15:00 | 667.50 | 2025-01-21 11:00:00 | 669.71 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-23 09:35:00 | 659.50 | 2025-01-23 09:40:00 | 656.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-24 09:30:00 | 653.50 | 2025-01-24 09:45:00 | 649.37 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-01-24 09:30:00 | 653.50 | 2025-01-24 15:20:00 | 641.25 | TARGET_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2025-01-29 09:45:00 | 625.75 | 2025-01-29 10:20:00 | 622.16 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-02-18 09:30:00 | 568.80 | 2025-02-18 09:35:00 | 571.33 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-11 10:55:00 | 603.75 | 2025-03-11 11:05:00 | 608.06 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-03-11 10:55:00 | 603.75 | 2025-03-11 11:20:00 | 603.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 11:00:00 | 590.00 | 2025-03-12 11:25:00 | 586.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-12 11:00:00 | 590.00 | 2025-03-12 13:20:00 | 590.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-13 09:35:00 | 611.80 | 2025-03-13 09:40:00 | 607.96 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2025-03-17 09:45:00 | 607.65 | 2025-03-17 10:00:00 | 604.76 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-21 10:30:00 | 628.20 | 2025-03-21 10:50:00 | 631.64 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-21 10:30:00 | 628.20 | 2025-03-21 11:40:00 | 628.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-24 11:10:00 | 630.15 | 2025-03-24 11:20:00 | 631.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-03-25 11:10:00 | 616.35 | 2025-03-25 11:40:00 | 613.29 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-03-25 11:10:00 | 616.35 | 2025-03-25 11:55:00 | 616.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 10:55:00 | 608.05 | 2025-04-15 11:00:00 | 610.95 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-15 10:55:00 | 608.05 | 2025-04-15 11:15:00 | 608.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-17 10:25:00 | 614.10 | 2025-04-17 10:35:00 | 617.80 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-17 10:25:00 | 614.10 | 2025-04-17 10:50:00 | 614.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:30:00 | 616.00 | 2025-04-21 10:25:00 | 619.23 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-21 09:30:00 | 616.00 | 2025-04-21 10:55:00 | 616.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 10:30:00 | 633.90 | 2025-04-22 10:35:00 | 631.74 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-23 09:55:00 | 629.75 | 2025-04-23 10:00:00 | 627.54 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-28 09:30:00 | 609.40 | 2025-04-28 10:05:00 | 614.25 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-04-28 09:30:00 | 609.40 | 2025-04-28 15:20:00 | 617.40 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-04-29 09:35:00 | 616.65 | 2025-04-29 09:40:00 | 613.34 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-29 09:35:00 | 616.65 | 2025-04-29 15:20:00 | 608.35 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-05-05 09:35:00 | 613.40 | 2025-05-05 09:40:00 | 617.27 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-05-05 09:35:00 | 613.40 | 2025-05-05 13:20:00 | 661.30 | TARGET_HIT | 0.50 | 7.81% |
