# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16813 bars)
- **Last close:** 1122.00
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 11 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 45
- **Target hits / Stop hits / Partials:** 11 / 45 / 22
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 11.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 21 | 51.2% | 8 | 20 | 13 | 0.27% | 11.0% |
| BUY @ 2nd Alert (retest1) | 41 | 21 | 51.2% | 8 | 20 | 13 | 0.27% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 12 | 32.4% | 3 | 25 | 9 | 0.01% | 0.4% |
| SELL @ 2nd Alert (retest1) | 37 | 12 | 32.4% | 3 | 25 | 9 | 0.01% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 33 | 42.3% | 11 | 45 | 22 | 0.15% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:50:00 | 818.90 | 813.81 | 0.00 | ORB-long ORB[809.95,818.30] vol=2.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-05-29 11:00:00 | 815.73 | 814.10 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:45:00 | 829.95 | 821.41 | 0.00 | ORB-long ORB[811.95,821.50] vol=3.8x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:00:00 | 836.50 | 834.06 | 0.00 | T1 1.5R @ 836.50 |
| Target hit | 2025-05-30 11:00:00 | 846.00 | 849.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-06-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:00:00 | 902.10 | 891.61 | 0.00 | ORB-long ORB[881.45,891.80] vol=3.2x ATR=4.71 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 897.39 | 892.59 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:50:00 | 878.10 | 886.49 | 0.00 | ORB-short ORB[883.85,895.95] vol=2.4x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 882.65 | 885.80 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 887.50 | 893.61 | 0.00 | ORB-short ORB[890.90,902.00] vol=1.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-06-06 09:40:00 | 890.84 | 893.02 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:50:00 | 848.85 | 849.70 | 0.00 | ORB-short ORB[850.10,860.35] vol=10.8x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:10:00 | 845.22 | 848.66 | 0.00 | T1 1.5R @ 845.22 |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 848.85 | 848.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:05:00 | 861.65 | 865.83 | 0.00 | ORB-short ORB[865.95,874.65] vol=2.3x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-08-12 11:10:00 | 864.52 | 865.78 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 848.70 | 854.78 | 0.00 | ORB-short ORB[852.40,864.60] vol=2.9x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-08-13 09:55:00 | 851.38 | 853.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:10:00 | 853.55 | 856.53 | 0.00 | ORB-short ORB[857.05,862.90] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-08-14 11:30:00 | 855.22 | 855.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:45:00 | 878.75 | 871.75 | 0.00 | ORB-long ORB[866.05,878.00] vol=1.9x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:50:00 | 885.23 | 876.06 | 0.00 | T1 1.5R @ 885.23 |
| Target hit | 2025-08-18 15:20:00 | 899.95 | 895.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 09:55:00 | 860.80 | 855.94 | 0.00 | ORB-long ORB[850.70,857.90] vol=1.6x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:10:00 | 865.39 | 860.74 | 0.00 | T1 1.5R @ 865.39 |
| Target hit | 2025-08-29 15:20:00 | 872.55 | 870.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 912.60 | 919.13 | 0.00 | ORB-short ORB[915.35,927.00] vol=1.7x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:30:00 | 907.39 | 918.41 | 0.00 | T1 1.5R @ 907.39 |
| Stop hit — per-position SL triggered | 2025-09-05 11:55:00 | 912.60 | 917.04 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:50:00 | 898.90 | 901.24 | 0.00 | ORB-short ORB[901.20,911.00] vol=1.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-09-09 10:55:00 | 901.75 | 901.19 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-09-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:20:00 | 890.00 | 894.15 | 0.00 | ORB-short ORB[893.60,906.40] vol=10.8x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-09-18 10:25:00 | 894.26 | 894.09 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:55:00 | 917.50 | 909.92 | 0.00 | ORB-long ORB[898.85,911.45] vol=5.1x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:10:00 | 923.18 | 913.05 | 0.00 | T1 1.5R @ 923.18 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 917.50 | 913.24 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-10-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:05:00 | 863.60 | 859.94 | 0.00 | ORB-long ORB[852.05,862.00] vol=2.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-10-03 10:10:00 | 860.87 | 860.05 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:45:00 | 857.15 | 855.01 | 0.00 | ORB-long ORB[849.95,856.75] vol=4.9x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-10-06 09:50:00 | 854.18 | 855.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:45:00 | 820.95 | 822.48 | 0.00 | ORB-short ORB[822.35,834.25] vol=6.6x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-10-09 10:10:00 | 824.60 | 821.16 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:55:00 | 836.60 | 833.01 | 0.00 | ORB-long ORB[824.75,833.55] vol=1.6x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-10-10 11:10:00 | 833.62 | 833.07 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:40:00 | 838.00 | 836.06 | 0.00 | ORB-long ORB[830.05,835.30] vol=1.8x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-10-15 09:50:00 | 834.59 | 836.01 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 843.70 | 849.45 | 0.00 | ORB-short ORB[844.80,854.95] vol=1.8x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 13:20:00 | 839.38 | 847.65 | 0.00 | T1 1.5R @ 839.38 |
| Target hit | 2025-10-17 15:20:00 | 835.00 | 843.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:40:00 | 852.30 | 851.89 | 0.00 | ORB-long ORB[846.55,852.20] vol=4.3x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:55:00 | 856.39 | 853.82 | 0.00 | T1 1.5R @ 856.39 |
| Target hit | 2025-10-29 11:10:00 | 857.05 | 857.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 865.20 | 861.97 | 0.00 | ORB-long ORB[854.60,863.00] vol=3.1x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-10-31 09:55:00 | 862.65 | 864.92 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:35:00 | 870.95 | 865.41 | 0.00 | ORB-long ORB[857.60,869.40] vol=2.1x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:50:00 | 876.52 | 872.20 | 0.00 | T1 1.5R @ 876.52 |
| Target hit | 2025-11-03 12:45:00 | 872.25 | 873.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2025-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:55:00 | 887.10 | 881.26 | 0.00 | ORB-long ORB[874.00,883.80] vol=1.5x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 13:20:00 | 891.22 | 884.23 | 0.00 | T1 1.5R @ 891.22 |
| Stop hit — per-position SL triggered | 2025-11-04 14:25:00 | 887.10 | 886.71 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 11:15:00 | 1049.65 | 1050.28 | 0.00 | ORB-short ORB[1050.60,1062.00] vol=2.1x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:40:00 | 1043.74 | 1049.89 | 0.00 | T1 1.5R @ 1043.74 |
| Target hit | 2025-11-13 15:20:00 | 1046.65 | 1047.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-11-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:55:00 | 979.00 | 985.28 | 0.00 | ORB-short ORB[982.10,993.70] vol=3.0x ATR=4.35 |
| Stop hit — per-position SL triggered | 2025-11-25 10:05:00 | 983.35 | 984.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1004.50 | 1004.27 | 0.00 | ORB-long ORB[995.20,1003.00] vol=1.5x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:30:00 | 1010.96 | 1005.68 | 0.00 | T1 1.5R @ 1010.96 |
| Target hit | 2025-11-28 10:55:00 | 1007.55 | 1007.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-12-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:50:00 | 967.60 | 972.63 | 0.00 | ORB-short ORB[971.00,983.70] vol=1.6x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:55:00 | 962.02 | 969.72 | 0.00 | T1 1.5R @ 962.02 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 967.60 | 969.25 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:05:00 | 975.40 | 968.69 | 0.00 | ORB-long ORB[961.10,970.50] vol=1.7x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 970.94 | 969.16 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:50:00 | 971.50 | 980.64 | 0.00 | ORB-short ORB[982.10,995.00] vol=2.1x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-12-18 10:10:00 | 974.72 | 978.52 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:05:00 | 980.30 | 986.98 | 0.00 | ORB-short ORB[982.30,995.60] vol=2.8x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:25:00 | 975.49 | 984.58 | 0.00 | T1 1.5R @ 975.49 |
| Target hit | 2025-12-19 15:20:00 | 971.10 | 974.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-12-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:50:00 | 985.50 | 978.22 | 0.00 | ORB-long ORB[969.90,978.00] vol=1.7x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-12-22 10:00:00 | 981.93 | 979.47 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:30:00 | 982.00 | 986.58 | 0.00 | ORB-short ORB[982.70,994.00] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-12-23 09:50:00 | 984.96 | 985.62 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:45:00 | 941.00 | 948.20 | 0.00 | ORB-short ORB[947.00,956.40] vol=1.6x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-12-26 09:50:00 | 944.81 | 947.93 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 935.30 | 939.20 | 0.00 | ORB-short ORB[938.05,948.00] vol=2.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-01-01 11:55:00 | 937.09 | 938.45 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:45:00 | 916.45 | 911.85 | 0.00 | ORB-long ORB[906.20,916.25] vol=3.1x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-01-05 10:00:00 | 913.43 | 912.27 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 927.85 | 937.05 | 0.00 | ORB-short ORB[929.00,943.00] vol=1.7x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:20:00 | 922.42 | 932.93 | 0.00 | T1 1.5R @ 922.42 |
| Stop hit — per-position SL triggered | 2026-01-06 14:55:00 | 927.85 | 927.29 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:40:00 | 931.00 | 926.47 | 0.00 | ORB-long ORB[920.00,930.00] vol=3.3x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:55:00 | 936.94 | 928.32 | 0.00 | T1 1.5R @ 936.94 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 931.00 | 928.56 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 924.35 | 927.62 | 0.00 | ORB-short ORB[926.25,933.65] vol=2.4x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-01-08 09:40:00 | 926.31 | 926.34 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 915.65 | 919.60 | 0.00 | ORB-short ORB[918.00,929.00] vol=2.2x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 919.13 | 918.34 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:45:00 | 975.95 | 971.28 | 0.00 | ORB-long ORB[959.40,973.40] vol=2.5x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-01-16 10:45:00 | 972.72 | 973.45 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-01-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 09:30:00 | 985.95 | 973.74 | 0.00 | ORB-long ORB[964.35,975.40] vol=3.0x ATR=4.27 |
| Stop hit — per-position SL triggered | 2026-01-19 09:35:00 | 981.68 | 977.79 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 933.05 | 943.32 | 0.00 | ORB-short ORB[941.60,949.35] vol=1.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2026-01-22 11:20:00 | 935.35 | 943.05 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:55:00 | 925.75 | 928.91 | 0.00 | ORB-short ORB[928.20,940.90] vol=3.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-01-23 11:25:00 | 927.81 | 928.69 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:35:00 | 952.45 | 940.03 | 0.00 | ORB-long ORB[927.90,936.00] vol=1.6x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:40:00 | 959.59 | 946.61 | 0.00 | T1 1.5R @ 959.59 |
| Stop hit — per-position SL triggered | 2026-01-30 10:20:00 | 952.45 | 949.93 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 975.75 | 968.51 | 0.00 | ORB-long ORB[960.70,972.00] vol=2.0x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-02 10:20:00 | 969.61 | 974.24 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:35:00 | 998.15 | 994.43 | 0.00 | ORB-long ORB[980.95,995.00] vol=4.6x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:45:00 | 1005.57 | 997.12 | 0.00 | T1 1.5R @ 1005.57 |
| Target hit | 2026-02-04 12:40:00 | 1004.15 | 1008.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 990.90 | 996.03 | 0.00 | ORB-short ORB[994.85,1002.75] vol=2.5x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:45:00 | 986.35 | 992.03 | 0.00 | T1 1.5R @ 986.35 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 990.90 | 992.23 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 1041.30 | 1037.89 | 0.00 | ORB-long ORB[1027.20,1041.25] vol=1.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 1047.34 | 1040.00 | 0.00 | T1 1.5R @ 1047.34 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 1041.30 | 1040.39 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1018.00 | 1026.90 | 0.00 | ORB-short ORB[1027.10,1040.00] vol=1.8x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 1012.95 | 1024.87 | 0.00 | T1 1.5R @ 1012.95 |
| Stop hit — per-position SL triggered | 2026-03-06 11:25:00 | 1018.00 | 1022.43 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 1052.90 | 1050.33 | 0.00 | ORB-long ORB[1040.00,1049.40] vol=6.0x ATR=4.07 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 1048.83 | 1050.44 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:40:00 | 1045.40 | 1038.78 | 0.00 | ORB-long ORB[1027.20,1038.40] vol=2.0x ATR=4.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 12:15:00 | 1052.06 | 1043.93 | 0.00 | T1 1.5R @ 1052.06 |
| Target hit | 2026-03-12 15:20:00 | 1060.00 | 1052.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 1038.80 | 1044.00 | 0.00 | ORB-short ORB[1041.30,1050.90] vol=1.7x ATR=5.14 |
| Stop hit — per-position SL triggered | 2026-03-30 11:15:00 | 1043.94 | 1043.86 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1106.40 | 1114.71 | 0.00 | ORB-short ORB[1114.40,1129.20] vol=2.1x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1109.57 | 1114.16 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 1121.80 | 1112.52 | 0.00 | ORB-long ORB[1104.90,1112.00] vol=2.3x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 1117.66 | 1118.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-29 10:50:00 | 818.90 | 2025-05-29 11:00:00 | 815.73 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-30 09:45:00 | 829.95 | 2025-05-30 10:00:00 | 836.50 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-05-30 09:45:00 | 829.95 | 2025-05-30 11:00:00 | 846.00 | TARGET_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2025-06-03 10:00:00 | 902.10 | 2025-06-03 10:05:00 | 897.39 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-06-04 09:50:00 | 878.10 | 2025-06-04 10:00:00 | 882.65 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-06-06 09:35:00 | 887.50 | 2025-06-06 09:40:00 | 890.84 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-24 10:50:00 | 848.85 | 2025-07-24 11:10:00 | 845.22 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-24 10:50:00 | 848.85 | 2025-07-24 11:15:00 | 848.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-12 11:05:00 | 861.65 | 2025-08-12 11:10:00 | 864.52 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-13 09:30:00 | 848.70 | 2025-08-13 09:55:00 | 851.38 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-14 11:10:00 | 853.55 | 2025-08-14 11:30:00 | 855.22 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-18 09:45:00 | 878.75 | 2025-08-18 09:50:00 | 885.23 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-08-18 09:45:00 | 878.75 | 2025-08-18 15:20:00 | 899.95 | TARGET_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2025-08-29 09:55:00 | 860.80 | 2025-08-29 10:10:00 | 865.39 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-29 09:55:00 | 860.80 | 2025-08-29 15:20:00 | 872.55 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2025-09-05 10:15:00 | 912.60 | 2025-09-05 10:30:00 | 907.39 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-09-05 10:15:00 | 912.60 | 2025-09-05 11:55:00 | 912.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 10:50:00 | 898.90 | 2025-09-09 10:55:00 | 901.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-18 10:20:00 | 890.00 | 2025-09-18 10:25:00 | 894.26 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-09-24 09:55:00 | 917.50 | 2025-09-24 10:10:00 | 923.18 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-24 09:55:00 | 917.50 | 2025-09-24 10:15:00 | 917.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 10:05:00 | 863.60 | 2025-10-03 10:10:00 | 860.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-06 09:45:00 | 857.15 | 2025-10-06 09:50:00 | 854.18 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-09 09:45:00 | 820.95 | 2025-10-09 10:10:00 | 824.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-10-10 10:55:00 | 836.60 | 2025-10-10 11:10:00 | 833.62 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-15 09:40:00 | 838.00 | 2025-10-15 09:50:00 | 834.59 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-17 11:00:00 | 843.70 | 2025-10-17 13:20:00 | 839.38 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-17 11:00:00 | 843.70 | 2025-10-17 15:20:00 | 835.00 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2025-10-29 09:40:00 | 852.30 | 2025-10-29 09:55:00 | 856.39 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-29 09:40:00 | 852.30 | 2025-10-29 11:10:00 | 857.05 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-10-31 09:30:00 | 865.20 | 2025-10-31 09:55:00 | 862.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-03 09:35:00 | 870.95 | 2025-11-03 11:50:00 | 876.52 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-11-03 09:35:00 | 870.95 | 2025-11-03 12:45:00 | 872.25 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-11-04 10:55:00 | 887.10 | 2025-11-04 13:20:00 | 891.22 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-04 10:55:00 | 887.10 | 2025-11-04 14:25:00 | 887.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 11:15:00 | 1049.65 | 2025-11-13 11:40:00 | 1043.74 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-11-13 11:15:00 | 1049.65 | 2025-11-13 15:20:00 | 1046.65 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-25 09:55:00 | 979.00 | 2025-11-25 10:05:00 | 983.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1004.50 | 2025-11-28 10:30:00 | 1010.96 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-11-28 09:45:00 | 1004.50 | 2025-11-28 10:55:00 | 1007.55 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-05 09:50:00 | 967.60 | 2025-12-05 09:55:00 | 962.02 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-12-05 09:50:00 | 967.60 | 2025-12-05 10:05:00 | 967.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 10:05:00 | 975.40 | 2025-12-10 10:15:00 | 970.94 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-12-18 09:50:00 | 971.50 | 2025-12-18 10:10:00 | 974.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-19 11:05:00 | 980.30 | 2025-12-19 11:25:00 | 975.49 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-19 11:05:00 | 980.30 | 2025-12-19 15:20:00 | 971.10 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2025-12-22 09:50:00 | 985.50 | 2025-12-22 10:00:00 | 981.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-23 09:30:00 | 982.00 | 2025-12-23 09:50:00 | 984.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-26 09:45:00 | 941.00 | 2025-12-26 09:50:00 | 944.81 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-01-01 11:15:00 | 935.30 | 2026-01-01 11:55:00 | 937.09 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-05 09:45:00 | 916.45 | 2026-01-05 10:00:00 | 913.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-06 09:40:00 | 927.85 | 2026-01-06 10:20:00 | 922.42 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-01-06 09:40:00 | 927.85 | 2026-01-06 14:55:00 | 927.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:40:00 | 931.00 | 2026-01-07 09:55:00 | 936.94 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-01-07 09:40:00 | 931.00 | 2026-01-07 10:15:00 | 931.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 09:30:00 | 924.35 | 2026-01-08 09:40:00 | 926.31 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-09 09:35:00 | 915.65 | 2026-01-09 09:45:00 | 919.13 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-01-16 09:45:00 | 975.95 | 2026-01-16 10:45:00 | 972.72 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-19 09:30:00 | 985.95 | 2026-01-19 09:35:00 | 981.68 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-01-22 11:15:00 | 933.05 | 2026-01-22 11:20:00 | 935.35 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-23 10:55:00 | 925.75 | 2026-01-23 11:25:00 | 927.81 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-30 09:35:00 | 952.45 | 2026-01-30 09:40:00 | 959.59 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-01-30 09:35:00 | 952.45 | 2026-01-30 10:20:00 | 952.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-02 09:30:00 | 975.75 | 2026-02-02 10:20:00 | 969.61 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-02-04 09:35:00 | 998.15 | 2026-02-04 09:45:00 | 1005.57 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-02-04 09:35:00 | 998.15 | 2026-02-04 12:40:00 | 1004.15 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-19 09:30:00 | 990.90 | 2026-02-19 09:45:00 | 986.35 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-19 09:30:00 | 990.90 | 2026-02-19 09:50:00 | 990.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1041.30 | 2026-02-26 11:05:00 | 1047.34 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1041.30 | 2026-02-26 11:40:00 | 1041.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1018.00 | 2026-03-06 10:45:00 | 1012.95 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1018.00 | 2026-03-06 11:25:00 | 1018.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:30:00 | 1052.90 | 2026-03-11 09:35:00 | 1048.83 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-12 09:40:00 | 1045.40 | 2026-03-12 12:15:00 | 1052.06 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-12 09:40:00 | 1045.40 | 2026-03-12 15:20:00 | 1060.00 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-03-30 10:55:00 | 1038.80 | 2026-03-30 11:15:00 | 1043.94 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1106.40 | 2026-04-28 11:20:00 | 1109.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 09:40:00 | 1121.80 | 2026-04-29 10:35:00 | 1117.66 | STOP_HIT | 1.00 | -0.37% |
