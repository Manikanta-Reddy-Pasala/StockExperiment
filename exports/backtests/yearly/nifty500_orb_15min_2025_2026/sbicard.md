# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 645.00
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
| PARTIAL | 31 |
| TARGET_HIT | 14 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 76
- **Target hits / Stop hits / Partials:** 14 / 76 / 31
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 17.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 21 | 34.4% | 5 | 40 | 16 | 0.11% | 6.8% |
| BUY @ 2nd Alert (retest1) | 61 | 21 | 34.4% | 5 | 40 | 16 | 0.11% | 6.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 24 | 40.0% | 9 | 36 | 15 | 0.17% | 10.5% |
| SELL @ 2nd Alert (retest1) | 60 | 24 | 40.0% | 9 | 36 | 15 | 0.17% | 10.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 121 | 45 | 37.2% | 14 | 76 | 31 | 0.14% | 17.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:00:00 | 898.95 | 895.40 | 0.00 | ORB-long ORB[886.10,896.85] vol=2.5x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-05-14 11:40:00 | 897.24 | 896.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:00:00 | 901.80 | 898.72 | 0.00 | ORB-long ORB[895.40,901.20] vol=2.4x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 10:35:00 | 905.36 | 900.35 | 0.00 | T1 1.5R @ 905.36 |
| Target hit | 2025-05-15 12:55:00 | 902.60 | 903.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:15:00 | 916.15 | 911.36 | 0.00 | ORB-long ORB[908.10,914.95] vol=1.6x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-05-19 11:00:00 | 913.88 | 912.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:10:00 | 902.65 | 899.83 | 0.00 | ORB-long ORB[892.85,899.95] vol=1.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-05-21 11:35:00 | 900.67 | 900.01 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 900.00 | 894.95 | 0.00 | ORB-long ORB[888.35,897.85] vol=3.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-05-23 11:20:00 | 897.63 | 895.20 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:05:00 | 908.65 | 905.16 | 0.00 | ORB-long ORB[894.15,906.00] vol=2.0x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:15:00 | 911.78 | 906.49 | 0.00 | T1 1.5R @ 911.78 |
| Stop hit — per-position SL triggered | 2025-05-26 11:40:00 | 908.65 | 909.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-05-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:10:00 | 910.90 | 904.57 | 0.00 | ORB-long ORB[903.30,909.75] vol=3.2x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 12:10:00 | 914.09 | 906.65 | 0.00 | T1 1.5R @ 914.09 |
| Stop hit — per-position SL triggered | 2025-05-27 13:45:00 | 910.90 | 911.79 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:50:00 | 918.65 | 916.34 | 0.00 | ORB-long ORB[910.50,917.95] vol=2.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-05-28 11:30:00 | 916.41 | 916.78 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-05-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:50:00 | 920.10 | 915.47 | 0.00 | ORB-long ORB[909.05,916.00] vol=3.7x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-05-30 10:35:00 | 917.35 | 918.47 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:30:00 | 917.20 | 922.71 | 0.00 | ORB-short ORB[920.55,929.15] vol=2.4x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:40:00 | 914.21 | 921.74 | 0.00 | T1 1.5R @ 914.21 |
| Stop hit — per-position SL triggered | 2025-06-03 10:45:00 | 917.20 | 921.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 951.85 | 946.51 | 0.00 | ORB-long ORB[941.05,945.00] vol=6.6x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:20:00 | 956.12 | 949.10 | 0.00 | T1 1.5R @ 956.12 |
| Target hit | 2025-06-06 15:20:00 | 994.85 | 983.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:15:00 | 1007.70 | 1014.37 | 0.00 | ORB-short ORB[1012.70,1027.25] vol=2.2x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:35:00 | 1004.48 | 1013.42 | 0.00 | T1 1.5R @ 1004.48 |
| Target hit | 2025-06-10 15:20:00 | 1002.10 | 1007.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:05:00 | 1009.15 | 1002.08 | 0.00 | ORB-long ORB[992.00,998.85] vol=1.9x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-06-12 10:20:00 | 1006.63 | 1003.65 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:00:00 | 998.70 | 1001.25 | 0.00 | ORB-short ORB[1000.20,1009.75] vol=3.1x ATR=1.73 |
| Stop hit — per-position SL triggered | 2025-06-17 11:35:00 | 1000.43 | 1000.53 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 11:15:00 | 983.00 | 987.90 | 0.00 | ORB-short ORB[984.00,993.00] vol=1.8x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 11:50:00 | 980.60 | 985.11 | 0.00 | T1 1.5R @ 980.60 |
| Target hit | 2025-06-18 15:20:00 | 973.25 | 976.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:40:00 | 964.85 | 970.24 | 0.00 | ORB-short ORB[968.50,975.90] vol=1.8x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:50:00 | 960.51 | 964.47 | 0.00 | T1 1.5R @ 960.51 |
| Target hit | 2025-06-19 15:20:00 | 938.70 | 948.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:25:00 | 928.00 | 924.48 | 0.00 | ORB-long ORB[920.05,924.50] vol=2.3x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:00:00 | 930.78 | 926.26 | 0.00 | T1 1.5R @ 930.78 |
| Stop hit — per-position SL triggered | 2025-07-09 11:50:00 | 928.00 | 927.67 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 930.10 | 935.00 | 0.00 | ORB-short ORB[933.40,939.00] vol=1.5x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-07-10 11:05:00 | 931.81 | 934.69 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:00:00 | 914.25 | 922.01 | 0.00 | ORB-short ORB[923.85,929.05] vol=2.4x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 916.36 | 921.69 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 904.75 | 901.80 | 0.00 | ORB-long ORB[896.05,903.00] vol=2.2x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-07-17 09:40:00 | 902.57 | 902.22 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:25:00 | 896.10 | 900.73 | 0.00 | ORB-short ORB[904.05,908.15] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 898.31 | 899.55 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:55:00 | 892.30 | 895.29 | 0.00 | ORB-short ORB[894.20,902.40] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-07-22 10:10:00 | 894.15 | 894.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 888.35 | 892.67 | 0.00 | ORB-short ORB[892.45,898.80] vol=2.4x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-07-23 09:35:00 | 890.28 | 891.65 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 886.75 | 890.48 | 0.00 | ORB-short ORB[889.10,895.90] vol=1.7x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-07-24 09:45:00 | 888.74 | 889.14 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:15:00 | 804.00 | 806.94 | 0.00 | ORB-short ORB[805.85,812.00] vol=3.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-07-31 10:25:00 | 805.78 | 806.76 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:15:00 | 799.05 | 802.63 | 0.00 | ORB-short ORB[802.00,808.00] vol=1.7x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-08-01 12:00:00 | 800.89 | 802.17 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 796.60 | 802.31 | 0.00 | ORB-short ORB[801.25,811.00] vol=3.4x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:05:00 | 793.14 | 799.65 | 0.00 | T1 1.5R @ 793.14 |
| Target hit | 2025-08-06 13:55:00 | 791.35 | 790.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — BUY (started 2025-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:40:00 | 793.75 | 791.20 | 0.00 | ORB-long ORB[785.05,792.95] vol=1.7x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 10:20:00 | 796.95 | 792.95 | 0.00 | T1 1.5R @ 796.95 |
| Target hit | 2025-08-11 15:20:00 | 797.00 | 795.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-08-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:05:00 | 800.85 | 797.87 | 0.00 | ORB-long ORB[796.20,799.90] vol=2.2x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-08-12 10:30:00 | 798.96 | 799.07 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:10:00 | 815.10 | 811.78 | 0.00 | ORB-long ORB[809.70,814.60] vol=1.5x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-08-19 11:20:00 | 813.71 | 811.91 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:05:00 | 824.25 | 821.04 | 0.00 | ORB-long ORB[816.25,821.55] vol=3.5x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 10:10:00 | 826.77 | 821.54 | 0.00 | T1 1.5R @ 826.77 |
| Stop hit — per-position SL triggered | 2025-08-25 10:25:00 | 824.25 | 822.17 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 10:50:00 | 805.70 | 810.53 | 0.00 | ORB-short ORB[806.00,813.70] vol=2.1x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:30:00 | 802.60 | 809.13 | 0.00 | T1 1.5R @ 802.60 |
| Target hit | 2025-09-02 15:20:00 | 793.45 | 802.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:00:00 | 803.00 | 799.84 | 0.00 | ORB-long ORB[793.50,801.15] vol=3.0x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-09-03 10:40:00 | 801.06 | 800.87 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:30:00 | 793.50 | 796.28 | 0.00 | ORB-short ORB[794.75,799.50] vol=2.4x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-09-05 11:00:00 | 795.36 | 796.01 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:05:00 | 796.10 | 794.74 | 0.00 | ORB-long ORB[790.20,795.95] vol=2.3x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:10:00 | 798.80 | 794.89 | 0.00 | T1 1.5R @ 798.80 |
| Stop hit — per-position SL triggered | 2025-09-08 10:25:00 | 796.10 | 795.13 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:15:00 | 810.80 | 806.95 | 0.00 | ORB-long ORB[802.75,809.00] vol=3.5x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-09-09 10:50:00 | 809.11 | 808.52 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:55:00 | 840.50 | 833.19 | 0.00 | ORB-long ORB[823.00,833.20] vol=2.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 838.66 | 834.17 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 880.45 | 884.32 | 0.00 | ORB-short ORB[881.00,892.60] vol=2.4x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-09-18 09:40:00 | 883.32 | 882.58 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:00:00 | 883.15 | 888.94 | 0.00 | ORB-short ORB[884.55,897.20] vol=1.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:35:00 | 879.93 | 887.67 | 0.00 | T1 1.5R @ 879.93 |
| Stop hit — per-position SL triggered | 2025-09-19 11:55:00 | 883.15 | 886.72 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:20:00 | 865.00 | 870.76 | 0.00 | ORB-short ORB[867.85,874.50] vol=1.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-09-23 10:35:00 | 867.55 | 870.18 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 883.50 | 878.62 | 0.00 | ORB-long ORB[870.75,882.80] vol=2.3x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-09-24 09:40:00 | 880.69 | 878.84 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:50:00 | 892.20 | 885.63 | 0.00 | ORB-long ORB[875.30,886.05] vol=1.8x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:20:00 | 896.07 | 887.60 | 0.00 | T1 1.5R @ 896.07 |
| Stop hit — per-position SL triggered | 2025-09-25 11:25:00 | 892.20 | 887.65 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 11:05:00 | 904.85 | 902.84 | 0.00 | ORB-long ORB[891.95,904.10] vol=4.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 12:45:00 | 908.70 | 903.34 | 0.00 | T1 1.5R @ 908.70 |
| Stop hit — per-position SL triggered | 2025-10-07 15:05:00 | 904.85 | 905.71 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 913.85 | 921.22 | 0.00 | ORB-short ORB[920.95,929.60] vol=2.4x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 915.94 | 919.32 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 938.25 | 935.27 | 0.00 | ORB-long ORB[931.00,936.00] vol=1.8x ATR=2.15 |
| Stop hit — per-position SL triggered | 2025-10-16 10:00:00 | 936.10 | 937.18 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:40:00 | 937.10 | 931.90 | 0.00 | ORB-long ORB[927.05,935.00] vol=1.5x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:50:00 | 940.57 | 932.72 | 0.00 | T1 1.5R @ 940.57 |
| Stop hit — per-position SL triggered | 2025-10-20 11:25:00 | 937.10 | 934.34 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 886.00 | 884.22 | 0.00 | ORB-long ORB[877.35,883.15] vol=4.1x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-11-03 12:05:00 | 884.37 | 885.00 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:40:00 | 894.60 | 887.78 | 0.00 | ORB-long ORB[881.00,886.95] vol=2.8x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-11-04 09:45:00 | 892.18 | 888.45 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:15:00 | 874.75 | 870.24 | 0.00 | ORB-long ORB[864.30,872.00] vol=4.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-11-10 11:25:00 | 872.88 | 870.38 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 11:00:00 | 883.70 | 879.47 | 0.00 | ORB-long ORB[873.95,879.00] vol=1.5x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-11-17 11:10:00 | 881.75 | 879.46 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:55:00 | 872.40 | 869.31 | 0.00 | ORB-long ORB[863.70,869.80] vol=1.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:25:00 | 874.41 | 870.29 | 0.00 | T1 1.5R @ 874.41 |
| Stop hit — per-position SL triggered | 2025-11-20 14:50:00 | 872.40 | 872.88 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:50:00 | 869.10 | 871.92 | 0.00 | ORB-short ORB[870.40,875.50] vol=3.5x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-11-21 10:55:00 | 870.97 | 871.79 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 890.80 | 886.90 | 0.00 | ORB-long ORB[880.35,887.00] vol=4.5x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:45:00 | 894.44 | 889.82 | 0.00 | T1 1.5R @ 894.44 |
| Target hit | 2025-11-24 10:55:00 | 895.35 | 895.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 871.35 | 874.36 | 0.00 | ORB-short ORB[875.35,883.15] vol=3.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-01 10:55:00 | 872.94 | 874.22 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 877.70 | 880.87 | 0.00 | ORB-short ORB[879.35,884.95] vol=1.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:30:00 | 874.87 | 880.06 | 0.00 | T1 1.5R @ 874.87 |
| Target hit | 2025-12-03 15:20:00 | 867.20 | 875.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 866.10 | 862.14 | 0.00 | ORB-long ORB[857.30,861.75] vol=1.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 870.79 | 863.50 | 0.00 | T1 1.5R @ 870.79 |
| Target hit | 2025-12-05 15:20:00 | 887.05 | 879.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-12-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:45:00 | 860.40 | 860.68 | 0.00 | ORB-short ORB[861.80,868.75] vol=2.3x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-12-09 11:10:00 | 862.67 | 860.76 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 863.85 | 867.62 | 0.00 | ORB-short ORB[865.35,872.05] vol=2.7x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:55:00 | 860.81 | 865.78 | 0.00 | T1 1.5R @ 860.81 |
| Stop hit — per-position SL triggered | 2025-12-10 14:50:00 | 863.85 | 862.95 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 873.95 | 871.44 | 0.00 | ORB-long ORB[860.60,869.05] vol=10.6x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-12-11 11:30:00 | 871.82 | 871.48 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:35:00 | 866.30 | 868.26 | 0.00 | ORB-short ORB[867.40,874.60] vol=3.8x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-12-15 09:40:00 | 868.46 | 868.16 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:25:00 | 853.35 | 858.87 | 0.00 | ORB-short ORB[860.95,868.35] vol=3.4x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 10:30:00 | 850.65 | 855.41 | 0.00 | T1 1.5R @ 850.65 |
| Stop hit — per-position SL triggered | 2025-12-16 10:35:00 | 853.35 | 855.14 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 835.90 | 842.81 | 0.00 | ORB-short ORB[845.35,851.40] vol=2.1x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-12-17 11:05:00 | 837.99 | 842.60 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:15:00 | 848.35 | 840.70 | 0.00 | ORB-long ORB[834.05,846.35] vol=1.7x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:20:00 | 852.01 | 841.09 | 0.00 | T1 1.5R @ 852.01 |
| Stop hit — per-position SL triggered | 2025-12-18 12:00:00 | 848.35 | 843.94 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 859.30 | 855.34 | 0.00 | ORB-long ORB[849.00,855.95] vol=2.0x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-12-19 09:35:00 | 857.16 | 855.49 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:25:00 | 876.25 | 872.31 | 0.00 | ORB-long ORB[865.65,872.65] vol=1.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-12-24 11:05:00 | 874.44 | 874.26 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 858.40 | 859.13 | 0.00 | ORB-short ORB[858.95,866.35] vol=1.7x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:20:00 | 855.75 | 858.69 | 0.00 | T1 1.5R @ 855.75 |
| Target hit | 2025-12-29 15:20:00 | 851.30 | 851.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2025-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:30:00 | 841.10 | 842.70 | 0.00 | ORB-short ORB[847.00,852.50] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:40:00 | 838.37 | 841.79 | 0.00 | T1 1.5R @ 838.37 |
| Target hit | 2025-12-30 12:15:00 | 836.00 | 835.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 864.75 | 862.56 | 0.00 | ORB-long ORB[859.55,864.35] vol=2.6x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-01-01 11:40:00 | 863.40 | 862.64 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:55:00 | 875.20 | 867.48 | 0.00 | ORB-long ORB[856.00,863.10] vol=2.8x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-01-02 11:00:00 | 873.13 | 868.53 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:55:00 | 867.20 | 869.02 | 0.00 | ORB-short ORB[869.00,873.95] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-01-05 11:00:00 | 868.81 | 869.00 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:50:00 | 888.65 | 892.64 | 0.00 | ORB-short ORB[891.75,901.45] vol=1.5x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-01-07 10:05:00 | 891.97 | 892.56 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 872.10 | 877.04 | 0.00 | ORB-short ORB[876.55,885.60] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-01-08 11:40:00 | 874.07 | 874.65 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:30:00 | 852.35 | 853.12 | 0.00 | ORB-short ORB[852.55,858.30] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 854.56 | 852.97 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:25:00 | 834.05 | 836.91 | 0.00 | ORB-short ORB[834.55,841.40] vol=1.5x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-01-19 10:45:00 | 836.07 | 836.77 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:45:00 | 776.95 | 773.02 | 0.00 | ORB-long ORB[764.65,775.00] vol=3.9x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-01-27 11:05:00 | 773.58 | 774.94 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:35:00 | 743.60 | 745.99 | 0.00 | ORB-short ORB[745.10,750.50] vol=2.5x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-02-06 10:55:00 | 745.82 | 745.61 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:35:00 | 763.60 | 760.59 | 0.00 | ORB-long ORB[755.80,763.00] vol=1.9x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 09:55:00 | 766.91 | 762.28 | 0.00 | T1 1.5R @ 766.91 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 763.60 | 763.60 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 773.70 | 766.42 | 0.00 | ORB-long ORB[760.95,767.05] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 771.99 | 767.99 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:35:00 | 729.95 | 733.23 | 0.00 | ORB-short ORB[731.55,740.00] vol=2.1x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:40:00 | 725.11 | 730.83 | 0.00 | T1 1.5R @ 725.11 |
| Stop hit — per-position SL triggered | 2026-03-04 13:10:00 | 729.95 | 727.47 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 725.95 | 728.29 | 0.00 | ORB-short ORB[729.95,738.00] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-03-05 10:50:00 | 727.81 | 728.21 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:35:00 | 698.00 | 693.71 | 0.00 | ORB-long ORB[687.40,693.90] vol=2.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-03-20 11:30:00 | 695.87 | 695.37 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 684.60 | 687.46 | 0.00 | ORB-short ORB[690.20,698.05] vol=1.9x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:40:00 | 681.28 | 686.82 | 0.00 | T1 1.5R @ 681.28 |
| Stop hit — per-position SL triggered | 2026-03-27 13:40:00 | 684.60 | 683.57 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-03-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:50:00 | 655.00 | 660.06 | 0.00 | ORB-short ORB[661.00,667.00] vol=5.3x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:05:00 | 651.19 | 657.56 | 0.00 | T1 1.5R @ 651.19 |
| Target hit | 2026-03-30 15:20:00 | 635.80 | 646.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2026-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:25:00 | 696.50 | 691.07 | 0.00 | ORB-long ORB[680.75,690.75] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 694.29 | 691.93 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 676.65 | 679.89 | 0.00 | ORB-short ORB[679.65,687.90] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-04-24 10:05:00 | 679.04 | 678.94 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:35:00 | 665.00 | 659.90 | 0.00 | ORB-long ORB[651.30,657.95] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 661.56 | 660.98 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:45:00 | 634.50 | 640.45 | 0.00 | ORB-short ORB[642.10,650.00] vol=5.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-04-30 11:20:00 | 636.49 | 638.67 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-05-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:05:00 | 643.35 | 645.74 | 0.00 | ORB-short ORB[646.05,649.80] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 644.91 | 645.45 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:20:00 | 640.00 | 641.77 | 0.00 | ORB-short ORB[641.10,645.95] vol=2.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 641.60 | 640.56 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 645.00 | 648.16 | 0.00 | ORB-short ORB[649.75,655.75] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 646.45 | 647.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 11:00:00 | 898.95 | 2025-05-14 11:40:00 | 897.24 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-15 10:00:00 | 901.80 | 2025-05-15 10:35:00 | 905.36 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-15 10:00:00 | 901.80 | 2025-05-15 12:55:00 | 902.60 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-05-19 10:15:00 | 916.15 | 2025-05-19 11:00:00 | 913.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-21 11:10:00 | 902.65 | 2025-05-21 11:35:00 | 900.67 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-23 11:00:00 | 900.00 | 2025-05-23 11:20:00 | 897.63 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-26 10:05:00 | 908.65 | 2025-05-26 10:15:00 | 911.78 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-26 10:05:00 | 908.65 | 2025-05-26 11:40:00 | 908.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-27 11:10:00 | 910.90 | 2025-05-27 12:10:00 | 914.09 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-27 11:10:00 | 910.90 | 2025-05-27 13:45:00 | 910.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 10:50:00 | 918.65 | 2025-05-28 11:30:00 | 916.41 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-30 09:50:00 | 920.10 | 2025-05-30 10:35:00 | 917.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-03 10:30:00 | 917.20 | 2025-06-03 10:40:00 | 914.21 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-03 10:30:00 | 917.20 | 2025-06-03 10:45:00 | 917.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 10:05:00 | 951.85 | 2025-06-06 10:20:00 | 956.12 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-06 10:05:00 | 951.85 | 2025-06-06 15:20:00 | 994.85 | TARGET_HIT | 0.50 | 4.52% |
| SELL | retest1 | 2025-06-10 11:15:00 | 1007.70 | 2025-06-10 11:35:00 | 1004.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-06-10 11:15:00 | 1007.70 | 2025-06-10 15:20:00 | 1002.10 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-12 10:05:00 | 1009.15 | 2025-06-12 10:20:00 | 1006.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-17 11:00:00 | 998.70 | 2025-06-17 11:35:00 | 1000.43 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-18 11:15:00 | 983.00 | 2025-06-18 11:50:00 | 980.60 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-06-18 11:15:00 | 983.00 | 2025-06-18 15:20:00 | 973.25 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2025-06-19 09:40:00 | 964.85 | 2025-06-19 09:50:00 | 960.51 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-19 09:40:00 | 964.85 | 2025-06-19 15:20:00 | 938.70 | TARGET_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2025-07-09 10:25:00 | 928.00 | 2025-07-09 11:00:00 | 930.78 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-09 10:25:00 | 928.00 | 2025-07-09 11:50:00 | 928.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 11:00:00 | 930.10 | 2025-07-10 11:05:00 | 931.81 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-11 11:00:00 | 914.25 | 2025-07-11 11:10:00 | 916.36 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-17 09:30:00 | 904.75 | 2025-07-17 09:40:00 | 902.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-18 10:25:00 | 896.10 | 2025-07-18 11:05:00 | 898.31 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-22 09:55:00 | 892.30 | 2025-07-22 10:10:00 | 894.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-23 09:30:00 | 888.35 | 2025-07-23 09:35:00 | 890.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-24 09:30:00 | 886.75 | 2025-07-24 09:45:00 | 888.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-31 10:15:00 | 804.00 | 2025-07-31 10:25:00 | 805.78 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-01 11:15:00 | 799.05 | 2025-08-01 12:00:00 | 800.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-06 10:00:00 | 796.60 | 2025-08-06 10:05:00 | 793.14 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-08-06 10:00:00 | 796.60 | 2025-08-06 13:55:00 | 791.35 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-08-11 09:40:00 | 793.75 | 2025-08-11 10:20:00 | 796.95 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-11 09:40:00 | 793.75 | 2025-08-11 15:20:00 | 797.00 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-12 10:05:00 | 800.85 | 2025-08-12 10:30:00 | 798.96 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-19 11:10:00 | 815.10 | 2025-08-19 11:20:00 | 813.71 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-08-25 10:05:00 | 824.25 | 2025-08-25 10:10:00 | 826.77 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-08-25 10:05:00 | 824.25 | 2025-08-25 10:25:00 | 824.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-02 10:50:00 | 805.70 | 2025-09-02 11:30:00 | 802.60 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-02 10:50:00 | 805.70 | 2025-09-02 15:20:00 | 793.45 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2025-09-03 10:00:00 | 803.00 | 2025-09-03 10:40:00 | 801.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-05 10:30:00 | 793.50 | 2025-09-05 11:00:00 | 795.36 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-08 10:05:00 | 796.10 | 2025-09-08 10:10:00 | 798.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-08 10:05:00 | 796.10 | 2025-09-08 10:25:00 | 796.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-09 10:15:00 | 810.80 | 2025-09-09 10:50:00 | 809.11 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-10 10:55:00 | 840.50 | 2025-09-10 11:20:00 | 838.66 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-18 09:30:00 | 880.45 | 2025-09-18 09:40:00 | 883.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-19 11:00:00 | 883.15 | 2025-09-19 11:35:00 | 879.93 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-19 11:00:00 | 883.15 | 2025-09-19 11:55:00 | 883.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 10:20:00 | 865.00 | 2025-09-23 10:35:00 | 867.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-24 09:35:00 | 883.50 | 2025-09-24 09:40:00 | 880.69 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-25 10:50:00 | 892.20 | 2025-09-25 11:20:00 | 896.07 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-25 10:50:00 | 892.20 | 2025-09-25 11:25:00 | 892.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 11:05:00 | 904.85 | 2025-10-07 12:45:00 | 908.70 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-07 11:05:00 | 904.85 | 2025-10-07 15:05:00 | 904.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:00:00 | 913.85 | 2025-10-14 11:15:00 | 915.94 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-16 09:30:00 | 938.25 | 2025-10-16 10:00:00 | 936.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-20 10:40:00 | 937.10 | 2025-10-20 10:50:00 | 940.57 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-20 10:40:00 | 937.10 | 2025-10-20 11:25:00 | 937.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:50:00 | 886.00 | 2025-11-03 12:05:00 | 884.37 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-04 09:40:00 | 894.60 | 2025-11-04 09:45:00 | 892.18 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-10 11:15:00 | 874.75 | 2025-11-10 11:25:00 | 872.88 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-17 11:00:00 | 883.70 | 2025-11-17 11:10:00 | 881.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-20 10:55:00 | 872.40 | 2025-11-20 12:25:00 | 874.41 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-11-20 10:55:00 | 872.40 | 2025-11-20 14:50:00 | 872.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:50:00 | 869.10 | 2025-11-21 10:55:00 | 870.97 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-24 09:30:00 | 890.80 | 2025-11-24 09:45:00 | 894.44 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-24 09:30:00 | 890.80 | 2025-11-24 10:55:00 | 895.35 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-01 10:50:00 | 871.35 | 2025-12-01 10:55:00 | 872.94 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-03 11:10:00 | 877.70 | 2025-12-03 12:30:00 | 874.87 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-03 11:10:00 | 877.70 | 2025-12-03 15:20:00 | 867.20 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-12-05 10:05:00 | 866.10 | 2025-12-05 10:15:00 | 870.79 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-12-05 10:05:00 | 866.10 | 2025-12-05 15:20:00 | 887.05 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2025-12-09 10:45:00 | 860.40 | 2025-12-09 11:10:00 | 862.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-10 10:15:00 | 863.85 | 2025-12-10 10:55:00 | 860.81 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-10 10:15:00 | 863.85 | 2025-12-10 14:50:00 | 863.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 11:10:00 | 873.95 | 2025-12-11 11:30:00 | 871.82 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-15 09:35:00 | 866.30 | 2025-12-15 09:40:00 | 868.46 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-16 10:25:00 | 853.35 | 2025-12-16 10:30:00 | 850.65 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-16 10:25:00 | 853.35 | 2025-12-16 10:35:00 | 853.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-17 10:55:00 | 835.90 | 2025-12-17 11:05:00 | 837.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-18 11:15:00 | 848.35 | 2025-12-18 11:20:00 | 852.01 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-12-18 11:15:00 | 848.35 | 2025-12-18 12:00:00 | 848.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:30:00 | 859.30 | 2025-12-19 09:35:00 | 857.16 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-24 10:25:00 | 876.25 | 2025-12-24 11:05:00 | 874.44 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-29 11:05:00 | 858.40 | 2025-12-29 11:20:00 | 855.75 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-29 11:05:00 | 858.40 | 2025-12-29 15:20:00 | 851.30 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-12-30 10:30:00 | 841.10 | 2025-12-30 10:40:00 | 838.37 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-30 10:30:00 | 841.10 | 2025-12-30 12:15:00 | 836.00 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2026-01-01 11:15:00 | 864.75 | 2026-01-01 11:40:00 | 863.40 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-02 10:55:00 | 875.20 | 2026-01-02 11:00:00 | 873.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-05 10:55:00 | 867.20 | 2026-01-05 11:00:00 | 868.81 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-07 09:50:00 | 888.65 | 2026-01-07 10:05:00 | 891.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-01-08 10:55:00 | 872.10 | 2026-01-08 11:40:00 | 874.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-14 10:30:00 | 852.35 | 2026-01-14 11:15:00 | 854.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-19 10:25:00 | 834.05 | 2026-01-19 10:45:00 | 836.07 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-27 09:45:00 | 776.95 | 2026-01-27 11:05:00 | 773.58 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-06 10:35:00 | 743.60 | 2026-02-06 10:55:00 | 745.82 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-09 09:35:00 | 763.60 | 2026-02-09 09:55:00 | 766.91 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-09 09:35:00 | 763.60 | 2026-02-09 10:15:00 | 763.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:50:00 | 773.70 | 2026-02-12 11:15:00 | 771.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-04 09:35:00 | 729.95 | 2026-03-04 10:40:00 | 725.11 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-04 09:35:00 | 729.95 | 2026-03-04 13:10:00 | 729.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:45:00 | 725.95 | 2026-03-05 10:50:00 | 727.81 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-20 10:35:00 | 698.00 | 2026-03-20 11:30:00 | 695.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-27 11:05:00 | 684.60 | 2026-03-27 11:40:00 | 681.28 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-27 11:05:00 | 684.60 | 2026-03-27 13:40:00 | 684.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 10:50:00 | 655.00 | 2026-03-30 11:05:00 | 651.19 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-30 10:50:00 | 655.00 | 2026-03-30 15:20:00 | 635.80 | TARGET_HIT | 0.50 | 2.93% |
| BUY | retest1 | 2026-04-17 10:25:00 | 696.50 | 2026-04-17 10:45:00 | 694.29 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-24 09:45:00 | 676.65 | 2026-04-24 10:05:00 | 679.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-28 10:35:00 | 665.00 | 2026-04-28 11:00:00 | 661.56 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-30 10:45:00 | 634.50 | 2026-04-30 11:20:00 | 636.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-04 11:05:00 | 643.35 | 2026-05-04 11:30:00 | 644.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-05 10:20:00 | 640.00 | 2026-05-05 11:25:00 | 641.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-07 11:00:00 | 645.00 | 2026-05-07 11:30:00 | 646.45 | STOP_HIT | 1.00 | -0.23% |
