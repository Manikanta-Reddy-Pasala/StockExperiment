# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 954.50
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
| ENTRY1 | 70 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 12 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 58
- **Target hits / Stop hits / Partials:** 12 / 58 / 31
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 14.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 20 | 35.7% | 6 | 36 | 14 | 0.09% | 5.3% |
| BUY @ 2nd Alert (retest1) | 56 | 20 | 35.7% | 6 | 36 | 14 | 0.09% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 23 | 51.1% | 6 | 22 | 17 | 0.21% | 9.5% |
| SELL @ 2nd Alert (retest1) | 45 | 23 | 51.1% | 6 | 22 | 17 | 0.21% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 101 | 43 | 42.6% | 12 | 58 | 31 | 0.15% | 14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:40:00 | 923.35 | 912.77 | 0.00 | ORB-long ORB[902.85,911.20] vol=1.5x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 11:15:00 | 926.93 | 916.16 | 0.00 | T1 1.5R @ 926.93 |
| Stop hit — per-position SL triggered | 2025-05-21 11:35:00 | 923.35 | 916.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 924.45 | 919.11 | 0.00 | ORB-long ORB[911.10,919.15] vol=1.7x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 922.42 | 920.31 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 918.65 | 923.11 | 0.00 | ORB-short ORB[919.80,927.95] vol=1.6x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-05-29 09:50:00 | 920.40 | 921.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 899.25 | 892.63 | 0.00 | ORB-long ORB[887.60,897.95] vol=2.8x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 896.91 | 893.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:15:00 | 932.50 | 927.40 | 0.00 | ORB-long ORB[919.00,931.50] vol=2.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-06-24 11:20:00 | 929.72 | 927.55 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:05:00 | 926.00 | 929.88 | 0.00 | ORB-short ORB[930.20,939.85] vol=2.2x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:35:00 | 923.24 | 929.14 | 0.00 | T1 1.5R @ 923.24 |
| Target hit | 2025-07-02 15:05:00 | 924.45 | 924.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 938.20 | 931.57 | 0.00 | ORB-long ORB[922.00,935.55] vol=2.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2025-07-04 09:50:00 | 934.35 | 933.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:25:00 | 936.85 | 931.63 | 0.00 | ORB-long ORB[924.00,933.15] vol=2.3x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:30:00 | 940.15 | 933.40 | 0.00 | T1 1.5R @ 940.15 |
| Stop hit — per-position SL triggered | 2025-07-09 10:50:00 | 936.85 | 934.37 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:50:00 | 938.50 | 941.43 | 0.00 | ORB-short ORB[938.65,947.15] vol=4.0x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 935.67 | 940.97 | 0.00 | T1 1.5R @ 935.67 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 938.50 | 940.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:50:00 | 923.75 | 920.84 | 0.00 | ORB-long ORB[918.85,922.35] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-07-15 11:20:00 | 922.14 | 921.52 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:40:00 | 919.60 | 922.04 | 0.00 | ORB-short ORB[921.20,925.95] vol=1.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:10:00 | 917.03 | 921.01 | 0.00 | T1 1.5R @ 917.03 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 919.60 | 920.06 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:50:00 | 931.20 | 924.68 | 0.00 | ORB-long ORB[920.00,923.75] vol=3.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-07-18 10:00:00 | 928.47 | 925.52 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 11:05:00 | 968.95 | 963.00 | 0.00 | ORB-long ORB[953.65,965.50] vol=2.3x ATR=2.74 |
| Stop hit — per-position SL triggered | 2025-07-23 11:20:00 | 966.21 | 963.33 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:55:00 | 877.45 | 884.65 | 0.00 | ORB-short ORB[884.20,892.60] vol=3.0x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 879.89 | 883.77 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:00:00 | 895.20 | 885.71 | 0.00 | ORB-long ORB[874.30,884.95] vol=2.5x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-08-01 11:05:00 | 892.74 | 886.28 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:05:00 | 882.55 | 888.27 | 0.00 | ORB-short ORB[887.60,895.50] vol=2.1x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-08-06 10:25:00 | 885.15 | 887.37 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:30:00 | 861.90 | 872.58 | 0.00 | ORB-short ORB[873.05,884.90] vol=1.5x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:35:00 | 857.76 | 870.67 | 0.00 | T1 1.5R @ 857.76 |
| Stop hit — per-position SL triggered | 2025-08-12 10:45:00 | 861.90 | 870.04 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:05:00 | 851.05 | 853.57 | 0.00 | ORB-short ORB[851.20,858.90] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 853.30 | 852.72 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:45:00 | 893.00 | 896.15 | 0.00 | ORB-short ORB[896.80,909.90] vol=1.6x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-08-19 11:20:00 | 895.12 | 895.16 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:45:00 | 902.50 | 898.65 | 0.00 | ORB-long ORB[892.00,899.30] vol=2.0x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-08-22 09:55:00 | 900.29 | 899.27 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 11:10:00 | 889.25 | 895.34 | 0.00 | ORB-short ORB[895.00,901.90] vol=2.0x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:40:00 | 886.80 | 894.18 | 0.00 | T1 1.5R @ 886.80 |
| Target hit | 2025-08-26 15:20:00 | 874.60 | 882.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-09-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:35:00 | 891.30 | 885.47 | 0.00 | ORB-long ORB[877.85,884.70] vol=1.8x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:50:00 | 894.56 | 887.60 | 0.00 | T1 1.5R @ 894.56 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 891.30 | 888.13 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:55:00 | 897.95 | 895.17 | 0.00 | ORB-long ORB[891.45,896.45] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-09-02 10:45:00 | 896.18 | 895.97 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:35:00 | 964.95 | 956.52 | 0.00 | ORB-long ORB[947.70,955.00] vol=1.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-09-10 10:40:00 | 962.67 | 957.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 974.75 | 971.93 | 0.00 | ORB-long ORB[965.60,973.60] vol=1.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-09-11 09:35:00 | 972.52 | 972.07 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:35:00 | 977.40 | 974.30 | 0.00 | ORB-long ORB[970.85,974.85] vol=1.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-09-12 09:40:00 | 975.48 | 974.49 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:35:00 | 1016.50 | 1009.87 | 0.00 | ORB-long ORB[999.00,1011.45] vol=2.4x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:10:00 | 1020.83 | 1013.53 | 0.00 | T1 1.5R @ 1020.83 |
| Target hit | 2025-09-23 15:20:00 | 1025.20 | 1022.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:15:00 | 982.75 | 990.42 | 0.00 | ORB-short ORB[988.55,993.35] vol=3.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-09-29 11:50:00 | 984.78 | 989.30 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:10:00 | 1026.70 | 1019.46 | 0.00 | ORB-long ORB[1015.15,1024.90] vol=1.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-10-09 11:20:00 | 1024.63 | 1019.71 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:10:00 | 1026.35 | 1022.86 | 0.00 | ORB-long ORB[1017.00,1025.65] vol=2.0x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 1024.38 | 1022.93 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:05:00 | 1027.00 | 1031.32 | 0.00 | ORB-short ORB[1031.60,1042.10] vol=1.9x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 1022.94 | 1030.50 | 0.00 | T1 1.5R @ 1022.94 |
| Target hit | 2025-10-14 15:15:00 | 1020.60 | 1020.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2025-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:35:00 | 1035.00 | 1030.99 | 0.00 | ORB-long ORB[1018.40,1031.90] vol=3.3x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:00:00 | 1038.40 | 1032.55 | 0.00 | T1 1.5R @ 1038.40 |
| Target hit | 2025-10-15 15:20:00 | 1061.10 | 1049.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1071.25 | 1070.90 | 0.00 | ORB-long ORB[1058.85,1068.15] vol=5.7x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 1068.34 | 1070.78 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1071.55 | 1066.99 | 0.00 | ORB-long ORB[1056.10,1066.40] vol=2.2x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:30:00 | 1074.55 | 1067.78 | 0.00 | T1 1.5R @ 1074.55 |
| Stop hit — per-position SL triggered | 2025-10-17 12:05:00 | 1071.55 | 1069.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 1079.80 | 1082.52 | 0.00 | ORB-short ORB[1081.55,1094.00] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 12:50:00 | 1077.57 | 1081.83 | 0.00 | T1 1.5R @ 1077.57 |
| Stop hit — per-position SL triggered | 2025-10-27 13:30:00 | 1079.80 | 1081.32 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 1065.90 | 1069.17 | 0.00 | ORB-short ORB[1067.20,1079.15] vol=1.6x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 12:00:00 | 1062.49 | 1067.27 | 0.00 | T1 1.5R @ 1062.49 |
| Stop hit — per-position SL triggered | 2025-10-29 12:50:00 | 1065.90 | 1066.00 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:20:00 | 1060.40 | 1061.70 | 0.00 | ORB-short ORB[1061.25,1066.80] vol=2.0x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:50:00 | 1057.67 | 1061.00 | 0.00 | T1 1.5R @ 1057.67 |
| Target hit | 2025-10-30 15:20:00 | 1053.00 | 1053.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 1078.70 | 1071.26 | 0.00 | ORB-long ORB[1062.90,1073.60] vol=1.6x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-11-10 13:20:00 | 1075.80 | 1075.06 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 11:00:00 | 1015.80 | 1011.87 | 0.00 | ORB-long ORB[1007.30,1015.20] vol=2.4x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:25:00 | 1018.17 | 1012.93 | 0.00 | T1 1.5R @ 1018.17 |
| Stop hit — per-position SL triggered | 2025-11-20 11:45:00 | 1015.80 | 1013.34 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:00:00 | 1011.60 | 1020.31 | 0.00 | ORB-short ORB[1020.00,1027.60] vol=3.2x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 13:15:00 | 1007.92 | 1016.64 | 0.00 | T1 1.5R @ 1007.92 |
| Target hit | 2025-11-21 15:20:00 | 1004.80 | 1012.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 989.00 | 992.90 | 0.00 | ORB-short ORB[989.20,996.90] vol=1.8x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-11-25 11:05:00 | 990.67 | 992.70 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:05:00 | 1001.00 | 994.69 | 0.00 | ORB-long ORB[981.30,992.90] vol=2.8x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-11-26 10:10:00 | 998.42 | 995.15 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:25:00 | 1039.60 | 1036.26 | 0.00 | ORB-long ORB[1031.00,1038.90] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-11-28 10:30:00 | 1037.38 | 1036.39 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 1033.40 | 1034.90 | 0.00 | ORB-short ORB[1034.00,1042.00] vol=2.2x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 09:50:00 | 1030.42 | 1034.19 | 0.00 | T1 1.5R @ 1030.42 |
| Stop hit — per-position SL triggered | 2025-12-01 10:25:00 | 1033.40 | 1032.68 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:50:00 | 1035.00 | 1026.22 | 0.00 | ORB-long ORB[1013.90,1028.40] vol=1.5x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-12-02 11:05:00 | 1032.68 | 1027.25 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:55:00 | 1022.50 | 1022.08 | 0.00 | ORB-long ORB[1016.00,1021.90] vol=2.4x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-12-04 11:45:00 | 1020.58 | 1022.55 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 1041.80 | 1037.34 | 0.00 | ORB-long ORB[1026.50,1037.50] vol=5.5x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:05:00 | 1045.75 | 1039.22 | 0.00 | T1 1.5R @ 1045.75 |
| Stop hit — per-position SL triggered | 2025-12-05 10:10:00 | 1041.80 | 1039.54 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:15:00 | 1017.90 | 1019.07 | 0.00 | ORB-short ORB[1019.20,1026.00] vol=1.7x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:35:00 | 1014.29 | 1017.99 | 0.00 | T1 1.5R @ 1014.29 |
| Stop hit — per-position SL triggered | 2025-12-09 10:50:00 | 1017.90 | 1017.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:40:00 | 1019.10 | 1013.01 | 0.00 | ORB-long ORB[1005.10,1016.80] vol=1.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 1016.55 | 1013.26 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1006.20 | 1006.49 | 0.00 | ORB-short ORB[1006.40,1010.60] vol=2.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-12-16 11:05:00 | 1007.47 | 1006.54 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 1012.40 | 1008.34 | 0.00 | ORB-long ORB[1002.50,1009.80] vol=1.8x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-12-19 09:35:00 | 1010.28 | 1008.64 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 1024.00 | 1018.90 | 0.00 | ORB-long ORB[1010.00,1023.20] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 09:40:00 | 1027.41 | 1021.54 | 0.00 | T1 1.5R @ 1027.41 |
| Target hit | 2025-12-24 11:30:00 | 1029.90 | 1029.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 992.50 | 997.53 | 0.00 | ORB-short ORB[996.20,1003.90] vol=1.6x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:25:00 | 990.01 | 996.07 | 0.00 | T1 1.5R @ 990.01 |
| Stop hit — per-position SL triggered | 2025-12-29 11:40:00 | 992.50 | 995.62 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 982.10 | 984.84 | 0.00 | ORB-short ORB[983.20,994.20] vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:40:00 | 979.46 | 983.46 | 0.00 | T1 1.5R @ 979.46 |
| Stop hit — per-position SL triggered | 2025-12-31 11:50:00 | 982.10 | 982.95 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:15:00 | 980.25 | 983.30 | 0.00 | ORB-short ORB[981.50,989.00] vol=1.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2026-01-01 10:25:00 | 981.90 | 983.12 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:45:00 | 975.15 | 972.01 | 0.00 | ORB-long ORB[966.75,974.00] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-01-08 09:55:00 | 973.48 | 972.30 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 968.05 | 974.54 | 0.00 | ORB-short ORB[974.00,985.15] vol=1.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 964.97 | 972.06 | 0.00 | T1 1.5R @ 964.97 |
| Stop hit — per-position SL triggered | 2026-02-10 10:35:00 | 968.05 | 970.02 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 971.00 | 966.09 | 0.00 | ORB-long ORB[964.20,968.20] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:40:00 | 973.98 | 966.83 | 0.00 | T1 1.5R @ 973.98 |
| Stop hit — per-position SL triggered | 2026-02-11 10:50:00 | 971.00 | 967.72 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:45:00 | 975.10 | 971.91 | 0.00 | ORB-long ORB[965.00,972.45] vol=1.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:10:00 | 978.02 | 973.37 | 0.00 | T1 1.5R @ 978.02 |
| Target hit | 2026-02-12 15:20:00 | 999.90 | 990.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 1005.90 | 1003.19 | 0.00 | ORB-long ORB[992.80,1004.00] vol=1.8x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-02-13 10:20:00 | 1003.13 | 1003.58 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 1020.35 | 1017.25 | 0.00 | ORB-long ORB[1008.75,1019.90] vol=2.4x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:20:00 | 1024.37 | 1017.94 | 0.00 | T1 1.5R @ 1024.37 |
| Target hit | 2026-02-20 15:15:00 | 1027.20 | 1027.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — SELL (started 2026-02-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:50:00 | 1013.00 | 1016.60 | 0.00 | ORB-short ORB[1015.50,1028.75] vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 1016.50 | 1016.46 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:10:00 | 1023.50 | 1020.88 | 0.00 | ORB-long ORB[1015.75,1022.45] vol=1.6x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 1026.94 | 1021.46 | 0.00 | T1 1.5R @ 1026.94 |
| Stop hit — per-position SL triggered | 2026-02-26 11:50:00 | 1023.50 | 1023.04 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 916.05 | 926.52 | 0.00 | ORB-short ORB[932.20,941.25] vol=4.5x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:45:00 | 912.62 | 924.08 | 0.00 | T1 1.5R @ 912.62 |
| Target hit | 2026-03-11 15:20:00 | 890.80 | 905.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 867.30 | 873.13 | 0.00 | ORB-short ORB[871.05,881.00] vol=2.2x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:25:00 | 863.25 | 872.44 | 0.00 | T1 1.5R @ 863.25 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 867.30 | 869.66 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 921.25 | 918.75 | 0.00 | ORB-long ORB[912.30,919.50] vol=4.4x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 917.15 | 920.37 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 928.25 | 925.42 | 0.00 | ORB-long ORB[919.00,927.95] vol=1.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 925.85 | 926.70 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 908.95 | 917.70 | 0.00 | ORB-short ORB[916.15,925.00] vol=1.6x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-04-27 10:40:00 | 911.77 | 917.53 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:05:00 | 929.45 | 926.55 | 0.00 | ORB-long ORB[916.60,925.95] vol=4.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-04-29 11:30:00 | 926.72 | 926.96 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 952.65 | 947.45 | 0.00 | ORB-long ORB[938.10,951.85] vol=1.7x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:35:00 | 958.94 | 950.93 | 0.00 | T1 1.5R @ 958.94 |
| Target hit | 2026-05-04 12:40:00 | 953.55 | 954.21 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-21 10:40:00 | 923.35 | 2025-05-21 11:15:00 | 926.93 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-21 10:40:00 | 923.35 | 2025-05-21 11:35:00 | 923.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 10:45:00 | 924.45 | 2025-05-23 11:45:00 | 922.42 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-29 09:30:00 | 918.65 | 2025-05-29 09:50:00 | 920.40 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-06 10:05:00 | 899.25 | 2025-06-06 10:10:00 | 896.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-24 11:15:00 | 932.50 | 2025-06-24 11:20:00 | 929.72 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-02 11:05:00 | 926.00 | 2025-07-02 11:35:00 | 923.24 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-02 11:05:00 | 926.00 | 2025-07-02 15:05:00 | 924.45 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-07-04 09:30:00 | 938.20 | 2025-07-04 09:50:00 | 934.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-07-09 10:25:00 | 936.85 | 2025-07-09 10:30:00 | 940.15 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-09 10:25:00 | 936.85 | 2025-07-09 10:50:00 | 936.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:50:00 | 938.50 | 2025-07-11 11:00:00 | 935.67 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-11 10:50:00 | 938.50 | 2025-07-11 11:10:00 | 938.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:50:00 | 923.75 | 2025-07-15 11:20:00 | 922.14 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-17 09:40:00 | 919.60 | 2025-07-17 10:10:00 | 917.03 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-17 09:40:00 | 919.60 | 2025-07-17 11:10:00 | 919.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-18 09:50:00 | 931.20 | 2025-07-18 10:00:00 | 928.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-23 11:05:00 | 968.95 | 2025-07-23 11:20:00 | 966.21 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-30 10:55:00 | 877.45 | 2025-07-30 11:15:00 | 879.89 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-01 11:00:00 | 895.20 | 2025-08-01 11:05:00 | 892.74 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-06 10:05:00 | 882.55 | 2025-08-06 10:25:00 | 885.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-12 10:30:00 | 861.90 | 2025-08-12 10:35:00 | 857.76 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-12 10:30:00 | 861.90 | 2025-08-12 10:45:00 | 861.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 11:05:00 | 851.05 | 2025-08-13 12:15:00 | 853.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-19 10:45:00 | 893.00 | 2025-08-19 11:20:00 | 895.12 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-22 09:45:00 | 902.50 | 2025-08-22 09:55:00 | 900.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-26 11:10:00 | 889.25 | 2025-08-26 11:40:00 | 886.80 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-26 11:10:00 | 889.25 | 2025-08-26 15:20:00 | 874.60 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2025-09-01 09:35:00 | 891.30 | 2025-09-01 09:50:00 | 894.56 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-01 09:35:00 | 891.30 | 2025-09-01 09:55:00 | 891.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:55:00 | 897.95 | 2025-09-02 10:45:00 | 896.18 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-10 10:35:00 | 964.95 | 2025-09-10 10:40:00 | 962.67 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-11 09:30:00 | 974.75 | 2025-09-11 09:35:00 | 972.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-12 09:35:00 | 977.40 | 2025-09-12 09:40:00 | 975.48 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-23 09:35:00 | 1016.50 | 2025-09-23 10:10:00 | 1020.83 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-23 09:35:00 | 1016.50 | 2025-09-23 15:20:00 | 1025.20 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-09-29 11:15:00 | 982.75 | 2025-09-29 11:50:00 | 984.78 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-09 11:10:00 | 1026.70 | 2025-10-09 11:20:00 | 1024.63 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-10 11:10:00 | 1026.35 | 2025-10-10 11:15:00 | 1024.38 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-14 10:05:00 | 1027.00 | 2025-10-14 10:15:00 | 1022.94 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-14 10:05:00 | 1027.00 | 2025-10-14 15:15:00 | 1020.60 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-10-15 10:35:00 | 1035.00 | 2025-10-15 11:00:00 | 1038.40 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-15 10:35:00 | 1035.00 | 2025-10-15 15:20:00 | 1061.10 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2025-10-16 11:00:00 | 1071.25 | 2025-10-16 11:15:00 | 1068.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-17 11:15:00 | 1071.55 | 2025-10-17 11:30:00 | 1074.55 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-17 11:15:00 | 1071.55 | 2025-10-17 12:05:00 | 1071.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 11:05:00 | 1079.80 | 2025-10-27 12:50:00 | 1077.57 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-10-27 11:05:00 | 1079.80 | 2025-10-27 13:30:00 | 1079.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 10:55:00 | 1065.90 | 2025-10-29 12:00:00 | 1062.49 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-29 10:55:00 | 1065.90 | 2025-10-29 12:50:00 | 1065.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:20:00 | 1060.40 | 2025-10-30 10:50:00 | 1057.67 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-30 10:20:00 | 1060.40 | 2025-10-30 15:20:00 | 1053.00 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2025-11-10 11:00:00 | 1078.70 | 2025-11-10 13:20:00 | 1075.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-20 11:00:00 | 1015.80 | 2025-11-20 11:25:00 | 1018.17 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-11-20 11:00:00 | 1015.80 | 2025-11-20 11:45:00 | 1015.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 11:00:00 | 1011.60 | 2025-11-21 13:15:00 | 1007.92 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-21 11:00:00 | 1011.60 | 2025-11-21 15:20:00 | 1004.80 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2025-11-25 10:55:00 | 989.00 | 2025-11-25 11:05:00 | 990.67 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-26 10:05:00 | 1001.00 | 2025-11-26 10:10:00 | 998.42 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-28 10:25:00 | 1039.60 | 2025-11-28 10:30:00 | 1037.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-01 09:35:00 | 1033.40 | 2025-12-01 09:50:00 | 1030.42 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-01 09:35:00 | 1033.40 | 2025-12-01 10:25:00 | 1033.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-02 10:50:00 | 1035.00 | 2025-12-02 11:05:00 | 1032.68 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-04 10:55:00 | 1022.50 | 2025-12-04 11:45:00 | 1020.58 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-05 10:00:00 | 1041.80 | 2025-12-05 10:05:00 | 1045.75 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-05 10:00:00 | 1041.80 | 2025-12-05 10:10:00 | 1041.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-09 10:15:00 | 1017.90 | 2025-12-09 10:35:00 | 1014.29 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-09 10:15:00 | 1017.90 | 2025-12-09 10:50:00 | 1017.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:40:00 | 1019.10 | 2025-12-11 10:55:00 | 1016.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-16 11:00:00 | 1006.20 | 2025-12-16 11:05:00 | 1007.47 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-19 09:30:00 | 1012.40 | 2025-12-19 09:35:00 | 1010.28 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-24 09:30:00 | 1024.00 | 2025-12-24 09:40:00 | 1027.41 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-24 09:30:00 | 1024.00 | 2025-12-24 11:30:00 | 1029.90 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-12-29 11:00:00 | 992.50 | 2025-12-29 11:25:00 | 990.01 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-29 11:00:00 | 992.50 | 2025-12-29 11:40:00 | 992.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 11:00:00 | 982.10 | 2025-12-31 11:40:00 | 979.46 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-31 11:00:00 | 982.10 | 2025-12-31 11:50:00 | 982.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 10:15:00 | 980.25 | 2026-01-01 10:25:00 | 981.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-08 09:45:00 | 975.15 | 2026-01-08 09:55:00 | 973.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-10 09:50:00 | 968.05 | 2026-02-10 10:10:00 | 964.97 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 09:50:00 | 968.05 | 2026-02-10 10:35:00 | 968.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:35:00 | 971.00 | 2026-02-11 10:40:00 | 973.98 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-11 10:35:00 | 971.00 | 2026-02-11 10:50:00 | 971.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 09:45:00 | 975.10 | 2026-02-12 10:10:00 | 978.02 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-12 09:45:00 | 975.10 | 2026-02-12 15:20:00 | 999.90 | TARGET_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2026-02-13 10:10:00 | 1005.90 | 2026-02-13 10:20:00 | 1003.13 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-20 11:00:00 | 1020.35 | 2026-02-20 11:20:00 | 1024.37 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-20 11:00:00 | 1020.35 | 2026-02-20 15:15:00 | 1027.20 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-24 09:50:00 | 1013.00 | 2026-02-24 09:55:00 | 1016.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 11:10:00 | 1023.50 | 2026-02-26 11:15:00 | 1026.94 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-26 11:10:00 | 1023.50 | 2026-02-26 11:50:00 | 1023.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:15:00 | 916.05 | 2026-03-11 11:45:00 | 912.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-11 11:15:00 | 916.05 | 2026-03-11 15:20:00 | 890.80 | TARGET_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2026-03-17 11:15:00 | 867.30 | 2026-03-17 11:25:00 | 863.25 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-17 11:15:00 | 867.30 | 2026-03-17 13:15:00 | 867.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 921.25 | 2026-04-10 10:05:00 | 917.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-21 09:45:00 | 928.25 | 2026-04-21 10:25:00 | 925.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-27 10:35:00 | 908.95 | 2026-04-27 10:40:00 | 911.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-29 11:05:00 | 929.45 | 2026-04-29 11:30:00 | 926.72 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-04 09:35:00 | 952.65 | 2026-05-04 10:35:00 | 958.94 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-05-04 09:35:00 | 952.65 | 2026-05-04 12:40:00 | 953.55 | TARGET_HIT | 0.50 | 0.09% |
