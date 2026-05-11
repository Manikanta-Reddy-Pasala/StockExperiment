# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2024-06-10 09:15:00 → 2026-05-08 15:25:00 (35425 bars)
- **Last close:** 565.50
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
| ENTRY1 | 81 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 15 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 66
- **Target hits / Stop hits / Partials:** 15 / 66 / 37
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 20.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 26 | 40.0% | 7 | 39 | 19 | 0.11% | 6.9% |
| BUY @ 2nd Alert (retest1) | 65 | 26 | 40.0% | 7 | 39 | 19 | 0.11% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 26 | 49.1% | 8 | 27 | 18 | 0.26% | 13.9% |
| SELL @ 2nd Alert (retest1) | 53 | 26 | 49.1% | 8 | 27 | 18 | 0.26% | 13.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 118 | 52 | 44.1% | 15 | 66 | 37 | 0.18% | 20.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:10:00 | 1021.75 | 1028.71 | 0.00 | ORB-short ORB[1027.25,1037.00] vol=1.8x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 1018.76 | 1027.97 | 0.00 | T1 1.5R @ 1018.76 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 1021.75 | 1027.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 1029.60 | 1024.09 | 0.00 | ORB-long ORB[1018.70,1028.90] vol=2.1x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:35:00 | 1033.67 | 1026.21 | 0.00 | T1 1.5R @ 1033.67 |
| Stop hit — per-position SL triggered | 2024-06-18 09:55:00 | 1029.60 | 1029.43 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 1023.45 | 1015.28 | 0.00 | ORB-long ORB[1007.20,1017.35] vol=1.9x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-06-20 09:55:00 | 1019.84 | 1017.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 1025.25 | 1022.39 | 0.00 | ORB-long ORB[1016.10,1024.45] vol=1.8x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:35:00 | 1028.82 | 1024.96 | 0.00 | T1 1.5R @ 1028.82 |
| Target hit | 2024-06-21 10:45:00 | 1037.40 | 1037.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 1006.30 | 1011.02 | 0.00 | ORB-short ORB[1008.45,1018.80] vol=1.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:45:00 | 1001.73 | 1008.95 | 0.00 | T1 1.5R @ 1001.73 |
| Target hit | 2024-06-25 15:20:00 | 995.50 | 999.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:55:00 | 1003.70 | 999.18 | 0.00 | ORB-long ORB[990.25,999.95] vol=2.0x ATR=3.05 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 1000.65 | 999.32 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 1000.40 | 996.82 | 0.00 | ORB-long ORB[993.30,997.85] vol=1.8x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:45:00 | 1003.80 | 1000.08 | 0.00 | T1 1.5R @ 1003.80 |
| Target hit | 2024-07-02 11:35:00 | 1005.70 | 1005.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2024-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:50:00 | 1003.80 | 1004.63 | 0.00 | ORB-short ORB[1003.85,1010.00] vol=3.2x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-07-03 12:50:00 | 1006.20 | 1004.00 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:35:00 | 1015.90 | 1010.87 | 0.00 | ORB-long ORB[1006.85,1011.35] vol=2.1x ATR=2.72 |
| Stop hit — per-position SL triggered | 2024-07-04 09:45:00 | 1013.18 | 1011.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 1015.60 | 1012.23 | 0.00 | ORB-long ORB[1003.05,1014.00] vol=5.3x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:05:00 | 1019.58 | 1014.18 | 0.00 | T1 1.5R @ 1019.58 |
| Target hit | 2024-07-05 15:20:00 | 1025.95 | 1022.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:35:00 | 1040.55 | 1035.58 | 0.00 | ORB-long ORB[1027.80,1038.00] vol=3.2x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 1036.72 | 1037.35 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:00:00 | 1026.05 | 1030.82 | 0.00 | ORB-short ORB[1029.75,1037.30] vol=2.0x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 1021.59 | 1028.68 | 0.00 | T1 1.5R @ 1021.59 |
| Target hit | 2024-07-10 11:50:00 | 1018.85 | 1017.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:40:00 | 1037.70 | 1032.72 | 0.00 | ORB-long ORB[1026.95,1033.70] vol=3.9x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 1034.16 | 1035.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:00:00 | 1037.80 | 1033.84 | 0.00 | ORB-long ORB[1028.60,1035.00] vol=3.5x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-07-12 10:30:00 | 1034.90 | 1034.41 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:10:00 | 991.65 | 1003.37 | 0.00 | ORB-short ORB[998.00,1012.65] vol=2.1x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-07-19 10:20:00 | 995.95 | 1002.42 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1004.55 | 1009.16 | 0.00 | ORB-short ORB[1005.00,1015.70] vol=3.8x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 1008.35 | 1009.01 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:55:00 | 996.35 | 991.98 | 0.00 | ORB-long ORB[985.30,991.90] vol=3.4x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-07-29 10:30:00 | 993.38 | 993.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:15:00 | 989.15 | 991.06 | 0.00 | ORB-short ORB[990.00,994.40] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-07-31 10:35:00 | 990.71 | 990.98 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:50:00 | 992.95 | 990.50 | 0.00 | ORB-long ORB[988.05,991.70] vol=1.8x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-08-01 10:20:00 | 990.96 | 990.96 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:45:00 | 964.65 | 969.39 | 0.00 | ORB-short ORB[965.10,974.90] vol=2.1x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-08-02 09:55:00 | 968.02 | 968.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:15:00 | 929.90 | 931.96 | 0.00 | ORB-short ORB[930.00,935.65] vol=1.6x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 12:50:00 | 927.31 | 931.39 | 0.00 | T1 1.5R @ 927.31 |
| Stop hit — per-position SL triggered | 2024-08-09 13:45:00 | 929.90 | 931.03 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:00:00 | 927.80 | 921.06 | 0.00 | ORB-long ORB[916.00,926.50] vol=2.2x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:05:00 | 931.58 | 921.88 | 0.00 | T1 1.5R @ 931.58 |
| Stop hit — per-position SL triggered | 2024-08-12 13:45:00 | 927.80 | 924.83 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:50:00 | 922.10 | 926.73 | 0.00 | ORB-short ORB[925.15,931.45] vol=1.5x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:30:00 | 918.02 | 923.86 | 0.00 | T1 1.5R @ 918.02 |
| Stop hit — per-position SL triggered | 2024-08-13 13:25:00 | 922.10 | 922.70 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 909.45 | 917.26 | 0.00 | ORB-short ORB[916.05,929.00] vol=1.7x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-08-14 11:20:00 | 912.89 | 916.34 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:45:00 | 937.85 | 934.46 | 0.00 | ORB-long ORB[925.15,937.60] vol=1.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-08-19 10:35:00 | 935.56 | 935.69 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:45:00 | 929.80 | 935.84 | 0.00 | ORB-short ORB[936.05,942.40] vol=1.9x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:25:00 | 926.94 | 934.12 | 0.00 | T1 1.5R @ 926.94 |
| Stop hit — per-position SL triggered | 2024-08-20 12:05:00 | 929.80 | 933.49 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:35:00 | 931.55 | 934.58 | 0.00 | ORB-short ORB[932.40,938.10] vol=1.5x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-08-23 09:50:00 | 933.45 | 933.70 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 928.55 | 930.61 | 0.00 | ORB-short ORB[928.65,933.50] vol=1.6x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:25:00 | 926.45 | 929.28 | 0.00 | T1 1.5R @ 926.45 |
| Stop hit — per-position SL triggered | 2024-08-27 11:20:00 | 928.55 | 928.74 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 916.60 | 922.21 | 0.00 | ORB-short ORB[922.30,928.50] vol=2.0x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 918.37 | 921.75 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 11:15:00 | 939.90 | 936.96 | 0.00 | ORB-long ORB[932.20,938.85] vol=2.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-09-04 11:50:00 | 938.21 | 937.46 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 931.10 | 936.17 | 0.00 | ORB-short ORB[941.50,948.20] vol=1.9x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-09-06 11:35:00 | 933.95 | 935.19 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:10:00 | 931.95 | 928.87 | 0.00 | ORB-long ORB[924.75,928.80] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 930.34 | 928.91 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 901.05 | 907.20 | 0.00 | ORB-short ORB[906.20,914.25] vol=1.6x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:45:00 | 896.16 | 904.21 | 0.00 | T1 1.5R @ 896.16 |
| Target hit | 2024-09-19 15:10:00 | 883.55 | 883.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:50:00 | 904.45 | 899.56 | 0.00 | ORB-long ORB[895.05,900.30] vol=3.3x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:05:00 | 907.15 | 900.23 | 0.00 | T1 1.5R @ 907.15 |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 904.45 | 900.59 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:25:00 | 915.70 | 910.54 | 0.00 | ORB-long ORB[902.05,912.15] vol=2.2x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:30:00 | 919.19 | 911.43 | 0.00 | T1 1.5R @ 919.19 |
| Stop hit — per-position SL triggered | 2024-09-27 10:35:00 | 915.70 | 911.81 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:30:00 | 906.65 | 912.95 | 0.00 | ORB-short ORB[911.55,922.80] vol=7.0x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:20:00 | 902.09 | 908.76 | 0.00 | T1 1.5R @ 902.09 |
| Target hit | 2024-10-03 15:20:00 | 885.50 | 895.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:30:00 | 870.00 | 874.12 | 0.00 | ORB-short ORB[875.15,880.75] vol=4.0x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:40:00 | 865.63 | 872.93 | 0.00 | T1 1.5R @ 865.63 |
| Target hit | 2024-10-07 11:20:00 | 860.30 | 859.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2024-10-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:30:00 | 882.20 | 876.55 | 0.00 | ORB-long ORB[873.35,879.30] vol=2.4x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:20:00 | 886.42 | 878.92 | 0.00 | T1 1.5R @ 886.42 |
| Stop hit — per-position SL triggered | 2024-10-09 11:50:00 | 882.20 | 879.81 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:00:00 | 889.30 | 887.07 | 0.00 | ORB-long ORB[880.40,886.40] vol=1.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 887.03 | 887.27 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:00:00 | 890.55 | 885.18 | 0.00 | ORB-long ORB[878.05,883.10] vol=1.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:10:00 | 893.50 | 886.81 | 0.00 | T1 1.5R @ 893.50 |
| Stop hit — per-position SL triggered | 2024-10-11 10:35:00 | 890.55 | 888.30 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:05:00 | 896.05 | 891.57 | 0.00 | ORB-long ORB[883.60,893.00] vol=3.4x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-10-15 10:10:00 | 894.01 | 891.89 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:45:00 | 898.50 | 896.28 | 0.00 | ORB-long ORB[891.00,897.95] vol=1.9x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-10-16 09:55:00 | 896.49 | 896.43 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:05:00 | 840.00 | 832.46 | 0.00 | ORB-long ORB[827.15,835.90] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-10-24 12:15:00 | 836.51 | 836.14 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 822.55 | 826.86 | 0.00 | ORB-short ORB[825.05,832.85] vol=1.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:35:00 | 819.15 | 825.40 | 0.00 | T1 1.5R @ 819.15 |
| Target hit | 2024-10-25 11:15:00 | 813.50 | 811.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2024-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:45:00 | 808.15 | 817.33 | 0.00 | ORB-short ORB[818.60,825.60] vol=2.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-10-29 11:05:00 | 810.93 | 816.06 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:30:00 | 834.20 | 827.94 | 0.00 | ORB-long ORB[822.15,831.25] vol=1.8x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-11-11 09:40:00 | 831.03 | 829.01 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:55:00 | 825.15 | 832.18 | 0.00 | ORB-short ORB[834.15,843.80] vol=1.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 827.60 | 831.07 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 810.80 | 805.60 | 0.00 | ORB-long ORB[796.00,807.50] vol=1.9x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:35:00 | 814.80 | 809.20 | 0.00 | T1 1.5R @ 814.80 |
| Stop hit — per-position SL triggered | 2024-11-19 12:40:00 | 810.80 | 810.67 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:35:00 | 783.50 | 792.63 | 0.00 | ORB-short ORB[789.60,801.30] vol=1.5x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 786.44 | 791.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 827.05 | 819.54 | 0.00 | ORB-long ORB[812.00,816.70] vol=1.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:45:00 | 830.57 | 823.61 | 0.00 | T1 1.5R @ 830.57 |
| Stop hit — per-position SL triggered | 2024-11-27 11:10:00 | 827.05 | 824.11 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 839.05 | 837.09 | 0.00 | ORB-long ORB[831.40,838.00] vol=2.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-12-05 09:35:00 | 837.13 | 837.13 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:45:00 | 839.65 | 836.62 | 0.00 | ORB-long ORB[829.85,839.60] vol=1.9x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-12-09 10:50:00 | 837.95 | 836.65 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:30:00 | 843.80 | 841.71 | 0.00 | ORB-long ORB[835.30,843.15] vol=3.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-12-10 09:35:00 | 841.78 | 841.74 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:10:00 | 839.45 | 836.77 | 0.00 | ORB-long ORB[835.00,838.90] vol=1.7x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:15:00 | 841.79 | 837.20 | 0.00 | T1 1.5R @ 841.79 |
| Stop hit — per-position SL triggered | 2024-12-11 10:20:00 | 839.45 | 837.34 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 825.20 | 831.84 | 0.00 | ORB-short ORB[832.20,839.25] vol=1.6x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:30:00 | 821.49 | 829.89 | 0.00 | T1 1.5R @ 821.49 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 825.20 | 828.94 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 779.95 | 784.11 | 0.00 | ORB-short ORB[782.15,792.95] vol=2.3x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 782.36 | 783.71 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 780.90 | 784.65 | 0.00 | ORB-short ORB[785.55,792.20] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 782.73 | 784.40 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:40:00 | 793.50 | 790.33 | 0.00 | ORB-long ORB[784.20,791.00] vol=1.8x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-01-01 09:45:00 | 791.33 | 790.42 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:45:00 | 803.35 | 801.19 | 0.00 | ORB-long ORB[791.00,802.60] vol=1.8x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 801.48 | 801.26 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:05:00 | 767.20 | 770.40 | 0.00 | ORB-short ORB[767.55,773.20] vol=1.6x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 769.18 | 770.01 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:15:00 | 787.55 | 779.34 | 0.00 | ORB-long ORB[766.80,772.60] vol=3.7x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:30:00 | 791.20 | 781.27 | 0.00 | T1 1.5R @ 791.20 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 787.55 | 784.63 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:45:00 | 742.60 | 747.72 | 0.00 | ORB-short ORB[748.10,757.00] vol=1.9x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:50:00 | 738.25 | 746.69 | 0.00 | T1 1.5R @ 738.25 |
| Stop hit — per-position SL triggered | 2025-01-28 10:00:00 | 742.60 | 745.53 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 773.20 | 768.71 | 0.00 | ORB-long ORB[763.05,772.75] vol=1.9x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:40:00 | 776.40 | 771.69 | 0.00 | T1 1.5R @ 776.40 |
| Target hit | 2025-01-30 10:55:00 | 776.00 | 777.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2025-01-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:35:00 | 801.60 | 790.71 | 0.00 | ORB-long ORB[775.00,785.95] vol=1.8x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:10:00 | 806.01 | 794.55 | 0.00 | T1 1.5R @ 806.01 |
| Target hit | 2025-01-31 15:20:00 | 821.80 | 809.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-02-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:30:00 | 790.80 | 787.60 | 0.00 | ORB-long ORB[781.00,790.00] vol=1.5x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 12:10:00 | 793.89 | 789.97 | 0.00 | T1 1.5R @ 793.89 |
| Target hit | 2025-02-05 14:55:00 | 791.75 | 791.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2025-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 10:25:00 | 779.15 | 773.10 | 0.00 | ORB-long ORB[767.35,777.00] vol=1.8x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:50:00 | 782.95 | 775.42 | 0.00 | T1 1.5R @ 782.95 |
| Stop hit — per-position SL triggered | 2025-02-10 12:00:00 | 779.15 | 780.33 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:05:00 | 730.60 | 724.69 | 0.00 | ORB-long ORB[716.00,724.35] vol=1.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-02-20 11:25:00 | 728.83 | 725.14 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-03-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 10:50:00 | 661.10 | 666.84 | 0.00 | ORB-short ORB[670.00,677.95] vol=2.3x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 12:00:00 | 656.45 | 664.94 | 0.00 | T1 1.5R @ 656.45 |
| Stop hit — per-position SL triggered | 2025-03-03 12:20:00 | 661.10 | 664.33 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:30:00 | 683.00 | 680.03 | 0.00 | ORB-long ORB[673.05,682.90] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 680.92 | 681.96 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 697.65 | 701.00 | 0.00 | ORB-short ORB[698.05,705.60] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-03-06 09:35:00 | 700.09 | 701.00 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:15:00 | 701.70 | 699.09 | 0.00 | ORB-long ORB[693.15,700.00] vol=1.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-03-18 12:40:00 | 700.21 | 699.67 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 714.50 | 717.39 | 0.00 | ORB-short ORB[715.30,721.95] vol=1.7x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:40:00 | 711.19 | 716.67 | 0.00 | T1 1.5R @ 711.19 |
| Stop hit — per-position SL triggered | 2025-03-26 09:50:00 | 714.50 | 715.61 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 11:00:00 | 729.25 | 724.97 | 0.00 | ORB-long ORB[723.00,729.00] vol=2.7x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 11:10:00 | 732.16 | 726.07 | 0.00 | T1 1.5R @ 732.16 |
| Stop hit — per-position SL triggered | 2025-04-11 13:50:00 | 729.25 | 728.68 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:00:00 | 764.15 | 760.67 | 0.00 | ORB-long ORB[755.00,762.90] vol=2.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 12:45:00 | 766.65 | 762.74 | 0.00 | T1 1.5R @ 766.65 |
| Target hit | 2025-04-17 15:20:00 | 770.15 | 765.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:10:00 | 779.00 | 775.05 | 0.00 | ORB-long ORB[769.00,778.00] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-04-22 10:30:00 | 777.12 | 775.47 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 772.75 | 774.78 | 0.00 | ORB-short ORB[773.05,778.85] vol=1.7x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:50:00 | 770.27 | 773.77 | 0.00 | T1 1.5R @ 770.27 |
| Target hit | 2025-04-23 11:30:00 | 770.65 | 770.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 77 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 778.95 | 781.21 | 0.00 | ORB-short ORB[779.10,785.00] vol=2.0x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-04-24 10:25:00 | 781.13 | 780.56 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 771.75 | 776.15 | 0.00 | ORB-short ORB[773.95,781.30] vol=1.8x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 768.05 | 772.89 | 0.00 | T1 1.5R @ 768.05 |
| Target hit | 2025-04-25 15:20:00 | 751.15 | 760.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 772.65 | 770.25 | 0.00 | ORB-long ORB[765.50,772.00] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 770.51 | 770.27 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-05-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:05:00 | 758.45 | 754.18 | 0.00 | ORB-long ORB[749.95,754.65] vol=1.7x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-05-05 10:55:00 | 756.28 | 755.21 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:15:00 | 751.10 | 756.96 | 0.00 | ORB-short ORB[757.10,766.95] vol=2.3x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:25:00 | 748.47 | 755.91 | 0.00 | T1 1.5R @ 748.47 |
| Stop hit — per-position SL triggered | 2025-05-06 11:55:00 | 751.10 | 755.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-13 11:10:00 | 1021.75 | 2024-06-13 11:20:00 | 1018.76 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-06-13 11:10:00 | 1021.75 | 2024-06-13 11:25:00 | 1021.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 09:30:00 | 1029.60 | 2024-06-18 09:35:00 | 1033.67 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-18 09:30:00 | 1029.60 | 2024-06-18 09:55:00 | 1029.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:30:00 | 1023.45 | 2024-06-20 09:55:00 | 1019.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1025.25 | 2024-06-21 09:35:00 | 1028.82 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-06-21 09:30:00 | 1025.25 | 2024-06-21 10:45:00 | 1037.40 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1006.30 | 2024-06-25 09:45:00 | 1001.73 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1006.30 | 2024-06-25 15:20:00 | 995.50 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2024-06-28 09:55:00 | 1003.70 | 2024-06-28 10:00:00 | 1000.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-02 09:35:00 | 1000.40 | 2024-07-02 09:45:00 | 1003.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-07-02 09:35:00 | 1000.40 | 2024-07-02 11:35:00 | 1005.70 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-03 10:50:00 | 1003.80 | 2024-07-03 12:50:00 | 1006.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-04 09:35:00 | 1015.90 | 2024-07-04 09:45:00 | 1013.18 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-05 09:50:00 | 1015.60 | 2024-07-05 10:05:00 | 1019.58 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-05 09:50:00 | 1015.60 | 2024-07-05 15:20:00 | 1025.95 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-07-08 09:35:00 | 1040.55 | 2024-07-08 09:45:00 | 1036.72 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-10 10:00:00 | 1026.05 | 2024-07-10 10:05:00 | 1021.59 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-10 10:00:00 | 1026.05 | 2024-07-10 11:50:00 | 1018.85 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-07-11 09:40:00 | 1037.70 | 2024-07-11 10:15:00 | 1034.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-12 10:00:00 | 1037.80 | 2024-07-12 10:30:00 | 1034.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-19 10:10:00 | 991.65 | 2024-07-19 10:20:00 | 995.95 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1004.55 | 2024-07-23 11:20:00 | 1008.35 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-29 09:55:00 | 996.35 | 2024-07-29 10:30:00 | 993.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-31 10:15:00 | 989.15 | 2024-07-31 10:35:00 | 990.71 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-08-01 09:50:00 | 992.95 | 2024-08-01 10:20:00 | 990.96 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-02 09:45:00 | 964.65 | 2024-08-02 09:55:00 | 968.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-09 11:15:00 | 929.90 | 2024-08-09 12:50:00 | 927.31 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-08-09 11:15:00 | 929.90 | 2024-08-09 13:45:00 | 929.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 11:00:00 | 927.80 | 2024-08-12 11:05:00 | 931.58 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-12 11:00:00 | 927.80 | 2024-08-12 13:45:00 | 927.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-13 09:50:00 | 922.10 | 2024-08-13 12:30:00 | 918.02 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-13 09:50:00 | 922.10 | 2024-08-13 13:25:00 | 922.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 10:55:00 | 909.45 | 2024-08-14 11:20:00 | 912.89 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-19 09:45:00 | 937.85 | 2024-08-19 10:35:00 | 935.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-20 10:45:00 | 929.80 | 2024-08-20 11:25:00 | 926.94 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-20 10:45:00 | 929.80 | 2024-08-20 12:05:00 | 929.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 09:35:00 | 931.55 | 2024-08-23 09:50:00 | 933.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-27 09:35:00 | 928.55 | 2024-08-27 10:25:00 | 926.45 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-08-27 09:35:00 | 928.55 | 2024-08-27 11:20:00 | 928.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 10:55:00 | 916.60 | 2024-08-29 11:10:00 | 918.37 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-04 11:15:00 | 939.90 | 2024-09-04 11:50:00 | 938.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-06 11:05:00 | 931.10 | 2024-09-06 11:35:00 | 933.95 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-11 11:10:00 | 931.95 | 2024-09-11 11:15:00 | 930.34 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-19 09:40:00 | 901.05 | 2024-09-19 09:45:00 | 896.16 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-19 09:40:00 | 901.05 | 2024-09-19 15:10:00 | 883.55 | TARGET_HIT | 0.50 | 1.94% |
| BUY | retest1 | 2024-09-23 10:50:00 | 904.45 | 2024-09-23 11:05:00 | 907.15 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-09-23 10:50:00 | 904.45 | 2024-09-23 11:15:00 | 904.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:25:00 | 915.70 | 2024-09-27 10:30:00 | 919.19 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-27 10:25:00 | 915.70 | 2024-09-27 10:35:00 | 915.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-03 10:30:00 | 906.65 | 2024-10-03 11:20:00 | 902.09 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-03 10:30:00 | 906.65 | 2024-10-03 15:20:00 | 885.50 | TARGET_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2024-10-07 09:30:00 | 870.00 | 2024-10-07 09:40:00 | 865.63 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-07 09:30:00 | 870.00 | 2024-10-07 11:20:00 | 860.30 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2024-10-09 10:30:00 | 882.20 | 2024-10-09 11:20:00 | 886.42 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-09 10:30:00 | 882.20 | 2024-10-09 11:50:00 | 882.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 10:00:00 | 889.30 | 2024-10-10 10:35:00 | 887.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-11 10:00:00 | 890.55 | 2024-10-11 10:10:00 | 893.50 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-10-11 10:00:00 | 890.55 | 2024-10-11 10:35:00 | 890.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 10:05:00 | 896.05 | 2024-10-15 10:10:00 | 894.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-16 09:45:00 | 898.50 | 2024-10-16 09:55:00 | 896.49 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-10-24 10:05:00 | 840.00 | 2024-10-24 12:15:00 | 836.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-25 09:30:00 | 822.55 | 2024-10-25 09:35:00 | 819.15 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-25 09:30:00 | 822.55 | 2024-10-25 11:15:00 | 813.50 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2024-10-29 10:45:00 | 808.15 | 2024-10-29 11:05:00 | 810.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-11 09:30:00 | 834.20 | 2024-11-11 09:40:00 | 831.03 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-12 10:55:00 | 825.15 | 2024-11-12 11:20:00 | 827.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-19 09:30:00 | 810.80 | 2024-11-19 10:35:00 | 814.80 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-11-19 09:30:00 | 810.80 | 2024-11-19 12:40:00 | 810.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-21 09:35:00 | 783.50 | 2024-11-21 09:40:00 | 786.44 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-27 09:55:00 | 827.05 | 2024-11-27 10:45:00 | 830.57 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-27 09:55:00 | 827.05 | 2024-11-27 11:10:00 | 827.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-05 09:30:00 | 839.05 | 2024-12-05 09:35:00 | 837.13 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-09 10:45:00 | 839.65 | 2024-12-09 10:50:00 | 837.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-10 09:30:00 | 843.80 | 2024-12-10 09:35:00 | 841.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-11 10:10:00 | 839.45 | 2024-12-11 10:15:00 | 841.79 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-12-11 10:10:00 | 839.45 | 2024-12-11 10:20:00 | 839.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:15:00 | 825.20 | 2024-12-13 10:30:00 | 821.49 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-13 10:15:00 | 825.20 | 2024-12-13 10:55:00 | 825.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 09:30:00 | 779.95 | 2024-12-24 09:40:00 | 782.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-26 10:50:00 | 780.90 | 2024-12-26 11:00:00 | 782.73 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-01 09:40:00 | 793.50 | 2025-01-01 09:45:00 | 791.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-03 09:45:00 | 803.35 | 2025-01-03 09:50:00 | 801.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-16 10:05:00 | 767.20 | 2025-01-16 10:15:00 | 769.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-23 10:15:00 | 787.55 | 2025-01-23 10:30:00 | 791.20 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-23 10:15:00 | 787.55 | 2025-01-23 11:40:00 | 787.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 09:45:00 | 742.60 | 2025-01-28 09:50:00 | 738.25 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-28 09:45:00 | 742.60 | 2025-01-28 10:00:00 | 742.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:30:00 | 773.20 | 2025-01-30 09:40:00 | 776.40 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-01-30 09:30:00 | 773.20 | 2025-01-30 10:55:00 | 776.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-31 10:35:00 | 801.60 | 2025-01-31 11:10:00 | 806.01 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-01-31 10:35:00 | 801.60 | 2025-01-31 15:20:00 | 821.80 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2025-02-05 10:30:00 | 790.80 | 2025-02-05 12:10:00 | 793.89 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-02-05 10:30:00 | 790.80 | 2025-02-05 14:55:00 | 791.75 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-02-10 10:25:00 | 779.15 | 2025-02-10 10:50:00 | 782.95 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-10 10:25:00 | 779.15 | 2025-02-10 12:00:00 | 779.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 11:05:00 | 730.60 | 2025-02-20 11:25:00 | 728.83 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-03 10:50:00 | 661.10 | 2025-03-03 12:00:00 | 656.45 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-03-03 10:50:00 | 661.10 | 2025-03-03 12:20:00 | 661.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 09:30:00 | 683.00 | 2025-03-05 10:15:00 | 680.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-06 09:30:00 | 697.65 | 2025-03-06 09:35:00 | 700.09 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-18 11:15:00 | 701.70 | 2025-03-18 12:40:00 | 700.21 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-03-26 09:35:00 | 714.50 | 2025-03-26 09:40:00 | 711.19 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-03-26 09:35:00 | 714.50 | 2025-03-26 09:50:00 | 714.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-11 11:00:00 | 729.25 | 2025-04-11 11:10:00 | 732.16 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-11 11:00:00 | 729.25 | 2025-04-11 13:50:00 | 729.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-17 11:00:00 | 764.15 | 2025-04-17 12:45:00 | 766.65 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-04-17 11:00:00 | 764.15 | 2025-04-17 15:20:00 | 770.15 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-04-22 10:10:00 | 779.00 | 2025-04-22 10:30:00 | 777.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-04-23 09:30:00 | 772.75 | 2025-04-23 09:50:00 | 770.27 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-04-23 09:30:00 | 772.75 | 2025-04-23 11:30:00 | 770.65 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-04-24 09:45:00 | 778.95 | 2025-04-24 10:25:00 | 781.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-25 09:30:00 | 771.75 | 2025-04-25 09:55:00 | 768.05 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-25 09:30:00 | 771.75 | 2025-04-25 15:20:00 | 751.15 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2025-04-29 09:30:00 | 772.65 | 2025-04-29 09:40:00 | 770.51 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-05 10:05:00 | 758.45 | 2025-05-05 10:55:00 | 756.28 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-06 11:15:00 | 751.10 | 2025-05-06 11:25:00 | 748.47 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-05-06 11:15:00 | 751.10 | 2025-05-06 11:55:00 | 751.10 | STOP_HIT | 0.50 | 0.00% |
