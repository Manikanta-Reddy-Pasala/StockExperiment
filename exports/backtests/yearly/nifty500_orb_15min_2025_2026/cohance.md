# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 487.90
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
| PARTIAL | 21 |
| TARGET_HIT | 12 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 40
- **Target hits / Stop hits / Partials:** 12 / 40 / 21
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 12.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 10 | 40.0% | 4 | 15 | 6 | -0.01% | -0.3% |
| BUY @ 2nd Alert (retest1) | 25 | 10 | 40.0% | 4 | 15 | 6 | -0.01% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 23 | 47.9% | 8 | 25 | 15 | 0.27% | 12.8% |
| SELL @ 2nd Alert (retest1) | 48 | 23 | 47.9% | 8 | 25 | 15 | 0.27% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 73 | 33 | 45.2% | 12 | 40 | 21 | 0.17% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:35:00 | 1081.60 | 1071.43 | 0.00 | ORB-long ORB[1057.00,1070.00] vol=1.6x ATR=6.96 |
| Stop hit — per-position SL triggered | 2025-05-20 10:30:00 | 1074.64 | 1077.10 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:45:00 | 1062.70 | 1066.66 | 0.00 | ORB-short ORB[1070.00,1077.70] vol=2.0x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-05-22 12:20:00 | 1066.06 | 1064.22 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1090.40 | 1084.78 | 0.00 | ORB-long ORB[1069.70,1084.20] vol=2.1x ATR=5.82 |
| Stop hit — per-position SL triggered | 2025-05-23 09:45:00 | 1084.58 | 1086.96 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:50:00 | 1113.80 | 1100.34 | 0.00 | ORB-long ORB[1089.10,1105.00] vol=4.2x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-05-28 11:00:00 | 1108.58 | 1102.00 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1040.60 | 1034.68 | 0.00 | ORB-long ORB[1021.50,1036.60] vol=2.7x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:55:00 | 1047.47 | 1040.18 | 0.00 | T1 1.5R @ 1047.47 |
| Stop hit — per-position SL triggered | 2025-06-05 10:05:00 | 1040.60 | 1040.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:50:00 | 1011.90 | 1019.80 | 0.00 | ORB-short ORB[1022.00,1030.60] vol=2.0x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:00:00 | 1008.01 | 1019.11 | 0.00 | T1 1.5R @ 1008.01 |
| Stop hit — per-position SL triggered | 2025-06-10 11:05:00 | 1011.90 | 1018.79 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 996.90 | 991.38 | 0.00 | ORB-long ORB[983.20,994.40] vol=2.0x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-06-20 11:00:00 | 992.80 | 992.39 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 11:00:00 | 984.90 | 994.69 | 0.00 | ORB-short ORB[988.10,1002.70] vol=2.3x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:15:00 | 980.39 | 990.43 | 0.00 | T1 1.5R @ 980.39 |
| Target hit | 2025-06-27 15:20:00 | 951.10 | 955.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:30:00 | 980.00 | 975.81 | 0.00 | ORB-long ORB[968.75,979.00] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:35:00 | 984.80 | 979.59 | 0.00 | T1 1.5R @ 984.80 |
| Target hit | 2025-07-03 10:00:00 | 986.00 | 986.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 1002.00 | 1008.97 | 0.00 | ORB-short ORB[1006.55,1018.40] vol=2.0x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:15:00 | 997.34 | 1007.46 | 0.00 | T1 1.5R @ 997.34 |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1002.00 | 1000.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 1003.90 | 998.65 | 0.00 | ORB-long ORB[988.00,1000.75] vol=1.7x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:50:00 | 1008.74 | 1002.55 | 0.00 | T1 1.5R @ 1008.74 |
| Target hit | 2025-07-11 10:55:00 | 1017.85 | 1018.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2025-07-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:55:00 | 1061.85 | 1067.54 | 0.00 | ORB-short ORB[1067.05,1079.00] vol=1.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-07-22 10:00:00 | 1064.96 | 1067.07 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:25:00 | 1007.15 | 1011.48 | 0.00 | ORB-short ORB[1009.85,1020.00] vol=1.7x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:00:00 | 1001.97 | 1008.12 | 0.00 | T1 1.5R @ 1001.97 |
| Target hit | 2025-07-30 15:20:00 | 997.30 | 1004.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:40:00 | 984.80 | 988.19 | 0.00 | ORB-short ORB[985.05,994.95] vol=2.8x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-07-31 10:05:00 | 987.85 | 987.38 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 936.70 | 948.12 | 0.00 | ORB-short ORB[946.55,959.40] vol=1.9x ATR=4.09 |
| Stop hit — per-position SL triggered | 2025-08-06 09:55:00 | 940.79 | 945.81 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:55:00 | 906.70 | 910.37 | 0.00 | ORB-short ORB[908.25,920.00] vol=1.5x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:05:00 | 903.22 | 909.06 | 0.00 | T1 1.5R @ 903.22 |
| Target hit | 2025-08-20 10:25:00 | 905.40 | 904.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2025-08-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:50:00 | 902.60 | 895.61 | 0.00 | ORB-long ORB[885.00,897.60] vol=3.5x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:05:00 | 906.80 | 897.25 | 0.00 | T1 1.5R @ 906.80 |
| Stop hit — per-position SL triggered | 2025-08-21 12:40:00 | 902.60 | 901.14 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:00:00 | 895.00 | 884.57 | 0.00 | ORB-long ORB[872.60,880.60] vol=6.9x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 891.81 | 885.76 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:50:00 | 907.50 | 900.48 | 0.00 | ORB-long ORB[891.00,901.05] vol=2.0x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-09-02 11:40:00 | 904.69 | 902.86 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:00:00 | 919.95 | 909.26 | 0.00 | ORB-long ORB[896.55,910.00] vol=2.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-09-03 10:45:00 | 916.38 | 915.78 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 11:10:00 | 978.90 | 975.58 | 0.00 | ORB-long ORB[971.20,978.25] vol=1.8x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-09-15 11:40:00 | 976.65 | 975.95 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:15:00 | 901.55 | 892.69 | 0.00 | ORB-long ORB[886.80,899.00] vol=2.8x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-09-24 10:20:00 | 896.50 | 892.77 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:00:00 | 867.00 | 868.79 | 0.00 | ORB-short ORB[867.20,874.95] vol=2.3x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 868.96 | 868.54 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:30:00 | 873.05 | 868.88 | 0.00 | ORB-long ORB[861.35,868.90] vol=2.7x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-10-09 09:40:00 | 870.18 | 869.33 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:55:00 | 890.15 | 893.30 | 0.00 | ORB-short ORB[892.10,904.00] vol=1.8x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:30:00 | 885.05 | 891.64 | 0.00 | T1 1.5R @ 885.05 |
| Stop hit — per-position SL triggered | 2025-10-13 10:55:00 | 890.15 | 891.20 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:00:00 | 875.70 | 881.19 | 0.00 | ORB-short ORB[880.30,888.05] vol=1.9x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-10-14 11:20:00 | 878.54 | 878.70 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:05:00 | 879.30 | 881.46 | 0.00 | ORB-short ORB[885.85,894.55] vol=2.0x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:30:00 | 875.58 | 880.57 | 0.00 | T1 1.5R @ 875.58 |
| Stop hit — per-position SL triggered | 2025-10-20 11:35:00 | 879.30 | 880.37 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:20:00 | 883.10 | 886.83 | 0.00 | ORB-short ORB[883.60,892.20] vol=2.0x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-10-24 10:25:00 | 886.41 | 886.41 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:05:00 | 866.60 | 869.89 | 0.00 | ORB-short ORB[869.40,875.00] vol=1.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:50:00 | 863.66 | 867.75 | 0.00 | T1 1.5R @ 863.66 |
| Target hit | 2025-10-27 15:20:00 | 850.60 | 856.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2025-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:30:00 | 756.85 | 764.44 | 0.00 | ORB-short ORB[769.00,779.45] vol=1.7x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-10-31 10:40:00 | 758.96 | 763.88 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-11-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:50:00 | 730.50 | 735.82 | 0.00 | ORB-short ORB[739.00,748.95] vol=2.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-11-04 11:35:00 | 732.77 | 734.43 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-11-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 10:10:00 | 714.50 | 717.07 | 0.00 | ORB-short ORB[715.10,724.05] vol=2.0x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-11-07 10:40:00 | 716.69 | 716.67 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:55:00 | 706.65 | 710.01 | 0.00 | ORB-short ORB[707.60,715.00] vol=1.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:05:00 | 702.49 | 709.59 | 0.00 | T1 1.5R @ 702.49 |
| Stop hit — per-position SL triggered | 2025-11-10 10:20:00 | 706.65 | 709.21 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:15:00 | 689.30 | 698.14 | 0.00 | ORB-short ORB[698.90,708.05] vol=6.2x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-11-11 11:20:00 | 691.65 | 697.49 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:40:00 | 589.95 | 593.13 | 0.00 | ORB-short ORB[591.85,597.85] vol=1.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:55:00 | 586.56 | 591.61 | 0.00 | T1 1.5R @ 586.56 |
| Target hit | 2025-11-19 11:15:00 | 586.25 | 582.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — SELL (started 2025-11-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:05:00 | 577.55 | 581.23 | 0.00 | ORB-short ORB[578.85,587.00] vol=1.5x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:20:00 | 574.61 | 580.47 | 0.00 | T1 1.5R @ 574.61 |
| Target hit | 2025-11-24 15:20:00 | 572.40 | 573.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:05:00 | 547.50 | 544.98 | 0.00 | ORB-long ORB[542.55,547.30] vol=2.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-12-04 13:35:00 | 545.93 | 546.15 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 531.00 | 535.14 | 0.00 | ORB-short ORB[536.00,541.00] vol=1.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-12-08 10:10:00 | 532.49 | 534.60 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 527.45 | 532.79 | 0.00 | ORB-short ORB[532.05,538.80] vol=8.2x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-12-10 11:20:00 | 529.40 | 531.43 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:55:00 | 536.25 | 535.17 | 0.00 | ORB-long ORB[528.80,536.00] vol=5.4x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 10:00:00 | 538.53 | 535.71 | 0.00 | T1 1.5R @ 538.53 |
| Target hit | 2025-12-16 11:40:00 | 537.00 | 537.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:05:00 | 535.05 | 536.53 | 0.00 | ORB-short ORB[535.55,543.10] vol=1.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 536.25 | 536.47 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:50:00 | 538.70 | 537.25 | 0.00 | ORB-long ORB[534.60,538.05] vol=6.8x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:05:00 | 540.35 | 538.11 | 0.00 | T1 1.5R @ 540.35 |
| Target hit | 2025-12-24 12:40:00 | 539.00 | 539.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2026-01-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:20:00 | 525.75 | 527.16 | 0.00 | ORB-short ORB[526.95,531.35] vol=1.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-01-01 10:30:00 | 526.69 | 527.10 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:05:00 | 521.05 | 521.40 | 0.00 | ORB-short ORB[522.10,525.50] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-01-02 10:20:00 | 522.14 | 521.41 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 499.45 | 505.20 | 0.00 | ORB-short ORB[502.15,507.90] vol=2.3x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 497.71 | 504.65 | 0.00 | T1 1.5R @ 497.71 |
| Target hit | 2026-01-08 13:45:00 | 495.00 | 494.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2026-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:45:00 | 437.50 | 440.81 | 0.00 | ORB-short ORB[440.35,445.30] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:55:00 | 435.79 | 439.87 | 0.00 | T1 1.5R @ 435.79 |
| Target hit | 2026-01-16 15:20:00 | 425.15 | 431.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2026-01-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:40:00 | 371.00 | 372.15 | 0.00 | ORB-short ORB[371.40,375.70] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:15:00 | 369.58 | 371.56 | 0.00 | T1 1.5R @ 369.58 |
| Stop hit — per-position SL triggered | 2026-01-28 11:25:00 | 371.00 | 371.50 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 383.20 | 380.94 | 0.00 | ORB-long ORB[377.80,383.00] vol=4.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 381.91 | 381.05 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-02-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 10:30:00 | 373.65 | 375.73 | 0.00 | ORB-short ORB[374.30,379.00] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:45:00 | 371.72 | 374.88 | 0.00 | T1 1.5R @ 371.72 |
| Stop hit — per-position SL triggered | 2026-02-04 11:30:00 | 373.65 | 373.51 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 348.75 | 350.61 | 0.00 | ORB-short ORB[350.30,355.00] vol=2.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 350.31 | 349.83 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 308.85 | 305.34 | 0.00 | ORB-long ORB[300.45,305.00] vol=4.4x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-03-18 09:50:00 | 307.51 | 305.52 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-03-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:55:00 | 299.15 | 302.59 | 0.00 | ORB-short ORB[301.10,305.10] vol=1.7x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-03-27 10:05:00 | 300.65 | 302.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-20 09:35:00 | 1081.60 | 2025-05-20 10:30:00 | 1074.64 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2025-05-22 10:45:00 | 1062.70 | 2025-05-22 12:20:00 | 1066.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1090.40 | 2025-05-23 09:45:00 | 1084.58 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-05-28 10:50:00 | 1113.80 | 2025-05-28 11:00:00 | 1108.58 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-06-05 09:30:00 | 1040.60 | 2025-06-05 09:55:00 | 1047.47 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-06-05 09:30:00 | 1040.60 | 2025-06-05 10:05:00 | 1040.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 10:50:00 | 1011.90 | 2025-06-10 11:00:00 | 1008.01 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-10 10:50:00 | 1011.90 | 2025-06-10 11:05:00 | 1011.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 10:15:00 | 996.90 | 2025-06-20 11:00:00 | 992.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-06-27 11:00:00 | 984.90 | 2025-06-27 11:15:00 | 980.39 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-06-27 11:00:00 | 984.90 | 2025-06-27 15:20:00 | 951.10 | TARGET_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2025-07-03 09:30:00 | 980.00 | 2025-07-03 09:35:00 | 984.80 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-03 09:30:00 | 980.00 | 2025-07-03 10:00:00 | 986.00 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-08 10:45:00 | 1002.00 | 2025-07-08 11:15:00 | 997.34 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-08 10:45:00 | 1002.00 | 2025-07-08 15:15:00 | 1002.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1003.90 | 2025-07-11 09:50:00 | 1008.74 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1003.90 | 2025-07-11 10:55:00 | 1017.85 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2025-07-22 09:55:00 | 1061.85 | 2025-07-22 10:00:00 | 1064.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-30 10:25:00 | 1007.15 | 2025-07-30 12:00:00 | 1001.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-30 10:25:00 | 1007.15 | 2025-07-30 15:20:00 | 997.30 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2025-07-31 09:40:00 | 984.80 | 2025-07-31 10:05:00 | 987.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-06 09:35:00 | 936.70 | 2025-08-06 09:55:00 | 940.79 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-08-20 09:55:00 | 906.70 | 2025-08-20 10:05:00 | 903.22 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-20 09:55:00 | 906.70 | 2025-08-20 10:25:00 | 905.40 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-08-21 10:50:00 | 902.60 | 2025-08-21 11:05:00 | 906.80 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-21 10:50:00 | 902.60 | 2025-08-21 12:40:00 | 902.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 11:00:00 | 895.00 | 2025-09-01 11:15:00 | 891.81 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-02 10:50:00 | 907.50 | 2025-09-02 11:40:00 | 904.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-03 10:00:00 | 919.95 | 2025-09-03 10:45:00 | 916.38 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-15 11:10:00 | 978.90 | 2025-09-15 11:40:00 | 976.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-24 10:15:00 | 901.55 | 2025-09-24 10:20:00 | 896.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-10-08 10:00:00 | 867.00 | 2025-10-08 10:15:00 | 868.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-09 09:30:00 | 873.05 | 2025-10-09 09:40:00 | 870.18 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-13 09:55:00 | 890.15 | 2025-10-13 10:30:00 | 885.05 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-10-13 09:55:00 | 890.15 | 2025-10-13 10:55:00 | 890.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 10:00:00 | 875.70 | 2025-10-14 11:20:00 | 878.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-20 11:05:00 | 879.30 | 2025-10-20 11:30:00 | 875.58 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-20 11:05:00 | 879.30 | 2025-10-20 11:35:00 | 879.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:20:00 | 883.10 | 2025-10-24 10:25:00 | 886.41 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-27 10:05:00 | 866.60 | 2025-10-27 10:50:00 | 863.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-27 10:05:00 | 866.60 | 2025-10-27 15:20:00 | 850.60 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2025-10-31 10:30:00 | 756.85 | 2025-10-31 10:40:00 | 758.96 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-04 10:50:00 | 730.50 | 2025-11-04 11:35:00 | 732.77 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-07 10:10:00 | 714.50 | 2025-11-07 10:40:00 | 716.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-10 09:55:00 | 706.65 | 2025-11-10 10:05:00 | 702.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-11-10 09:55:00 | 706.65 | 2025-11-10 10:20:00 | 706.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 11:15:00 | 689.30 | 2025-11-11 11:20:00 | 691.65 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-19 09:40:00 | 589.95 | 2025-11-19 09:55:00 | 586.56 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-11-19 09:40:00 | 589.95 | 2025-11-19 11:15:00 | 586.25 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-11-24 11:05:00 | 577.55 | 2025-11-24 11:20:00 | 574.61 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-11-24 11:05:00 | 577.55 | 2025-11-24 15:20:00 | 572.40 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-12-04 11:05:00 | 547.50 | 2025-12-04 13:35:00 | 545.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-08 10:00:00 | 531.00 | 2025-12-08 10:10:00 | 532.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-10 10:45:00 | 527.45 | 2025-12-10 11:20:00 | 529.40 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-16 09:55:00 | 536.25 | 2025-12-16 10:00:00 | 538.53 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-16 09:55:00 | 536.25 | 2025-12-16 11:40:00 | 537.00 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-12-22 11:05:00 | 535.05 | 2025-12-22 11:15:00 | 536.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-24 10:50:00 | 538.70 | 2025-12-24 11:05:00 | 540.35 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-12-24 10:50:00 | 538.70 | 2025-12-24 12:40:00 | 539.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-01-01 10:20:00 | 525.75 | 2026-01-01 10:30:00 | 526.69 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-02 10:05:00 | 521.05 | 2026-01-02 10:20:00 | 522.14 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-08 11:10:00 | 499.45 | 2026-01-08 11:20:00 | 497.71 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-08 11:10:00 | 499.45 | 2026-01-08 13:45:00 | 495.00 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-01-16 10:45:00 | 437.50 | 2026-01-16 10:55:00 | 435.79 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-16 10:45:00 | 437.50 | 2026-01-16 15:20:00 | 425.15 | TARGET_HIT | 0.50 | 2.82% |
| SELL | retest1 | 2026-01-28 10:40:00 | 371.00 | 2026-01-28 11:15:00 | 369.58 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-28 10:40:00 | 371.00 | 2026-01-28 11:25:00 | 371.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 383.20 | 2026-02-01 11:15:00 | 381.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-04 10:30:00 | 373.65 | 2026-02-04 10:45:00 | 371.72 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-04 10:30:00 | 373.65 | 2026-02-04 11:30:00 | 373.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 09:35:00 | 348.75 | 2026-02-11 10:10:00 | 350.31 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-18 09:45:00 | 308.85 | 2026-03-18 09:50:00 | 307.51 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-27 09:55:00 | 299.15 | 2026-03-27 10:05:00 | 300.65 | STOP_HIT | 1.00 | -0.50% |
