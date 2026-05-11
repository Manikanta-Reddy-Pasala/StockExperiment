# Ramkrishna Forgings Ltd. (RKFORGE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 607.80
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
| ENTRY1 | 58 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 46
- **Target hits / Stop hits / Partials:** 11 / 47 / 20
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 22.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 20 | 50.0% | 7 | 21 | 12 | 0.49% | 19.7% |
| BUY @ 2nd Alert (retest1) | 40 | 20 | 50.0% | 7 | 21 | 12 | 0.49% | 19.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 12 | 31.6% | 4 | 26 | 8 | 0.07% | 2.7% |
| SELL @ 2nd Alert (retest1) | 38 | 12 | 31.6% | 4 | 26 | 8 | 0.07% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 32 | 41.0% | 11 | 47 | 20 | 0.29% | 22.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:30:00 | 722.00 | 718.55 | 0.00 | ORB-long ORB[713.30,719.00] vol=2.1x ATR=2.72 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 719.28 | 718.91 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:45:00 | 739.45 | 730.28 | 0.00 | ORB-long ORB[719.30,727.75] vol=8.0x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 744.00 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 3 — SELL (started 2024-05-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:50:00 | 722.00 | 727.03 | 0.00 | ORB-short ORB[724.95,732.60] vol=2.3x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:55:00 | 717.44 | 725.03 | 0.00 | T1 1.5R @ 717.44 |
| Target hit | 2024-05-23 15:20:00 | 715.00 | 721.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 697.35 | 701.60 | 0.00 | ORB-short ORB[697.90,705.20] vol=2.7x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-05-27 09:50:00 | 700.10 | 701.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:35:00 | 677.25 | 679.32 | 0.00 | ORB-short ORB[678.05,684.95] vol=1.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-05-29 10:40:00 | 679.46 | 679.30 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:55:00 | 673.05 | 674.31 | 0.00 | ORB-short ORB[674.55,684.00] vol=11.4x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-30 10:20:00 | 675.79 | 674.16 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 668.45 | 674.05 | 0.00 | ORB-short ORB[673.60,681.30] vol=2.5x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-05-31 10:05:00 | 671.30 | 673.81 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 708.00 | 712.97 | 0.00 | ORB-short ORB[710.90,720.95] vol=2.1x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 12:10:00 | 703.25 | 708.63 | 0.00 | T1 1.5R @ 703.25 |
| Target hit | 2024-06-10 15:20:00 | 699.00 | 704.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 712.60 | 708.14 | 0.00 | ORB-long ORB[701.20,710.00] vol=4.7x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-06-11 09:35:00 | 709.16 | 708.45 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:15:00 | 736.85 | 723.76 | 0.00 | ORB-long ORB[716.55,724.80] vol=3.3x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-06-12 10:20:00 | 733.27 | 724.91 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 746.90 | 751.07 | 0.00 | ORB-short ORB[747.45,756.15] vol=3.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 749.52 | 751.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:00:00 | 900.75 | 891.53 | 0.00 | ORB-long ORB[883.95,891.90] vol=3.1x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-07-01 10:15:00 | 897.01 | 892.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:35:00 | 928.60 | 921.31 | 0.00 | ORB-long ORB[909.75,923.25] vol=4.7x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-07-03 09:45:00 | 923.38 | 922.41 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:20:00 | 908.40 | 899.10 | 0.00 | ORB-long ORB[895.05,904.40] vol=4.9x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:25:00 | 916.12 | 904.56 | 0.00 | T1 1.5R @ 916.12 |
| Target hit | 2024-07-12 12:30:00 | 925.05 | 925.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 897.40 | 908.95 | 0.00 | ORB-short ORB[904.50,914.50] vol=1.6x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 901.85 | 905.20 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:40:00 | 891.00 | 885.91 | 0.00 | ORB-long ORB[877.35,889.00] vol=1.9x ATR=6.11 |
| Stop hit — per-position SL triggered | 2024-07-24 09:45:00 | 884.89 | 886.11 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:30:00 | 878.85 | 872.74 | 0.00 | ORB-long ORB[865.20,878.00] vol=2.7x ATR=5.62 |
| Stop hit — per-position SL triggered | 2024-07-25 09:40:00 | 873.23 | 873.03 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 867.50 | 869.67 | 0.00 | ORB-short ORB[867.95,875.45] vol=2.5x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:35:00 | 864.38 | 868.57 | 0.00 | T1 1.5R @ 864.38 |
| Target hit | 2024-07-26 15:20:00 | 839.95 | 853.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 947.10 | 935.07 | 0.00 | ORB-long ORB[925.15,938.70] vol=3.0x ATR=7.64 |
| Stop hit — per-position SL triggered | 2024-08-08 10:00:00 | 939.46 | 939.76 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:40:00 | 941.60 | 933.83 | 0.00 | ORB-long ORB[927.25,933.60] vol=2.6x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:55:00 | 949.25 | 939.12 | 0.00 | T1 1.5R @ 949.25 |
| Stop hit — per-position SL triggered | 2024-08-13 10:20:00 | 941.60 | 940.02 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:40:00 | 937.10 | 933.24 | 0.00 | ORB-long ORB[928.00,935.35] vol=2.3x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:50:00 | 942.15 | 934.56 | 0.00 | T1 1.5R @ 942.15 |
| Stop hit — per-position SL triggered | 2024-08-22 09:55:00 | 937.10 | 934.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:45:00 | 951.80 | 960.17 | 0.00 | ORB-short ORB[957.35,970.00] vol=2.3x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-08-26 11:30:00 | 955.54 | 959.46 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 971.85 | 986.27 | 0.00 | ORB-short ORB[987.00,998.05] vol=1.6x ATR=5.17 |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 977.02 | 984.67 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 947.10 | 953.12 | 0.00 | ORB-short ORB[950.15,964.05] vol=3.7x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 951.48 | 953.09 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:40:00 | 972.95 | 969.04 | 0.00 | ORB-long ORB[963.90,972.00] vol=1.8x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-09-11 09:50:00 | 969.29 | 969.12 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:55:00 | 994.30 | 1021.23 | 0.00 | ORB-short ORB[1020.45,1035.45] vol=1.7x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 12:00:00 | 986.56 | 1017.24 | 0.00 | T1 1.5R @ 986.56 |
| Stop hit — per-position SL triggered | 2024-09-16 12:15:00 | 994.30 | 1016.37 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 977.95 | 982.77 | 0.00 | ORB-short ORB[980.05,991.85] vol=2.2x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:30:00 | 972.11 | 981.59 | 0.00 | T1 1.5R @ 972.11 |
| Stop hit — per-position SL triggered | 2024-09-17 13:00:00 | 977.95 | 980.06 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 979.15 | 983.16 | 0.00 | ORB-short ORB[980.00,994.45] vol=1.6x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-09-18 09:50:00 | 982.49 | 982.88 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 1005.50 | 1012.96 | 0.00 | ORB-short ORB[1012.55,1025.00] vol=1.9x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 1008.91 | 1010.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:45:00 | 1011.20 | 1004.85 | 0.00 | ORB-long ORB[996.70,1007.95] vol=1.6x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:50:00 | 1015.66 | 1006.07 | 0.00 | T1 1.5R @ 1015.66 |
| Stop hit — per-position SL triggered | 2024-09-26 10:55:00 | 1011.20 | 1006.65 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:35:00 | 1004.60 | 1006.64 | 0.00 | ORB-short ORB[1006.05,1015.00] vol=2.8x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-09-27 09:45:00 | 1007.16 | 1006.02 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:00:00 | 1015.85 | 1018.18 | 0.00 | ORB-short ORB[1021.00,1029.90] vol=3.0x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-10-21 11:50:00 | 1019.85 | 1017.35 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:40:00 | 896.25 | 892.11 | 0.00 | ORB-long ORB[883.25,896.00] vol=2.0x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 904.27 | 896.42 | 0.00 | T1 1.5R @ 904.27 |
| Target hit | 2024-10-30 11:50:00 | 903.55 | 904.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 931.10 | 929.30 | 0.00 | ORB-long ORB[919.55,931.00] vol=4.4x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:10:00 | 935.84 | 930.50 | 0.00 | T1 1.5R @ 935.84 |
| Target hit | 2024-11-06 15:20:00 | 949.85 | 938.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:15:00 | 968.05 | 961.48 | 0.00 | ORB-long ORB[953.50,964.85] vol=2.1x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-11-12 10:50:00 | 964.06 | 962.08 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 10:35:00 | 968.05 | 970.35 | 0.00 | ORB-short ORB[968.80,977.95] vol=1.8x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-11-25 10:50:00 | 971.49 | 970.28 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:10:00 | 963.35 | 968.56 | 0.00 | ORB-short ORB[965.05,977.00] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:15:00 | 958.92 | 968.01 | 0.00 | T1 1.5R @ 958.92 |
| Stop hit — per-position SL triggered | 2024-11-26 14:45:00 | 963.35 | 963.85 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 971.55 | 966.54 | 0.00 | ORB-long ORB[962.50,967.70] vol=1.9x ATR=3.78 |
| Stop hit — per-position SL triggered | 2024-11-27 09:40:00 | 967.77 | 966.75 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 969.50 | 974.23 | 0.00 | ORB-short ORB[971.80,980.00] vol=2.1x ATR=3.46 |
| Stop hit — per-position SL triggered | 2024-11-29 11:25:00 | 972.96 | 973.62 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:30:00 | 965.60 | 973.16 | 0.00 | ORB-short ORB[969.65,984.00] vol=2.8x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:40:00 | 959.86 | 971.66 | 0.00 | T1 1.5R @ 959.86 |
| Stop hit — per-position SL triggered | 2024-12-03 14:55:00 | 965.60 | 967.79 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:10:00 | 981.85 | 973.61 | 0.00 | ORB-long ORB[965.00,974.85] vol=8.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-12-11 11:20:00 | 978.61 | 974.03 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:40:00 | 948.10 | 959.76 | 0.00 | ORB-short ORB[960.00,971.25] vol=1.7x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 951.99 | 958.12 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 950.05 | 955.33 | 0.00 | ORB-short ORB[952.05,960.50] vol=2.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-12-16 10:25:00 | 952.69 | 955.08 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:15:00 | 917.15 | 917.96 | 0.00 | ORB-short ORB[918.30,925.10] vol=13.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-12-18 10:35:00 | 920.95 | 918.05 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:50:00 | 916.40 | 905.83 | 0.00 | ORB-long ORB[894.10,905.95] vol=2.7x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-12-19 09:55:00 | 911.51 | 907.27 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 11:15:00 | 903.90 | 905.30 | 0.00 | ORB-short ORB[904.20,913.95] vol=2.2x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-12-20 12:10:00 | 906.77 | 904.99 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 891.35 | 900.91 | 0.00 | ORB-short ORB[904.00,911.10] vol=3.1x ATR=3.05 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 894.40 | 900.62 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:45:00 | 890.10 | 894.79 | 0.00 | ORB-short ORB[894.40,902.95] vol=2.4x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-12-27 10:00:00 | 893.46 | 894.30 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:40:00 | 911.00 | 904.30 | 0.00 | ORB-long ORB[895.95,906.00] vol=5.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-01-01 10:50:00 | 907.81 | 905.38 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 918.50 | 918.15 | 0.00 | ORB-long ORB[908.00,918.35] vol=5.1x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:55:00 | 923.94 | 919.32 | 0.00 | T1 1.5R @ 923.94 |
| Target hit | 2025-01-02 10:10:00 | 918.60 | 919.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:15:00 | 913.00 | 909.96 | 0.00 | ORB-long ORB[904.10,910.00] vol=1.9x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-01-09 10:20:00 | 910.14 | 909.97 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:30:00 | 911.00 | 902.52 | 0.00 | ORB-long ORB[899.00,910.45] vol=2.4x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:35:00 | 916.61 | 904.47 | 0.00 | T1 1.5R @ 916.61 |
| Stop hit — per-position SL triggered | 2025-01-10 10:40:00 | 911.00 | 907.28 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:50:00 | 953.85 | 945.70 | 0.00 | ORB-long ORB[939.05,951.80] vol=2.1x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:15:00 | 960.81 | 950.76 | 0.00 | T1 1.5R @ 960.81 |
| Target hit | 2025-01-15 11:20:00 | 956.40 | 957.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — BUY (started 2025-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:10:00 | 969.00 | 965.22 | 0.00 | ORB-long ORB[954.95,968.00] vol=1.7x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:15:00 | 974.40 | 968.68 | 0.00 | T1 1.5R @ 974.40 |
| Target hit | 2025-01-16 15:20:00 | 995.45 | 983.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 765.15 | 773.91 | 0.00 | ORB-short ORB[771.55,781.80] vol=2.4x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-01-30 09:40:00 | 769.36 | 772.70 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 766.70 | 756.01 | 0.00 | ORB-long ORB[747.05,754.45] vol=2.5x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:10:00 | 773.35 | 763.70 | 0.00 | T1 1.5R @ 773.35 |
| Target hit | 2025-03-21 15:20:00 | 847.60 | 817.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:25:00 | 767.05 | 764.10 | 0.00 | ORB-long ORB[756.00,765.75] vol=2.5x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:30:00 | 771.96 | 764.88 | 0.00 | T1 1.5R @ 771.96 |
| Stop hit — per-position SL triggered | 2025-04-16 10:35:00 | 767.05 | 765.09 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:15:00 | 686.70 | 693.49 | 0.00 | ORB-short ORB[695.50,702.00] vol=1.9x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:40:00 | 682.35 | 691.75 | 0.00 | T1 1.5R @ 682.35 |
| Target hit | 2025-04-23 15:20:00 | 679.00 | 683.54 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 09:30:00 | 722.00 | 2024-05-17 09:40:00 | 719.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-18 09:45:00 | 739.45 | 2024-05-21 09:15:00 | 744.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest1 | 2024-05-23 10:50:00 | 722.00 | 2024-05-23 11:55:00 | 717.44 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-05-23 10:50:00 | 722.00 | 2024-05-23 15:20:00 | 715.00 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-05-27 09:45:00 | 697.35 | 2024-05-27 09:50:00 | 700.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-29 10:35:00 | 677.25 | 2024-05-29 10:40:00 | 679.46 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-30 09:55:00 | 673.05 | 2024-05-30 10:20:00 | 675.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-31 10:00:00 | 668.45 | 2024-05-31 10:05:00 | 671.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-06-10 09:30:00 | 708.00 | 2024-06-10 12:10:00 | 703.25 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-06-10 09:30:00 | 708.00 | 2024-06-10 15:20:00 | 699.00 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2024-06-11 09:30:00 | 712.60 | 2024-06-11 09:35:00 | 709.16 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-06-12 10:15:00 | 736.85 | 2024-06-12 10:20:00 | 733.27 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-06-13 11:00:00 | 746.90 | 2024-06-13 11:05:00 | 749.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-01 10:00:00 | 900.75 | 2024-07-01 10:15:00 | 897.01 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-03 09:35:00 | 928.60 | 2024-07-03 09:45:00 | 923.38 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-07-12 10:20:00 | 908.40 | 2024-07-12 10:25:00 | 916.12 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2024-07-12 10:20:00 | 908.40 | 2024-07-12 12:30:00 | 925.05 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2024-07-18 09:30:00 | 897.40 | 2024-07-18 09:45:00 | 901.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-07-24 09:40:00 | 891.00 | 2024-07-24 09:45:00 | 884.89 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2024-07-25 09:30:00 | 878.85 | 2024-07-25 09:40:00 | 873.23 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-07-26 09:30:00 | 867.50 | 2024-07-26 09:35:00 | 864.38 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-26 09:30:00 | 867.50 | 2024-07-26 15:20:00 | 839.95 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2024-08-08 09:30:00 | 947.10 | 2024-08-08 10:00:00 | 939.46 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2024-08-13 09:40:00 | 941.60 | 2024-08-13 09:55:00 | 949.25 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2024-08-13 09:40:00 | 941.60 | 2024-08-13 10:20:00 | 941.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:40:00 | 937.10 | 2024-08-22 09:50:00 | 942.15 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-22 09:40:00 | 937.10 | 2024-08-22 09:55:00 | 937.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 10:45:00 | 951.80 | 2024-08-26 11:30:00 | 955.54 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-06 10:05:00 | 971.85 | 2024-09-06 10:15:00 | 977.02 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-09-09 09:30:00 | 947.10 | 2024-09-09 09:35:00 | 951.48 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-09-11 09:40:00 | 972.95 | 2024-09-11 09:50:00 | 969.29 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-16 10:55:00 | 994.30 | 2024-09-16 12:00:00 | 986.56 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-09-16 10:55:00 | 994.30 | 2024-09-16 12:15:00 | 994.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:10:00 | 977.95 | 2024-09-17 10:30:00 | 972.11 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-09-17 10:10:00 | 977.95 | 2024-09-17 13:00:00 | 977.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 09:30:00 | 979.15 | 2024-09-18 09:50:00 | 982.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-24 09:50:00 | 1005.50 | 2024-09-24 10:15:00 | 1008.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-26 10:45:00 | 1011.20 | 2024-09-26 10:50:00 | 1015.66 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-26 10:45:00 | 1011.20 | 2024-09-26 10:55:00 | 1011.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 09:35:00 | 1004.60 | 2024-09-27 09:45:00 | 1007.16 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-21 11:00:00 | 1015.85 | 2024-10-21 11:50:00 | 1019.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-30 09:40:00 | 896.25 | 2024-10-30 09:50:00 | 904.27 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-10-30 09:40:00 | 896.25 | 2024-10-30 11:50:00 | 903.55 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2024-11-06 09:45:00 | 931.10 | 2024-11-06 10:10:00 | 935.84 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-06 09:45:00 | 931.10 | 2024-11-06 15:20:00 | 949.85 | TARGET_HIT | 0.50 | 2.01% |
| BUY | retest1 | 2024-11-12 10:15:00 | 968.05 | 2024-11-12 10:50:00 | 964.06 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-11-25 10:35:00 | 968.05 | 2024-11-25 10:50:00 | 971.49 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-26 11:10:00 | 963.35 | 2024-11-26 11:15:00 | 958.92 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-26 11:10:00 | 963.35 | 2024-11-26 14:45:00 | 963.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:35:00 | 971.55 | 2024-11-27 09:40:00 | 967.77 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-29 10:50:00 | 969.50 | 2024-11-29 11:25:00 | 972.96 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-03 10:30:00 | 965.60 | 2024-12-03 11:40:00 | 959.86 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-12-03 10:30:00 | 965.60 | 2024-12-03 14:55:00 | 965.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:10:00 | 981.85 | 2024-12-11 11:20:00 | 978.61 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-13 10:40:00 | 948.10 | 2024-12-13 11:10:00 | 951.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-16 10:15:00 | 950.05 | 2024-12-16 10:25:00 | 952.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-18 10:15:00 | 917.15 | 2024-12-18 10:35:00 | 920.95 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-19 09:50:00 | 916.40 | 2024-12-19 09:55:00 | 911.51 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-12-20 11:15:00 | 903.90 | 2024-12-20 12:10:00 | 906.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-26 10:50:00 | 891.35 | 2024-12-26 11:00:00 | 894.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-27 09:45:00 | 890.10 | 2024-12-27 10:00:00 | 893.46 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-01 10:40:00 | 911.00 | 2025-01-01 10:50:00 | 907.81 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-02 09:30:00 | 918.50 | 2025-01-02 09:55:00 | 923.94 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-01-02 09:30:00 | 918.50 | 2025-01-02 10:10:00 | 918.60 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-01-09 10:15:00 | 913.00 | 2025-01-09 10:20:00 | 910.14 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-10 10:30:00 | 911.00 | 2025-01-10 10:35:00 | 916.61 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-10 10:30:00 | 911.00 | 2025-01-10 10:40:00 | 911.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-15 09:50:00 | 953.85 | 2025-01-15 10:15:00 | 960.81 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-01-15 09:50:00 | 953.85 | 2025-01-15 11:20:00 | 956.40 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-01-16 10:10:00 | 969.00 | 2025-01-16 11:15:00 | 974.40 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-01-16 10:10:00 | 969.00 | 2025-01-16 15:20:00 | 995.45 | TARGET_HIT | 0.50 | 2.73% |
| SELL | retest1 | 2025-01-30 09:30:00 | 765.15 | 2025-01-30 09:40:00 | 769.36 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-03-21 09:30:00 | 766.70 | 2025-03-21 10:10:00 | 773.35 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2025-03-21 09:30:00 | 766.70 | 2025-03-21 15:20:00 | 847.60 | TARGET_HIT | 0.50 | 10.55% |
| BUY | retest1 | 2025-04-16 10:25:00 | 767.05 | 2025-04-16 10:30:00 | 771.96 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-16 10:25:00 | 767.05 | 2025-04-16 10:35:00 | 767.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 10:15:00 | 686.70 | 2025-04-23 10:40:00 | 682.35 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-04-23 10:15:00 | 686.70 | 2025-04-23 15:20:00 | 679.00 | TARGET_HIT | 0.50 | 1.12% |
