# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2025-10-08 09:15:00 → 2026-04-02 15:25:00 (7363 bars)
- **Last close:** 817.00
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
| ENTRY1 | 31 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 23
- **Target hits / Stop hits / Partials:** 8 / 23 / 13
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 7.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 2 | 11 | 4 | 0.07% | 1.2% |
| BUY @ 2nd Alert (retest1) | 17 | 6 | 35.3% | 2 | 11 | 4 | 0.07% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 15 | 55.6% | 6 | 12 | 9 | 0.24% | 6.6% |
| SELL @ 2nd Alert (retest1) | 27 | 15 | 55.6% | 6 | 12 | 9 | 0.24% | 6.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 44 | 21 | 47.7% | 8 | 23 | 13 | 0.18% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:40:00 | 1093.40 | 1087.57 | 0.00 | ORB-long ORB[1078.00,1092.50] vol=1.8x ATR=3.89 |
| Stop hit — per-position SL triggered | 2025-10-15 10:25:00 | 1089.51 | 1089.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:55:00 | 1088.00 | 1083.98 | 0.00 | ORB-long ORB[1076.80,1085.00] vol=1.8x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-10-17 10:05:00 | 1085.27 | 1084.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-10-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:20:00 | 1111.20 | 1116.85 | 0.00 | ORB-short ORB[1113.80,1123.90] vol=1.9x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-10-23 11:25:00 | 1114.86 | 1115.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:15:00 | 1114.60 | 1111.84 | 0.00 | ORB-long ORB[1104.00,1114.30] vol=1.9x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-10-31 10:35:00 | 1112.07 | 1111.93 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1083.60 | 1091.08 | 0.00 | ORB-short ORB[1091.40,1100.70] vol=1.9x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-11-04 11:25:00 | 1085.90 | 1089.48 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:55:00 | 998.90 | 1002.36 | 0.00 | ORB-short ORB[999.10,1005.00] vol=1.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-11-13 10:10:00 | 1001.09 | 1001.95 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-11-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:45:00 | 990.00 | 984.45 | 0.00 | ORB-long ORB[979.70,987.10] vol=1.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-11-17 11:00:00 | 987.46 | 984.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:40:00 | 969.00 | 973.49 | 0.00 | ORB-short ORB[972.60,977.90] vol=1.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-11-20 10:35:00 | 971.51 | 970.64 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:00:00 | 960.50 | 963.40 | 0.00 | ORB-short ORB[963.00,970.70] vol=3.1x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:50:00 | 957.63 | 961.19 | 0.00 | T1 1.5R @ 957.63 |
| Stop hit — per-position SL triggered | 2025-11-21 11:00:00 | 960.50 | 961.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 11:00:00 | 981.00 | 977.38 | 0.00 | ORB-long ORB[971.40,979.60] vol=4.0x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-11-27 12:00:00 | 978.84 | 978.28 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-11-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:45:00 | 986.00 | 981.39 | 0.00 | ORB-long ORB[975.00,985.50] vol=3.8x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 13:05:00 | 989.61 | 985.63 | 0.00 | T1 1.5R @ 989.61 |
| Target hit | 2025-11-28 15:20:00 | 992.30 | 988.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:00:00 | 984.70 | 989.61 | 0.00 | ORB-short ORB[987.95,997.80] vol=1.5x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 12:05:00 | 981.56 | 987.06 | 0.00 | T1 1.5R @ 981.56 |
| Target hit | 2025-12-01 15:20:00 | 979.55 | 985.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-12-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:55:00 | 969.40 | 972.21 | 0.00 | ORB-short ORB[971.30,984.95] vol=1.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-12-02 10:00:00 | 971.42 | 972.13 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:55:00 | 970.95 | 973.98 | 0.00 | ORB-short ORB[971.00,985.10] vol=1.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 973.04 | 973.74 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 965.30 | 968.03 | 0.00 | ORB-short ORB[966.20,976.65] vol=2.3x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 14:55:00 | 960.86 | 964.71 | 0.00 | T1 1.5R @ 960.86 |
| Target hit | 2025-12-05 15:20:00 | 962.55 | 964.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:30:00 | 950.80 | 952.76 | 0.00 | ORB-short ORB[951.05,965.20] vol=1.6x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:35:00 | 948.36 | 952.38 | 0.00 | T1 1.5R @ 948.36 |
| Target hit | 2025-12-08 15:20:00 | 924.35 | 935.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-12-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:40:00 | 914.65 | 920.27 | 0.00 | ORB-short ORB[920.00,926.90] vol=1.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:00:00 | 911.04 | 916.65 | 0.00 | T1 1.5R @ 911.04 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 914.65 | 915.89 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 976.00 | 981.50 | 0.00 | ORB-short ORB[980.40,993.00] vol=2.1x ATR=2.92 |
| Stop hit — per-position SL triggered | 2025-12-16 11:20:00 | 978.92 | 981.35 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:05:00 | 965.80 | 972.22 | 0.00 | ORB-short ORB[969.90,981.40] vol=1.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 12:10:00 | 962.19 | 970.75 | 0.00 | T1 1.5R @ 962.19 |
| Target hit | 2025-12-24 15:20:00 | 960.50 | 967.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-12-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:35:00 | 940.80 | 945.94 | 0.00 | ORB-short ORB[943.70,955.00] vol=1.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:10:00 | 937.01 | 944.58 | 0.00 | T1 1.5R @ 937.01 |
| Target hit | 2025-12-29 15:20:00 | 934.95 | 937.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 949.65 | 945.65 | 0.00 | ORB-long ORB[940.55,948.40] vol=3.8x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:35:00 | 953.82 | 950.75 | 0.00 | T1 1.5R @ 953.82 |
| Stop hit — per-position SL triggered | 2026-01-02 09:40:00 | 949.65 | 951.27 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 863.00 | 870.39 | 0.00 | ORB-short ORB[868.15,879.50] vol=1.8x ATR=6.69 |
| Stop hit — per-position SL triggered | 2026-02-05 11:00:00 | 869.69 | 868.46 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:45:00 | 899.70 | 891.82 | 0.00 | ORB-long ORB[883.65,897.00] vol=2.2x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-02-06 10:00:00 | 895.71 | 893.37 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 913.65 | 917.42 | 0.00 | ORB-short ORB[915.00,922.05] vol=1.9x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 910.16 | 916.08 | 0.00 | T1 1.5R @ 910.16 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 913.65 | 915.88 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 915.00 | 908.97 | 0.00 | ORB-long ORB[903.10,914.00] vol=3.3x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 912.05 | 909.46 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 906.00 | 893.68 | 0.00 | ORB-long ORB[882.55,893.80] vol=2.2x ATR=4.89 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 901.11 | 899.25 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 910.05 | 914.30 | 0.00 | ORB-short ORB[913.50,919.30] vol=1.9x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 906.15 | 912.04 | 0.00 | T1 1.5R @ 906.15 |
| Target hit | 2026-02-18 15:20:00 | 897.20 | 901.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 888.85 | 881.90 | 0.00 | ORB-long ORB[871.20,883.15] vol=2.2x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 892.84 | 885.73 | 0.00 | T1 1.5R @ 892.84 |
| Target hit | 2026-02-26 11:00:00 | 903.00 | 905.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 834.85 | 840.89 | 0.00 | ORB-short ORB[838.80,846.30] vol=1.8x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-03-12 09:40:00 | 837.91 | 839.61 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2026-03-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:45:00 | 830.85 | 826.32 | 0.00 | ORB-long ORB[817.30,828.95] vol=2.4x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-03-17 11:10:00 | 827.94 | 826.67 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 841.45 | 836.36 | 0.00 | ORB-long ORB[826.10,838.40] vol=1.5x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:45:00 | 847.44 | 839.08 | 0.00 | T1 1.5R @ 847.44 |
| Stop hit — per-position SL triggered | 2026-03-18 10:00:00 | 841.45 | 841.57 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-15 09:40:00 | 1093.40 | 2025-10-15 10:25:00 | 1089.51 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-17 09:55:00 | 1088.00 | 2025-10-17 10:05:00 | 1085.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-23 10:20:00 | 1111.20 | 2025-10-23 11:25:00 | 1114.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-31 10:15:00 | 1114.60 | 2025-10-31 10:35:00 | 1112.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-04 11:00:00 | 1083.60 | 2025-11-04 11:25:00 | 1085.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-13 09:55:00 | 998.90 | 2025-11-13 10:10:00 | 1001.09 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-17 10:45:00 | 990.00 | 2025-11-17 11:00:00 | 987.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-20 09:40:00 | 969.00 | 2025-11-20 10:35:00 | 971.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 10:00:00 | 960.50 | 2025-11-21 10:50:00 | 957.63 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-21 10:00:00 | 960.50 | 2025-11-21 11:00:00 | 960.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-27 11:00:00 | 981.00 | 2025-11-27 12:00:00 | 978.84 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-28 10:45:00 | 986.00 | 2025-11-28 13:05:00 | 989.61 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-28 10:45:00 | 986.00 | 2025-11-28 15:20:00 | 992.30 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-12-01 11:00:00 | 984.70 | 2025-12-01 12:05:00 | 981.56 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-01 11:00:00 | 984.70 | 2025-12-01 15:20:00 | 979.55 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-12-02 09:55:00 | 969.40 | 2025-12-02 10:00:00 | 971.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-03 10:55:00 | 970.95 | 2025-12-03 11:05:00 | 973.04 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-05 09:30:00 | 965.30 | 2025-12-05 14:55:00 | 960.86 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-05 09:30:00 | 965.30 | 2025-12-05 15:20:00 | 962.55 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-08 10:30:00 | 950.80 | 2025-12-08 10:35:00 | 948.36 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-08 10:30:00 | 950.80 | 2025-12-08 15:20:00 | 924.35 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2025-12-09 09:40:00 | 914.65 | 2025-12-09 10:00:00 | 911.04 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-09 09:40:00 | 914.65 | 2025-12-09 10:15:00 | 914.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 11:15:00 | 976.00 | 2025-12-16 11:20:00 | 978.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-24 11:05:00 | 965.80 | 2025-12-24 12:10:00 | 962.19 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-24 11:05:00 | 965.80 | 2025-12-24 15:20:00 | 960.50 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-12-29 10:35:00 | 940.80 | 2025-12-29 11:10:00 | 937.01 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-29 10:35:00 | 940.80 | 2025-12-29 15:20:00 | 934.95 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-01-02 09:30:00 | 949.65 | 2026-01-02 09:35:00 | 953.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-01-02 09:30:00 | 949.65 | 2026-01-02 09:40:00 | 949.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 09:40:00 | 863.00 | 2026-02-05 11:00:00 | 869.69 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2026-02-06 09:45:00 | 899.70 | 2026-02-06 10:00:00 | 895.71 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-11 09:30:00 | 913.65 | 2026-02-11 09:40:00 | 910.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-11 09:30:00 | 913.65 | 2026-02-11 09:45:00 | 913.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:10:00 | 915.00 | 2026-02-12 10:15:00 | 912.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-16 09:30:00 | 906.00 | 2026-02-16 09:55:00 | 901.11 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-02-18 09:35:00 | 910.05 | 2026-02-18 09:45:00 | 906.15 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-18 09:35:00 | 910.05 | 2026-02-18 15:20:00 | 897.20 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2026-02-26 09:40:00 | 888.85 | 2026-02-26 10:00:00 | 892.84 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-26 09:40:00 | 888.85 | 2026-02-26 11:00:00 | 903.00 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2026-03-12 09:30:00 | 834.85 | 2026-03-12 09:40:00 | 837.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-17 10:45:00 | 830.85 | 2026-03-17 11:10:00 | 827.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-18 09:30:00 | 841.45 | 2026-03-18 09:45:00 | 847.44 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-18 09:30:00 | 841.45 | 2026-03-18 10:00:00 | 841.45 | STOP_HIT | 0.50 | 0.00% |
