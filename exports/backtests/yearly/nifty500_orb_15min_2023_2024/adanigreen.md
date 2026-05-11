# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 1350.00
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
| PARTIAL | 25 |
| TARGET_HIT | 7 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 45
- **Target hits / Stop hits / Partials:** 7 / 45 / 25
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 16.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 19 | 46.3% | 5 | 22 | 14 | 0.34% | 14.1% |
| BUY @ 2nd Alert (retest1) | 41 | 19 | 46.3% | 5 | 22 | 14 | 0.34% | 14.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 13 | 36.1% | 2 | 23 | 11 | 0.07% | 2.4% |
| SELL @ 2nd Alert (retest1) | 36 | 13 | 36.1% | 2 | 23 | 11 | 0.07% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 32 | 41.6% | 7 | 45 | 25 | 0.21% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 968.65 | 972.60 | 0.00 | ORB-short ORB[970.85,977.00] vol=2.0x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:35:00 | 965.03 | 970.89 | 0.00 | T1 1.5R @ 965.03 |
| Stop hit — per-position SL triggered | 2023-06-09 09:40:00 | 968.65 | 970.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 10:05:00 | 962.95 | 965.70 | 0.00 | ORB-short ORB[963.00,968.95] vol=1.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:10:00 | 959.14 | 965.07 | 0.00 | T1 1.5R @ 959.14 |
| Stop hit — per-position SL triggered | 2023-06-12 13:50:00 | 962.95 | 961.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:40:00 | 954.90 | 957.66 | 0.00 | ORB-short ORB[955.55,963.90] vol=2.1x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:00:00 | 950.85 | 956.23 | 0.00 | T1 1.5R @ 950.85 |
| Stop hit — per-position SL triggered | 2023-06-13 10:05:00 | 954.90 | 956.02 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 10:40:00 | 950.05 | 954.02 | 0.00 | ORB-short ORB[953.05,961.00] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-06-14 10:50:00 | 951.90 | 953.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:05:00 | 973.85 | 962.92 | 0.00 | ORB-long ORB[958.10,963.95] vol=4.2x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:10:00 | 978.90 | 966.69 | 0.00 | T1 1.5R @ 978.90 |
| Stop hit — per-position SL triggered | 2023-06-21 10:15:00 | 973.85 | 968.42 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-07-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:10:00 | 950.05 | 945.00 | 0.00 | ORB-long ORB[941.00,949.95] vol=2.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2023-07-04 10:25:00 | 946.74 | 945.44 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 11:15:00 | 955.40 | 957.31 | 0.00 | ORB-short ORB[955.55,964.90] vol=1.6x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 13:15:00 | 951.44 | 955.86 | 0.00 | T1 1.5R @ 951.44 |
| Stop hit — per-position SL triggered | 2023-07-07 13:55:00 | 955.40 | 955.73 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-07-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:30:00 | 968.75 | 964.33 | 0.00 | ORB-long ORB[955.00,964.65] vol=3.3x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-07-13 09:40:00 | 965.64 | 965.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 957.80 | 961.37 | 0.00 | ORB-short ORB[959.05,968.90] vol=1.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-07-18 09:45:00 | 960.50 | 961.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:45:00 | 994.00 | 987.96 | 0.00 | ORB-long ORB[979.70,988.70] vol=2.4x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-07-21 10:15:00 | 991.14 | 990.74 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:45:00 | 1007.10 | 1002.50 | 0.00 | ORB-long ORB[990.00,1003.05] vol=4.2x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 09:55:00 | 1012.77 | 1004.95 | 0.00 | T1 1.5R @ 1012.77 |
| Target hit | 2023-07-25 15:20:00 | 1088.05 | 1049.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-07-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 11:10:00 | 1116.45 | 1126.45 | 0.00 | ORB-short ORB[1122.50,1134.80] vol=2.1x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 11:15:00 | 1110.22 | 1125.58 | 0.00 | T1 1.5R @ 1110.22 |
| Stop hit — per-position SL triggered | 2023-07-28 11:40:00 | 1116.45 | 1124.37 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-14 09:30:00 | 942.00 | 950.32 | 0.00 | ORB-short ORB[945.00,958.00] vol=1.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2023-08-14 09:35:00 | 946.15 | 949.62 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 11:10:00 | 964.00 | 960.72 | 0.00 | ORB-long ORB[954.40,963.00] vol=6.6x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 11:15:00 | 966.86 | 961.93 | 0.00 | T1 1.5R @ 966.86 |
| Stop hit — per-position SL triggered | 2023-09-06 11:45:00 | 964.00 | 964.56 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-09-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:50:00 | 999.90 | 1001.88 | 0.00 | ORB-short ORB[1002.00,1015.15] vol=2.6x ATR=3.64 |
| Stop hit — per-position SL triggered | 2023-09-07 11:25:00 | 1003.54 | 1001.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 10:55:00 | 999.65 | 1002.50 | 0.00 | ORB-short ORB[1012.10,1026.95] vol=3.1x ATR=5.63 |
| Stop hit — per-position SL triggered | 2023-09-12 15:15:00 | 1005.28 | 1001.76 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:15:00 | 1003.80 | 992.32 | 0.00 | ORB-long ORB[983.80,992.85] vol=3.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-09-15 10:20:00 | 1000.63 | 996.61 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 11:05:00 | 1011.55 | 1011.75 | 0.00 | ORB-short ORB[1013.05,1025.45] vol=1.6x ATR=3.40 |
| Stop hit — per-position SL triggered | 2023-09-22 11:20:00 | 1014.95 | 1012.03 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:50:00 | 1004.95 | 1008.24 | 0.00 | ORB-short ORB[1012.00,1021.95] vol=3.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-09-25 11:00:00 | 1007.44 | 1007.85 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-09-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:10:00 | 1009.75 | 1004.06 | 0.00 | ORB-long ORB[999.00,1006.85] vol=1.9x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 10:25:00 | 1014.21 | 1008.26 | 0.00 | T1 1.5R @ 1014.21 |
| Target hit | 2023-09-27 15:20:00 | 1016.55 | 1014.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-09-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:35:00 | 1013.30 | 1013.37 | 0.00 | ORB-short ORB[1014.10,1020.95] vol=3.1x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-09-28 10:45:00 | 1015.84 | 1013.95 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-10-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 11:05:00 | 975.80 | 985.55 | 0.00 | ORB-short ORB[986.10,997.95] vol=1.8x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-10-03 15:10:00 | 978.66 | 980.92 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-10-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 10:45:00 | 980.75 | 973.41 | 0.00 | ORB-long ORB[969.00,979.40] vol=1.7x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:00:00 | 985.67 | 975.54 | 0.00 | T1 1.5R @ 985.67 |
| Stop hit — per-position SL triggered | 2023-10-04 11:40:00 | 980.75 | 980.53 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-10-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:30:00 | 970.25 | 975.14 | 0.00 | ORB-short ORB[975.60,982.10] vol=1.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-10-05 10:45:00 | 972.63 | 974.69 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 10:35:00 | 945.95 | 947.86 | 0.00 | ORB-short ORB[946.80,956.60] vol=1.9x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 13:45:00 | 942.78 | 946.81 | 0.00 | T1 1.5R @ 942.78 |
| Target hit | 2023-10-16 15:20:00 | 937.00 | 944.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2023-10-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 10:25:00 | 940.70 | 936.25 | 0.00 | ORB-long ORB[930.30,940.15] vol=1.6x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 10:35:00 | 944.66 | 937.31 | 0.00 | T1 1.5R @ 944.66 |
| Stop hit — per-position SL triggered | 2023-10-20 11:15:00 | 940.70 | 939.54 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-11-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:05:00 | 917.65 | 913.91 | 0.00 | ORB-long ORB[908.00,916.35] vol=1.7x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 10:55:00 | 921.37 | 915.45 | 0.00 | T1 1.5R @ 921.37 |
| Target hit | 2023-11-06 15:20:00 | 926.40 | 921.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2023-11-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:50:00 | 940.50 | 936.71 | 0.00 | ORB-long ORB[932.70,939.90] vol=1.7x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 13:15:00 | 944.67 | 939.10 | 0.00 | T1 1.5R @ 944.67 |
| Stop hit — per-position SL triggered | 2023-11-08 13:30:00 | 940.50 | 939.37 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-11-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 11:10:00 | 924.10 | 925.45 | 0.00 | ORB-short ORB[925.10,937.85] vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 12:05:00 | 921.45 | 924.88 | 0.00 | T1 1.5R @ 921.45 |
| Stop hit — per-position SL triggered | 2023-11-24 12:10:00 | 924.10 | 926.10 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:15:00 | 1516.20 | 1520.20 | 0.00 | ORB-short ORB[1520.50,1541.95] vol=2.6x ATR=5.27 |
| Stop hit — per-position SL triggered | 2023-12-19 11:05:00 | 1521.47 | 1519.67 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 1568.05 | 1552.33 | 0.00 | ORB-long ORB[1535.00,1549.40] vol=6.0x ATR=6.95 |
| Stop hit — per-position SL triggered | 2023-12-20 09:35:00 | 1561.10 | 1554.59 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:35:00 | 1605.15 | 1592.67 | 0.00 | ORB-long ORB[1585.00,1599.70] vol=1.9x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 10:45:00 | 1614.58 | 1599.62 | 0.00 | T1 1.5R @ 1614.58 |
| Stop hit — per-position SL triggered | 2024-01-01 11:00:00 | 1605.15 | 1601.23 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:30:00 | 1615.05 | 1604.34 | 0.00 | ORB-long ORB[1592.00,1607.40] vol=3.1x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-01-02 09:40:00 | 1607.93 | 1606.11 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:30:00 | 1678.90 | 1685.23 | 0.00 | ORB-short ORB[1680.00,1692.00] vol=1.8x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 09:35:00 | 1671.70 | 1683.44 | 0.00 | T1 1.5R @ 1671.70 |
| Stop hit — per-position SL triggered | 2024-01-08 09:55:00 | 1678.90 | 1679.96 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:00:00 | 1702.90 | 1696.51 | 0.00 | ORB-long ORB[1691.00,1700.00] vol=3.3x ATR=6.89 |
| Stop hit — per-position SL triggered | 2024-01-09 10:05:00 | 1696.01 | 1696.68 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:40:00 | 1717.00 | 1707.60 | 0.00 | ORB-long ORB[1692.00,1713.30] vol=5.1x ATR=7.92 |
| Stop hit — per-position SL triggered | 2024-01-10 09:45:00 | 1709.08 | 1708.08 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:40:00 | 1738.15 | 1728.50 | 0.00 | ORB-long ORB[1717.00,1730.75] vol=2.5x ATR=7.37 |
| Stop hit — per-position SL triggered | 2024-01-12 10:00:00 | 1730.78 | 1730.91 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-01-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:45:00 | 1706.95 | 1715.19 | 0.00 | ORB-short ORB[1711.00,1727.10] vol=2.0x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 09:55:00 | 1696.90 | 1712.22 | 0.00 | T1 1.5R @ 1696.90 |
| Stop hit — per-position SL triggered | 2024-01-15 10:05:00 | 1706.95 | 1710.32 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-02-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:20:00 | 1685.00 | 1676.10 | 0.00 | ORB-long ORB[1663.85,1681.90] vol=1.8x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:05:00 | 1695.73 | 1681.16 | 0.00 | T1 1.5R @ 1695.73 |
| Stop hit — per-position SL triggered | 2024-02-02 11:25:00 | 1685.00 | 1682.57 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-02-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:35:00 | 1688.90 | 1675.41 | 0.00 | ORB-long ORB[1653.00,1674.30] vol=1.5x ATR=6.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 10:40:00 | 1698.65 | 1680.09 | 0.00 | T1 1.5R @ 1698.65 |
| Target hit | 2024-02-06 12:55:00 | 1712.00 | 1713.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2024-02-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 09:30:00 | 1907.50 | 1881.81 | 0.00 | ORB-long ORB[1857.15,1879.90] vol=4.1x ATR=10.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 09:40:00 | 1922.91 | 1909.60 | 0.00 | T1 1.5R @ 1922.91 |
| Target hit | 2024-02-15 10:15:00 | 1916.20 | 1916.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:35:00 | 1947.35 | 1929.94 | 0.00 | ORB-long ORB[1911.75,1939.00] vol=3.0x ATR=8.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:40:00 | 1959.84 | 1937.29 | 0.00 | T1 1.5R @ 1959.84 |
| Stop hit — per-position SL triggered | 2024-02-21 09:45:00 | 1947.35 | 1937.75 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:40:00 | 1936.85 | 1927.08 | 0.00 | ORB-long ORB[1915.00,1930.75] vol=2.1x ATR=6.78 |
| Stop hit — per-position SL triggered | 2024-02-23 09:45:00 | 1930.07 | 1928.30 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 1929.75 | 1935.96 | 0.00 | ORB-short ORB[1931.00,1953.50] vol=2.0x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:35:00 | 1920.07 | 1933.85 | 0.00 | T1 1.5R @ 1920.07 |
| Stop hit — per-position SL triggered | 2024-03-06 09:40:00 | 1929.75 | 1933.16 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 10:15:00 | 1958.80 | 1929.79 | 0.00 | ORB-long ORB[1904.10,1932.00] vol=4.1x ATR=9.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:20:00 | 1972.65 | 1936.86 | 0.00 | T1 1.5R @ 1972.65 |
| Stop hit — per-position SL triggered | 2024-03-11 10:25:00 | 1958.80 | 1942.39 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1895.45 | 1910.10 | 0.00 | ORB-short ORB[1900.00,1922.95] vol=3.5x ATR=6.80 |
| Stop hit — per-position SL triggered | 2024-04-04 11:00:00 | 1902.25 | 1909.56 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-04-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:45:00 | 1918.35 | 1901.26 | 0.00 | ORB-long ORB[1875.00,1896.00] vol=5.8x ATR=9.01 |
| Stop hit — per-position SL triggered | 2024-04-05 09:50:00 | 1909.34 | 1902.43 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 1903.35 | 1911.68 | 0.00 | ORB-short ORB[1907.50,1925.00] vol=2.4x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 12:45:00 | 1895.92 | 1904.18 | 0.00 | T1 1.5R @ 1895.92 |
| Target hit | 2024-04-12 15:20:00 | 1882.10 | 1897.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-04-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 11:05:00 | 1799.25 | 1805.99 | 0.00 | ORB-short ORB[1801.10,1819.65] vol=2.5x ATR=4.67 |
| Stop hit — per-position SL triggered | 2024-04-25 14:05:00 | 1803.92 | 1801.85 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-04-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 11:00:00 | 1801.30 | 1805.85 | 0.00 | ORB-short ORB[1810.00,1823.30] vol=1.6x ATR=3.91 |
| Stop hit — per-position SL triggered | 2024-04-26 14:10:00 | 1805.21 | 1803.61 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:35:00 | 1837.35 | 1817.86 | 0.00 | ORB-long ORB[1803.40,1814.00] vol=3.2x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-04-30 09:40:00 | 1831.83 | 1822.95 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:45:00 | 1781.20 | 1769.86 | 0.00 | ORB-long ORB[1752.05,1775.00] vol=2.6x ATR=7.42 |
| Stop hit — per-position SL triggered | 2024-05-07 09:50:00 | 1773.78 | 1770.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-06-09 09:30:00 | 968.65 | 2023-06-09 09:35:00 | 965.03 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-06-09 09:30:00 | 968.65 | 2023-06-09 09:40:00 | 968.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-12 10:05:00 | 962.95 | 2023-06-12 10:10:00 | 959.14 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-06-12 10:05:00 | 962.95 | 2023-06-12 13:50:00 | 962.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-13 09:40:00 | 954.90 | 2023-06-13 10:00:00 | 950.85 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-06-13 09:40:00 | 954.90 | 2023-06-13 10:05:00 | 954.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 10:40:00 | 950.05 | 2023-06-14 10:50:00 | 951.90 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-21 10:05:00 | 973.85 | 2023-06-21 10:10:00 | 978.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-06-21 10:05:00 | 973.85 | 2023-06-21 10:15:00 | 973.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-04 10:10:00 | 950.05 | 2023-07-04 10:25:00 | 946.74 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-07 11:15:00 | 955.40 | 2023-07-07 13:15:00 | 951.44 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-07-07 11:15:00 | 955.40 | 2023-07-07 13:55:00 | 955.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-13 09:30:00 | 968.75 | 2023-07-13 09:40:00 | 965.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-07-18 09:35:00 | 957.80 | 2023-07-18 09:45:00 | 960.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-07-21 09:45:00 | 994.00 | 2023-07-21 10:15:00 | 991.14 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-25 09:45:00 | 1007.10 | 2023-07-25 09:55:00 | 1012.77 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-07-25 09:45:00 | 1007.10 | 2023-07-25 15:20:00 | 1088.05 | TARGET_HIT | 0.50 | 8.04% |
| SELL | retest1 | 2023-07-28 11:10:00 | 1116.45 | 2023-07-28 11:15:00 | 1110.22 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-07-28 11:10:00 | 1116.45 | 2023-07-28 11:40:00 | 1116.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-14 09:30:00 | 942.00 | 2023-08-14 09:35:00 | 946.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-06 11:10:00 | 964.00 | 2023-09-06 11:15:00 | 966.86 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-06 11:10:00 | 964.00 | 2023-09-06 11:45:00 | 964.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-07 10:50:00 | 999.90 | 2023-09-07 11:25:00 | 1003.54 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-09-12 10:55:00 | 999.65 | 2023-09-12 15:15:00 | 1005.28 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-09-15 10:15:00 | 1003.80 | 2023-09-15 10:20:00 | 1000.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-22 11:05:00 | 1011.55 | 2023-09-22 11:20:00 | 1014.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-09-25 10:50:00 | 1004.95 | 2023-09-25 11:00:00 | 1007.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-27 10:10:00 | 1009.75 | 2023-09-27 10:25:00 | 1014.21 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-09-27 10:10:00 | 1009.75 | 2023-09-27 15:20:00 | 1016.55 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2023-09-28 10:35:00 | 1013.30 | 2023-09-28 10:45:00 | 1015.84 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-03 11:05:00 | 975.80 | 2023-10-03 15:10:00 | 978.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-04 10:45:00 | 980.75 | 2023-10-04 11:00:00 | 985.67 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-10-04 10:45:00 | 980.75 | 2023-10-04 11:40:00 | 980.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 10:30:00 | 970.25 | 2023-10-05 10:45:00 | 972.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-16 10:35:00 | 945.95 | 2023-10-16 13:45:00 | 942.78 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-16 10:35:00 | 945.95 | 2023-10-16 15:20:00 | 937.00 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2023-10-20 10:25:00 | 940.70 | 2023-10-20 10:35:00 | 944.66 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-10-20 10:25:00 | 940.70 | 2023-10-20 11:15:00 | 940.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 10:05:00 | 917.65 | 2023-11-06 10:55:00 | 921.37 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-11-06 10:05:00 | 917.65 | 2023-11-06 15:20:00 | 926.40 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2023-11-08 10:50:00 | 940.50 | 2023-11-08 13:15:00 | 944.67 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-08 10:50:00 | 940.50 | 2023-11-08 13:30:00 | 940.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 11:10:00 | 924.10 | 2023-11-24 12:05:00 | 921.45 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-11-24 11:10:00 | 924.10 | 2023-11-24 12:10:00 | 924.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-19 10:15:00 | 1516.20 | 2023-12-19 11:05:00 | 1521.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-20 09:30:00 | 1568.05 | 2023-12-20 09:35:00 | 1561.10 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-01-01 10:35:00 | 1605.15 | 2024-01-01 10:45:00 | 1614.58 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-01-01 10:35:00 | 1605.15 | 2024-01-01 11:00:00 | 1605.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-02 09:30:00 | 1615.05 | 2024-01-02 09:40:00 | 1607.93 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-01-08 09:30:00 | 1678.90 | 2024-01-08 09:35:00 | 1671.70 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-08 09:30:00 | 1678.90 | 2024-01-08 09:55:00 | 1678.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-09 10:00:00 | 1702.90 | 2024-01-09 10:05:00 | 1696.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-01-10 09:40:00 | 1717.00 | 2024-01-10 09:45:00 | 1709.08 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-01-12 09:40:00 | 1738.15 | 2024-01-12 10:00:00 | 1730.78 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-01-15 09:45:00 | 1706.95 | 2024-01-15 09:55:00 | 1696.90 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-01-15 09:45:00 | 1706.95 | 2024-01-15 10:05:00 | 1706.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 10:20:00 | 1685.00 | 2024-02-02 11:05:00 | 1695.73 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-02-02 10:20:00 | 1685.00 | 2024-02-02 11:25:00 | 1685.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-06 10:35:00 | 1688.90 | 2024-02-06 10:40:00 | 1698.65 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-02-06 10:35:00 | 1688.90 | 2024-02-06 12:55:00 | 1712.00 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2024-02-15 09:30:00 | 1907.50 | 2024-02-15 09:40:00 | 1922.91 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2024-02-15 09:30:00 | 1907.50 | 2024-02-15 10:15:00 | 1916.20 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-02-21 09:35:00 | 1947.35 | 2024-02-21 09:40:00 | 1959.84 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-02-21 09:35:00 | 1947.35 | 2024-02-21 09:45:00 | 1947.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-23 09:40:00 | 1936.85 | 2024-02-23 09:45:00 | 1930.07 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-06 09:30:00 | 1929.75 | 2024-03-06 09:35:00 | 1920.07 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-03-06 09:30:00 | 1929.75 | 2024-03-06 09:40:00 | 1929.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-11 10:15:00 | 1958.80 | 2024-03-11 10:20:00 | 1972.65 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-03-11 10:15:00 | 1958.80 | 2024-03-11 10:25:00 | 1958.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1895.45 | 2024-04-04 11:00:00 | 1902.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-04-05 09:45:00 | 1918.35 | 2024-04-05 09:50:00 | 1909.34 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-04-12 09:40:00 | 1903.35 | 2024-04-12 12:45:00 | 1895.92 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-12 09:40:00 | 1903.35 | 2024-04-12 15:20:00 | 1882.10 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2024-04-25 11:05:00 | 1799.25 | 2024-04-25 14:05:00 | 1803.92 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-26 11:00:00 | 1801.30 | 2024-04-26 14:10:00 | 1805.21 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-30 09:35:00 | 1837.35 | 2024-04-30 09:40:00 | 1831.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-07 09:45:00 | 1781.20 | 2024-05-07 09:50:00 | 1773.78 | STOP_HIT | 1.00 | -0.42% |
