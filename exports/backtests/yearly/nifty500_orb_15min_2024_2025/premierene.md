# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2024-09-03 09:40:00 → 2026-05-08 15:25:00 (31070 bars)
- **Last close:** 1014.00
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 10
- **Avg / median % per leg:** 0.44% / 0.00%
- **Sum % (uncompounded):** 11.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.35% | 6.3% |
| BUY @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.35% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.62% | 5.6% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.62% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 13 | 48.1% | 3 | 14 | 10 | 0.44% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:55:00 | 1045.00 | 1032.23 | 0.00 | ORB-long ORB[1025.25,1039.00] vol=5.1x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:00:00 | 1053.20 | 1036.27 | 0.00 | T1 1.5R @ 1053.20 |
| Target hit | 2024-10-09 15:20:00 | 1078.10 | 1061.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 1117.75 | 1127.92 | 0.00 | ORB-short ORB[1129.00,1144.70] vol=2.0x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:05:00 | 1108.84 | 1123.04 | 0.00 | T1 1.5R @ 1108.84 |
| Stop hit — per-position SL triggered | 2024-10-17 10:40:00 | 1117.75 | 1120.09 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:35:00 | 1020.60 | 1007.88 | 0.00 | ORB-long ORB[998.05,1010.00] vol=2.8x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 09:55:00 | 1029.74 | 1016.01 | 0.00 | T1 1.5R @ 1029.74 |
| Stop hit — per-position SL triggered | 2024-10-31 10:00:00 | 1020.60 | 1016.47 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1332.00 | 1323.49 | 0.00 | ORB-long ORB[1311.25,1330.00] vol=2.1x ATR=6.55 |
| Stop hit — per-position SL triggered | 2024-12-26 09:35:00 | 1325.45 | 1323.66 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1315.45 | 1309.47 | 0.00 | ORB-long ORB[1297.50,1314.00] vol=3.3x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:45:00 | 1324.46 | 1313.22 | 0.00 | T1 1.5R @ 1324.46 |
| Target hit | 2024-12-30 13:25:00 | 1337.15 | 1339.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-12-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:35:00 | 1311.80 | 1320.64 | 0.00 | ORB-short ORB[1315.00,1330.00] vol=2.3x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:50:00 | 1301.73 | 1316.27 | 0.00 | T1 1.5R @ 1301.73 |
| Stop hit — per-position SL triggered | 2024-12-31 09:55:00 | 1311.80 | 1315.71 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1317.20 | 1324.37 | 0.00 | ORB-short ORB[1320.55,1339.00] vol=2.6x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:00:00 | 1308.30 | 1322.67 | 0.00 | T1 1.5R @ 1308.30 |
| Target hit | 2025-01-06 15:20:00 | 1273.55 | 1297.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:10:00 | 1172.50 | 1165.33 | 0.00 | ORB-long ORB[1158.35,1170.50] vol=1.7x ATR=6.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:45:00 | 1182.00 | 1168.67 | 0.00 | T1 1.5R @ 1182.00 |
| Stop hit — per-position SL triggered | 2025-01-16 11:10:00 | 1172.50 | 1169.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 1166.00 | 1156.62 | 0.00 | ORB-long ORB[1145.05,1160.60] vol=2.3x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-01-20 10:00:00 | 1160.99 | 1159.87 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:40:00 | 1038.85 | 1030.31 | 0.00 | ORB-long ORB[1020.10,1032.00] vol=2.0x ATR=4.89 |
| Stop hit — per-position SL triggered | 2025-02-06 09:55:00 | 1033.96 | 1031.68 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 1027.40 | 1020.08 | 0.00 | ORB-long ORB[1012.10,1026.10] vol=2.3x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 09:35:00 | 1034.81 | 1022.94 | 0.00 | T1 1.5R @ 1034.81 |
| Stop hit — per-position SL triggered | 2025-02-07 09:40:00 | 1027.40 | 1023.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:45:00 | 999.95 | 1022.34 | 0.00 | ORB-short ORB[1021.00,1034.50] vol=1.7x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-02-10 10:50:00 | 1005.28 | 1020.77 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 958.90 | 952.28 | 0.00 | ORB-long ORB[943.80,958.00] vol=2.2x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-02-25 10:05:00 | 953.95 | 954.62 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-04-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 10:00:00 | 934.70 | 927.42 | 0.00 | ORB-long ORB[916.25,930.00] vol=1.9x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-04-01 10:10:00 | 929.51 | 927.77 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 1010.70 | 1016.85 | 0.00 | ORB-short ORB[1012.10,1024.00] vol=2.1x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:40:00 | 1004.83 | 1014.82 | 0.00 | T1 1.5R @ 1004.83 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 1010.70 | 1012.73 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 1011.85 | 1001.01 | 0.00 | ORB-long ORB[992.00,1002.65] vol=3.5x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 1006.40 | 1006.74 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:15:00 | 971.75 | 965.56 | 0.00 | ORB-long ORB[955.50,969.35] vol=2.0x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:25:00 | 977.90 | 967.91 | 0.00 | T1 1.5R @ 977.90 |
| Stop hit — per-position SL triggered | 2025-05-08 13:20:00 | 971.75 | 971.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-09 10:55:00 | 1045.00 | 2024-10-09 11:00:00 | 1053.20 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-10-09 10:55:00 | 1045.00 | 2024-10-09 15:20:00 | 1078.10 | TARGET_HIT | 0.50 | 3.17% |
| SELL | retest1 | 2024-10-17 09:40:00 | 1117.75 | 2024-10-17 10:05:00 | 1108.84 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-10-17 09:40:00 | 1117.75 | 2024-10-17 10:40:00 | 1117.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:35:00 | 1020.60 | 2024-10-31 09:55:00 | 1029.74 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2024-10-31 09:35:00 | 1020.60 | 2024-10-31 10:00:00 | 1020.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 09:30:00 | 1332.00 | 2024-12-26 09:35:00 | 1325.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1315.45 | 2024-12-30 09:45:00 | 1324.46 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1315.45 | 2024-12-30 13:25:00 | 1337.15 | TARGET_HIT | 0.50 | 1.65% |
| SELL | retest1 | 2024-12-31 09:35:00 | 1311.80 | 2024-12-31 09:50:00 | 1301.73 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-12-31 09:35:00 | 1311.80 | 2024-12-31 09:55:00 | 1311.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1317.20 | 2025-01-06 11:00:00 | 1308.30 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1317.20 | 2025-01-06 15:20:00 | 1273.55 | TARGET_HIT | 0.50 | 3.31% |
| BUY | retest1 | 2025-01-16 10:10:00 | 1172.50 | 2025-01-16 10:45:00 | 1182.00 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2025-01-16 10:10:00 | 1172.50 | 2025-01-16 11:10:00 | 1172.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-20 09:40:00 | 1166.00 | 2025-01-20 10:00:00 | 1160.99 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-06 09:40:00 | 1038.85 | 2025-02-06 09:55:00 | 1033.96 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-02-07 09:30:00 | 1027.40 | 2025-02-07 09:35:00 | 1034.81 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-02-07 09:30:00 | 1027.40 | 2025-02-07 09:40:00 | 1027.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 10:45:00 | 999.95 | 2025-02-10 10:50:00 | 1005.28 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-02-25 09:30:00 | 958.90 | 2025-02-25 10:05:00 | 953.95 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-04-01 10:00:00 | 934.70 | 2025-04-01 10:10:00 | 929.51 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-04-29 09:35:00 | 1010.70 | 2025-04-29 09:40:00 | 1004.83 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-29 09:35:00 | 1010.70 | 2025-04-29 09:55:00 | 1010.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:40:00 | 1011.85 | 2025-05-05 10:15:00 | 1006.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-05-08 10:15:00 | 971.75 | 2025-05-08 11:25:00 | 977.90 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-05-08 10:15:00 | 971.75 | 2025-05-08 13:20:00 | 971.75 | STOP_HIT | 0.50 | 0.00% |
