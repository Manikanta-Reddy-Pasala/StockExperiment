# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1589.10
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 10 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 33
- **Target hits / Stop hits / Partials:** 10 / 33 / 20
- **Avg / median % per leg:** 0.32% / 0.00%
- **Sum % (uncompounded):** 20.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 15 | 46.9% | 5 | 17 | 10 | 0.23% | 7.4% |
| BUY @ 2nd Alert (retest1) | 32 | 15 | 46.9% | 5 | 17 | 10 | 0.23% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 31 | 15 | 48.4% | 5 | 16 | 10 | 0.42% | 12.9% |
| SELL @ 2nd Alert (retest1) | 31 | 15 | 48.4% | 5 | 16 | 10 | 0.42% | 12.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 63 | 30 | 47.6% | 10 | 33 | 20 | 0.32% | 20.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 1168.00 | 1177.35 | 0.00 | ORB-short ORB[1174.25,1185.00] vol=1.8x ATR=4.60 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 1172.60 | 1176.66 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 1198.50 | 1205.17 | 0.00 | ORB-short ORB[1201.00,1213.00] vol=2.0x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:55:00 | 1190.81 | 1202.84 | 0.00 | T1 1.5R @ 1190.81 |
| Stop hit — per-position SL triggered | 2024-06-25 11:00:00 | 1198.50 | 1198.45 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:45:00 | 1209.25 | 1200.56 | 0.00 | ORB-long ORB[1190.00,1204.35] vol=1.6x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 1204.35 | 1201.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:25:00 | 1218.75 | 1225.40 | 0.00 | ORB-short ORB[1222.00,1235.00] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:45:00 | 1211.88 | 1223.10 | 0.00 | T1 1.5R @ 1211.88 |
| Stop hit — per-position SL triggered | 2024-07-03 11:25:00 | 1218.75 | 1222.06 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 1332.00 | 1314.70 | 0.00 | ORB-long ORB[1307.75,1321.45] vol=1.9x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:10:00 | 1341.63 | 1316.84 | 0.00 | T1 1.5R @ 1341.63 |
| Stop hit — per-position SL triggered | 2024-07-25 11:50:00 | 1332.00 | 1322.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 1352.45 | 1341.92 | 0.00 | ORB-long ORB[1331.00,1345.00] vol=2.2x ATR=7.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:40:00 | 1363.57 | 1348.26 | 0.00 | T1 1.5R @ 1363.57 |
| Target hit | 2024-08-13 14:00:00 | 1375.60 | 1381.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1295.55 | 1301.26 | 0.00 | ORB-short ORB[1296.00,1312.90] vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 1298.90 | 1301.24 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-08-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:35:00 | 1318.85 | 1307.54 | 0.00 | ORB-long ORB[1300.50,1317.65] vol=2.8x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-08-30 11:05:00 | 1314.38 | 1310.86 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:45:00 | 1272.35 | 1278.02 | 0.00 | ORB-short ORB[1276.05,1288.70] vol=2.8x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:00:00 | 1265.00 | 1274.76 | 0.00 | T1 1.5R @ 1265.00 |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 1272.35 | 1273.87 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:20:00 | 1212.00 | 1232.62 | 0.00 | ORB-short ORB[1237.00,1253.95] vol=1.6x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:45:00 | 1205.36 | 1224.23 | 0.00 | T1 1.5R @ 1205.36 |
| Target hit | 2024-09-19 15:20:00 | 1209.10 | 1210.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-09-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:45:00 | 1228.20 | 1238.61 | 0.00 | ORB-short ORB[1235.00,1245.70] vol=2.9x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-09-24 10:50:00 | 1232.61 | 1238.43 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 1215.90 | 1222.96 | 0.00 | ORB-short ORB[1220.00,1232.70] vol=1.5x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 1220.02 | 1222.57 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-10-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:05:00 | 1284.60 | 1276.01 | 0.00 | ORB-long ORB[1262.35,1278.20] vol=8.0x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-10-14 10:10:00 | 1279.72 | 1276.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:15:00 | 1304.65 | 1293.80 | 0.00 | ORB-long ORB[1275.05,1290.50] vol=8.0x ATR=6.74 |
| Stop hit — per-position SL triggered | 2024-10-16 10:20:00 | 1297.91 | 1295.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:15:00 | 1258.50 | 1264.74 | 0.00 | ORB-short ORB[1262.75,1281.60] vol=2.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-10-21 11:45:00 | 1262.10 | 1264.05 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:45:00 | 1225.00 | 1207.70 | 0.00 | ORB-long ORB[1200.15,1214.00] vol=1.5x ATR=5.45 |
| Stop hit — per-position SL triggered | 2024-10-24 10:50:00 | 1219.55 | 1208.98 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:10:00 | 1276.70 | 1281.52 | 0.00 | ORB-short ORB[1279.15,1292.95] vol=1.5x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:25:00 | 1268.88 | 1279.32 | 0.00 | T1 1.5R @ 1268.88 |
| Target hit | 2024-11-06 15:20:00 | 1257.60 | 1263.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 1181.75 | 1190.79 | 0.00 | ORB-short ORB[1184.25,1201.75] vol=2.1x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:50:00 | 1173.85 | 1185.33 | 0.00 | T1 1.5R @ 1173.85 |
| Target hit | 2024-11-12 15:20:00 | 1122.00 | 1148.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:55:00 | 1070.55 | 1067.16 | 0.00 | ORB-long ORB[1065.05,1069.90] vol=2.4x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-12-17 10:00:00 | 1067.90 | 1067.20 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 1014.65 | 1019.57 | 0.00 | ORB-short ORB[1019.00,1030.00] vol=1.8x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 1017.29 | 1018.85 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-12-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:00:00 | 961.15 | 966.61 | 0.00 | ORB-short ORB[961.85,972.80] vol=1.5x ATR=3.03 |
| Stop hit — per-position SL triggered | 2024-12-27 11:05:00 | 964.18 | 966.56 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:50:00 | 970.05 | 974.80 | 0.00 | ORB-short ORB[972.05,985.45] vol=2.4x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:55:00 | 965.19 | 973.95 | 0.00 | T1 1.5R @ 965.19 |
| Target hit | 2024-12-30 15:20:00 | 942.35 | 961.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 985.50 | 979.39 | 0.00 | ORB-long ORB[966.90,980.50] vol=4.5x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:50:00 | 989.61 | 981.97 | 0.00 | T1 1.5R @ 989.61 |
| Stop hit — per-position SL triggered | 2025-01-02 09:55:00 | 985.50 | 983.25 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 941.30 | 951.17 | 0.00 | ORB-short ORB[945.35,958.65] vol=1.6x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:00:00 | 935.21 | 945.77 | 0.00 | T1 1.5R @ 935.21 |
| Stop hit — per-position SL triggered | 2025-01-10 10:10:00 | 941.30 | 944.75 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:55:00 | 949.00 | 956.30 | 0.00 | ORB-short ORB[957.05,968.60] vol=2.3x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 13:00:00 | 943.84 | 950.45 | 0.00 | T1 1.5R @ 943.84 |
| Target hit | 2025-01-24 15:20:00 | 938.15 | 944.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:45:00 | 967.60 | 956.85 | 0.00 | ORB-long ORB[946.05,956.00] vol=3.3x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:00:00 | 972.36 | 960.46 | 0.00 | T1 1.5R @ 972.36 |
| Target hit | 2025-01-31 15:20:00 | 982.85 | 976.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-02-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:00:00 | 995.65 | 994.89 | 0.00 | ORB-long ORB[984.90,995.50] vol=2.3x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:30:00 | 1001.52 | 996.65 | 0.00 | T1 1.5R @ 1001.52 |
| Stop hit — per-position SL triggered | 2025-02-01 11:35:00 | 995.65 | 996.30 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:50:00 | 900.55 | 903.58 | 0.00 | ORB-short ORB[901.45,909.15] vol=3.1x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-02-11 10:05:00 | 903.73 | 903.36 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 905.55 | 902.23 | 0.00 | ORB-long ORB[898.00,905.00] vol=2.1x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-02-13 09:35:00 | 901.77 | 902.25 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:10:00 | 898.80 | 901.70 | 0.00 | ORB-short ORB[901.00,908.40] vol=1.8x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:50:00 | 894.75 | 900.62 | 0.00 | T1 1.5R @ 894.75 |
| Stop hit — per-position SL triggered | 2025-03-12 10:55:00 | 898.80 | 900.56 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-03-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 10:55:00 | 892.75 | 895.33 | 0.00 | ORB-short ORB[893.20,901.55] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-03-18 11:05:00 | 894.95 | 895.24 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:55:00 | 905.85 | 900.06 | 0.00 | ORB-long ORB[896.50,905.55] vol=2.9x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 11:05:00 | 909.87 | 900.86 | 0.00 | T1 1.5R @ 909.87 |
| Target hit | 2025-03-19 15:20:00 | 929.70 | 920.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-03-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:50:00 | 957.50 | 952.27 | 0.00 | ORB-long ORB[946.70,955.00] vol=2.1x ATR=5.54 |
| Stop hit — per-position SL triggered | 2025-03-24 11:45:00 | 951.96 | 954.15 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 921.35 | 928.71 | 0.00 | ORB-short ORB[926.25,935.85] vol=1.8x ATR=4.11 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 925.46 | 927.56 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-03-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:35:00 | 925.40 | 921.46 | 0.00 | ORB-long ORB[913.80,925.25] vol=3.0x ATR=3.06 |
| Stop hit — per-position SL triggered | 2025-03-27 11:00:00 | 922.34 | 922.34 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-03-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:05:00 | 938.55 | 929.66 | 0.00 | ORB-long ORB[920.15,931.00] vol=2.0x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 10:10:00 | 943.80 | 932.18 | 0.00 | T1 1.5R @ 943.80 |
| Stop hit — per-position SL triggered | 2025-03-28 12:35:00 | 938.55 | 940.78 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-04-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:50:00 | 929.60 | 923.79 | 0.00 | ORB-long ORB[916.05,922.80] vol=1.5x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 927.24 | 924.40 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-04-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:45:00 | 920.15 | 927.26 | 0.00 | ORB-short ORB[925.90,934.85] vol=2.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-04-03 09:50:00 | 923.49 | 927.05 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:00:00 | 921.00 | 916.09 | 0.00 | ORB-long ORB[907.45,918.45] vol=3.4x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:05:00 | 925.43 | 917.92 | 0.00 | T1 1.5R @ 925.43 |
| Target hit | 2025-04-22 11:40:00 | 923.10 | 923.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 921.60 | 917.59 | 0.00 | ORB-long ORB[913.00,921.45] vol=1.5x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:40:00 | 927.09 | 919.37 | 0.00 | T1 1.5R @ 927.09 |
| Stop hit — per-position SL triggered | 2025-04-23 10:05:00 | 921.60 | 923.60 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 926.35 | 919.53 | 0.00 | ORB-long ORB[914.10,923.35] vol=1.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 09:35:00 | 931.37 | 925.81 | 0.00 | T1 1.5R @ 931.37 |
| Target hit | 2025-04-24 10:00:00 | 927.50 | 928.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 890.00 | 885.59 | 0.00 | ORB-long ORB[878.15,888.25] vol=2.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-05-05 10:20:00 | 887.07 | 886.79 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 885.15 | 881.53 | 0.00 | ORB-long ORB[877.00,884.95] vol=2.0x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-05-08 09:50:00 | 882.32 | 881.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-21 09:35:00 | 1168.00 | 2024-06-21 09:40:00 | 1172.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1198.50 | 2024-06-25 09:55:00 | 1190.81 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1198.50 | 2024-06-25 11:00:00 | 1198.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 10:45:00 | 1209.25 | 2024-06-26 11:00:00 | 1204.35 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-03 10:25:00 | 1218.75 | 2024-07-03 10:45:00 | 1211.88 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-07-03 10:25:00 | 1218.75 | 2024-07-03 11:25:00 | 1218.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 11:05:00 | 1332.00 | 2024-07-25 11:10:00 | 1341.63 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-07-25 11:05:00 | 1332.00 | 2024-07-25 11:50:00 | 1332.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1352.45 | 2024-08-13 09:40:00 | 1363.57 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1352.45 | 2024-08-13 14:00:00 | 1375.60 | TARGET_HIT | 0.50 | 1.71% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1295.55 | 2024-08-29 11:10:00 | 1298.90 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-30 10:35:00 | 1318.85 | 2024-08-30 11:05:00 | 1314.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-12 09:45:00 | 1272.35 | 2024-09-12 11:00:00 | 1265.00 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-12 09:45:00 | 1272.35 | 2024-09-12 12:15:00 | 1272.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:20:00 | 1212.00 | 2024-09-19 10:45:00 | 1205.36 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-19 10:20:00 | 1212.00 | 2024-09-19 15:20:00 | 1209.10 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-09-24 10:45:00 | 1228.20 | 2024-09-24 10:50:00 | 1232.61 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-25 10:05:00 | 1215.90 | 2024-09-25 10:15:00 | 1220.02 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-14 10:05:00 | 1284.60 | 2024-10-14 10:10:00 | 1279.72 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-16 10:15:00 | 1304.65 | 2024-10-16 10:20:00 | 1297.91 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-10-21 11:15:00 | 1258.50 | 2024-10-21 11:45:00 | 1262.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-24 10:45:00 | 1225.00 | 2024-10-24 10:50:00 | 1219.55 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-06 10:10:00 | 1276.70 | 2024-11-06 10:25:00 | 1268.88 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-06 10:10:00 | 1276.70 | 2024-11-06 15:20:00 | 1257.60 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-11-12 09:30:00 | 1181.75 | 2024-11-12 09:50:00 | 1173.85 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-12 09:30:00 | 1181.75 | 2024-11-12 15:20:00 | 1122.00 | TARGET_HIT | 0.50 | 5.06% |
| BUY | retest1 | 2024-12-17 09:55:00 | 1070.55 | 2024-12-17 10:00:00 | 1067.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-20 10:00:00 | 1014.65 | 2024-12-20 10:15:00 | 1017.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-27 11:00:00 | 961.15 | 2024-12-27 11:05:00 | 964.18 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-30 10:50:00 | 970.05 | 2024-12-30 10:55:00 | 965.19 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-30 10:50:00 | 970.05 | 2024-12-30 15:20:00 | 942.35 | TARGET_HIT | 0.50 | 2.86% |
| BUY | retest1 | 2025-01-02 09:30:00 | 985.50 | 2025-01-02 09:50:00 | 989.61 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-02 09:30:00 | 985.50 | 2025-01-02 09:55:00 | 985.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:35:00 | 941.30 | 2025-01-10 10:00:00 | 935.21 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-01-10 09:35:00 | 941.30 | 2025-01-10 10:10:00 | 941.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:55:00 | 949.00 | 2025-01-24 13:00:00 | 943.84 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-24 09:55:00 | 949.00 | 2025-01-24 15:20:00 | 938.15 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-01-31 10:45:00 | 967.60 | 2025-01-31 11:00:00 | 972.36 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-01-31 10:45:00 | 967.60 | 2025-01-31 15:20:00 | 982.85 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2025-02-01 10:00:00 | 995.65 | 2025-02-01 11:30:00 | 1001.52 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-02-01 10:00:00 | 995.65 | 2025-02-01 11:35:00 | 995.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 09:50:00 | 900.55 | 2025-02-11 10:05:00 | 903.73 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-13 09:30:00 | 905.55 | 2025-02-13 09:35:00 | 901.77 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-12 10:10:00 | 898.80 | 2025-03-12 10:50:00 | 894.75 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-03-12 10:10:00 | 898.80 | 2025-03-12 10:55:00 | 898.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-18 10:55:00 | 892.75 | 2025-03-18 11:05:00 | 894.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-19 10:55:00 | 905.85 | 2025-03-19 11:05:00 | 909.87 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-19 10:55:00 | 905.85 | 2025-03-19 15:20:00 | 929.70 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2025-03-24 09:50:00 | 957.50 | 2025-03-24 11:45:00 | 951.96 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-03-26 09:40:00 | 921.35 | 2025-03-26 09:55:00 | 925.46 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-27 10:35:00 | 925.40 | 2025-03-27 11:00:00 | 922.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-28 10:05:00 | 938.55 | 2025-03-28 10:10:00 | 943.80 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-03-28 10:05:00 | 938.55 | 2025-03-28 12:35:00 | 938.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 10:50:00 | 929.60 | 2025-04-02 11:15:00 | 927.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-04-03 09:45:00 | 920.15 | 2025-04-03 09:50:00 | 923.49 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-22 10:00:00 | 921.00 | 2025-04-22 10:05:00 | 925.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-22 10:00:00 | 921.00 | 2025-04-22 11:40:00 | 923.10 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-04-23 09:35:00 | 921.60 | 2025-04-23 09:40:00 | 927.09 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-23 09:35:00 | 921.60 | 2025-04-23 10:05:00 | 921.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 09:30:00 | 926.35 | 2025-04-24 09:35:00 | 931.37 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-24 09:30:00 | 926.35 | 2025-04-24 10:00:00 | 927.50 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-05-05 09:45:00 | 890.00 | 2025-05-05 10:20:00 | 887.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-08 09:35:00 | 885.15 | 2025-05-08 09:50:00 | 882.32 | STOP_HIT | 1.00 | -0.32% |
