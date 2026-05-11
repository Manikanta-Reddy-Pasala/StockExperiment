# Lloyds Metals And Energy Ltd. (LLOYDSME)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (32275 bars)
- **Last close:** 1738.70
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
| ENTRY1 | 38 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 8 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 30
- **Target hits / Stop hits / Partials:** 8 / 30 / 21
- **Avg / median % per leg:** 0.33% / 0.00%
- **Sum % (uncompounded):** 19.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 24 | 58.5% | 8 | 17 | 16 | 0.49% | 20.0% |
| BUY @ 2nd Alert (retest1) | 41 | 24 | 58.5% | 8 | 17 | 16 | 0.49% | 20.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 5 | 27.8% | 0 | 13 | 5 | -0.03% | -0.6% |
| SELL @ 2nd Alert (retest1) | 18 | 5 | 27.8% | 0 | 13 | 5 | -0.03% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 59 | 29 | 49.2% | 8 | 30 | 21 | 0.33% | 19.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:25:00 | 741.35 | 742.88 | 0.00 | ORB-short ORB[742.25,748.75] vol=2.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 744.54 | 743.64 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 750.85 | 745.97 | 0.00 | ORB-long ORB[742.55,749.00] vol=1.9x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:00:00 | 755.41 | 748.45 | 0.00 | T1 1.5R @ 755.41 |
| Target hit | 2024-08-21 15:20:00 | 756.30 | 753.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-09-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:35:00 | 757.95 | 762.40 | 0.00 | ORB-short ORB[762.05,769.75] vol=2.6x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:40:00 | 754.62 | 760.96 | 0.00 | T1 1.5R @ 754.62 |
| Stop hit — per-position SL triggered | 2024-09-10 11:05:00 | 757.95 | 760.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 750.30 | 755.68 | 0.00 | ORB-short ORB[754.15,761.35] vol=3.1x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 753.54 | 754.45 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-09-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:05:00 | 805.00 | 797.71 | 0.00 | ORB-long ORB[790.00,801.50] vol=2.2x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-09-17 10:10:00 | 800.41 | 798.64 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 803.70 | 811.64 | 0.00 | ORB-short ORB[815.55,826.50] vol=2.2x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-09-19 10:35:00 | 807.59 | 810.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 817.00 | 812.08 | 0.00 | ORB-long ORB[805.00,814.00] vol=4.7x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:35:00 | 822.21 | 815.65 | 0.00 | T1 1.5R @ 822.21 |
| Stop hit — per-position SL triggered | 2024-09-20 10:00:00 | 817.00 | 818.76 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 891.10 | 885.17 | 0.00 | ORB-long ORB[875.15,887.85] vol=2.7x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:30:00 | 896.62 | 887.31 | 0.00 | T1 1.5R @ 896.62 |
| Target hit | 2024-09-26 14:10:00 | 908.85 | 909.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:15:00 | 986.15 | 994.08 | 0.00 | ORB-short ORB[988.65,999.00] vol=2.3x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 990.36 | 992.62 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 09:40:00 | 953.65 | 945.61 | 0.00 | ORB-long ORB[936.00,945.90] vol=5.3x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:45:00 | 961.10 | 950.73 | 0.00 | T1 1.5R @ 961.10 |
| Stop hit — per-position SL triggered | 2024-10-29 10:00:00 | 953.65 | 952.97 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:35:00 | 966.85 | 960.84 | 0.00 | ORB-long ORB[950.10,962.80] vol=2.3x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 972.88 | 964.18 | 0.00 | T1 1.5R @ 972.88 |
| Target hit | 2024-10-30 11:45:00 | 973.50 | 975.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:30:00 | 1026.95 | 1020.00 | 0.00 | ORB-long ORB[1010.35,1023.55] vol=2.3x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-11-07 09:50:00 | 1021.77 | 1021.73 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 997.00 | 989.77 | 0.00 | ORB-long ORB[981.05,995.75] vol=1.5x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 10:35:00 | 1003.45 | 992.18 | 0.00 | T1 1.5R @ 1003.45 |
| Target hit | 2024-11-08 15:00:00 | 1004.60 | 1009.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-11-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:00:00 | 956.80 | 947.82 | 0.00 | ORB-long ORB[939.15,949.55] vol=2.7x ATR=6.05 |
| Stop hit — per-position SL triggered | 2024-11-14 10:30:00 | 950.75 | 948.92 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:45:00 | 967.00 | 956.31 | 0.00 | ORB-long ORB[947.65,960.30] vol=1.7x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-11-27 09:55:00 | 962.51 | 957.15 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 1067.10 | 1060.97 | 0.00 | ORB-long ORB[1052.00,1062.70] vol=3.2x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:40:00 | 1073.38 | 1065.54 | 0.00 | T1 1.5R @ 1073.38 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 1067.10 | 1067.15 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:55:00 | 1058.65 | 1064.06 | 0.00 | ORB-short ORB[1060.30,1074.90] vol=1.5x ATR=3.18 |
| Stop hit — per-position SL triggered | 2024-12-05 10:20:00 | 1061.83 | 1063.62 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:05:00 | 1145.70 | 1130.92 | 0.00 | ORB-long ORB[1112.05,1128.95] vol=1.6x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:15:00 | 1154.49 | 1138.66 | 0.00 | T1 1.5R @ 1154.49 |
| Stop hit — per-position SL triggered | 2024-12-16 10:35:00 | 1145.70 | 1142.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:50:00 | 1166.95 | 1158.63 | 0.00 | ORB-long ORB[1141.40,1153.60] vol=5.2x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:00:00 | 1174.55 | 1160.46 | 0.00 | T1 1.5R @ 1174.55 |
| Stop hit — per-position SL triggered | 2024-12-20 12:45:00 | 1166.95 | 1167.17 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 1173.50 | 1166.90 | 0.00 | ORB-long ORB[1159.90,1173.25] vol=3.7x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-12-24 10:55:00 | 1168.91 | 1170.83 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1178.90 | 1170.52 | 0.00 | ORB-long ORB[1161.80,1173.45] vol=2.0x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-12-27 09:55:00 | 1173.57 | 1176.99 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:40:00 | 1242.80 | 1238.36 | 0.00 | ORB-long ORB[1233.95,1242.00] vol=2.1x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-01-01 09:45:00 | 1239.40 | 1238.37 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:55:00 | 1284.95 | 1278.64 | 0.00 | ORB-long ORB[1263.50,1278.00] vol=2.3x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:00:00 | 1290.60 | 1280.32 | 0.00 | T1 1.5R @ 1290.60 |
| Target hit | 2025-01-02 12:40:00 | 1319.25 | 1319.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2025-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 11:05:00 | 1428.40 | 1420.13 | 0.00 | ORB-long ORB[1412.20,1428.05] vol=3.4x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:20:00 | 1435.75 | 1423.90 | 0.00 | T1 1.5R @ 1435.75 |
| Stop hit — per-position SL triggered | 2025-01-09 11:50:00 | 1428.40 | 1426.80 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-01-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:45:00 | 1463.65 | 1441.34 | 0.00 | ORB-long ORB[1418.15,1439.85] vol=3.0x ATR=10.09 |
| Stop hit — per-position SL triggered | 2025-01-10 11:40:00 | 1453.56 | 1451.86 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-01-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:45:00 | 1422.10 | 1433.52 | 0.00 | ORB-short ORB[1425.00,1441.85] vol=1.5x ATR=6.69 |
| Stop hit — per-position SL triggered | 2025-01-15 09:50:00 | 1428.79 | 1433.05 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:30:00 | 1458.00 | 1452.75 | 0.00 | ORB-long ORB[1438.80,1455.00] vol=3.1x ATR=4.77 |
| Stop hit — per-position SL triggered | 2025-01-16 09:50:00 | 1453.23 | 1453.77 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:50:00 | 1401.20 | 1411.77 | 0.00 | ORB-short ORB[1410.00,1428.00] vol=3.7x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:55:00 | 1392.93 | 1410.14 | 0.00 | T1 1.5R @ 1392.93 |
| Stop hit — per-position SL triggered | 2025-01-20 12:45:00 | 1401.20 | 1403.09 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-01-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 11:00:00 | 1248.20 | 1259.06 | 0.00 | ORB-short ORB[1261.00,1272.45] vol=1.9x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-01-30 11:05:00 | 1254.20 | 1258.93 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 1202.45 | 1210.27 | 0.00 | ORB-short ORB[1206.70,1221.95] vol=1.9x ATR=6.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:45:00 | 1192.18 | 1207.45 | 0.00 | T1 1.5R @ 1192.18 |
| Stop hit — per-position SL triggered | 2025-02-10 10:30:00 | 1202.45 | 1201.84 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-02-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:55:00 | 1178.80 | 1191.24 | 0.00 | ORB-short ORB[1185.95,1201.50] vol=1.8x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 11:10:00 | 1171.03 | 1190.29 | 0.00 | T1 1.5R @ 1171.03 |
| Stop hit — per-position SL triggered | 2025-02-21 11:30:00 | 1178.80 | 1189.73 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-11 09:35:00 | 1120.45 | 1137.97 | 0.00 | ORB-short ORB[1136.80,1152.00] vol=1.9x ATR=6.91 |
| Stop hit — per-position SL triggered | 2025-03-11 09:45:00 | 1127.36 | 1133.65 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 1140.20 | 1131.52 | 0.00 | ORB-long ORB[1121.60,1135.00] vol=1.5x ATR=5.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:50:00 | 1148.05 | 1134.11 | 0.00 | T1 1.5R @ 1148.05 |
| Target hit | 2025-03-18 12:25:00 | 1165.15 | 1165.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1304.60 | 1290.73 | 0.00 | ORB-long ORB[1278.10,1295.00] vol=2.2x ATR=7.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:45:00 | 1316.23 | 1296.96 | 0.00 | T1 1.5R @ 1316.23 |
| Stop hit — per-position SL triggered | 2025-04-02 09:50:00 | 1304.60 | 1299.23 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:00:00 | 1245.40 | 1238.68 | 0.00 | ORB-long ORB[1227.70,1244.30] vol=2.3x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:45:00 | 1253.63 | 1241.49 | 0.00 | T1 1.5R @ 1253.63 |
| Target hit | 2025-04-16 15:20:00 | 1293.00 | 1265.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-05-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:50:00 | 1222.80 | 1211.02 | 0.00 | ORB-long ORB[1201.30,1212.70] vol=1.6x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 10:00:00 | 1228.69 | 1213.56 | 0.00 | T1 1.5R @ 1228.69 |
| Stop hit — per-position SL triggered | 2025-05-02 10:05:00 | 1222.80 | 1214.87 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 1209.00 | 1201.93 | 0.00 | ORB-long ORB[1193.30,1207.20] vol=1.9x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:00:00 | 1217.20 | 1207.37 | 0.00 | T1 1.5R @ 1217.20 |
| Target hit | 2025-05-05 11:55:00 | 1222.80 | 1223.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2025-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:40:00 | 1219.80 | 1225.48 | 0.00 | ORB-short ORB[1222.10,1232.70] vol=1.7x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:50:00 | 1213.48 | 1225.00 | 0.00 | T1 1.5R @ 1213.48 |
| Stop hit — per-position SL triggered | 2025-05-06 11:05:00 | 1219.80 | 1224.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-09 10:25:00 | 741.35 | 2024-08-09 10:50:00 | 744.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-21 10:10:00 | 750.85 | 2024-08-21 11:00:00 | 755.41 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-21 10:10:00 | 750.85 | 2024-08-21 15:20:00 | 756.30 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-09-10 10:35:00 | 757.95 | 2024-09-10 10:40:00 | 754.62 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-10 10:35:00 | 757.95 | 2024-09-10 11:05:00 | 757.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-13 09:35:00 | 750.30 | 2024-09-13 09:50:00 | 753.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-17 10:05:00 | 805.00 | 2024-09-17 10:10:00 | 800.41 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-09-19 10:15:00 | 803.70 | 2024-09-19 10:35:00 | 807.59 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-09-20 09:30:00 | 817.00 | 2024-09-20 09:35:00 | 822.21 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-09-20 09:30:00 | 817.00 | 2024-09-20 10:00:00 | 817.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:55:00 | 891.10 | 2024-09-26 11:30:00 | 896.62 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-09-26 10:55:00 | 891.10 | 2024-09-26 14:10:00 | 908.85 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2024-10-11 10:15:00 | 986.15 | 2024-10-11 10:55:00 | 990.36 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-29 09:40:00 | 953.65 | 2024-10-29 09:45:00 | 961.10 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-10-29 09:40:00 | 953.65 | 2024-10-29 10:00:00 | 953.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 09:35:00 | 966.85 | 2024-10-30 09:50:00 | 972.88 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-30 09:35:00 | 966.85 | 2024-10-30 11:45:00 | 973.50 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2024-11-07 09:30:00 | 1026.95 | 2024-11-07 09:50:00 | 1021.77 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-08 09:45:00 | 997.00 | 2024-11-08 10:35:00 | 1003.45 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-08 09:45:00 | 997.00 | 2024-11-08 15:00:00 | 1004.60 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-11-14 10:00:00 | 956.80 | 2024-11-14 10:30:00 | 950.75 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-11-27 09:45:00 | 967.00 | 2024-11-27 09:55:00 | 962.51 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-04 09:35:00 | 1067.10 | 2024-12-04 09:40:00 | 1073.38 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-12-04 09:35:00 | 1067.10 | 2024-12-04 09:55:00 | 1067.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 09:55:00 | 1058.65 | 2024-12-05 10:20:00 | 1061.83 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-16 10:05:00 | 1145.70 | 2024-12-16 10:15:00 | 1154.49 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-12-16 10:05:00 | 1145.70 | 2024-12-16 10:35:00 | 1145.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 10:50:00 | 1166.95 | 2024-12-20 11:00:00 | 1174.55 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-12-20 10:50:00 | 1166.95 | 2024-12-20 12:45:00 | 1166.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:35:00 | 1173.50 | 2024-12-24 10:55:00 | 1168.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-27 09:30:00 | 1178.90 | 2024-12-27 09:55:00 | 1173.57 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-01 09:40:00 | 1242.80 | 2025-01-01 09:45:00 | 1239.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1284.95 | 2025-01-02 10:00:00 | 1290.60 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1284.95 | 2025-01-02 12:40:00 | 1319.25 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2025-01-09 11:05:00 | 1428.40 | 2025-01-09 11:20:00 | 1435.75 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-09 11:05:00 | 1428.40 | 2025-01-09 11:50:00 | 1428.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-10 10:45:00 | 1463.65 | 2025-01-10 11:40:00 | 1453.56 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-01-15 09:45:00 | 1422.10 | 2025-01-15 09:50:00 | 1428.79 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-16 09:30:00 | 1458.00 | 2025-01-16 09:50:00 | 1453.23 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-20 09:50:00 | 1401.20 | 2025-01-20 09:55:00 | 1392.93 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-20 09:50:00 | 1401.20 | 2025-01-20 12:45:00 | 1401.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 11:00:00 | 1248.20 | 2025-01-30 11:05:00 | 1254.20 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-10 09:30:00 | 1202.45 | 2025-02-10 09:45:00 | 1192.18 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2025-02-10 09:30:00 | 1202.45 | 2025-02-10 10:30:00 | 1202.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-21 10:55:00 | 1178.80 | 2025-02-21 11:10:00 | 1171.03 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-02-21 10:55:00 | 1178.80 | 2025-02-21 11:30:00 | 1178.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-11 09:35:00 | 1120.45 | 2025-03-11 09:45:00 | 1127.36 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-03-18 09:35:00 | 1140.20 | 2025-03-18 09:50:00 | 1148.05 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-03-18 09:35:00 | 1140.20 | 2025-03-18 12:25:00 | 1165.15 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2025-04-02 09:30:00 | 1304.60 | 2025-04-02 09:45:00 | 1316.23 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-04-02 09:30:00 | 1304.60 | 2025-04-02 09:50:00 | 1304.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 10:00:00 | 1245.40 | 2025-04-16 10:45:00 | 1253.63 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-04-16 10:00:00 | 1245.40 | 2025-04-16 15:20:00 | 1293.00 | TARGET_HIT | 0.50 | 3.82% |
| BUY | retest1 | 2025-05-02 09:50:00 | 1222.80 | 2025-05-02 10:00:00 | 1228.69 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-02 09:50:00 | 1222.80 | 2025-05-02 10:05:00 | 1222.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:30:00 | 1209.00 | 2025-05-05 10:00:00 | 1217.20 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-05-05 09:30:00 | 1209.00 | 2025-05-05 11:55:00 | 1222.80 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-05-06 10:40:00 | 1219.80 | 2025-05-06 10:50:00 | 1213.48 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-05-06 10:40:00 | 1219.80 | 2025-05-06 11:05:00 | 1219.80 | STOP_HIT | 0.50 | 0.00% |
