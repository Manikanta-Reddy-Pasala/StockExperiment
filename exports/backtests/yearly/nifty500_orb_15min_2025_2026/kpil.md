# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1277.20
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 6 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 50
- **Target hits / Stop hits / Partials:** 6 / 50 / 20
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 5.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 12 | 32.4% | 4 | 25 | 8 | 0.05% | 2.0% |
| BUY @ 2nd Alert (retest1) | 37 | 12 | 32.4% | 4 | 25 | 8 | 0.05% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 14 | 35.9% | 2 | 25 | 12 | 0.08% | 3.3% |
| SELL @ 2nd Alert (retest1) | 39 | 14 | 35.9% | 2 | 25 | 12 | 0.08% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 26 | 34.2% | 6 | 50 | 20 | 0.07% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 1166.00 | 1159.51 | 0.00 | ORB-long ORB[1151.50,1163.20] vol=2.5x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-06-03 09:45:00 | 1162.19 | 1160.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-06-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:40:00 | 1157.30 | 1164.15 | 0.00 | ORB-short ORB[1165.40,1174.20] vol=2.9x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:15:00 | 1151.11 | 1162.31 | 0.00 | T1 1.5R @ 1151.11 |
| Stop hit — per-position SL triggered | 2025-06-05 11:20:00 | 1157.30 | 1161.19 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:15:00 | 1195.60 | 1189.79 | 0.00 | ORB-long ORB[1177.80,1194.30] vol=2.2x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-06-12 10:20:00 | 1190.86 | 1189.97 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:35:00 | 1212.00 | 1205.28 | 0.00 | ORB-long ORB[1192.60,1205.50] vol=4.7x ATR=4.05 |
| Stop hit — per-position SL triggered | 2025-06-19 09:40:00 | 1207.95 | 1205.96 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:35:00 | 1212.00 | 1207.78 | 0.00 | ORB-long ORB[1182.20,1198.70] vol=5.6x ATR=5.00 |
| Stop hit — per-position SL triggered | 2025-06-25 09:40:00 | 1207.00 | 1208.17 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-07-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:05:00 | 1203.00 | 1196.40 | 0.00 | ORB-long ORB[1186.60,1199.90] vol=2.4x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 1210.19 | 1199.48 | 0.00 | T1 1.5R @ 1210.19 |
| Stop hit — per-position SL triggered | 2025-07-03 11:00:00 | 1203.00 | 1205.55 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:20:00 | 1201.00 | 1198.91 | 0.00 | ORB-long ORB[1188.90,1197.00] vol=1.7x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:25:00 | 1205.39 | 1199.41 | 0.00 | T1 1.5R @ 1205.39 |
| Target hit | 2025-07-09 13:30:00 | 1204.10 | 1204.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:50:00 | 1198.60 | 1202.55 | 0.00 | ORB-short ORB[1199.50,1210.00] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-07-10 10:35:00 | 1202.09 | 1201.64 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 1207.20 | 1200.85 | 0.00 | ORB-long ORB[1191.40,1205.60] vol=2.1x ATR=3.95 |
| Stop hit — per-position SL triggered | 2025-07-11 09:45:00 | 1203.25 | 1201.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 11:15:00 | 1194.10 | 1198.25 | 0.00 | ORB-short ORB[1196.30,1207.50] vol=3.9x ATR=3.08 |
| Stop hit — per-position SL triggered | 2025-07-15 11:20:00 | 1197.18 | 1198.05 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:00:00 | 1198.50 | 1205.75 | 0.00 | ORB-short ORB[1209.90,1219.80] vol=2.7x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:40:00 | 1194.33 | 1203.13 | 0.00 | T1 1.5R @ 1194.33 |
| Stop hit — per-position SL triggered | 2025-07-16 11:55:00 | 1198.50 | 1202.87 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:10:00 | 1195.30 | 1192.64 | 0.00 | ORB-long ORB[1182.70,1192.80] vol=2.0x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-07-22 11:10:00 | 1191.39 | 1194.11 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:10:00 | 1114.20 | 1118.71 | 0.00 | ORB-short ORB[1118.60,1133.00] vol=2.0x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-07-30 11:50:00 | 1117.59 | 1118.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:50:00 | 1111.70 | 1122.80 | 0.00 | ORB-short ORB[1124.30,1136.50] vol=1.6x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:30:00 | 1105.97 | 1119.29 | 0.00 | T1 1.5R @ 1105.97 |
| Stop hit — per-position SL triggered | 2025-08-06 11:35:00 | 1111.70 | 1119.11 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1281.50 | 1270.59 | 0.00 | ORB-long ORB[1260.80,1279.70] vol=1.9x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-08-19 11:45:00 | 1277.37 | 1273.51 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:45:00 | 1277.70 | 1270.92 | 0.00 | ORB-long ORB[1261.00,1275.00] vol=3.3x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-08-20 11:25:00 | 1274.52 | 1272.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:25:00 | 1299.20 | 1289.87 | 0.00 | ORB-long ORB[1280.70,1293.50] vol=3.2x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:30:00 | 1304.91 | 1293.07 | 0.00 | T1 1.5R @ 1304.91 |
| Stop hit — per-position SL triggered | 2025-08-21 10:50:00 | 1299.20 | 1294.70 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:55:00 | 1299.70 | 1304.68 | 0.00 | ORB-short ORB[1302.40,1319.00] vol=6.6x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:05:00 | 1295.33 | 1303.89 | 0.00 | T1 1.5R @ 1295.33 |
| Stop hit — per-position SL triggered | 2025-08-25 12:20:00 | 1299.70 | 1300.34 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:50:00 | 1238.00 | 1250.26 | 0.00 | ORB-short ORB[1250.30,1265.70] vol=2.9x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:45:00 | 1230.96 | 1245.05 | 0.00 | T1 1.5R @ 1230.96 |
| Stop hit — per-position SL triggered | 2025-09-01 12:00:00 | 1238.00 | 1242.45 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 10:05:00 | 1234.40 | 1250.73 | 0.00 | ORB-short ORB[1235.80,1252.00] vol=1.8x ATR=7.54 |
| Stop hit — per-position SL triggered | 2025-09-02 10:10:00 | 1241.94 | 1248.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:40:00 | 1256.70 | 1266.17 | 0.00 | ORB-short ORB[1265.20,1282.00] vol=2.7x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-09-04 10:50:00 | 1260.48 | 1265.91 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 1287.00 | 1279.12 | 0.00 | ORB-long ORB[1267.90,1283.20] vol=2.7x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 1282.09 | 1280.00 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:55:00 | 1238.50 | 1245.91 | 0.00 | ORB-short ORB[1242.70,1254.20] vol=1.7x ATR=4.86 |
| Stop hit — per-position SL triggered | 2025-09-23 10:00:00 | 1243.36 | 1245.52 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1256.00 | 1250.10 | 0.00 | ORB-long ORB[1241.20,1253.50] vol=1.9x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:40:00 | 1262.05 | 1252.39 | 0.00 | T1 1.5R @ 1262.05 |
| Stop hit — per-position SL triggered | 2025-09-24 13:10:00 | 1256.00 | 1257.10 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1244.20 | 1247.86 | 0.00 | ORB-short ORB[1245.40,1260.50] vol=5.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-10-01 11:25:00 | 1247.86 | 1247.54 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:45:00 | 1242.20 | 1249.83 | 0.00 | ORB-short ORB[1245.50,1260.00] vol=2.1x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-10-06 11:40:00 | 1245.98 | 1247.87 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 11:15:00 | 1259.90 | 1254.55 | 0.00 | ORB-long ORB[1248.30,1258.70] vol=2.3x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-10-07 11:25:00 | 1256.74 | 1254.70 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 1253.30 | 1260.33 | 0.00 | ORB-short ORB[1259.40,1271.00] vol=1.9x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-10-08 11:20:00 | 1256.59 | 1259.75 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:45:00 | 1259.60 | 1256.32 | 0.00 | ORB-long ORB[1245.40,1257.40] vol=2.0x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-10-10 10:25:00 | 1255.67 | 1259.58 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 1283.00 | 1280.01 | 0.00 | ORB-long ORB[1270.00,1282.20] vol=9.6x ATR=5.95 |
| Stop hit — per-position SL triggered | 2025-10-23 09:45:00 | 1277.05 | 1279.80 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:35:00 | 1254.90 | 1258.92 | 0.00 | ORB-short ORB[1256.00,1269.90] vol=4.1x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:30:00 | 1249.08 | 1257.60 | 0.00 | T1 1.5R @ 1249.08 |
| Stop hit — per-position SL triggered | 2025-10-24 12:25:00 | 1254.90 | 1256.31 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:05:00 | 1258.00 | 1261.84 | 0.00 | ORB-short ORB[1264.20,1272.00] vol=1.5x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-10-29 12:45:00 | 1260.51 | 1260.44 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:45:00 | 1278.30 | 1272.56 | 0.00 | ORB-long ORB[1262.40,1272.70] vol=2.5x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-10-31 09:50:00 | 1274.78 | 1273.49 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:50:00 | 1261.80 | 1267.45 | 0.00 | ORB-short ORB[1266.40,1275.50] vol=2.0x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:15:00 | 1255.80 | 1262.26 | 0.00 | T1 1.5R @ 1255.80 |
| Target hit | 2025-11-13 15:20:00 | 1238.70 | 1250.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-11-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:05:00 | 1247.50 | 1243.44 | 0.00 | ORB-long ORB[1237.20,1245.00] vol=1.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-11-14 10:35:00 | 1244.40 | 1243.87 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1231.50 | 1238.90 | 0.00 | ORB-short ORB[1239.50,1256.10] vol=3.1x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:50:00 | 1226.20 | 1236.51 | 0.00 | T1 1.5R @ 1226.20 |
| Stop hit — per-position SL triggered | 2025-11-18 12:00:00 | 1231.50 | 1230.76 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:40:00 | 1208.50 | 1216.17 | 0.00 | ORB-short ORB[1216.00,1231.80] vol=1.6x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:45:00 | 1204.37 | 1212.77 | 0.00 | T1 1.5R @ 1204.37 |
| Target hit | 2025-11-24 15:20:00 | 1199.90 | 1203.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:55:00 | 1194.50 | 1200.73 | 0.00 | ORB-short ORB[1198.10,1210.00] vol=3.6x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-11-28 11:00:00 | 1197.11 | 1200.70 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:10:00 | 1195.60 | 1200.14 | 0.00 | ORB-short ORB[1196.80,1205.40] vol=4.6x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:40:00 | 1190.97 | 1197.41 | 0.00 | T1 1.5R @ 1190.97 |
| Stop hit — per-position SL triggered | 2025-12-01 14:10:00 | 1195.60 | 1195.50 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 11:10:00 | 1150.60 | 1154.77 | 0.00 | ORB-short ORB[1153.80,1168.80] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 1153.82 | 1154.79 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:50:00 | 1154.30 | 1149.14 | 0.00 | ORB-long ORB[1135.80,1149.30] vol=3.8x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:55:00 | 1161.88 | 1150.54 | 0.00 | T1 1.5R @ 1161.88 |
| Target hit | 2025-12-09 15:20:00 | 1171.40 | 1162.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-12-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:50:00 | 1148.00 | 1148.76 | 0.00 | ORB-short ORB[1151.00,1160.90] vol=2.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 1150.85 | 1149.06 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 1180.30 | 1185.04 | 0.00 | ORB-short ORB[1181.50,1194.00] vol=1.7x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:35:00 | 1174.89 | 1183.63 | 0.00 | T1 1.5R @ 1174.89 |
| Stop hit — per-position SL triggered | 2025-12-30 11:40:00 | 1180.30 | 1183.48 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 11:05:00 | 1205.10 | 1207.94 | 0.00 | ORB-short ORB[1205.20,1214.10] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2026-01-02 11:55:00 | 1207.75 | 1207.41 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:50:00 | 1148.50 | 1143.90 | 0.00 | ORB-long ORB[1133.00,1145.60] vol=2.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-01-16 10:55:00 | 1145.33 | 1144.11 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-01-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:25:00 | 1093.80 | 1103.35 | 0.00 | ORB-short ORB[1106.10,1121.50] vol=1.9x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-01-29 12:35:00 | 1098.32 | 1099.15 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:45:00 | 1105.90 | 1098.23 | 0.00 | ORB-long ORB[1091.00,1103.40] vol=2.7x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-02-06 10:50:00 | 1101.44 | 1101.50 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 1128.20 | 1117.23 | 0.00 | ORB-long ORB[1091.00,1102.00] vol=5.2x ATR=4.60 |
| Stop hit — per-position SL triggered | 2026-02-09 13:10:00 | 1123.60 | 1122.59 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1116.00 | 1109.39 | 0.00 | ORB-long ORB[1092.70,1103.50] vol=4.3x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1112.55 | 1110.65 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1203.80 | 1193.52 | 0.00 | ORB-long ORB[1186.50,1196.90] vol=1.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 1210.68 | 1201.80 | 0.00 | T1 1.5R @ 1210.68 |
| Target hit | 2026-02-27 15:20:00 | 1232.30 | 1229.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1116.40 | 1108.26 | 0.00 | ORB-long ORB[1098.30,1108.00] vol=3.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-10 11:50:00 | 1113.15 | 1108.84 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 1173.20 | 1164.51 | 0.00 | ORB-long ORB[1150.90,1157.30] vol=4.0x ATR=5.24 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1167.96 | 1164.74 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 1200.80 | 1195.73 | 0.00 | ORB-long ORB[1185.70,1199.00] vol=2.9x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:40:00 | 1207.86 | 1199.82 | 0.00 | T1 1.5R @ 1207.86 |
| Target hit | 2026-04-15 15:20:00 | 1210.50 | 1210.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 1250.00 | 1244.99 | 0.00 | ORB-long ORB[1238.40,1249.90] vol=2.0x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:40:00 | 1255.91 | 1247.42 | 0.00 | T1 1.5R @ 1255.91 |
| Stop hit — per-position SL triggered | 2026-04-21 10:45:00 | 1250.00 | 1247.42 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 1247.20 | 1257.18 | 0.00 | ORB-short ORB[1257.90,1272.90] vol=1.6x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 1241.47 | 1254.84 | 0.00 | T1 1.5R @ 1241.47 |
| Stop hit — per-position SL triggered | 2026-04-24 12:05:00 | 1247.20 | 1254.61 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 1260.00 | 1255.01 | 0.00 | ORB-long ORB[1243.70,1258.90] vol=1.8x ATR=4.69 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 1255.31 | 1255.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-03 09:35:00 | 1166.00 | 2025-06-03 09:45:00 | 1162.19 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-05 10:40:00 | 1157.30 | 2025-06-05 11:15:00 | 1151.11 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-06-05 10:40:00 | 1157.30 | 2025-06-05 11:20:00 | 1157.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-12 10:15:00 | 1195.60 | 2025-06-12 10:20:00 | 1190.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-19 09:35:00 | 1212.00 | 2025-06-19 09:40:00 | 1207.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-25 09:35:00 | 1212.00 | 2025-06-25 09:40:00 | 1207.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-07-03 10:05:00 | 1203.00 | 2025-07-03 10:15:00 | 1210.19 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-07-03 10:05:00 | 1203.00 | 2025-07-03 11:00:00 | 1203.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 10:20:00 | 1201.00 | 2025-07-09 10:25:00 | 1205.39 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-09 10:20:00 | 1201.00 | 2025-07-09 13:30:00 | 1204.10 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-10 09:50:00 | 1198.60 | 2025-07-10 10:35:00 | 1202.09 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1207.20 | 2025-07-11 09:45:00 | 1203.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-15 11:15:00 | 1194.10 | 2025-07-15 11:20:00 | 1197.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-16 11:00:00 | 1198.50 | 2025-07-16 11:40:00 | 1194.33 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-16 11:00:00 | 1198.50 | 2025-07-16 11:55:00 | 1198.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 10:10:00 | 1195.30 | 2025-07-22 11:10:00 | 1191.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-30 11:10:00 | 1114.20 | 2025-07-30 11:50:00 | 1117.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-06 10:50:00 | 1111.70 | 2025-08-06 11:30:00 | 1105.97 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-08-06 10:50:00 | 1111.70 | 2025-08-06 11:35:00 | 1111.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 11:15:00 | 1281.50 | 2025-08-19 11:45:00 | 1277.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-20 10:45:00 | 1277.70 | 2025-08-20 11:25:00 | 1274.52 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-21 10:25:00 | 1299.20 | 2025-08-21 10:30:00 | 1304.91 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-21 10:25:00 | 1299.20 | 2025-08-21 10:50:00 | 1299.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-25 10:55:00 | 1299.70 | 2025-08-25 11:05:00 | 1295.33 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-25 10:55:00 | 1299.70 | 2025-08-25 12:20:00 | 1299.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-01 10:50:00 | 1238.00 | 2025-09-01 11:45:00 | 1230.96 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-09-01 10:50:00 | 1238.00 | 2025-09-01 12:00:00 | 1238.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-02 10:05:00 | 1234.40 | 2025-09-02 10:10:00 | 1241.94 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-09-04 10:40:00 | 1256.70 | 2025-09-04 10:50:00 | 1260.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-10 09:40:00 | 1287.00 | 2025-09-10 09:45:00 | 1282.09 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-23 09:55:00 | 1238.50 | 2025-09-23 10:00:00 | 1243.36 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-24 11:15:00 | 1256.00 | 2025-09-24 11:40:00 | 1262.05 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-24 11:15:00 | 1256.00 | 2025-09-24 13:10:00 | 1256.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-01 11:00:00 | 1244.20 | 2025-10-01 11:25:00 | 1247.86 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-06 10:45:00 | 1242.20 | 2025-10-06 11:40:00 | 1245.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-07 11:15:00 | 1259.90 | 2025-10-07 11:25:00 | 1256.74 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-08 11:05:00 | 1253.30 | 2025-10-08 11:20:00 | 1256.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-10 09:45:00 | 1259.60 | 2025-10-10 10:25:00 | 1255.67 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-23 09:40:00 | 1283.00 | 2025-10-23 09:45:00 | 1277.05 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-10-24 10:35:00 | 1254.90 | 2025-10-24 11:30:00 | 1249.08 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-24 10:35:00 | 1254.90 | 2025-10-24 12:25:00 | 1254.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 11:05:00 | 1258.00 | 2025-10-29 12:45:00 | 1260.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-31 09:45:00 | 1278.30 | 2025-10-31 09:50:00 | 1274.78 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-13 09:50:00 | 1261.80 | 2025-11-13 10:15:00 | 1255.80 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-11-13 09:50:00 | 1261.80 | 2025-11-13 15:20:00 | 1238.70 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2025-11-14 10:05:00 | 1247.50 | 2025-11-14 10:35:00 | 1244.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-18 09:45:00 | 1231.50 | 2025-11-18 09:50:00 | 1226.20 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-18 09:45:00 | 1231.50 | 2025-11-18 12:00:00 | 1231.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-24 10:40:00 | 1208.50 | 2025-11-24 10:45:00 | 1204.37 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-24 10:40:00 | 1208.50 | 2025-11-24 15:20:00 | 1199.90 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-11-28 10:55:00 | 1194.50 | 2025-11-28 11:00:00 | 1197.11 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-01 11:10:00 | 1195.60 | 2025-12-01 11:40:00 | 1190.97 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-01 11:10:00 | 1195.60 | 2025-12-01 14:10:00 | 1195.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 11:10:00 | 1150.60 | 2025-12-05 11:15:00 | 1153.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-09 10:50:00 | 1154.30 | 2025-12-09 11:55:00 | 1161.88 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-12-09 10:50:00 | 1154.30 | 2025-12-09 15:20:00 | 1171.40 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2025-12-18 10:50:00 | 1148.00 | 2025-12-18 11:15:00 | 1150.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-30 10:55:00 | 1180.30 | 2025-12-30 11:35:00 | 1174.89 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-30 10:55:00 | 1180.30 | 2025-12-30 11:40:00 | 1180.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-02 11:05:00 | 1205.10 | 2026-01-02 11:55:00 | 1207.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-16 10:50:00 | 1148.50 | 2026-01-16 10:55:00 | 1145.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-29 10:25:00 | 1093.80 | 2026-01-29 12:35:00 | 1098.32 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-06 09:45:00 | 1105.90 | 2026-02-06 10:50:00 | 1101.44 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-09 10:45:00 | 1128.20 | 2026-02-09 13:10:00 | 1123.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-17 11:00:00 | 1116.00 | 2026-02-17 11:30:00 | 1112.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1203.80 | 2026-02-27 10:50:00 | 1210.68 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-27 09:30:00 | 1203.80 | 2026-02-27 15:20:00 | 1232.30 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2026-03-10 11:15:00 | 1116.40 | 2026-03-10 11:50:00 | 1113.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-10 10:00:00 | 1173.20 | 2026-04-10 10:05:00 | 1167.96 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:30:00 | 1200.80 | 2026-04-15 09:40:00 | 1207.86 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-15 09:30:00 | 1200.80 | 2026-04-15 15:20:00 | 1210.50 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-04-21 10:35:00 | 1250.00 | 2026-04-21 10:40:00 | 1255.91 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-21 10:35:00 | 1250.00 | 2026-04-21 10:45:00 | 1250.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:00:00 | 1247.20 | 2026-04-24 11:50:00 | 1241.47 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 11:00:00 | 1247.20 | 2026-04-24 12:05:00 | 1247.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:40:00 | 1260.00 | 2026-04-27 11:30:00 | 1255.31 | STOP_HIT | 1.00 | -0.37% |
