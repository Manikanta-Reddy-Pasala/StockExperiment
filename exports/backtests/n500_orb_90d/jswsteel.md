# JSW Steel Ltd. (JSWSTEEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1272.00
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 19
- **Target hits / Stop hits / Partials:** 3 / 19 / 10
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 4.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.11% | 1.6% |
| BUY @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.11% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.17% | 2.8% |
| SELL @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.17% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 13 | 40.6% | 3 | 19 | 10 | 0.14% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:15:00 | 1249.00 | 1242.63 | 0.00 | ORB-long ORB[1238.50,1248.50] vol=4.9x ATR=4.25 |
| Stop hit — per-position SL triggered | 2026-02-09 12:50:00 | 1244.75 | 1244.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:00:00 | 1252.40 | 1245.71 | 0.00 | ORB-long ORB[1236.70,1247.70] vol=1.7x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 1256.69 | 1247.67 | 0.00 | T1 1.5R @ 1256.69 |
| Stop hit — per-position SL triggered | 2026-02-10 11:35:00 | 1252.40 | 1251.84 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1243.00 | 1244.99 | 0.00 | ORB-short ORB[1243.40,1249.90] vol=2.8x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 1245.74 | 1244.95 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 1228.80 | 1232.64 | 0.00 | ORB-short ORB[1229.90,1247.80] vol=1.6x ATR=3.21 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 1232.01 | 1231.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1260.80 | 1255.37 | 0.00 | ORB-long ORB[1244.20,1257.00] vol=2.4x ATR=3.05 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 1257.75 | 1257.43 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1238.50 | 1241.72 | 0.00 | ORB-short ORB[1247.80,1255.50] vol=2.1x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:00:00 | 1234.48 | 1240.84 | 0.00 | T1 1.5R @ 1234.48 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 1238.50 | 1240.64 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 1240.40 | 1232.57 | 0.00 | ORB-long ORB[1223.20,1234.50] vol=2.5x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 1244.85 | 1234.30 | 0.00 | T1 1.5R @ 1244.85 |
| Stop hit — per-position SL triggered | 2026-02-20 15:05:00 | 1240.40 | 1241.60 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 1264.90 | 1269.36 | 0.00 | ORB-short ORB[1265.10,1277.80] vol=1.8x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 1267.93 | 1269.07 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 1203.80 | 1215.48 | 0.00 | ORB-short ORB[1212.00,1222.90] vol=1.9x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:00:00 | 1197.64 | 1211.88 | 0.00 | T1 1.5R @ 1197.64 |
| Target hit | 2026-03-11 15:20:00 | 1175.60 | 1196.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 1147.30 | 1155.82 | 0.00 | ORB-short ORB[1154.40,1169.20] vol=1.9x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 1140.16 | 1149.18 | 0.00 | T1 1.5R @ 1140.16 |
| Target hit | 2026-03-13 11:25:00 | 1146.80 | 1142.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1169.50 | 1166.13 | 0.00 | ORB-long ORB[1160.10,1167.20] vol=1.9x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 1165.93 | 1166.93 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 1149.50 | 1154.17 | 0.00 | ORB-short ORB[1155.20,1165.90] vol=2.8x ATR=4.26 |
| Stop hit — per-position SL triggered | 2026-03-19 10:25:00 | 1153.76 | 1154.05 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:50:00 | 1153.80 | 1149.98 | 0.00 | ORB-long ORB[1138.20,1150.30] vol=2.9x ATR=3.89 |
| Stop hit — per-position SL triggered | 2026-04-01 11:00:00 | 1149.91 | 1150.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:55:00 | 1218.00 | 1223.58 | 0.00 | ORB-short ORB[1218.70,1232.00] vol=3.3x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:20:00 | 1213.02 | 1221.62 | 0.00 | T1 1.5R @ 1213.02 |
| Stop hit — per-position SL triggered | 2026-04-15 12:55:00 | 1218.00 | 1218.48 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 1237.60 | 1233.56 | 0.00 | ORB-long ORB[1221.20,1236.00] vol=2.2x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 1242.41 | 1235.28 | 0.00 | T1 1.5R @ 1242.41 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 1237.60 | 1235.83 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1254.90 | 1249.08 | 0.00 | ORB-long ORB[1233.00,1246.90] vol=2.2x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:30:00 | 1259.30 | 1251.12 | 0.00 | T1 1.5R @ 1259.30 |
| Target hit | 2026-04-20 15:20:00 | 1271.50 | 1267.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 1270.00 | 1274.40 | 0.00 | ORB-short ORB[1270.10,1286.60] vol=1.7x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:55:00 | 1264.93 | 1272.42 | 0.00 | T1 1.5R @ 1264.93 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 1270.00 | 1270.07 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 1253.00 | 1263.59 | 0.00 | ORB-short ORB[1253.30,1268.50] vol=1.7x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-04-23 11:45:00 | 1256.36 | 1262.38 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 1250.10 | 1252.73 | 0.00 | ORB-short ORB[1254.20,1268.00] vol=4.1x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-04-24 12:10:00 | 1252.56 | 1250.96 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 1301.40 | 1294.46 | 0.00 | ORB-long ORB[1281.00,1295.00] vol=1.8x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 1306.70 | 1297.55 | 0.00 | T1 1.5R @ 1306.70 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 1301.40 | 1298.06 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 1254.00 | 1256.05 | 0.00 | ORB-short ORB[1257.20,1273.40] vol=1.8x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 1257.44 | 1255.95 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1280.10 | 1277.26 | 0.00 | ORB-long ORB[1268.60,1279.40] vol=1.6x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 1275.96 | 1277.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:15:00 | 1249.00 | 2026-02-09 12:50:00 | 1244.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-10 10:00:00 | 1252.40 | 2026-02-10 10:15:00 | 1256.69 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-10 10:00:00 | 1252.40 | 2026-02-10 11:35:00 | 1252.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 11:05:00 | 1243.00 | 2026-02-11 11:20:00 | 1245.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 09:40:00 | 1228.80 | 2026-02-13 10:25:00 | 1232.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-18 09:30:00 | 1260.80 | 2026-02-18 09:50:00 | 1257.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1238.50 | 2026-02-19 12:00:00 | 1234.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1238.50 | 2026-02-19 12:15:00 | 1238.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 1240.40 | 2026-02-20 11:15:00 | 1244.85 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-20 10:45:00 | 1240.40 | 2026-02-20 15:05:00 | 1240.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:10:00 | 1264.90 | 2026-02-27 10:35:00 | 1267.93 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-11 09:45:00 | 1203.80 | 2026-03-11 10:00:00 | 1197.64 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-11 09:45:00 | 1203.80 | 2026-03-11 15:20:00 | 1175.60 | TARGET_HIT | 0.50 | 2.34% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1147.30 | 2026-03-13 10:00:00 | 1140.16 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1147.30 | 2026-03-13 11:25:00 | 1146.80 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1169.50 | 2026-03-18 09:55:00 | 1165.93 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-19 10:20:00 | 1149.50 | 2026-03-19 10:25:00 | 1153.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-01 10:50:00 | 1153.80 | 2026-04-01 11:00:00 | 1149.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-15 10:55:00 | 1218.00 | 2026-04-15 11:20:00 | 1213.02 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-15 10:55:00 | 1218.00 | 2026-04-15 12:55:00 | 1218.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:30:00 | 1237.60 | 2026-04-16 09:40:00 | 1242.41 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-16 09:30:00 | 1237.60 | 2026-04-16 09:50:00 | 1237.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-20 11:15:00 | 1254.90 | 2026-04-20 11:30:00 | 1259.30 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-20 11:15:00 | 1254.90 | 2026-04-20 15:20:00 | 1271.50 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2026-04-22 09:40:00 | 1270.00 | 2026-04-22 09:55:00 | 1264.93 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-22 09:40:00 | 1270.00 | 2026-04-22 10:55:00 | 1270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1253.00 | 2026-04-23 11:45:00 | 1256.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1250.10 | 2026-04-24 12:10:00 | 1252.56 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-28 09:50:00 | 1301.40 | 2026-04-28 10:15:00 | 1306.70 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-28 09:50:00 | 1301.40 | 2026-04-28 10:25:00 | 1301.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:30:00 | 1254.00 | 2026-04-30 11:05:00 | 1257.44 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 09:45:00 | 1280.10 | 2026-05-07 10:05:00 | 1275.96 | STOP_HIT | 1.00 | -0.32% |
