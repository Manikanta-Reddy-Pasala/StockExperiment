# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1493.20
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 7
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 3.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.23% | 4.0% |
| BUY @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 2 | 9 | 6 | 0.23% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 10 | 47.6% | 3 | 11 | 7 | 0.19% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 1283.60 | 1287.57 | 0.00 | ORB-short ORB[1287.00,1303.00] vol=1.5x ATR=7.39 |
| Stop hit — per-position SL triggered | 2026-02-09 10:35:00 | 1290.99 | 1287.54 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1276.60 | 1281.18 | 0.00 | ORB-short ORB[1277.00,1295.00] vol=2.1x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 1270.79 | 1280.54 | 0.00 | T1 1.5R @ 1270.79 |
| Target hit | 2026-02-11 15:20:00 | 1270.80 | 1274.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 1267.20 | 1261.28 | 0.00 | ORB-long ORB[1247.90,1263.70] vol=2.1x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:25:00 | 1272.14 | 1262.77 | 0.00 | T1 1.5R @ 1272.14 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 1267.20 | 1262.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 1262.80 | 1255.78 | 0.00 | ORB-long ORB[1245.80,1259.00] vol=2.3x ATR=4.14 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 1258.66 | 1256.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 1286.60 | 1277.53 | 0.00 | ORB-long ORB[1267.00,1280.20] vol=2.8x ATR=3.62 |
| Stop hit — per-position SL triggered | 2026-02-20 13:05:00 | 1282.98 | 1283.45 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:35:00 | 1208.10 | 1200.93 | 0.00 | ORB-long ORB[1194.00,1204.80] vol=2.4x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:45:00 | 1216.17 | 1203.51 | 0.00 | T1 1.5R @ 1216.17 |
| Target hit | 2026-03-04 10:25:00 | 1240.60 | 1242.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-03-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 10:10:00 | 1161.70 | 1159.93 | 0.00 | ORB-long ORB[1147.80,1161.30] vol=2.3x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-03-16 10:20:00 | 1156.92 | 1159.65 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 1162.10 | 1165.89 | 0.00 | ORB-short ORB[1163.20,1176.10] vol=1.7x ATR=4.69 |
| Stop hit — per-position SL triggered | 2026-03-17 12:30:00 | 1166.79 | 1161.33 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:40:00 | 1164.10 | 1159.59 | 0.00 | ORB-long ORB[1152.90,1164.00] vol=2.3x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:45:00 | 1170.39 | 1160.80 | 0.00 | T1 1.5R @ 1170.39 |
| Stop hit — per-position SL triggered | 2026-03-27 11:25:00 | 1164.10 | 1162.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 09:55:00 | 1183.40 | 1171.81 | 0.00 | ORB-long ORB[1158.10,1174.60] vol=4.6x ATR=6.13 |
| Stop hit — per-position SL triggered | 2026-04-02 10:35:00 | 1177.27 | 1174.87 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 1252.50 | 1246.35 | 0.00 | ORB-long ORB[1237.60,1252.00] vol=2.3x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:05:00 | 1257.91 | 1249.51 | 0.00 | T1 1.5R @ 1257.91 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1252.50 | 1251.81 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 1255.00 | 1245.59 | 0.00 | ORB-long ORB[1237.10,1248.60] vol=1.7x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:00:00 | 1260.55 | 1248.13 | 0.00 | T1 1.5R @ 1260.55 |
| Target hit | 2026-04-28 11:15:00 | 1256.30 | 1259.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 1275.60 | 1267.99 | 0.00 | ORB-long ORB[1259.40,1270.30] vol=1.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 1272.03 | 1271.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:15:00 | 1331.20 | 1320.63 | 0.00 | ORB-long ORB[1305.90,1316.80] vol=1.7x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:35:00 | 1338.22 | 1326.20 | 0.00 | T1 1.5R @ 1338.22 |
| Stop hit — per-position SL triggered | 2026-05-04 10:45:00 | 1331.20 | 1327.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:25:00 | 1283.60 | 2026-02-09 10:35:00 | 1290.99 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-02-11 11:00:00 | 1276.60 | 2026-02-11 11:10:00 | 1270.79 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-11 11:00:00 | 1276.60 | 2026-02-11 15:20:00 | 1270.80 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-16 10:50:00 | 1267.20 | 2026-02-16 11:25:00 | 1272.14 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-16 10:50:00 | 1267.20 | 2026-02-16 11:30:00 | 1267.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:35:00 | 1262.80 | 2026-02-17 10:50:00 | 1258.66 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-20 10:40:00 | 1286.60 | 2026-02-20 13:05:00 | 1282.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-04 09:35:00 | 1208.10 | 2026-03-04 09:45:00 | 1216.17 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-04 09:35:00 | 1208.10 | 2026-03-04 10:25:00 | 1240.60 | TARGET_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2026-03-16 10:10:00 | 1161.70 | 2026-03-16 10:20:00 | 1156.92 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-17 09:35:00 | 1162.10 | 2026-03-17 12:30:00 | 1166.79 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-27 10:40:00 | 1164.10 | 2026-03-27 10:45:00 | 1170.39 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-27 10:40:00 | 1164.10 | 2026-03-27 11:25:00 | 1164.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-02 09:55:00 | 1183.40 | 2026-04-02 10:35:00 | 1177.27 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-16 10:25:00 | 1252.50 | 2026-04-16 11:05:00 | 1257.91 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-16 10:25:00 | 1252.50 | 2026-04-16 11:15:00 | 1252.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:55:00 | 1255.00 | 2026-04-28 10:00:00 | 1260.55 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-28 09:55:00 | 1255.00 | 2026-04-28 11:15:00 | 1256.30 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-04-29 10:00:00 | 1275.60 | 2026-04-29 10:45:00 | 1272.03 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-04 10:15:00 | 1331.20 | 2026-05-04 10:35:00 | 1338.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-04 10:15:00 | 1331.20 | 2026-05-04 10:45:00 | 1331.20 | STOP_HIT | 0.50 | 0.00% |
