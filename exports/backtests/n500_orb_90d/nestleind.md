# Nestle India Ltd. (NESTLEIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1475.30
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.06% | 0.6% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.06% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.09% | 1.1% |
| SELL @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.09% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.08% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 1293.30 | 1294.19 | 0.00 | ORB-short ORB[1295.00,1307.00] vol=1.9x ATR=4.60 |
| Stop hit — per-position SL triggered | 2026-02-09 11:35:00 | 1297.90 | 1294.28 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1311.10 | 1318.01 | 0.00 | ORB-short ORB[1312.10,1324.90] vol=2.6x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:30:00 | 1307.50 | 1317.10 | 0.00 | T1 1.5R @ 1307.50 |
| Target hit | 2026-02-11 15:20:00 | 1304.60 | 1308.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 1298.30 | 1304.60 | 0.00 | ORB-short ORB[1300.10,1312.10] vol=1.8x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-02-12 11:40:00 | 1301.32 | 1303.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 1288.20 | 1281.73 | 0.00 | ORB-long ORB[1276.70,1282.60] vol=2.2x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:10:00 | 1292.02 | 1283.46 | 0.00 | T1 1.5R @ 1292.02 |
| Stop hit — per-position SL triggered | 2026-02-16 12:50:00 | 1288.20 | 1287.07 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 1296.10 | 1288.26 | 0.00 | ORB-long ORB[1276.40,1291.20] vol=3.7x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 1299.55 | 1292.21 | 0.00 | T1 1.5R @ 1299.55 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 1296.10 | 1293.00 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 1321.50 | 1316.63 | 0.00 | ORB-long ORB[1305.60,1320.20] vol=1.5x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-24 10:45:00 | 1318.70 | 1317.39 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1332.10 | 1328.90 | 0.00 | ORB-long ORB[1324.00,1330.90] vol=2.1x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 1329.15 | 1329.65 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 1226.10 | 1235.47 | 0.00 | ORB-short ORB[1236.20,1253.10] vol=2.6x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-03-05 10:35:00 | 1229.46 | 1234.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 11:00:00 | 1232.80 | 1228.06 | 0.00 | ORB-long ORB[1212.00,1229.20] vol=2.2x ATR=4.19 |
| Stop hit — per-position SL triggered | 2026-03-13 11:10:00 | 1228.61 | 1228.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1213.10 | 1217.97 | 0.00 | ORB-short ORB[1213.40,1220.60] vol=1.6x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:30:00 | 1208.31 | 1217.21 | 0.00 | T1 1.5R @ 1208.31 |
| Target hit | 2026-03-17 13:15:00 | 1208.00 | 1207.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 1205.00 | 1207.64 | 0.00 | ORB-short ORB[1206.00,1213.30] vol=3.0x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:10:00 | 1200.41 | 1206.18 | 0.00 | T1 1.5R @ 1200.41 |
| Stop hit — per-position SL triggered | 2026-03-18 11:30:00 | 1205.00 | 1205.45 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 1169.20 | 1173.20 | 0.00 | ORB-short ORB[1172.00,1186.90] vol=3.4x ATR=3.56 |
| Stop hit — per-position SL triggered | 2026-03-24 11:10:00 | 1172.76 | 1172.94 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 1275.60 | 1268.39 | 0.00 | ORB-long ORB[1252.80,1266.60] vol=1.6x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:25:00 | 1280.97 | 1273.12 | 0.00 | T1 1.5R @ 1280.97 |
| Target hit | 2026-04-17 15:20:00 | 1285.80 | 1282.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1308.70 | 1298.30 | 0.00 | ORB-long ORB[1286.80,1306.40] vol=1.6x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 1303.69 | 1303.48 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 1474.90 | 1480.27 | 0.00 | ORB-short ORB[1481.50,1493.00] vol=5.9x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:25:00 | 1470.47 | 1479.29 | 0.00 | T1 1.5R @ 1470.47 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 1474.90 | 1479.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:55:00 | 1293.30 | 2026-02-09 11:35:00 | 1297.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-11 11:00:00 | 1311.10 | 2026-02-11 11:30:00 | 1307.50 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-11 11:00:00 | 1311.10 | 2026-02-11 15:20:00 | 1304.60 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-12 10:55:00 | 1298.30 | 2026-02-12 11:40:00 | 1301.32 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-16 10:50:00 | 1288.20 | 2026-02-16 11:10:00 | 1292.02 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-16 10:50:00 | 1288.20 | 2026-02-16 12:50:00 | 1288.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 11:10:00 | 1296.10 | 2026-02-20 12:15:00 | 1299.55 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-20 11:10:00 | 1296.10 | 2026-02-20 12:55:00 | 1296.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:25:00 | 1321.50 | 2026-02-24 10:45:00 | 1318.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-25 09:30:00 | 1332.10 | 2026-02-25 09:50:00 | 1329.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1226.10 | 2026-03-05 10:35:00 | 1229.46 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-13 11:00:00 | 1232.80 | 2026-03-13 11:10:00 | 1228.61 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-17 11:15:00 | 1213.10 | 2026-03-17 11:30:00 | 1208.31 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-17 11:15:00 | 1213.10 | 2026-03-17 13:15:00 | 1208.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-18 11:00:00 | 1205.00 | 2026-03-18 11:10:00 | 1200.41 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-18 11:00:00 | 1205.00 | 2026-03-18 11:30:00 | 1205.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 10:55:00 | 1169.20 | 2026-03-24 11:10:00 | 1172.76 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1275.60 | 2026-04-17 11:25:00 | 1280.97 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1275.60 | 2026-04-17 15:20:00 | 1285.80 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1308.70 | 2026-04-21 11:00:00 | 1303.69 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1474.90 | 2026-05-07 11:25:00 | 1470.47 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1474.90 | 2026-05-07 11:30:00 | 1474.90 | STOP_HIT | 0.50 | 0.00% |
