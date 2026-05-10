# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1294.50
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
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.11% | 1.1% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.11% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.05% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.08% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 1265.50 | 1260.47 | 0.00 | ORB-long ORB[1243.50,1260.80] vol=1.6x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:40:00 | 1275.54 | 1264.05 | 0.00 | T1 1.5R @ 1275.54 |
| Target hit | 2026-02-09 15:20:00 | 1272.40 | 1266.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1266.40 | 1260.20 | 0.00 | ORB-long ORB[1253.20,1264.80] vol=1.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:40:00 | 1270.19 | 1262.34 | 0.00 | T1 1.5R @ 1270.19 |
| Stop hit — per-position SL triggered | 2026-02-11 12:35:00 | 1266.40 | 1264.12 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 1265.30 | 1269.59 | 0.00 | ORB-short ORB[1266.70,1274.00] vol=1.9x ATR=2.65 |
| Stop hit — per-position SL triggered | 2026-02-13 11:35:00 | 1267.95 | 1269.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 1280.50 | 1285.70 | 0.00 | ORB-short ORB[1282.20,1291.50] vol=1.7x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 1283.75 | 1285.52 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1325.40 | 1316.16 | 0.00 | ORB-long ORB[1305.00,1314.90] vol=1.6x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 1320.64 | 1316.95 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 1291.30 | 1298.83 | 0.00 | ORB-short ORB[1303.00,1317.20] vol=1.9x ATR=3.77 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 1295.07 | 1298.16 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 1284.00 | 1281.72 | 0.00 | ORB-long ORB[1272.10,1283.50] vol=1.5x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-03-04 09:55:00 | 1278.79 | 1281.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1294.80 | 1303.29 | 0.00 | ORB-short ORB[1304.80,1316.40] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 1298.13 | 1303.05 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:45:00 | 1271.60 | 1283.51 | 0.00 | ORB-short ORB[1286.30,1297.00] vol=2.5x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:55:00 | 1265.73 | 1283.01 | 0.00 | T1 1.5R @ 1265.73 |
| Stop hit — per-position SL triggered | 2026-03-16 11:10:00 | 1271.60 | 1282.25 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1258.50 | 1264.60 | 0.00 | ORB-short ORB[1263.10,1279.80] vol=1.6x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:40:00 | 1250.77 | 1259.98 | 0.00 | T1 1.5R @ 1250.77 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1258.50 | 1258.88 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 1300.60 | 1295.11 | 0.00 | ORB-long ORB[1282.30,1297.00] vol=2.1x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-03-27 11:25:00 | 1296.12 | 1296.67 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:10:00 | 1207.20 | 1204.27 | 0.00 | ORB-long ORB[1189.00,1205.00] vol=7.0x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:40:00 | 1211.43 | 1204.86 | 0.00 | T1 1.5R @ 1211.43 |
| Target hit | 2026-04-09 15:20:00 | 1212.80 | 1207.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1222.80 | 1225.63 | 0.00 | ORB-short ORB[1227.30,1234.90] vol=1.7x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 1219.71 | 1224.61 | 0.00 | T1 1.5R @ 1219.71 |
| Target hit | 2026-04-21 15:20:00 | 1220.70 | 1222.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:40:00 | 1289.60 | 1283.70 | 0.00 | ORB-long ORB[1277.10,1285.30] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 1286.43 | 1284.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 1265.50 | 2026-02-09 14:40:00 | 1275.54 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-09 10:40:00 | 1265.50 | 2026-02-09 15:20:00 | 1272.40 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1266.40 | 2026-02-11 11:40:00 | 1270.19 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1266.40 | 2026-02-11 12:35:00 | 1266.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:55:00 | 1265.30 | 2026-02-13 11:35:00 | 1267.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-18 09:40:00 | 1280.50 | 2026-02-18 09:45:00 | 1283.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1325.40 | 2026-02-26 09:50:00 | 1320.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-27 10:45:00 | 1291.30 | 2026-02-27 11:00:00 | 1295.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-04 09:45:00 | 1284.00 | 2026-03-04 09:55:00 | 1278.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1294.80 | 2026-03-06 10:50:00 | 1298.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-16 10:45:00 | 1271.60 | 2026-03-16 10:55:00 | 1265.73 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-16 10:45:00 | 1271.60 | 2026-03-16 11:10:00 | 1271.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 10:15:00 | 1258.50 | 2026-03-24 10:40:00 | 1250.77 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-24 10:15:00 | 1258.50 | 2026-03-24 11:15:00 | 1258.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 10:00:00 | 1300.60 | 2026-03-27 11:25:00 | 1296.12 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-09 11:10:00 | 1207.20 | 2026-04-09 11:40:00 | 1211.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-09 11:10:00 | 1207.20 | 2026-04-09 15:20:00 | 1212.80 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1222.80 | 2026-04-21 11:35:00 | 1219.71 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1222.80 | 2026-04-21 15:20:00 | 1220.70 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-05-06 10:40:00 | 1289.60 | 2026-05-06 10:55:00 | 1286.43 | STOP_HIT | 1.00 | -0.25% |
