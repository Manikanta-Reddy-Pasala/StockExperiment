# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1447.20
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 1
- **Avg / median % per leg:** -0.24% / -0.36%
- **Sum % (uncompounded):** -2.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.37% | -2.2% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.37% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.01% | 0.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.01% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.24% | -2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1248.40 | 1255.89 | 0.00 | ORB-short ORB[1251.20,1267.90] vol=2.1x ATR=5.44 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 1253.84 | 1254.42 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 1246.00 | 1240.63 | 0.00 | ORB-long ORB[1227.70,1243.90] vol=2.0x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-02-16 11:00:00 | 1241.03 | 1241.48 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 1292.90 | 1295.93 | 0.00 | ORB-short ORB[1299.00,1314.20] vol=1.6x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 1286.87 | 1294.96 | 0.00 | T1 1.5R @ 1286.87 |
| Stop hit — per-position SL triggered | 2026-02-19 12:25:00 | 1292.90 | 1294.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:25:00 | 1258.00 | 1252.28 | 0.00 | ORB-long ORB[1245.00,1256.00] vol=1.5x ATR=3.72 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 1254.28 | 1253.35 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1311.70 | 1303.68 | 0.00 | ORB-long ORB[1297.60,1306.80] vol=1.7x ATR=4.71 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 1306.99 | 1306.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1395.80 | 1382.12 | 0.00 | ORB-long ORB[1367.50,1384.20] vol=5.0x ATR=6.25 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 1389.55 | 1385.74 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1384.20 | 1374.28 | 0.00 | ORB-long ORB[1365.40,1377.20] vol=2.5x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-05-05 09:35:00 | 1378.58 | 1375.72 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1416.00 | 1411.74 | 0.00 | ORB-long ORB[1403.80,1412.00] vol=4.2x ATR=4.59 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1411.41 | 1412.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 1248.40 | 2026-02-13 09:40:00 | 1253.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-16 10:00:00 | 1246.00 | 2026-02-16 11:00:00 | 1241.03 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-19 11:05:00 | 1292.90 | 2026-02-19 11:50:00 | 1286.87 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-19 11:05:00 | 1292.90 | 2026-02-19 12:25:00 | 1292.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:25:00 | 1258.00 | 2026-02-26 10:55:00 | 1254.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1311.70 | 2026-03-18 09:45:00 | 1306.99 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1395.80 | 2026-04-21 09:40:00 | 1389.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-05 09:30:00 | 1384.20 | 2026-05-05 09:35:00 | 1378.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1416.00 | 2026-05-06 10:15:00 | 1411.41 | STOP_HIT | 1.00 | -0.32% |
