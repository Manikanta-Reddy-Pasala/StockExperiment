# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1631.10
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 3
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 0.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.15% | -1.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.15% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.30% | 1.5% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.30% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.03% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 1398.90 | 1388.04 | 0.00 | ORB-long ORB[1365.00,1384.70] vol=2.4x ATR=7.09 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 1391.81 | 1389.38 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1276.50 | 1285.39 | 0.00 | ORB-short ORB[1280.20,1289.90] vol=2.6x ATR=4.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:00:00 | 1269.17 | 1284.41 | 0.00 | T1 1.5R @ 1269.17 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1276.50 | 1283.63 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1304.80 | 1297.94 | 0.00 | ORB-long ORB[1288.20,1303.70] vol=2.1x ATR=5.12 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1299.68 | 1300.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1279.40 | 1270.61 | 0.00 | ORB-long ORB[1258.10,1273.90] vol=2.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 1274.94 | 1271.74 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1259.80 | 1265.97 | 0.00 | ORB-short ORB[1262.60,1280.00] vol=2.2x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:35:00 | 1253.32 | 1263.68 | 0.00 | T1 1.5R @ 1253.32 |
| Target hit | 2026-02-27 10:25:00 | 1250.30 | 1250.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 1257.40 | 1252.68 | 0.00 | ORB-long ORB[1235.50,1251.00] vol=2.8x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-03-06 09:40:00 | 1251.26 | 1252.74 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 1231.60 | 1237.64 | 0.00 | ORB-short ORB[1235.00,1246.80] vol=2.4x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-03-11 10:00:00 | 1236.02 | 1236.86 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 1367.20 | 1356.35 | 0.00 | ORB-long ORB[1347.00,1363.80] vol=4.9x ATR=4.87 |
| Stop hit — per-position SL triggered | 2026-04-27 12:35:00 | 1362.33 | 1359.71 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 1381.80 | 1376.20 | 0.00 | ORB-long ORB[1360.70,1372.10] vol=6.7x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 1392.22 | 1381.14 | 0.00 | T1 1.5R @ 1392.22 |
| Target hit | 2026-05-06 12:25:00 | 1385.40 | 1385.82 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 1398.90 | 2026-02-09 12:00:00 | 1391.81 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1276.50 | 2026-02-17 11:00:00 | 1269.17 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1276.50 | 2026-02-17 11:30:00 | 1276.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:30:00 | 1304.80 | 2026-02-20 13:15:00 | 1299.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1279.40 | 2026-02-24 11:40:00 | 1274.94 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 09:30:00 | 1259.80 | 2026-02-27 09:35:00 | 1253.32 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-27 09:30:00 | 1259.80 | 2026-02-27 10:25:00 | 1250.30 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2026-03-06 09:35:00 | 1257.40 | 2026-03-06 09:40:00 | 1251.26 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-11 09:40:00 | 1231.60 | 2026-03-11 10:00:00 | 1236.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 11:05:00 | 1367.20 | 2026-04-27 12:35:00 | 1362.33 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 09:55:00 | 1381.80 | 2026-05-06 11:00:00 | 1392.22 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-05-06 09:55:00 | 1381.80 | 2026-05-06 12:25:00 | 1385.40 | TARGET_HIT | 0.50 | 0.26% |
