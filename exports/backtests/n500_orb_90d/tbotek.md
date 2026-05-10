# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1227.20
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 5
- **Avg / median % per leg:** 0.33% / 0.42%
- **Sum % (uncompounded):** 3.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.40% | 4.4% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 1 | 5 | 5 | 0.40% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 1 | 6 | 5 | 0.33% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 1492.50 | 1481.73 | 0.00 | ORB-long ORB[1460.00,1475.30] vol=2.3x ATR=7.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:55:00 | 1504.24 | 1485.11 | 0.00 | T1 1.5R @ 1504.24 |
| Target hit | 2026-02-09 15:20:00 | 1518.00 | 1496.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 1107.00 | 1120.28 | 0.00 | ORB-short ORB[1128.10,1144.00] vol=2.5x ATR=5.15 |
| Stop hit — per-position SL triggered | 2026-03-05 10:55:00 | 1112.15 | 1118.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 1181.50 | 1178.60 | 0.00 | ORB-long ORB[1169.40,1180.00] vol=1.8x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 1177.12 | 1178.68 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 1300.00 | 1287.13 | 0.00 | ORB-long ORB[1261.00,1274.40] vol=1.6x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:30:00 | 1309.30 | 1292.13 | 0.00 | T1 1.5R @ 1309.30 |
| Stop hit — per-position SL triggered | 2026-04-17 12:45:00 | 1300.00 | 1294.96 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 1261.00 | 1256.08 | 0.00 | ORB-long ORB[1241.10,1255.30] vol=7.2x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:50:00 | 1266.32 | 1257.50 | 0.00 | T1 1.5R @ 1266.32 |
| Stop hit — per-position SL triggered | 2026-04-27 12:50:00 | 1261.00 | 1260.91 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:55:00 | 1276.30 | 1269.17 | 0.00 | ORB-long ORB[1256.50,1269.00] vol=1.5x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:15:00 | 1284.39 | 1275.31 | 0.00 | T1 1.5R @ 1284.39 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 1276.30 | 1276.30 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 1294.80 | 1287.69 | 0.00 | ORB-long ORB[1279.60,1292.10] vol=2.0x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:25:00 | 1300.91 | 1292.05 | 0.00 | T1 1.5R @ 1300.91 |
| Stop hit — per-position SL triggered | 2026-05-07 12:35:00 | 1294.80 | 1295.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 1492.50 | 2026-02-09 11:55:00 | 1504.24 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-09 10:55:00 | 1492.50 | 2026-02-09 15:20:00 | 1518.00 | TARGET_HIT | 0.50 | 1.71% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1107.00 | 2026-03-05 10:55:00 | 1112.15 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-17 10:25:00 | 1181.50 | 2026-03-17 10:35:00 | 1177.12 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-17 10:35:00 | 1300.00 | 2026-04-17 11:30:00 | 1309.30 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-17 10:35:00 | 1300.00 | 2026-04-17 12:45:00 | 1300.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:35:00 | 1261.00 | 2026-04-27 10:50:00 | 1266.32 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-27 10:35:00 | 1261.00 | 2026-04-27 12:50:00 | 1261.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:55:00 | 1276.30 | 2026-05-04 11:15:00 | 1284.39 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-05-04 09:55:00 | 1276.30 | 2026-05-04 12:10:00 | 1276.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:15:00 | 1294.80 | 2026-05-07 10:25:00 | 1300.91 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-07 10:15:00 | 1294.80 | 2026-05-07 12:35:00 | 1294.80 | STOP_HIT | 0.50 | 0.00% |
