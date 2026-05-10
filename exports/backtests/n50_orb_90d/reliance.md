# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1436.00
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
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 5
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.05% | 0.6% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.05% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.22% | 1.8% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.22% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 7 | 35.0% | 2 | 13 | 5 | 0.12% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1459.70 | 1464.41 | 0.00 | ORB-short ORB[1464.50,1473.00] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 1457.01 | 1463.98 | 0.00 | T1 1.5R @ 1457.01 |
| Stop hit — per-position SL triggered | 2026-02-12 11:55:00 | 1459.70 | 1462.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1431.00 | 1433.10 | 0.00 | ORB-short ORB[1432.00,1440.50] vol=2.2x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:10:00 | 1427.95 | 1432.33 | 0.00 | T1 1.5R @ 1427.95 |
| Target hit | 2026-02-25 15:20:00 | 1398.40 | 1414.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 1404.20 | 1396.41 | 0.00 | ORB-long ORB[1381.10,1397.00] vol=1.5x ATR=3.95 |
| Stop hit — per-position SL triggered | 2026-03-12 13:05:00 | 1400.25 | 1399.02 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 1411.60 | 1407.87 | 0.00 | ORB-long ORB[1397.20,1409.70] vol=1.7x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 1408.64 | 1408.20 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:00:00 | 1427.50 | 1420.65 | 0.00 | ORB-long ORB[1414.10,1421.90] vol=1.9x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-03-25 11:35:00 | 1424.83 | 1422.38 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:40:00 | 1332.20 | 1341.29 | 0.00 | ORB-short ORB[1338.00,1350.00] vol=1.6x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-04-09 10:50:00 | 1335.22 | 1340.80 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 1347.50 | 1344.73 | 0.00 | ORB-long ORB[1331.50,1344.50] vol=1.5x ATR=3.43 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1344.07 | 1344.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 1335.50 | 1338.71 | 0.00 | ORB-short ORB[1340.00,1353.80] vol=2.1x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 1338.60 | 1338.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1357.00 | 1348.90 | 0.00 | ORB-long ORB[1340.00,1351.90] vol=2.1x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1360.46 | 1350.09 | 0.00 | T1 1.5R @ 1360.46 |
| Stop hit — per-position SL triggered | 2026-04-17 13:20:00 | 1357.00 | 1356.90 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1361.60 | 1355.43 | 0.00 | ORB-long ORB[1350.20,1358.00] vol=1.5x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:05:00 | 1365.24 | 1357.24 | 0.00 | T1 1.5R @ 1365.24 |
| Stop hit — per-position SL triggered | 2026-04-22 11:35:00 | 1361.60 | 1358.30 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 1331.60 | 1337.45 | 0.00 | ORB-short ORB[1337.30,1345.90] vol=1.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 1334.02 | 1337.34 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 1342.40 | 1327.75 | 0.00 | ORB-long ORB[1311.00,1331.00] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-27 10:20:00 | 1338.38 | 1329.95 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1448.10 | 1443.16 | 0.00 | ORB-long ORB[1433.40,1446.50] vol=1.5x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:35:00 | 1452.74 | 1445.68 | 0.00 | T1 1.5R @ 1452.74 |
| Target hit | 2026-05-04 15:20:00 | 1465.10 | 1453.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1445.10 | 1460.09 | 0.00 | ORB-short ORB[1463.00,1473.30] vol=1.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 1448.67 | 1458.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 1429.90 | 1425.40 | 0.00 | ORB-long ORB[1417.50,1429.50] vol=2.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 1426.76 | 1425.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 1459.70 | 2026-02-12 11:20:00 | 1457.01 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2026-02-12 11:15:00 | 1459.70 | 2026-02-12 11:55:00 | 1459.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:15:00 | 1431.00 | 2026-02-25 12:10:00 | 1427.95 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2026-02-25 11:15:00 | 1431.00 | 2026-02-25 15:20:00 | 1398.40 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2026-03-12 11:15:00 | 1404.20 | 2026-03-12 13:05:00 | 1400.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-18 11:00:00 | 1411.60 | 2026-03-18 11:25:00 | 1408.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-25 11:00:00 | 1427.50 | 2026-03-25 11:35:00 | 1424.83 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-04-09 10:40:00 | 1332.20 | 2026-04-09 10:50:00 | 1335.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-10 09:55:00 | 1347.50 | 2026-04-10 10:05:00 | 1344.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1335.50 | 2026-04-16 11:00:00 | 1338.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1357.00 | 2026-04-17 11:10:00 | 1360.46 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1357.00 | 2026-04-17 13:20:00 | 1357.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1361.60 | 2026-04-22 11:05:00 | 1365.24 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1361.60 | 2026-04-22 11:35:00 | 1361.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1331.60 | 2026-04-24 11:20:00 | 1334.02 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-04-27 09:55:00 | 1342.40 | 2026-04-27 10:20:00 | 1338.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 11:15:00 | 1448.10 | 2026-05-04 12:35:00 | 1452.74 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-05-04 11:15:00 | 1448.10 | 2026-05-04 15:20:00 | 1465.10 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1445.10 | 2026-05-06 11:10:00 | 1448.67 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-08 11:00:00 | 1429.90 | 2026-05-08 11:15:00 | 1426.76 | STOP_HIT | 1.00 | -0.22% |
