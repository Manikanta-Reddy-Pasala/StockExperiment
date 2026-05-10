# TECHM (TECHM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1460.90
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 4
- **Avg / median % per leg:** 0.33% / 0.33%
- **Sum % (uncompounded):** 3.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.57% | 4.0% |
| SELL @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.57% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.33% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1647.20 | 1639.25 | 0.00 | ORB-long ORB[1611.10,1627.90] vol=6.1x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 1654.80 | 1641.90 | 0.00 | T1 1.5R @ 1654.80 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 1647.20 | 1642.77 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 1521.80 | 1525.02 | 0.00 | ORB-short ORB[1522.00,1541.90] vol=3.8x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:25:00 | 1512.77 | 1522.76 | 0.00 | T1 1.5R @ 1512.77 |
| Target hit | 2026-02-16 15:15:00 | 1513.90 | 1513.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 1340.90 | 1344.98 | 0.00 | ORB-short ORB[1343.90,1355.40] vol=3.0x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-03-11 11:35:00 | 1344.93 | 1344.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-23 10:35:00 | 1396.10 | 1382.85 | 0.00 | ORB-long ORB[1357.60,1376.50] vol=1.5x ATR=5.81 |
| Stop hit — per-position SL triggered | 2026-03-23 10:40:00 | 1390.29 | 1383.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:50:00 | 1415.50 | 1412.81 | 0.00 | ORB-long ORB[1402.00,1414.00] vol=1.8x ATR=5.58 |
| Stop hit — per-position SL triggered | 2026-03-24 10:00:00 | 1409.92 | 1412.49 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 1392.90 | 1408.79 | 0.00 | ORB-short ORB[1406.00,1426.70] vol=1.7x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:50:00 | 1383.09 | 1403.51 | 0.00 | T1 1.5R @ 1383.09 |
| Target hit | 2026-04-24 15:20:00 | 1362.80 | 1367.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 1457.70 | 1459.43 | 0.00 | ORB-short ORB[1460.00,1475.90] vol=2.9x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:10:00 | 1452.88 | 1459.06 | 0.00 | T1 1.5R @ 1452.88 |
| Stop hit — per-position SL triggered | 2026-05-07 12:00:00 | 1457.70 | 1457.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 1647.20 | 2026-02-10 09:45:00 | 1654.80 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-10 09:40:00 | 1647.20 | 2026-02-10 09:55:00 | 1647.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-16 10:00:00 | 1521.80 | 2026-02-16 10:25:00 | 1512.77 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-16 10:00:00 | 1521.80 | 2026-02-16 15:15:00 | 1513.90 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-03-11 10:55:00 | 1340.90 | 2026-03-11 11:35:00 | 1344.93 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-23 10:35:00 | 1396.10 | 2026-03-23 10:40:00 | 1390.29 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-24 09:50:00 | 1415.50 | 2026-03-24 10:00:00 | 1409.92 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-24 09:40:00 | 1392.90 | 2026-04-24 09:50:00 | 1383.09 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-24 09:40:00 | 1392.90 | 2026-04-24 15:20:00 | 1362.80 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1457.70 | 2026-05-07 11:10:00 | 1452.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-05-07 10:55:00 | 1457.70 | 2026-05-07 12:00:00 | 1457.70 | STOP_HIT | 0.50 | 0.00% |
