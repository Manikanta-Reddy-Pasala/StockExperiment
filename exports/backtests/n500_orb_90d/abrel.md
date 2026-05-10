# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1479.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 2
- **Avg / median % per leg:** 0.17% / -0.35%
- **Sum % (uncompounded):** 2.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.24% | -1.4% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.24% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.53% | 3.7% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.53% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 3 | 23.1% | 1 | 10 | 2 | 0.17% | 2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 1377.10 | 1384.03 | 0.00 | ORB-short ORB[1387.10,1400.40] vol=4.9x ATR=4.86 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 1381.96 | 1383.17 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 1358.90 | 1363.58 | 0.00 | ORB-short ORB[1362.60,1372.40] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 1362.50 | 1362.35 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1285.00 | 1289.04 | 0.00 | ORB-short ORB[1286.10,1297.40] vol=1.7x ATR=4.39 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 1289.39 | 1288.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1294.00 | 1280.08 | 0.00 | ORB-long ORB[1262.40,1278.70] vol=3.2x ATR=5.25 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 1288.75 | 1282.15 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 1442.60 | 1431.56 | 0.00 | ORB-long ORB[1418.10,1434.80] vol=1.8x ATR=7.49 |
| Stop hit — per-position SL triggered | 2026-04-27 10:20:00 | 1435.11 | 1438.89 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 1509.80 | 1494.07 | 0.00 | ORB-long ORB[1477.40,1498.90] vol=1.7x ATR=8.53 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 1501.27 | 1498.57 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 1541.00 | 1519.80 | 0.00 | ORB-long ORB[1500.00,1520.00] vol=1.7x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:50:00 | 1549.58 | 1522.94 | 0.00 | T1 1.5R @ 1549.58 |
| Stop hit — per-position SL triggered | 2026-04-29 13:20:00 | 1541.00 | 1525.20 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:10:00 | 1494.90 | 1512.43 | 0.00 | ORB-short ORB[1500.80,1523.30] vol=1.5x ATR=6.91 |
| Stop hit — per-position SL triggered | 2026-05-04 10:25:00 | 1501.81 | 1511.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1464.20 | 1486.31 | 0.00 | ORB-short ORB[1488.20,1507.80] vol=1.8x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 1469.98 | 1480.02 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:20:00 | 1519.50 | 1502.07 | 0.00 | ORB-long ORB[1491.00,1501.50] vol=3.3x ATR=7.67 |
| Stop hit — per-position SL triggered | 2026-05-06 10:25:00 | 1511.83 | 1503.20 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 1551.50 | 1572.57 | 0.00 | ORB-short ORB[1580.00,1593.80] vol=3.8x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:00:00 | 1538.63 | 1564.26 | 0.00 | T1 1.5R @ 1538.63 |
| Target hit | 2026-05-08 15:20:00 | 1479.00 | 1512.00 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:40:00 | 1377.10 | 2026-02-18 11:30:00 | 1381.96 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-19 09:40:00 | 1358.90 | 2026-02-19 09:55:00 | 1362.50 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1285.00 | 2026-02-24 09:45:00 | 1289.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1294.00 | 2026-02-26 09:50:00 | 1288.75 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-27 09:45:00 | 1442.60 | 2026-04-27 10:20:00 | 1435.11 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-28 09:55:00 | 1509.80 | 2026-04-28 10:35:00 | 1501.27 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-29 11:10:00 | 1541.00 | 2026-04-29 11:50:00 | 1549.58 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-29 11:10:00 | 1541.00 | 2026-04-29 13:20:00 | 1541.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:10:00 | 1494.90 | 2026-05-04 10:25:00 | 1501.81 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-05-05 11:00:00 | 1464.20 | 2026-05-05 11:10:00 | 1469.98 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-06 10:20:00 | 1519.50 | 2026-05-06 10:25:00 | 1511.83 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1551.50 | 2026-05-08 10:00:00 | 1538.63 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1551.50 | 2026-05-08 15:20:00 | 1479.00 | TARGET_HIT | 0.50 | 4.67% |
