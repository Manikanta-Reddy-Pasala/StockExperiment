# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1717.50
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 1.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.21% | 1.8% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.21% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.14% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:10:00 | 1464.60 | 1450.86 | 0.00 | ORB-long ORB[1440.00,1454.30] vol=1.9x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:25:00 | 1473.64 | 1455.54 | 0.00 | T1 1.5R @ 1473.64 |
| Stop hit — per-position SL triggered | 2026-02-16 11:50:00 | 1464.60 | 1458.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 1545.50 | 1543.10 | 0.00 | ORB-long ORB[1525.20,1543.60] vol=2.4x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 1541.45 | 1543.02 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 1592.00 | 1576.73 | 0.00 | ORB-long ORB[1561.30,1578.60] vol=3.0x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:05:00 | 1599.19 | 1581.95 | 0.00 | T1 1.5R @ 1599.19 |
| Stop hit — per-position SL triggered | 2026-02-27 11:10:00 | 1592.00 | 1582.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 1502.70 | 1499.60 | 0.00 | ORB-long ORB[1479.90,1499.80] vol=1.5x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-03-05 11:30:00 | 1498.79 | 1500.26 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1482.40 | 1490.41 | 0.00 | ORB-short ORB[1485.20,1497.00] vol=1.9x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-03-06 11:05:00 | 1486.98 | 1489.91 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 1485.10 | 1469.04 | 0.00 | ORB-long ORB[1432.70,1454.00] vol=1.7x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:50:00 | 1495.85 | 1472.14 | 0.00 | T1 1.5R @ 1495.85 |
| Target hit | 2026-03-12 15:20:00 | 1500.10 | 1490.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 1442.90 | 1458.85 | 0.00 | ORB-short ORB[1455.80,1475.40] vol=3.7x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:45:00 | 1433.51 | 1454.14 | 0.00 | T1 1.5R @ 1433.51 |
| Stop hit — per-position SL triggered | 2026-03-16 11:20:00 | 1442.90 | 1450.57 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:40:00 | 1464.20 | 1454.75 | 0.00 | ORB-long ORB[1430.00,1451.70] vol=3.4x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 1457.89 | 1455.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 1732.30 | 1743.69 | 0.00 | ORB-short ORB[1739.60,1761.90] vol=1.7x ATR=7.19 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 1739.49 | 1740.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 10:10:00 | 1464.60 | 2026-02-16 10:25:00 | 1473.64 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-16 10:10:00 | 1464.60 | 2026-02-16 11:50:00 | 1464.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:55:00 | 1545.50 | 2026-02-24 11:10:00 | 1541.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-27 10:45:00 | 1592.00 | 2026-02-27 11:05:00 | 1599.19 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-27 10:45:00 | 1592.00 | 2026-02-27 11:10:00 | 1592.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:55:00 | 1502.70 | 2026-03-05 11:30:00 | 1498.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1482.40 | 2026-03-06 11:05:00 | 1486.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-12 11:15:00 | 1485.10 | 2026-03-12 11:50:00 | 1495.85 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-03-12 11:15:00 | 1485.10 | 2026-03-12 15:20:00 | 1500.10 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-03-16 10:20:00 | 1442.90 | 2026-03-16 10:45:00 | 1433.51 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-16 10:20:00 | 1442.90 | 2026-03-16 11:20:00 | 1442.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:40:00 | 1464.20 | 2026-03-18 09:45:00 | 1457.89 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-05-07 09:40:00 | 1732.30 | 2026-05-07 10:15:00 | 1739.49 | STOP_HIT | 1.00 | -0.42% |
