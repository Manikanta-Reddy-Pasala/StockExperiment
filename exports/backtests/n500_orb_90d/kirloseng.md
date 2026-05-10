# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1736.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 1
- **Avg / median % per leg:** -0.18% / -0.34%
- **Sum % (uncompounded):** -1.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.39% | -1.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.39% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.08% | -0.6% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.08% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.18% | -1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1397.20 | 1413.58 | 0.00 | ORB-short ORB[1413.20,1429.30] vol=3.2x ATR=4.54 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 1401.74 | 1413.18 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:25:00 | 1425.00 | 1414.67 | 0.00 | ORB-long ORB[1397.10,1414.00] vol=1.8x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 1420.21 | 1415.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1443.10 | 1453.87 | 0.00 | ORB-short ORB[1450.20,1468.00] vol=3.3x ATR=5.50 |
| Stop hit — per-position SL triggered | 2026-03-17 11:55:00 | 1448.60 | 1451.63 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1464.80 | 1456.48 | 0.00 | ORB-long ORB[1445.00,1459.90] vol=1.8x ATR=5.93 |
| Stop hit — per-position SL triggered | 2026-03-18 09:35:00 | 1458.87 | 1457.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:55:00 | 1419.10 | 1430.42 | 0.00 | ORB-short ORB[1424.80,1442.50] vol=2.8x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:20:00 | 1410.12 | 1427.35 | 0.00 | T1 1.5R @ 1410.12 |
| Target hit | 2026-03-20 13:25:00 | 1411.00 | 1403.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 1454.50 | 1459.09 | 0.00 | ORB-short ORB[1458.70,1473.00] vol=5.8x ATR=6.24 |
| Stop hit — per-position SL triggered | 2026-04-10 10:10:00 | 1460.74 | 1458.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1633.70 | 1640.89 | 0.00 | ORB-short ORB[1639.00,1653.40] vol=2.0x ATR=5.87 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 1639.57 | 1640.84 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1643.80 | 1652.27 | 0.00 | ORB-short ORB[1649.80,1670.00] vol=2.4x ATR=4.87 |
| Stop hit — per-position SL triggered | 2026-04-22 09:35:00 | 1648.67 | 1651.87 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1670.30 | 1655.63 | 0.00 | ORB-long ORB[1630.00,1654.80] vol=4.0x ATR=7.32 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 1662.98 | 1657.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-23 10:55:00 | 1397.20 | 2026-02-23 11:00:00 | 1401.74 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:25:00 | 1425.00 | 2026-02-26 10:50:00 | 1420.21 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-17 11:15:00 | 1443.10 | 2026-03-17 11:55:00 | 1448.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1464.80 | 2026-03-18 09:35:00 | 1458.87 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-20 09:55:00 | 1419.10 | 2026-03-20 10:20:00 | 1410.12 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-20 09:55:00 | 1419.10 | 2026-03-20 13:25:00 | 1411.00 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-10 09:55:00 | 1454.50 | 2026-04-10 10:10:00 | 1460.74 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-21 09:40:00 | 1633.70 | 2026-04-21 09:45:00 | 1639.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-22 09:30:00 | 1643.80 | 2026-04-22 09:35:00 | 1648.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1670.30 | 2026-04-23 09:40:00 | 1662.98 | STOP_HIT | 1.00 | -0.44% |
