# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1769.40
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 4
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 0.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.11% | 0.7% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.11% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.04% | 0.2% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.04% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.08% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1509.00 | 1517.74 | 0.00 | ORB-short ORB[1512.50,1534.90] vol=2.3x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 1514.09 | 1515.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1481.60 | 1488.22 | 0.00 | ORB-short ORB[1482.80,1501.80] vol=1.5x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 1487.22 | 1486.92 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 1493.00 | 1488.01 | 0.00 | ORB-long ORB[1480.10,1492.70] vol=2.7x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:40:00 | 1499.92 | 1491.33 | 0.00 | T1 1.5R @ 1499.92 |
| Target hit | 2026-02-26 15:20:00 | 1497.00 | 1501.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 1479.00 | 1483.50 | 0.00 | ORB-short ORB[1482.10,1498.00] vol=2.0x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:50:00 | 1474.72 | 1482.18 | 0.00 | T1 1.5R @ 1474.72 |
| Stop hit — per-position SL triggered | 2026-02-27 13:55:00 | 1479.00 | 1480.92 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:00:00 | 1428.60 | 1413.62 | 0.00 | ORB-long ORB[1400.20,1419.00] vol=1.7x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:05:00 | 1438.59 | 1417.96 | 0.00 | T1 1.5R @ 1438.59 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 1428.60 | 1418.47 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 1489.30 | 1500.78 | 0.00 | ORB-short ORB[1490.50,1511.80] vol=3.0x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 1479.23 | 1494.65 | 0.00 | T1 1.5R @ 1479.23 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 1489.30 | 1488.32 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:55:00 | 1596.40 | 1587.93 | 0.00 | ORB-long ORB[1571.80,1592.20] vol=2.1x ATR=5.61 |
| Stop hit — per-position SL triggered | 2026-04-23 10:45:00 | 1590.79 | 1591.75 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 1749.30 | 1734.21 | 0.00 | ORB-long ORB[1721.90,1737.50] vol=2.4x ATR=6.83 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1742.47 | 1740.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 1509.00 | 2026-02-11 09:45:00 | 1514.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1481.60 | 2026-02-24 09:45:00 | 1487.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-26 10:10:00 | 1493.00 | 2026-02-26 10:40:00 | 1499.92 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-26 10:10:00 | 1493.00 | 2026-02-26 15:20:00 | 1497.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-27 11:15:00 | 1479.00 | 2026-02-27 11:50:00 | 1474.72 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-27 11:15:00 | 1479.00 | 2026-02-27 13:55:00 | 1479.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:00:00 | 1428.60 | 2026-03-05 10:05:00 | 1438.59 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-05 10:00:00 | 1428.60 | 2026-03-05 10:10:00 | 1428.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:30:00 | 1489.30 | 2026-04-16 09:45:00 | 1479.23 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-16 09:30:00 | 1489.30 | 2026-04-16 11:00:00 | 1489.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:55:00 | 1596.40 | 2026-04-23 10:45:00 | 1590.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-06 09:45:00 | 1749.30 | 2026-05-06 10:15:00 | 1742.47 | STOP_HIT | 1.00 | -0.39% |
