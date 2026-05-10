# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1686.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 7
- **Avg / median % per leg:** 0.27% / 0.32%
- **Sum % (uncompounded):** 4.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.31% | 3.5% |
| BUY @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.31% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.17% | 0.8% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.17% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 10 | 62.5% | 3 | 6 | 7 | 0.27% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 1714.00 | 1702.09 | 0.00 | ORB-long ORB[1685.20,1705.00] vol=5.9x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 1724.41 | 1708.20 | 0.00 | T1 1.5R @ 1724.41 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 1714.00 | 1711.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 1712.30 | 1708.63 | 0.00 | ORB-long ORB[1692.70,1709.00] vol=3.3x ATR=6.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 1722.48 | 1712.31 | 0.00 | T1 1.5R @ 1722.48 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 1712.30 | 1719.51 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 1642.50 | 1644.40 | 0.00 | ORB-short ORB[1647.70,1659.90] vol=5.1x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:45:00 | 1632.34 | 1643.17 | 0.00 | T1 1.5R @ 1632.34 |
| Stop hit — per-position SL triggered | 2026-02-18 12:50:00 | 1642.50 | 1641.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:00:00 | 1343.00 | 1331.38 | 0.00 | ORB-long ORB[1323.70,1340.00] vol=3.3x ATR=5.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:10:00 | 1351.92 | 1335.69 | 0.00 | T1 1.5R @ 1351.92 |
| Target hit | 2026-03-04 11:55:00 | 1344.80 | 1346.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 1504.80 | 1513.66 | 0.00 | ORB-short ORB[1512.40,1525.40] vol=2.0x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 1496.42 | 1505.68 | 0.00 | T1 1.5R @ 1496.42 |
| Stop hit — per-position SL triggered | 2026-04-17 11:40:00 | 1504.80 | 1503.17 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1435.20 | 1428.81 | 0.00 | ORB-long ORB[1414.10,1434.50] vol=1.6x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:40:00 | 1443.92 | 1432.47 | 0.00 | T1 1.5R @ 1443.92 |
| Target hit | 2026-04-21 12:15:00 | 1437.40 | 1437.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 1508.70 | 1519.60 | 0.00 | ORB-short ORB[1511.70,1530.90] vol=2.0x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 1513.74 | 1519.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1687.70 | 1680.10 | 0.00 | ORB-long ORB[1660.00,1683.90] vol=2.3x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:35:00 | 1699.26 | 1685.08 | 0.00 | T1 1.5R @ 1699.26 |
| Target hit | 2026-05-05 13:10:00 | 1693.10 | 1693.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 1686.70 | 1682.16 | 0.00 | ORB-long ORB[1668.60,1685.70] vol=2.4x ATR=5.10 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 1681.60 | 1682.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 1714.00 | 2026-02-09 11:20:00 | 1724.41 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-09 11:00:00 | 1714.00 | 2026-02-09 11:50:00 | 1714.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1712.30 | 2026-02-10 09:35:00 | 1722.48 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1712.30 | 2026-02-10 10:50:00 | 1712.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1642.50 | 2026-02-18 11:45:00 | 1632.34 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1642.50 | 2026-02-18 12:50:00 | 1642.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 10:00:00 | 1343.00 | 2026-03-04 10:10:00 | 1351.92 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-04 10:00:00 | 1343.00 | 2026-03-04 11:55:00 | 1344.80 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2026-04-17 09:35:00 | 1504.80 | 2026-04-17 10:35:00 | 1496.42 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-17 09:35:00 | 1504.80 | 2026-04-17 11:40:00 | 1504.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 1435.20 | 2026-04-21 10:40:00 | 1443.92 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 10:00:00 | 1435.20 | 2026-04-21 12:15:00 | 1437.40 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-04-29 10:30:00 | 1508.70 | 2026-04-29 10:40:00 | 1513.74 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-05 11:05:00 | 1687.70 | 2026-05-05 12:35:00 | 1699.26 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-05 11:05:00 | 1687.70 | 2026-05-05 13:10:00 | 1693.10 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-05-07 10:55:00 | 1686.70 | 2026-05-07 11:25:00 | 1681.60 | STOP_HIT | 1.00 | -0.30% |
