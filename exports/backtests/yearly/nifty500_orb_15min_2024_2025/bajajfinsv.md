# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-07-09 15:25:00 (3021 bars)
- **Last close:** 1581.00
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 4
- **Avg / median % per leg:** 0.10% / -0.19%
- **Sum % (uncompounded):** 1.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.24% | 1.7% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.24% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.02% | 0.2% |
| SELL @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.02% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 7 | 36.8% | 3 | 12 | 4 | 0.10% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1567.60 | 1570.25 | 0.00 | ORB-short ORB[1570.75,1579.35] vol=2.1x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-05-16 10:30:00 | 1570.85 | 1570.12 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 1573.55 | 1579.96 | 0.00 | ORB-short ORB[1574.00,1595.30] vol=2.7x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-05-21 10:00:00 | 1577.88 | 1577.93 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 1593.80 | 1588.85 | 0.00 | ORB-long ORB[1581.25,1591.85] vol=2.0x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 1590.35 | 1589.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 1587.70 | 1591.07 | 0.00 | ORB-short ORB[1590.40,1602.35] vol=1.6x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:00:00 | 1582.53 | 1588.57 | 0.00 | T1 1.5R @ 1582.53 |
| Target hit | 2024-05-29 15:20:00 | 1569.40 | 1576.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1547.25 | 1562.11 | 0.00 | ORB-short ORB[1558.00,1575.20] vol=1.8x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:20:00 | 1539.95 | 1553.28 | 0.00 | T1 1.5R @ 1539.95 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 1547.25 | 1552.55 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:15:00 | 1504.55 | 1490.19 | 0.00 | ORB-long ORB[1476.95,1498.95] vol=3.1x ATR=8.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 12:00:00 | 1517.01 | 1493.42 | 0.00 | T1 1.5R @ 1517.01 |
| Target hit | 2024-06-05 15:20:00 | 1520.00 | 1508.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:45:00 | 1537.45 | 1529.86 | 0.00 | ORB-long ORB[1515.20,1530.00] vol=3.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-06-06 11:20:00 | 1531.93 | 1532.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:40:00 | 1555.90 | 1546.44 | 0.00 | ORB-long ORB[1530.55,1543.90] vol=1.8x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-06-07 10:55:00 | 1550.52 | 1547.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:00:00 | 1575.70 | 1568.94 | 0.00 | ORB-long ORB[1564.65,1573.90] vol=4.0x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:20:00 | 1582.46 | 1569.74 | 0.00 | T1 1.5R @ 1582.46 |
| Target hit | 2024-06-12 15:20:00 | 1580.55 | 1576.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 1584.55 | 1590.99 | 0.00 | ORB-short ORB[1590.00,1600.00] vol=3.0x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-06-19 10:40:00 | 1588.56 | 1590.75 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 1580.00 | 1583.45 | 0.00 | ORB-short ORB[1580.55,1590.50] vol=1.8x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-06-25 09:55:00 | 1583.28 | 1582.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1596.30 | 1603.91 | 0.00 | ORB-short ORB[1603.20,1616.50] vol=2.9x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-06-26 11:30:00 | 1599.31 | 1603.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 10:35:00 | 1581.50 | 1585.98 | 0.00 | ORB-short ORB[1583.00,1590.95] vol=2.2x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-07-01 10:55:00 | 1584.51 | 1585.21 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 1568.00 | 1568.33 | 0.00 | ORB-short ORB[1572.75,1585.00] vol=3.3x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-07-02 10:45:00 | 1571.41 | 1568.40 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 1565.50 | 1570.58 | 0.00 | ORB-short ORB[1571.85,1579.60] vol=1.6x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 1568.79 | 1570.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:20:00 | 1567.60 | 2024-05-16 10:30:00 | 1570.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-21 09:30:00 | 1573.55 | 2024-05-21 10:00:00 | 1577.88 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-23 09:40:00 | 1593.80 | 2024-05-23 09:50:00 | 1590.35 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-29 10:45:00 | 1587.70 | 2024-05-29 11:00:00 | 1582.53 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-29 10:45:00 | 1587.70 | 2024-05-29 15:20:00 | 1569.40 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1547.25 | 2024-05-30 10:20:00 | 1539.95 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1547.25 | 2024-05-30 10:35:00 | 1547.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1504.55 | 2024-06-05 12:00:00 | 1517.01 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1504.55 | 2024-06-05 15:20:00 | 1520.00 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-06-06 10:45:00 | 1537.45 | 2024-06-06 11:20:00 | 1531.93 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-07 10:40:00 | 1555.90 | 2024-06-07 10:55:00 | 1550.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-12 10:00:00 | 1575.70 | 2024-06-12 10:20:00 | 1582.46 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-12 10:00:00 | 1575.70 | 2024-06-12 15:20:00 | 1580.55 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-06-19 10:35:00 | 1584.55 | 2024-06-19 10:40:00 | 1588.56 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-25 09:35:00 | 1580.00 | 2024-06-25 09:55:00 | 1583.28 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-26 11:15:00 | 1596.30 | 2024-06-26 11:30:00 | 1599.31 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-01 10:35:00 | 1581.50 | 2024-07-01 10:55:00 | 1584.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-02 10:35:00 | 1568.00 | 2024-07-02 10:45:00 | 1571.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-08 09:55:00 | 1565.50 | 2024-07-08 10:00:00 | 1568.79 | STOP_HIT | 1.00 | -0.21% |
