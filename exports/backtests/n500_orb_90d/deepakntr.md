# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1875.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 3
- **Avg / median % per leg:** -0.01% / -0.18%
- **Sum % (uncompounded):** -0.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.50% | -1.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.50% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.01% | -0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 1644.30 | 1631.73 | 0.00 | ORB-long ORB[1612.00,1625.00] vol=3.6x ATR=5.53 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1638.77 | 1633.89 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 1634.00 | 1642.72 | 0.00 | ORB-short ORB[1646.90,1657.30] vol=2.4x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 1637.84 | 1642.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:40:00 | 1586.90 | 1593.92 | 0.00 | ORB-short ORB[1592.30,1613.10] vol=1.8x ATR=5.67 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 1592.57 | 1593.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 1587.10 | 1590.72 | 0.00 | ORB-short ORB[1588.80,1600.00] vol=1.5x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:20:00 | 1581.84 | 1588.17 | 0.00 | T1 1.5R @ 1581.84 |
| Stop hit — per-position SL triggered | 2026-02-25 12:30:00 | 1587.10 | 1584.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 1573.80 | 1578.38 | 0.00 | ORB-short ORB[1576.90,1593.70] vol=2.3x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-27 12:30:00 | 1576.60 | 1576.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 1508.30 | 1514.65 | 0.00 | ORB-short ORB[1509.60,1526.00] vol=1.5x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:50:00 | 1500.39 | 1509.05 | 0.00 | T1 1.5R @ 1500.39 |
| Target hit | 2026-03-04 14:55:00 | 1487.70 | 1486.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 1487.50 | 1496.47 | 0.00 | ORB-short ORB[1488.00,1504.90] vol=1.7x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-03-11 10:05:00 | 1492.27 | 1493.98 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 1458.50 | 1468.02 | 0.00 | ORB-short ORB[1462.50,1480.70] vol=1.7x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-03-12 09:40:00 | 1463.47 | 1465.50 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 1520.00 | 1526.51 | 0.00 | ORB-short ORB[1523.40,1544.90] vol=2.4x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 1512.29 | 1524.17 | 0.00 | T1 1.5R @ 1512.29 |
| Stop hit — per-position SL triggered | 2026-04-16 11:10:00 | 1520.00 | 1519.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 1680.20 | 1663.70 | 0.00 | ORB-long ORB[1622.10,1644.80] vol=3.1x ATR=11.43 |
| Stop hit — per-position SL triggered | 2026-04-22 10:05:00 | 1668.77 | 1665.10 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:55:00 | 1744.80 | 1730.51 | 0.00 | ORB-long ORB[1709.50,1735.00] vol=3.2x ATR=8.41 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 1736.39 | 1732.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:00:00 | 1644.30 | 2026-02-17 10:15:00 | 1638.77 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-18 10:40:00 | 1634.00 | 2026-02-18 10:50:00 | 1637.84 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-24 09:40:00 | 1586.90 | 2026-02-24 09:45:00 | 1592.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1587.10 | 2026-02-25 11:20:00 | 1581.84 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1587.10 | 2026-02-25 12:30:00 | 1587.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:55:00 | 1573.80 | 2026-02-27 12:30:00 | 1576.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-03-04 09:30:00 | 1508.30 | 2026-03-04 09:50:00 | 1500.39 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-03-04 09:30:00 | 1508.30 | 2026-03-04 14:55:00 | 1487.70 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2026-03-11 09:35:00 | 1487.50 | 2026-03-11 10:05:00 | 1492.27 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-12 09:30:00 | 1458.50 | 2026-03-12 09:40:00 | 1463.47 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 09:35:00 | 1520.00 | 2026-04-16 09:45:00 | 1512.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-16 09:35:00 | 1520.00 | 2026-04-16 11:10:00 | 1520.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:55:00 | 1680.20 | 2026-04-22 10:05:00 | 1668.77 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2026-04-30 09:55:00 | 1744.80 | 2026-04-30 10:10:00 | 1736.39 | STOP_HIT | 1.00 | -0.48% |
