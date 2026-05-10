# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1710.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 5
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 1.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.16% | 2.0% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.16% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.10% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 1534.60 | 1541.42 | 0.00 | ORB-short ORB[1545.10,1557.40] vol=2.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 1538.05 | 1539.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1525.00 | 1528.54 | 0.00 | ORB-short ORB[1528.00,1540.00] vol=4.6x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 1518.86 | 1527.12 | 0.00 | T1 1.5R @ 1518.86 |
| Target hit | 2026-02-19 15:20:00 | 1504.00 | 1517.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1510.70 | 1517.56 | 0.00 | ORB-short ORB[1519.30,1532.10] vol=2.0x ATR=4.55 |
| Stop hit — per-position SL triggered | 2026-02-23 11:40:00 | 1515.25 | 1516.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1524.40 | 1515.36 | 0.00 | ORB-long ORB[1510.00,1519.00] vol=1.7x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1520.35 | 1517.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 1527.20 | 1529.14 | 0.00 | ORB-short ORB[1528.90,1542.50] vol=2.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 1520.87 | 1528.03 | 0.00 | T1 1.5R @ 1520.87 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1527.20 | 1527.67 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 1498.50 | 1505.85 | 0.00 | ORB-short ORB[1500.50,1512.70] vol=2.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2026-03-11 10:30:00 | 1502.78 | 1505.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 1448.90 | 1453.08 | 0.00 | ORB-short ORB[1458.10,1470.40] vol=6.7x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 1454.42 | 1452.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 1458.00 | 1443.81 | 0.00 | ORB-long ORB[1427.30,1444.20] vol=1.8x ATR=7.47 |
| Stop hit — per-position SL triggered | 2026-03-17 09:40:00 | 1450.53 | 1444.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 1477.70 | 1470.14 | 0.00 | ORB-long ORB[1449.00,1466.80] vol=3.0x ATR=6.12 |
| Stop hit — per-position SL triggered | 2026-03-20 09:45:00 | 1471.58 | 1470.40 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 1688.70 | 1671.53 | 0.00 | ORB-long ORB[1652.50,1676.00] vol=1.6x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:05:00 | 1698.28 | 1673.95 | 0.00 | T1 1.5R @ 1698.28 |
| Stop hit — per-position SL triggered | 2026-04-10 13:25:00 | 1688.70 | 1687.55 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 1699.00 | 1704.45 | 0.00 | ORB-short ORB[1702.10,1720.00] vol=2.0x ATR=6.97 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1705.97 | 1701.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 1744.00 | 1731.93 | 0.00 | ORB-long ORB[1712.40,1737.30] vol=2.3x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:50:00 | 1753.22 | 1735.16 | 0.00 | T1 1.5R @ 1753.22 |
| Stop hit — per-position SL triggered | 2026-04-21 10:55:00 | 1744.00 | 1735.41 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1712.30 | 1722.20 | 0.00 | ORB-short ORB[1713.20,1736.90] vol=1.8x ATR=4.59 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 1716.89 | 1722.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 1709.80 | 1716.12 | 0.00 | ORB-short ORB[1711.80,1735.50] vol=2.6x ATR=6.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:20:00 | 1700.25 | 1714.79 | 0.00 | T1 1.5R @ 1700.25 |
| Target hit | 2026-04-24 15:20:00 | 1691.20 | 1697.67 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:50:00 | 1534.60 | 2026-02-18 11:00:00 | 1538.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1525.00 | 2026-02-19 11:15:00 | 1518.86 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1525.00 | 2026-02-19 15:20:00 | 1504.00 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2026-02-23 10:55:00 | 1510.70 | 2026-02-23 11:40:00 | 1515.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1524.40 | 2026-02-24 10:15:00 | 1520.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1527.20 | 2026-02-27 10:50:00 | 1520.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1527.20 | 2026-02-27 11:15:00 | 1527.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:25:00 | 1498.50 | 2026-03-11 10:30:00 | 1502.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-13 10:50:00 | 1448.90 | 2026-03-13 11:20:00 | 1454.42 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-17 09:35:00 | 1458.00 | 2026-03-17 09:40:00 | 1450.53 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-03-20 09:35:00 | 1477.70 | 2026-03-20 09:45:00 | 1471.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1688.70 | 2026-04-10 11:05:00 | 1698.28 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1688.70 | 2026-04-10 13:25:00 | 1688.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 10:05:00 | 1699.00 | 2026-04-15 11:15:00 | 1705.97 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-21 10:40:00 | 1744.00 | 2026-04-21 10:50:00 | 1753.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-21 10:40:00 | 1744.00 | 2026-04-21 10:55:00 | 1744.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1712.30 | 2026-04-23 11:25:00 | 1716.89 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1709.80 | 2026-04-24 10:20:00 | 1700.25 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1709.80 | 2026-04-24 15:20:00 | 1691.20 | TARGET_HIT | 0.50 | 1.09% |
