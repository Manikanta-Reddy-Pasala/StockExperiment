# Gland Pharma Ltd. (GLAND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1906.00
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 6
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 2.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.8% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.22% | 1.8% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.22% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 7 | 35.0% | 1 | 13 | 6 | 0.13% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1879.60 | 1881.84 | 0.00 | ORB-short ORB[1882.40,1896.00] vol=18.8x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:55:00 | 1871.73 | 1881.78 | 0.00 | T1 1.5R @ 1871.73 |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 1879.60 | 1881.43 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 1842.00 | 1835.94 | 0.00 | ORB-long ORB[1825.00,1841.40] vol=2.5x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 1851.91 | 1840.89 | 0.00 | T1 1.5R @ 1851.91 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 1842.00 | 1844.72 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1779.10 | 1783.45 | 0.00 | ORB-short ORB[1787.10,1803.00] vol=10.7x ATR=4.85 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 1783.95 | 1783.11 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 1808.30 | 1804.42 | 0.00 | ORB-long ORB[1791.60,1807.90] vol=3.8x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:40:00 | 1816.82 | 1807.81 | 0.00 | T1 1.5R @ 1816.82 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 1808.30 | 1808.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 1860.30 | 1851.80 | 0.00 | ORB-long ORB[1836.90,1854.90] vol=3.2x ATR=7.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:30:00 | 1871.95 | 1855.68 | 0.00 | T1 1.5R @ 1871.95 |
| Stop hit — per-position SL triggered | 2026-02-26 10:35:00 | 1860.30 | 1856.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:50:00 | 1743.60 | 1761.96 | 0.00 | ORB-short ORB[1761.00,1779.90] vol=2.1x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 12:20:00 | 1730.88 | 1746.73 | 0.00 | T1 1.5R @ 1730.88 |
| Target hit | 2026-03-04 15:20:00 | 1709.70 | 1727.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 1691.10 | 1714.58 | 0.00 | ORB-short ORB[1711.00,1732.90] vol=1.8x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-03-05 12:05:00 | 1696.94 | 1710.39 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 1683.70 | 1692.21 | 0.00 | ORB-short ORB[1685.00,1705.80] vol=2.0x ATR=6.06 |
| Stop hit — per-position SL triggered | 2026-03-10 14:45:00 | 1689.76 | 1686.31 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:05:00 | 1707.80 | 1700.93 | 0.00 | ORB-long ORB[1673.70,1695.20] vol=1.6x ATR=7.91 |
| Stop hit — per-position SL triggered | 2026-03-11 10:25:00 | 1699.89 | 1701.10 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 1587.40 | 1595.27 | 0.00 | ORB-short ORB[1600.40,1618.30] vol=1.5x ATR=5.79 |
| Stop hit — per-position SL triggered | 2026-03-16 11:35:00 | 1593.19 | 1594.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 1706.50 | 1691.42 | 0.00 | ORB-long ORB[1680.10,1703.40] vol=2.0x ATR=5.94 |
| Stop hit — per-position SL triggered | 2026-03-27 11:45:00 | 1700.56 | 1696.61 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1768.30 | 1757.32 | 0.00 | ORB-long ORB[1745.00,1760.90] vol=2.3x ATR=5.67 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 1762.63 | 1759.19 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1789.20 | 1785.02 | 0.00 | ORB-long ORB[1770.00,1788.70] vol=2.1x ATR=5.82 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 1783.38 | 1785.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 1816.00 | 1803.90 | 0.00 | ORB-long ORB[1782.00,1807.90] vol=4.6x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:50:00 | 1826.59 | 1810.08 | 0.00 | T1 1.5R @ 1826.59 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 1816.00 | 1815.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:45:00 | 1879.60 | 2026-02-10 10:55:00 | 1871.73 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-10 10:45:00 | 1879.60 | 2026-02-10 14:15:00 | 1879.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:40:00 | 1842.00 | 2026-02-18 09:45:00 | 1851.91 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-18 09:40:00 | 1842.00 | 2026-02-18 09:55:00 | 1842.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1779.10 | 2026-02-24 09:40:00 | 1783.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-25 09:50:00 | 1808.30 | 2026-02-25 10:40:00 | 1816.82 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-25 09:50:00 | 1808.30 | 2026-02-25 10:55:00 | 1808.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:15:00 | 1860.30 | 2026-02-26 10:30:00 | 1871.95 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-26 10:15:00 | 1860.30 | 2026-02-26 10:35:00 | 1860.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:50:00 | 1743.60 | 2026-03-04 12:20:00 | 1730.88 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-04 09:50:00 | 1743.60 | 2026-03-04 15:20:00 | 1709.70 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2026-03-05 10:55:00 | 1691.10 | 2026-03-05 12:05:00 | 1696.94 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-10 10:50:00 | 1683.70 | 2026-03-10 14:45:00 | 1689.76 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-11 10:05:00 | 1707.80 | 2026-03-11 10:25:00 | 1699.89 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-16 11:15:00 | 1587.40 | 2026-03-16 11:35:00 | 1593.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-27 11:00:00 | 1706.50 | 2026-03-27 11:45:00 | 1700.56 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 09:30:00 | 1768.30 | 2026-04-10 10:00:00 | 1762.63 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 10:00:00 | 1789.20 | 2026-04-21 10:05:00 | 1783.38 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1816.00 | 2026-04-23 09:50:00 | 1826.59 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1816.00 | 2026-04-23 10:10:00 | 1816.00 | STOP_HIT | 0.50 | 0.00% |
