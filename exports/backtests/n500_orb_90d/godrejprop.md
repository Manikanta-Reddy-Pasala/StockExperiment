# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1874.80
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 6
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 2.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.26% | 2.8% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.26% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.16% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 1821.40 | 1818.55 | 0.00 | ORB-long ORB[1801.50,1820.60] vol=3.9x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:45:00 | 1831.11 | 1819.75 | 0.00 | T1 1.5R @ 1831.11 |
| Stop hit — per-position SL triggered | 2026-02-10 13:00:00 | 1821.40 | 1821.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:00:00 | 1818.40 | 1820.28 | 0.00 | ORB-short ORB[1819.80,1829.90] vol=1.9x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1823.12 | 1820.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 1869.50 | 1874.56 | 0.00 | ORB-short ORB[1869.70,1887.00] vol=6.8x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 1875.85 | 1874.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 1803.00 | 1807.93 | 0.00 | ORB-short ORB[1806.30,1824.90] vol=2.4x ATR=7.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:00:00 | 1791.73 | 1805.96 | 0.00 | T1 1.5R @ 1791.73 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 1803.00 | 1805.52 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1826.10 | 1834.23 | 0.00 | ORB-short ORB[1827.50,1848.90] vol=2.2x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:20:00 | 1816.40 | 1832.65 | 0.00 | T1 1.5R @ 1816.40 |
| Target hit | 2026-02-23 14:40:00 | 1821.10 | 1820.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 1804.30 | 1807.05 | 0.00 | ORB-short ORB[1805.10,1831.20] vol=3.9x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 1795.91 | 1804.84 | 0.00 | T1 1.5R @ 1795.91 |
| Target hit | 2026-02-24 15:20:00 | 1777.00 | 1778.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1786.00 | 1794.93 | 0.00 | ORB-short ORB[1788.00,1810.00] vol=1.5x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 1778.17 | 1792.97 | 0.00 | T1 1.5R @ 1778.17 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 1786.00 | 1788.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 1578.30 | 1588.74 | 0.00 | ORB-short ORB[1586.90,1606.00] vol=1.9x ATR=6.50 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 1584.80 | 1588.04 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1601.80 | 1583.07 | 0.00 | ORB-long ORB[1562.70,1586.00] vol=1.7x ATR=8.45 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 1593.35 | 1587.82 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 1813.10 | 1805.49 | 0.00 | ORB-long ORB[1796.10,1812.70] vol=1.5x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 1807.26 | 1805.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 1843.90 | 1835.01 | 0.00 | ORB-long ORB[1822.70,1839.30] vol=2.7x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1838.85 | 1835.16 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 1861.50 | 1849.29 | 0.00 | ORB-long ORB[1831.00,1853.90] vol=2.8x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 1872.26 | 1855.53 | 0.00 | T1 1.5R @ 1872.26 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 1861.50 | 1857.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:15:00 | 1821.40 | 2026-02-10 11:45:00 | 1831.11 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-10 10:15:00 | 1821.40 | 2026-02-10 13:00:00 | 1821.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:00:00 | 1818.40 | 2026-02-11 10:15:00 | 1823.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 09:40:00 | 1869.50 | 2026-02-19 09:50:00 | 1875.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-20 10:45:00 | 1803.00 | 2026-02-20 11:00:00 | 1791.73 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-02-20 10:45:00 | 1803.00 | 2026-02-20 11:30:00 | 1803.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1826.10 | 2026-02-23 11:20:00 | 1816.40 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1826.10 | 2026-02-23 14:40:00 | 1821.10 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1804.30 | 2026-02-24 11:45:00 | 1795.91 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1804.30 | 2026-02-24 15:20:00 | 1777.00 | TARGET_HIT | 0.50 | 1.51% |
| SELL | retest1 | 2026-02-26 10:45:00 | 1786.00 | 2026-02-26 10:55:00 | 1778.17 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-26 10:45:00 | 1786.00 | 2026-02-26 11:25:00 | 1786.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 1578.30 | 2026-03-13 10:30:00 | 1584.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1601.80 | 2026-03-20 09:50:00 | 1593.35 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-22 10:35:00 | 1813.10 | 2026-04-22 10:45:00 | 1807.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 11:15:00 | 1843.90 | 2026-04-28 11:20:00 | 1838.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1861.50 | 2026-04-29 10:00:00 | 1872.26 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1861.50 | 2026-04-29 10:45:00 | 1861.50 | STOP_HIT | 0.50 | 0.00% |
