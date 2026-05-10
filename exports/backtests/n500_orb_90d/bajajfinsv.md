# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1814.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 9
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 3.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.08% | 1.3% |
| BUY @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.08% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.20% | 2.6% |
| SELL @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.20% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 13 | 44.8% | 4 | 16 | 9 | 0.13% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 2038.80 | 2031.72 | 0.00 | ORB-long ORB[2027.90,2034.90] vol=2.6x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:45:00 | 2043.53 | 2034.36 | 0.00 | T1 1.5R @ 2043.53 |
| Stop hit — per-position SL triggered | 2026-02-11 11:55:00 | 2038.80 | 2040.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:30:00 | 2033.80 | 2027.62 | 0.00 | ORB-long ORB[2012.10,2028.60] vol=2.5x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-02-12 10:40:00 | 2030.47 | 2028.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:10:00 | 2022.90 | 2027.70 | 0.00 | ORB-short ORB[2025.00,2035.00] vol=1.8x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:20:00 | 2017.28 | 2027.10 | 0.00 | T1 1.5R @ 2017.28 |
| Stop hit — per-position SL triggered | 2026-02-13 13:40:00 | 2022.90 | 2022.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 2045.30 | 2037.01 | 0.00 | ORB-long ORB[2022.20,2034.00] vol=1.8x ATR=4.28 |
| Stop hit — per-position SL triggered | 2026-02-16 10:45:00 | 2041.02 | 2038.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 2061.20 | 2056.10 | 0.00 | ORB-long ORB[2040.00,2058.50] vol=1.5x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-18 12:35:00 | 2057.75 | 2058.96 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 2048.90 | 2055.29 | 0.00 | ORB-short ORB[2055.80,2068.00] vol=2.5x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-02-19 11:10:00 | 2051.63 | 2055.00 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 2045.30 | 2036.18 | 0.00 | ORB-long ORB[2021.10,2042.00] vol=1.7x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:45:00 | 2051.27 | 2038.99 | 0.00 | T1 1.5R @ 2051.27 |
| Target hit | 2026-02-20 15:20:00 | 2055.40 | 2054.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 2061.20 | 2054.72 | 0.00 | ORB-long ORB[2039.80,2060.50] vol=1.7x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:40:00 | 2067.90 | 2057.90 | 0.00 | T1 1.5R @ 2067.90 |
| Stop hit — per-position SL triggered | 2026-02-23 10:20:00 | 2061.20 | 2061.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:10:00 | 2071.40 | 2061.26 | 0.00 | ORB-long ORB[2045.00,2057.80] vol=1.5x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-02-25 10:35:00 | 2066.61 | 2064.84 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:15:00 | 2062.00 | 2053.25 | 0.00 | ORB-long ORB[2047.80,2057.00] vol=3.9x ATR=4.36 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 2057.64 | 2054.57 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 2017.10 | 2021.42 | 0.00 | ORB-short ORB[2021.00,2039.60] vol=2.2x ATR=4.13 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 2021.23 | 2021.00 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1874.00 | 1887.21 | 0.00 | ORB-short ORB[1890.40,1908.90] vol=2.8x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:50:00 | 1867.37 | 1877.49 | 0.00 | T1 1.5R @ 1867.37 |
| Stop hit — per-position SL triggered | 2026-03-06 12:00:00 | 1874.00 | 1877.38 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:50:00 | 1841.00 | 1849.38 | 0.00 | ORB-short ORB[1853.70,1874.80] vol=3.3x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:00:00 | 1835.34 | 1844.99 | 0.00 | T1 1.5R @ 1835.34 |
| Target hit | 2026-03-11 15:20:00 | 1796.50 | 1815.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1767.30 | 1772.69 | 0.00 | ORB-short ORB[1770.60,1794.40] vol=1.7x ATR=5.58 |
| Stop hit — per-position SL triggered | 2026-03-12 10:25:00 | 1772.88 | 1772.36 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:55:00 | 1675.10 | 1682.53 | 0.00 | ORB-short ORB[1676.30,1700.00] vol=1.7x ATR=5.00 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1680.10 | 1681.92 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:45:00 | 1830.50 | 1828.22 | 0.00 | ORB-long ORB[1811.70,1828.80] vol=2.4x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:25:00 | 1836.79 | 1829.35 | 0.00 | T1 1.5R @ 1836.79 |
| Target hit | 2026-04-17 15:20:00 | 1839.70 | 1833.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 1776.10 | 1786.80 | 0.00 | ORB-short ORB[1780.00,1799.90] vol=2.6x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:10:00 | 1770.14 | 1780.45 | 0.00 | T1 1.5R @ 1770.14 |
| Target hit | 2026-04-24 14:25:00 | 1774.40 | 1774.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2026-04-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:30:00 | 1773.00 | 1771.85 | 0.00 | ORB-long ORB[1757.70,1772.10] vol=1.7x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:50:00 | 1779.95 | 1772.42 | 0.00 | T1 1.5R @ 1779.95 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1773.00 | 1772.86 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1782.70 | 1780.84 | 0.00 | ORB-long ORB[1767.70,1778.10] vol=3.7x ATR=4.82 |
| Stop hit — per-position SL triggered | 2026-04-29 11:05:00 | 1777.88 | 1780.82 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 1751.90 | 1776.55 | 0.00 | ORB-short ORB[1770.70,1792.20] vol=1.9x ATR=4.86 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 1756.76 | 1775.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:35:00 | 2038.80 | 2026-02-11 10:45:00 | 2043.53 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2026-02-11 10:35:00 | 2038.80 | 2026-02-11 11:55:00 | 2038.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:30:00 | 2033.80 | 2026-02-12 10:40:00 | 2030.47 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-13 11:10:00 | 2022.90 | 2026-02-13 11:20:00 | 2017.28 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-13 11:10:00 | 2022.90 | 2026-02-13 13:40:00 | 2022.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:15:00 | 2045.30 | 2026-02-16 10:45:00 | 2041.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-18 10:35:00 | 2061.20 | 2026-02-18 12:35:00 | 2057.75 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-19 11:00:00 | 2048.90 | 2026-02-19 11:10:00 | 2051.63 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2026-02-20 11:05:00 | 2045.30 | 2026-02-20 11:45:00 | 2051.27 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-20 11:05:00 | 2045.30 | 2026-02-20 15:20:00 | 2055.40 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-23 09:30:00 | 2061.20 | 2026-02-23 09:40:00 | 2067.90 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-23 09:30:00 | 2061.20 | 2026-02-23 10:20:00 | 2061.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:10:00 | 2071.40 | 2026-02-25 10:35:00 | 2066.61 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-26 11:15:00 | 2062.00 | 2026-02-26 11:30:00 | 2057.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2017.10 | 2026-02-27 11:15:00 | 2021.23 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1874.00 | 2026-03-06 11:50:00 | 1867.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1874.00 | 2026-03-06 12:00:00 | 1874.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:50:00 | 1841.00 | 2026-03-11 11:00:00 | 1835.34 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-11 10:50:00 | 1841.00 | 2026-03-11 15:20:00 | 1796.50 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2026-03-12 10:15:00 | 1767.30 | 2026-03-12 10:25:00 | 1772.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-23 10:55:00 | 1675.10 | 2026-03-23 11:05:00 | 1680.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 10:45:00 | 1830.50 | 2026-04-17 12:25:00 | 1836.79 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-04-17 10:45:00 | 1830.50 | 2026-04-17 15:20:00 | 1839.70 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-24 10:55:00 | 1776.10 | 2026-04-24 12:10:00 | 1770.14 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-24 10:55:00 | 1776.10 | 2026-04-24 14:25:00 | 1774.40 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-04-28 10:30:00 | 1773.00 | 2026-04-28 10:50:00 | 1779.95 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-28 10:30:00 | 1773.00 | 2026-04-28 11:15:00 | 1773.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1782.70 | 2026-04-29 11:05:00 | 1777.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-30 11:15:00 | 1751.90 | 2026-04-30 11:25:00 | 1756.76 | STOP_HIT | 1.00 | -0.28% |
