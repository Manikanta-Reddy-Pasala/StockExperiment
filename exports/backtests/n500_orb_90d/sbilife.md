# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1871.10
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 15
- **Target hits / Stop hits / Partials:** 2 / 15 / 6
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 0.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.07% | 1.0% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.07% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.04% | -0.4% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.04% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 8 | 34.8% | 2 | 15 | 6 | 0.03% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 2021.80 | 2025.32 | 0.00 | ORB-short ORB[2022.10,2039.10] vol=3.0x ATR=3.93 |
| Stop hit — per-position SL triggered | 2026-02-11 11:40:00 | 2025.73 | 2023.68 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 2047.70 | 2044.27 | 0.00 | ORB-long ORB[2034.40,2043.10] vol=3.1x ATR=3.52 |
| Stop hit — per-position SL triggered | 2026-02-18 10:00:00 | 2044.18 | 2044.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:50:00 | 2051.50 | 2054.77 | 0.00 | ORB-short ORB[2053.50,2068.00] vol=1.6x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:20:00 | 2045.50 | 2052.91 | 0.00 | T1 1.5R @ 2045.50 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 2051.50 | 2051.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 2070.20 | 2056.34 | 0.00 | ORB-long ORB[2036.00,2051.70] vol=1.8x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-20 12:10:00 | 2066.72 | 2060.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 2099.80 | 2093.26 | 0.00 | ORB-long ORB[2075.60,2095.00] vol=3.1x ATR=3.89 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 2095.91 | 2095.11 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 2055.80 | 2066.30 | 0.00 | ORB-short ORB[2064.00,2093.80] vol=1.6x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:05:00 | 2048.39 | 2057.63 | 0.00 | T1 1.5R @ 2048.39 |
| Target hit | 2026-02-27 14:40:00 | 2048.80 | 2048.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-03-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:10:00 | 1924.10 | 1927.96 | 0.00 | ORB-short ORB[1930.90,1950.00] vol=8.0x ATR=5.08 |
| Stop hit — per-position SL triggered | 2026-03-05 10:35:00 | 1929.18 | 1926.72 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1946.20 | 1934.22 | 0.00 | ORB-long ORB[1926.50,1939.80] vol=1.6x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:35:00 | 1952.92 | 1935.85 | 0.00 | T1 1.5R @ 1952.92 |
| Target hit | 2026-03-10 15:20:00 | 1965.00 | 1957.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 1906.40 | 1914.67 | 0.00 | ORB-short ORB[1913.20,1938.90] vol=1.5x ATR=6.02 |
| Stop hit — per-position SL triggered | 2026-03-13 10:00:00 | 1912.42 | 1914.49 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:05:00 | 1929.50 | 1919.15 | 0.00 | ORB-long ORB[1901.50,1919.40] vol=1.8x ATR=6.42 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 1923.08 | 1923.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:05:00 | 1817.10 | 1838.41 | 0.00 | ORB-short ORB[1842.00,1866.50] vol=2.2x ATR=5.96 |
| Stop hit — per-position SL triggered | 2026-03-24 12:00:00 | 1823.06 | 1832.42 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:55:00 | 1861.70 | 1859.78 | 0.00 | ORB-long ORB[1839.20,1861.40] vol=1.5x ATR=4.08 |
| Stop hit — per-position SL triggered | 2026-03-25 12:55:00 | 1857.62 | 1861.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:30:00 | 1786.60 | 1774.95 | 0.00 | ORB-long ORB[1757.50,1781.80] vol=1.8x ATR=7.32 |
| Stop hit — per-position SL triggered | 2026-04-06 09:40:00 | 1779.28 | 1776.30 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 1913.40 | 1905.09 | 0.00 | ORB-long ORB[1895.50,1910.90] vol=4.2x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:10:00 | 1922.13 | 1906.14 | 0.00 | T1 1.5R @ 1922.13 |
| Stop hit — per-position SL triggered | 2026-04-13 15:00:00 | 1913.40 | 1913.51 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 1882.80 | 1904.30 | 0.00 | ORB-short ORB[1906.10,1930.60] vol=2.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 1887.58 | 1903.88 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 1835.00 | 1828.20 | 0.00 | ORB-long ORB[1810.10,1834.80] vol=2.4x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:35:00 | 1842.81 | 1829.43 | 0.00 | T1 1.5R @ 1842.81 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 1835.00 | 1832.64 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1847.00 | 1843.09 | 0.00 | ORB-long ORB[1825.90,1839.00] vol=5.8x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:40:00 | 1852.68 | 1844.06 | 0.00 | T1 1.5R @ 1852.68 |
| Stop hit — per-position SL triggered | 2026-05-06 12:20:00 | 1847.00 | 1845.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:00:00 | 2021.80 | 2026-02-11 11:40:00 | 2025.73 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-18 09:50:00 | 2047.70 | 2026-02-18 10:00:00 | 2044.18 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-19 09:50:00 | 2051.50 | 2026-02-19 10:20:00 | 2045.50 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-19 09:50:00 | 2051.50 | 2026-02-19 11:00:00 | 2051.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 11:05:00 | 2070.20 | 2026-02-20 12:10:00 | 2066.72 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-02-26 10:15:00 | 2099.80 | 2026-02-26 10:55:00 | 2095.91 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2055.80 | 2026-02-27 11:05:00 | 2048.39 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2055.80 | 2026-02-27 14:40:00 | 2048.80 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-05 10:10:00 | 1924.10 | 2026-03-05 10:35:00 | 1929.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-10 11:15:00 | 1946.20 | 2026-03-10 11:35:00 | 1952.92 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-03-10 11:15:00 | 1946.20 | 2026-03-10 15:20:00 | 1965.00 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2026-03-13 09:55:00 | 1906.40 | 2026-03-13 10:00:00 | 1912.42 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-17 10:05:00 | 1929.50 | 2026-03-17 10:30:00 | 1923.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-24 11:05:00 | 1817.10 | 2026-03-24 12:00:00 | 1823.06 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-25 10:55:00 | 1861.70 | 2026-03-25 12:55:00 | 1857.62 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-06 09:30:00 | 1786.60 | 2026-04-06 09:40:00 | 1779.28 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-13 10:50:00 | 1913.40 | 2026-04-13 11:10:00 | 1922.13 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-13 10:50:00 | 1913.40 | 2026-04-13 15:00:00 | 1913.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:50:00 | 1882.80 | 2026-04-22 10:55:00 | 1887.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-04 09:30:00 | 1835.00 | 2026-05-04 09:35:00 | 1842.81 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-05-04 09:30:00 | 1835.00 | 2026-05-04 10:00:00 | 1835.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 11:15:00 | 1847.00 | 2026-05-06 11:40:00 | 1852.68 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-05-06 11:15:00 | 1847.00 | 2026-05-06 12:20:00 | 1847.00 | STOP_HIT | 0.50 | 0.00% |
