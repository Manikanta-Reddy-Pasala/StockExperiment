# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1820.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 6
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 2.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.20% | 2.6% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.20% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 2 | 7 | 2 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 2 | 7 | 2 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 10 | 41.7% | 4 | 14 | 6 | 0.10% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 1905.00 | 1886.25 | 0.00 | ORB-long ORB[1875.00,1895.00] vol=1.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 1900.77 | 1887.55 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1948.90 | 1953.38 | 0.00 | ORB-short ORB[1951.80,1960.80] vol=4.2x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 1952.78 | 1951.03 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:50:00 | 1954.60 | 1953.45 | 0.00 | ORB-long ORB[1936.20,1954.40] vol=7.5x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:00:00 | 1961.92 | 1954.08 | 0.00 | T1 1.5R @ 1961.92 |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 1954.60 | 1955.80 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1930.10 | 1922.52 | 0.00 | ORB-long ORB[1898.00,1913.40] vol=1.5x ATR=5.50 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 1924.60 | 1923.50 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:00:00 | 1915.30 | 1924.68 | 0.00 | ORB-short ORB[1930.00,1945.00] vol=1.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:05:00 | 1908.97 | 1921.54 | 0.00 | T1 1.5R @ 1908.97 |
| Target hit | 2026-02-27 14:40:00 | 1911.70 | 1910.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 1845.90 | 1851.35 | 0.00 | ORB-short ORB[1852.60,1868.70] vol=5.2x ATR=4.01 |
| Stop hit — per-position SL triggered | 2026-03-05 12:55:00 | 1849.91 | 1847.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1861.50 | 1865.74 | 0.00 | ORB-short ORB[1863.30,1874.90] vol=2.5x ATR=5.03 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 1866.53 | 1865.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:20:00 | 1869.90 | 1875.33 | 0.00 | ORB-short ORB[1871.70,1894.00] vol=3.4x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-03-11 10:40:00 | 1874.70 | 1875.18 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:25:00 | 1847.80 | 1855.44 | 0.00 | ORB-short ORB[1854.60,1872.90] vol=1.5x ATR=4.19 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 1851.99 | 1854.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:35:00 | 1802.80 | 1815.91 | 0.00 | ORB-short ORB[1832.50,1841.70] vol=1.6x ATR=5.03 |
| Stop hit — per-position SL triggered | 2026-03-18 11:40:00 | 1807.83 | 1807.78 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:40:00 | 1723.00 | 1734.01 | 0.00 | ORB-short ORB[1736.50,1758.00] vol=2.5x ATR=7.01 |
| Stop hit — per-position SL triggered | 2026-03-24 09:55:00 | 1730.01 | 1731.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:55:00 | 1759.50 | 1757.71 | 0.00 | ORB-long ORB[1736.00,1754.90] vol=3.9x ATR=4.18 |
| Stop hit — per-position SL triggered | 2026-03-25 11:05:00 | 1755.32 | 1757.71 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1815.70 | 1808.85 | 0.00 | ORB-long ORB[1795.00,1815.00] vol=1.6x ATR=9.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:00:00 | 1829.80 | 1815.83 | 0.00 | T1 1.5R @ 1829.80 |
| Target hit | 2026-04-15 12:45:00 | 1834.00 | 1834.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 1786.20 | 1802.29 | 0.00 | ORB-short ORB[1797.60,1819.30] vol=9.4x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:50:00 | 1780.59 | 1797.00 | 0.00 | T1 1.5R @ 1780.59 |
| Target hit | 2026-04-24 15:20:00 | 1768.90 | 1785.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 1797.80 | 1787.76 | 0.00 | ORB-long ORB[1775.00,1784.10] vol=1.9x ATR=5.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 1806.23 | 1793.80 | 0.00 | T1 1.5R @ 1806.23 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 1797.80 | 1795.49 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1764.90 | 1758.33 | 0.00 | ORB-long ORB[1745.00,1759.40] vol=4.4x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-05-05 09:35:00 | 1759.50 | 1762.57 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 1795.90 | 1790.01 | 0.00 | ORB-long ORB[1775.30,1795.10] vol=7.4x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 1803.82 | 1793.81 | 0.00 | T1 1.5R @ 1803.82 |
| Target hit | 2026-05-06 15:20:00 | 1810.00 | 1802.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1827.00 | 1821.79 | 0.00 | ORB-long ORB[1810.50,1821.60] vol=2.2x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 1822.54 | 1823.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 1905.00 | 2026-02-10 11:10:00 | 1900.77 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1948.90 | 2026-02-19 11:00:00 | 1952.78 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-23 09:50:00 | 1954.60 | 2026-02-23 10:00:00 | 1961.92 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-23 09:50:00 | 1954.60 | 2026-02-23 10:15:00 | 1954.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1930.10 | 2026-02-25 11:20:00 | 1924.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-27 11:00:00 | 1915.30 | 2026-02-27 11:05:00 | 1908.97 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-27 11:00:00 | 1915.30 | 2026-02-27 14:40:00 | 1911.70 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1845.90 | 2026-03-05 12:55:00 | 1849.91 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1861.50 | 2026-03-06 10:50:00 | 1866.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-11 10:20:00 | 1869.90 | 2026-03-11 10:40:00 | 1874.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-12 10:25:00 | 1847.80 | 2026-03-12 11:00:00 | 1851.99 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-18 10:35:00 | 1802.80 | 2026-03-18 11:40:00 | 1807.83 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-24 09:40:00 | 1723.00 | 2026-03-24 09:55:00 | 1730.01 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-25 10:55:00 | 1759.50 | 2026-03-25 11:05:00 | 1755.32 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-15 09:35:00 | 1815.70 | 2026-04-15 10:00:00 | 1829.80 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-15 09:35:00 | 1815.70 | 2026-04-15 12:45:00 | 1834.00 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1786.20 | 2026-04-24 11:50:00 | 1780.59 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-24 11:15:00 | 1786.20 | 2026-04-24 15:20:00 | 1768.90 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2026-04-29 09:40:00 | 1797.80 | 2026-04-29 10:15:00 | 1806.23 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-29 09:40:00 | 1797.80 | 2026-04-29 11:00:00 | 1797.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 1764.90 | 2026-05-05 09:35:00 | 1759.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-06 10:05:00 | 1795.90 | 2026-05-06 10:55:00 | 1803.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-05-06 10:05:00 | 1795.90 | 2026-05-06 15:20:00 | 1810.00 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2026-05-07 09:45:00 | 1827.00 | 2026-05-07 09:50:00 | 1822.54 | STOP_HIT | 1.00 | -0.24% |
