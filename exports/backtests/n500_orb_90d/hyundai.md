# Hyundai Motor India Ltd. (HYUNDAI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1833.10
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
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 7
- **Avg / median % per leg:** 0.31% / 0.00%
- **Sum % (uncompounded):** 6.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.18% | 1.8% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.18% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.43% | 4.7% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.43% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 2 | 12 | 7 | 0.31% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 2170.30 | 2183.59 | 0.00 | ORB-short ORB[2173.30,2189.30] vol=1.6x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 2175.67 | 2178.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 2169.70 | 2173.31 | 0.00 | ORB-short ORB[2176.10,2190.00] vol=12.1x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 2173.18 | 2173.23 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 2175.00 | 2163.59 | 0.00 | ORB-long ORB[2156.00,2168.00] vol=2.3x ATR=4.16 |
| Stop hit — per-position SL triggered | 2026-02-17 10:20:00 | 2170.84 | 2165.16 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 2185.90 | 2176.15 | 0.00 | ORB-long ORB[2162.60,2180.00] vol=1.8x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:25:00 | 2194.72 | 2177.26 | 0.00 | T1 1.5R @ 2194.72 |
| Stop hit — per-position SL triggered | 2026-02-20 11:50:00 | 2185.90 | 2179.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:40:00 | 2201.40 | 2232.17 | 0.00 | ORB-short ORB[2239.10,2265.50] vol=3.6x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:50:00 | 2188.52 | 2223.40 | 0.00 | T1 1.5R @ 2188.52 |
| Target hit | 2026-02-24 15:20:00 | 2156.50 | 2185.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 2067.60 | 2076.50 | 0.00 | ORB-short ORB[2072.50,2098.00] vol=3.0x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 2073.38 | 2075.47 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 2069.10 | 2087.24 | 0.00 | ORB-short ORB[2077.80,2100.00] vol=1.9x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 2061.16 | 2081.14 | 0.00 | T1 1.5R @ 2061.16 |
| Target hit | 2026-03-11 15:20:00 | 2020.00 | 2023.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 11:00:00 | 2004.90 | 1995.05 | 0.00 | ORB-long ORB[1981.20,2003.80] vol=3.9x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-03-13 12:25:00 | 1999.12 | 1996.95 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 1952.30 | 1960.57 | 0.00 | ORB-short ORB[1954.90,1968.90] vol=2.2x ATR=5.14 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 1957.44 | 1960.04 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 1777.50 | 1761.70 | 0.00 | ORB-long ORB[1743.30,1765.00] vol=1.8x ATR=7.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:45:00 | 1789.48 | 1767.53 | 0.00 | T1 1.5R @ 1789.48 |
| Stop hit — per-position SL triggered | 2026-04-08 11:30:00 | 1777.50 | 1777.47 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:55:00 | 1788.60 | 1779.99 | 0.00 | ORB-long ORB[1768.80,1785.00] vol=1.6x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:20:00 | 1798.76 | 1785.08 | 0.00 | T1 1.5R @ 1798.76 |
| Stop hit — per-position SL triggered | 2026-04-15 11:45:00 | 1788.60 | 1790.56 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 1832.10 | 1812.30 | 0.00 | ORB-long ORB[1790.00,1812.30] vol=1.9x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 1842.85 | 1816.88 | 0.00 | T1 1.5R @ 1842.85 |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 1832.10 | 1827.35 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 1799.90 | 1803.22 | 0.00 | ORB-short ORB[1801.10,1815.50] vol=1.9x ATR=4.16 |
| Stop hit — per-position SL triggered | 2026-04-30 12:45:00 | 1804.06 | 1802.56 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 1810.70 | 1822.60 | 0.00 | ORB-short ORB[1815.10,1840.40] vol=1.8x ATR=6.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:10:00 | 1801.15 | 1816.20 | 0.00 | T1 1.5R @ 1801.15 |
| Stop hit — per-position SL triggered | 2026-05-05 12:30:00 | 1810.70 | 1813.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:45:00 | 2170.30 | 2026-02-10 11:00:00 | 2175.67 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-11 11:00:00 | 2169.70 | 2026-02-11 11:20:00 | 2173.18 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-17 10:15:00 | 2175.00 | 2026-02-17 10:20:00 | 2170.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 11:15:00 | 2185.90 | 2026-02-20 11:25:00 | 2194.72 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-20 11:15:00 | 2185.90 | 2026-02-20 11:50:00 | 2185.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 10:40:00 | 2201.40 | 2026-02-24 10:50:00 | 2188.52 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-24 10:40:00 | 2201.40 | 2026-02-24 15:20:00 | 2156.50 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2026-03-05 10:55:00 | 2067.60 | 2026-03-05 11:00:00 | 2073.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-11 10:40:00 | 2069.10 | 2026-03-11 11:15:00 | 2061.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-11 10:40:00 | 2069.10 | 2026-03-11 15:20:00 | 2020.00 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2026-03-13 11:00:00 | 2004.90 | 2026-03-13 12:25:00 | 1999.12 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-20 09:35:00 | 1952.30 | 2026-03-20 09:40:00 | 1957.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-08 09:35:00 | 1777.50 | 2026-04-08 09:45:00 | 1789.48 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-08 09:35:00 | 1777.50 | 2026-04-08 11:30:00 | 1777.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:55:00 | 1788.60 | 2026-04-15 10:20:00 | 1798.76 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-15 09:55:00 | 1788.60 | 2026-04-15 11:45:00 | 1788.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1832.10 | 2026-04-27 09:55:00 | 1842.85 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1832.10 | 2026-04-27 11:15:00 | 1832.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 11:05:00 | 1799.90 | 2026-04-30 12:45:00 | 1804.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-05 10:00:00 | 1810.70 | 2026-05-05 11:10:00 | 1801.15 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-05 10:00:00 | 1810.70 | 2026-05-05 12:30:00 | 1810.70 | STOP_HIT | 0.50 | 0.00% |
