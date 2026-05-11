# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2025-01-06 09:15:00 → 2026-05-08 15:25:00 (24763 bars)
- **Last close:** 1479.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 3
- **Avg / median % per leg:** 0.12% / -0.35%
- **Sum % (uncompounded):** 1.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.07% | -0.9% |
| BUY @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.07% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.96% | 2.9% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.96% | 2.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 2 | 11 | 3 | 0.12% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:20:00 | 2082.20 | 2066.32 | 0.00 | ORB-long ORB[2044.60,2073.60] vol=3.9x ATR=9.36 |
| Stop hit — per-position SL triggered | 2025-01-17 10:35:00 | 2072.84 | 2069.33 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:55:00 | 1962.65 | 1910.27 | 0.00 | ORB-long ORB[1832.15,1861.10] vol=2.6x ATR=14.33 |
| Stop hit — per-position SL triggered | 2025-01-23 11:00:00 | 1948.32 | 1912.83 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1933.45 | 1939.67 | 0.00 | ORB-short ORB[1936.00,1955.60] vol=1.5x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:10:00 | 1919.88 | 1936.09 | 0.00 | T1 1.5R @ 1919.88 |
| Target hit | 2025-01-24 15:20:00 | 1882.00 | 1910.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-01-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:35:00 | 2016.00 | 2002.48 | 0.00 | ORB-long ORB[1975.55,1991.95] vol=2.2x ATR=9.38 |
| Stop hit — per-position SL triggered | 2025-01-31 10:55:00 | 2006.62 | 2002.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-02-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:50:00 | 2160.50 | 2136.07 | 0.00 | ORB-long ORB[2102.20,2128.15] vol=5.0x ATR=12.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:10:00 | 2178.98 | 2153.04 | 0.00 | T1 1.5R @ 2178.98 |
| Target hit | 2025-02-04 12:50:00 | 2194.55 | 2199.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 2277.70 | 2267.53 | 0.00 | ORB-long ORB[2242.05,2276.00] vol=1.7x ATR=8.19 |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 2269.51 | 2267.60 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:50:00 | 2155.85 | 2183.94 | 0.00 | ORB-short ORB[2185.50,2205.15] vol=1.5x ATR=10.30 |
| Stop hit — per-position SL triggered | 2025-02-11 10:10:00 | 2166.15 | 2175.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-04-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:50:00 | 1968.75 | 1956.56 | 0.00 | ORB-long ORB[1937.75,1966.70] vol=2.5x ATR=8.21 |
| Stop hit — per-position SL triggered | 2025-04-02 11:20:00 | 1960.54 | 1957.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 1904.50 | 1893.09 | 0.00 | ORB-long ORB[1881.70,1899.00] vol=1.7x ATR=6.60 |
| Stop hit — per-position SL triggered | 2025-04-21 09:40:00 | 1897.90 | 1894.75 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:10:00 | 1938.00 | 1934.28 | 0.00 | ORB-long ORB[1908.90,1936.00] vol=2.0x ATR=7.05 |
| Stop hit — per-position SL triggered | 2025-04-22 11:15:00 | 1930.95 | 1934.27 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:40:00 | 1919.00 | 1911.01 | 0.00 | ORB-long ORB[1892.70,1918.00] vol=1.6x ATR=10.91 |
| Stop hit — per-position SL triggered | 2025-04-28 13:10:00 | 1908.09 | 1916.19 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 11:15:00 | 1937.70 | 1915.43 | 0.00 | ORB-long ORB[1910.10,1935.50] vol=4.4x ATR=7.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:20:00 | 1949.41 | 1919.65 | 0.00 | T1 1.5R @ 1949.41 |
| Stop hit — per-position SL triggered | 2025-04-29 11:45:00 | 1937.70 | 1926.71 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 1917.10 | 1898.97 | 0.00 | ORB-long ORB[1888.40,1909.90] vol=1.9x ATR=5.37 |
| Stop hit — per-position SL triggered | 2025-05-05 11:05:00 | 1911.73 | 1899.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-01-17 10:20:00 | 2082.20 | 2025-01-17 10:35:00 | 2072.84 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-23 10:55:00 | 1962.65 | 2025-01-23 11:00:00 | 1948.32 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest1 | 2025-01-24 09:45:00 | 1933.45 | 2025-01-24 10:10:00 | 1919.88 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-01-24 09:45:00 | 1933.45 | 2025-01-24 15:20:00 | 1882.00 | TARGET_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2025-01-31 10:35:00 | 2016.00 | 2025-01-31 10:55:00 | 2006.62 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-02-04 09:50:00 | 2160.50 | 2025-02-04 10:10:00 | 2178.98 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-02-04 09:50:00 | 2160.50 | 2025-02-04 12:50:00 | 2194.55 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2025-02-07 10:10:00 | 2277.70 | 2025-02-07 10:15:00 | 2269.51 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-11 09:50:00 | 2155.85 | 2025-02-11 10:10:00 | 2166.15 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-04-02 10:50:00 | 1968.75 | 2025-04-02 11:20:00 | 1960.54 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-21 09:30:00 | 1904.50 | 2025-04-21 09:40:00 | 1897.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 11:10:00 | 1938.00 | 2025-04-22 11:15:00 | 1930.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-28 09:40:00 | 1919.00 | 2025-04-28 13:10:00 | 1908.09 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-04-29 11:15:00 | 1937.70 | 2025-04-29 11:20:00 | 1949.41 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-29 11:15:00 | 1937.70 | 2025-04-29 11:45:00 | 1937.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 11:00:00 | 1917.10 | 2025-05-05 11:05:00 | 1911.73 | STOP_HIT | 1.00 | -0.28% |
