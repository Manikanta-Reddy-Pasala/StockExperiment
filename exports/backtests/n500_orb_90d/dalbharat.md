# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1840.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 7
- **Avg / median % per leg:** 0.31% / 0.37%
- **Sum % (uncompounded):** 5.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.68% | 4.8% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.68% | 4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.08% | 0.9% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.08% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.31% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2196.80 | 2196.89 | 0.00 | ORB-short ORB[2198.00,2214.00] vol=2.1x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-02-10 11:55:00 | 2202.17 | 2196.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 2131.80 | 2119.51 | 0.00 | ORB-long ORB[2107.00,2128.70] vol=1.7x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 2141.60 | 2124.06 | 0.00 | T1 1.5R @ 2141.60 |
| Stop hit — per-position SL triggered | 2026-02-17 11:40:00 | 2131.80 | 2134.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 2022.10 | 2040.59 | 0.00 | ORB-short ORB[2048.50,2070.40] vol=2.1x ATR=7.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:05:00 | 2011.30 | 2030.98 | 0.00 | T1 1.5R @ 2011.30 |
| Stop hit — per-position SL triggered | 2026-02-27 10:10:00 | 2022.10 | 2030.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 1883.40 | 1886.92 | 0.00 | ORB-short ORB[1891.20,1910.10] vol=1.9x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 1875.31 | 1883.80 | 0.00 | T1 1.5R @ 1875.31 |
| Stop hit — per-position SL triggered | 2026-03-05 12:10:00 | 1883.40 | 1881.06 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:35:00 | 1912.20 | 1912.85 | 0.00 | ORB-short ORB[1914.30,1930.50] vol=11.1x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 1905.04 | 1911.31 | 0.00 | T1 1.5R @ 1905.04 |
| Target hit | 2026-03-06 13:10:00 | 1910.60 | 1908.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 1863.40 | 1869.52 | 0.00 | ORB-short ORB[1870.30,1886.30] vol=3.4x ATR=8.44 |
| Stop hit — per-position SL triggered | 2026-03-10 10:40:00 | 1871.84 | 1867.60 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 1856.60 | 1837.99 | 0.00 | ORB-long ORB[1826.60,1850.00] vol=2.1x ATR=8.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:40:00 | 1869.39 | 1851.04 | 0.00 | T1 1.5R @ 1869.39 |
| Target hit | 2026-03-12 15:20:00 | 1898.70 | 1880.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:25:00 | 1933.70 | 1947.95 | 0.00 | ORB-short ORB[1955.00,1975.20] vol=1.9x ATR=5.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:45:00 | 1925.24 | 1945.73 | 0.00 | T1 1.5R @ 1925.24 |
| Stop hit — per-position SL triggered | 2026-04-24 12:20:00 | 1933.70 | 1941.41 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 1984.60 | 1979.60 | 0.00 | ORB-long ORB[1964.70,1981.80] vol=2.4x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-04-27 11:20:00 | 1979.20 | 1979.75 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:50:00 | 1955.40 | 1940.12 | 0.00 | ORB-long ORB[1920.60,1937.20] vol=2.4x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:00:00 | 1966.97 | 1945.71 | 0.00 | T1 1.5R @ 1966.97 |
| Target hit | 2026-05-04 12:30:00 | 1975.70 | 1979.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 1944.90 | 1953.80 | 0.00 | ORB-short ORB[1950.60,1977.50] vol=3.1x ATR=5.67 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 1950.57 | 1953.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:00:00 | 2196.80 | 2026-02-10 11:55:00 | 2202.17 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-17 09:45:00 | 2131.80 | 2026-02-17 10:10:00 | 2141.60 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 09:45:00 | 2131.80 | 2026-02-17 11:40:00 | 2131.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2022.10 | 2026-02-27 10:05:00 | 2011.30 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-27 09:45:00 | 2022.10 | 2026-02-27 10:10:00 | 2022.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1883.40 | 2026-03-05 11:25:00 | 1875.31 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1883.40 | 2026-03-05 12:10:00 | 1883.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:35:00 | 1912.20 | 2026-03-06 10:45:00 | 1905.04 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-06 10:35:00 | 1912.20 | 2026-03-06 13:10:00 | 1910.60 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2026-03-10 09:40:00 | 1863.40 | 2026-03-10 10:40:00 | 1871.84 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-12 09:45:00 | 1856.60 | 2026-03-12 11:40:00 | 1869.39 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-03-12 09:45:00 | 1856.60 | 2026-03-12 15:20:00 | 1898.70 | TARGET_HIT | 0.50 | 2.27% |
| SELL | retest1 | 2026-04-24 10:25:00 | 1933.70 | 2026-04-24 10:45:00 | 1925.24 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-24 10:25:00 | 1933.70 | 2026-04-24 12:20:00 | 1933.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 11:10:00 | 1984.60 | 2026-04-27 11:20:00 | 1979.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-04 09:50:00 | 1955.40 | 2026-05-04 10:00:00 | 1966.97 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-05-04 09:50:00 | 1955.40 | 2026-05-04 12:30:00 | 1975.70 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2026-05-08 09:45:00 | 1944.90 | 2026-05-08 09:50:00 | 1950.57 | STOP_HIT | 1.00 | -0.29% |
