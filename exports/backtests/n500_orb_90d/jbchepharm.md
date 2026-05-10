# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2155.00
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 2 / 17 / 6
- **Avg / median % per leg:** -0.03% / -0.23%
- **Sum % (uncompounded):** -0.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 1 | 7.1% | 0 | 13 | 1 | -0.24% | -3.3% |
| BUY @ 2nd Alert (retest1) | 14 | 1 | 7.1% | 0 | 13 | 1 | -0.24% | -3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 7 | 63.6% | 2 | 4 | 5 | 0.24% | 2.7% |
| SELL @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 2 | 4 | 5 | 0.24% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 8 | 32.0% | 2 | 17 | 6 | -0.03% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 1891.10 | 1871.35 | 0.00 | ORB-long ORB[1848.70,1865.70] vol=2.7x ATR=6.70 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 1884.40 | 1874.45 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 1909.50 | 1900.18 | 0.00 | ORB-long ORB[1883.50,1902.60] vol=2.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-02-10 11:50:00 | 1905.04 | 1904.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1978.10 | 1988.11 | 0.00 | ORB-short ORB[1982.00,1997.50] vol=2.2x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:20:00 | 1971.04 | 1984.25 | 0.00 | T1 1.5R @ 1971.04 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 1978.10 | 1980.28 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 2033.10 | 2019.65 | 0.00 | ORB-long ORB[2004.00,2023.00] vol=2.2x ATR=6.45 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 2026.65 | 2020.93 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 2106.00 | 2097.27 | 0.00 | ORB-long ORB[2072.00,2092.90] vol=1.9x ATR=6.26 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 2099.74 | 2097.91 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 2071.90 | 2079.14 | 0.00 | ORB-short ORB[2072.00,2092.60] vol=1.7x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:25:00 | 2066.27 | 2075.34 | 0.00 | T1 1.5R @ 2066.27 |
| Target hit | 2026-02-27 15:20:00 | 2052.40 | 2063.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 2068.40 | 2058.46 | 0.00 | ORB-long ORB[2048.20,2066.00] vol=2.3x ATR=7.69 |
| Stop hit — per-position SL triggered | 2026-03-05 10:25:00 | 2060.71 | 2065.58 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2044.10 | 2058.63 | 0.00 | ORB-short ORB[2066.40,2080.00] vol=2.7x ATR=5.17 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 2049.27 | 2054.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:20:00 | 2105.10 | 2099.77 | 0.00 | ORB-long ORB[2077.20,2104.80] vol=1.9x ATR=6.27 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 2098.83 | 2100.91 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:10:00 | 2113.20 | 2123.86 | 0.00 | ORB-short ORB[2118.30,2139.20] vol=1.5x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:40:00 | 2104.02 | 2118.56 | 0.00 | T1 1.5R @ 2104.02 |
| Target hit | 2026-03-13 11:35:00 | 2112.40 | 2110.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 2112.00 | 2102.99 | 0.00 | ORB-long ORB[2083.10,2099.00] vol=6.9x ATR=7.09 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 2104.91 | 2103.25 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 2092.50 | 2104.64 | 0.00 | ORB-short ORB[2100.10,2126.50] vol=1.6x ATR=7.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:15:00 | 2081.07 | 2100.04 | 0.00 | T1 1.5R @ 2081.07 |
| Stop hit — per-position SL triggered | 2026-03-18 11:40:00 | 2092.50 | 2099.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:55:00 | 2096.90 | 2090.51 | 0.00 | ORB-long ORB[2063.10,2092.00] vol=3.1x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 2091.38 | 2091.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 11:00:00 | 2064.80 | 2060.03 | 0.00 | ORB-long ORB[2030.20,2055.00] vol=1.6x ATR=6.82 |
| Stop hit — per-position SL triggered | 2026-03-30 11:10:00 | 2057.98 | 2060.05 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 1972.60 | 1976.67 | 0.00 | ORB-short ORB[1975.20,1999.80] vol=2.4x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:35:00 | 1966.33 | 1976.36 | 0.00 | T1 1.5R @ 1966.33 |
| Stop hit — per-position SL triggered | 2026-04-16 11:50:00 | 1972.60 | 1974.31 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 2041.40 | 2025.30 | 0.00 | ORB-long ORB[2008.20,2031.80] vol=1.8x ATR=6.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 2051.34 | 2028.83 | 0.00 | T1 1.5R @ 2051.34 |
| Stop hit — per-position SL triggered | 2026-04-27 09:55:00 | 2041.40 | 2030.06 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 2073.00 | 2066.66 | 0.00 | ORB-long ORB[2040.00,2068.00] vol=1.6x ATR=7.43 |
| Stop hit — per-position SL triggered | 2026-04-29 09:50:00 | 2065.57 | 2067.80 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 2089.00 | 2077.46 | 0.00 | ORB-long ORB[2064.00,2085.40] vol=2.2x ATR=7.47 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 2081.53 | 2079.67 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 2120.00 | 2110.93 | 0.00 | ORB-long ORB[2100.00,2116.70] vol=1.9x ATR=6.36 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 2113.64 | 2113.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 1891.10 | 2026-02-09 11:25:00 | 1884.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-10 11:00:00 | 1909.50 | 2026-02-10 11:50:00 | 1905.04 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1978.10 | 2026-02-18 11:20:00 | 1971.04 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1978.10 | 2026-02-18 11:45:00 | 1978.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 09:45:00 | 2033.10 | 2026-02-23 09:50:00 | 2026.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:05:00 | 2106.00 | 2026-02-26 10:20:00 | 2099.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2071.90 | 2026-02-27 11:25:00 | 2066.27 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2071.90 | 2026-02-27 15:20:00 | 2052.40 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2026-03-05 09:30:00 | 2068.40 | 2026-03-05 10:25:00 | 2060.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2044.10 | 2026-03-06 11:15:00 | 2049.27 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-10 10:20:00 | 2105.10 | 2026-03-10 10:50:00 | 2098.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2113.20 | 2026-03-13 10:40:00 | 2104.02 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2113.20 | 2026-03-13 11:35:00 | 2112.40 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-03-17 10:15:00 | 2112.00 | 2026-03-17 10:25:00 | 2104.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-18 10:00:00 | 2092.50 | 2026-03-18 11:15:00 | 2081.07 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-18 10:00:00 | 2092.50 | 2026-03-18 11:40:00 | 2092.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:55:00 | 2096.90 | 2026-03-25 11:25:00 | 2091.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-30 11:00:00 | 2064.80 | 2026-03-30 11:10:00 | 2057.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1972.60 | 2026-04-16 10:35:00 | 1966.33 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1972.60 | 2026-04-16 11:50:00 | 1972.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 2041.40 | 2026-04-27 09:45:00 | 2051.34 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-27 09:40:00 | 2041.40 | 2026-04-27 09:55:00 | 2041.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:40:00 | 2073.00 | 2026-04-29 09:50:00 | 2065.57 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-04 09:40:00 | 2089.00 | 2026-05-04 10:00:00 | 2081.53 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 09:40:00 | 2120.00 | 2026-05-06 09:45:00 | 2113.64 | STOP_HIT | 1.00 | -0.30% |
