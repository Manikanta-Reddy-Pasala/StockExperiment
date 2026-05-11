# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 1760.60
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 18
- **Target hits / Stop hits / Partials:** 5 / 18 / 10
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 4.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 10 | 45.5% | 4 | 12 | 6 | 0.16% | 3.6% |
| BUY @ 2nd Alert (retest1) | 22 | 10 | 45.5% | 4 | 12 | 6 | 0.16% | 3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.12% | 1.3% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.12% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 15 | 45.5% | 5 | 18 | 10 | 0.15% | 4.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:50:00 | 920.95 | 912.87 | 0.00 | ORB-long ORB[906.00,919.40] vol=1.6x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 13:10:00 | 926.59 | 916.16 | 0.00 | T1 1.5R @ 926.59 |
| Target hit | 2024-05-14 15:20:00 | 946.80 | 929.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:05:00 | 952.95 | 964.74 | 0.00 | ORB-short ORB[961.95,975.85] vol=2.2x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 958.22 | 964.14 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:10:00 | 1058.85 | 1049.40 | 0.00 | ORB-long ORB[1035.00,1047.00] vol=3.1x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:25:00 | 1066.26 | 1051.50 | 0.00 | T1 1.5R @ 1066.26 |
| Target hit | 2024-05-24 15:15:00 | 1071.90 | 1075.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 1091.30 | 1083.01 | 0.00 | ORB-long ORB[1070.05,1085.80] vol=2.1x ATR=5.69 |
| Stop hit — per-position SL triggered | 2024-05-31 09:35:00 | 1085.61 | 1084.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 1144.75 | 1133.21 | 0.00 | ORB-long ORB[1125.00,1136.15] vol=2.8x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-06-11 10:00:00 | 1138.76 | 1136.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 1503.00 | 1494.45 | 0.00 | ORB-long ORB[1477.90,1497.80] vol=1.8x ATR=7.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:35:00 | 1513.91 | 1499.09 | 0.00 | T1 1.5R @ 1513.91 |
| Target hit | 2024-06-27 10:15:00 | 1510.00 | 1516.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 1411.00 | 1387.32 | 0.00 | ORB-long ORB[1367.95,1379.95] vol=3.0x ATR=8.69 |
| Stop hit — per-position SL triggered | 2024-07-09 09:50:00 | 1402.31 | 1389.95 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:45:00 | 1375.75 | 1359.43 | 0.00 | ORB-long ORB[1345.00,1362.00] vol=2.7x ATR=8.05 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 1367.70 | 1362.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:35:00 | 2551.70 | 2528.56 | 0.00 | ORB-long ORB[2512.65,2542.40] vol=1.9x ATR=13.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:40:00 | 2572.41 | 2537.63 | 0.00 | T1 1.5R @ 2572.41 |
| Target hit | 2024-09-25 10:45:00 | 2557.55 | 2566.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 2555.60 | 2551.57 | 0.00 | ORB-long ORB[2533.20,2554.00] vol=2.4x ATR=13.69 |
| Stop hit — per-position SL triggered | 2024-09-26 09:40:00 | 2541.91 | 2551.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:00:00 | 2509.50 | 2537.52 | 0.00 | ORB-short ORB[2529.60,2565.30] vol=1.7x ATR=10.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 11:10:00 | 2493.16 | 2524.30 | 0.00 | T1 1.5R @ 2493.16 |
| Stop hit — per-position SL triggered | 2024-10-15 11:40:00 | 2509.50 | 2514.90 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:35:00 | 2193.45 | 2175.42 | 0.00 | ORB-long ORB[2155.00,2182.00] vol=2.9x ATR=10.76 |
| Stop hit — per-position SL triggered | 2024-12-02 09:50:00 | 2182.69 | 2177.30 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 2165.30 | 2153.76 | 0.00 | ORB-long ORB[2136.15,2165.10] vol=1.8x ATR=9.09 |
| Stop hit — per-position SL triggered | 2024-12-03 09:45:00 | 2156.21 | 2158.45 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:15:00 | 2125.15 | 2137.28 | 0.00 | ORB-short ORB[2128.00,2154.85] vol=2.3x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:50:00 | 2117.23 | 2134.65 | 0.00 | T1 1.5R @ 2117.23 |
| Target hit | 2024-12-04 15:20:00 | 2120.15 | 2127.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 2152.60 | 2144.73 | 0.00 | ORB-long ORB[2125.05,2149.95] vol=3.4x ATR=8.22 |
| Stop hit — per-position SL triggered | 2024-12-05 09:35:00 | 2144.38 | 2144.88 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-12-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:45:00 | 2227.00 | 2220.46 | 0.00 | ORB-long ORB[2195.80,2220.00] vol=10.0x ATR=9.94 |
| Stop hit — per-position SL triggered | 2024-12-16 10:00:00 | 2217.06 | 2220.65 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-12-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:40:00 | 2177.40 | 2193.83 | 0.00 | ORB-short ORB[2188.55,2220.10] vol=2.4x ATR=13.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 10:05:00 | 2157.34 | 2188.66 | 0.00 | T1 1.5R @ 2157.34 |
| Stop hit — per-position SL triggered | 2024-12-31 11:10:00 | 2177.40 | 2185.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 2230.15 | 2208.54 | 0.00 | ORB-long ORB[2185.55,2214.95] vol=3.0x ATR=9.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:40:00 | 2244.97 | 2216.83 | 0.00 | T1 1.5R @ 2244.97 |
| Stop hit — per-position SL triggered | 2025-01-02 09:45:00 | 2230.15 | 2217.58 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-01-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:25:00 | 2295.50 | 2279.10 | 0.00 | ORB-long ORB[2251.00,2276.95] vol=2.8x ATR=9.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:30:00 | 2310.13 | 2283.94 | 0.00 | T1 1.5R @ 2310.13 |
| Stop hit — per-position SL triggered | 2025-01-03 10:50:00 | 2295.50 | 2293.68 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-01-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:30:00 | 2201.30 | 2229.19 | 0.00 | ORB-short ORB[2227.10,2259.95] vol=3.5x ATR=11.07 |
| Stop hit — per-position SL triggered | 2025-01-21 10:35:00 | 2212.37 | 2228.49 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 2081.00 | 2065.22 | 0.00 | ORB-long ORB[2048.65,2075.00] vol=2.5x ATR=15.78 |
| Stop hit — per-position SL triggered | 2025-02-01 09:35:00 | 2065.22 | 2065.65 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 1700.70 | 1731.85 | 0.00 | ORB-short ORB[1739.10,1764.45] vol=1.7x ATR=11.42 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1712.12 | 1728.98 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 1821.50 | 1849.35 | 0.00 | ORB-short ORB[1850.70,1874.00] vol=5.1x ATR=9.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:35:00 | 1806.68 | 1840.04 | 0.00 | T1 1.5R @ 1806.68 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 1821.50 | 1838.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:50:00 | 920.95 | 2024-05-14 13:10:00 | 926.59 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-05-14 10:50:00 | 920.95 | 2024-05-14 15:20:00 | 946.80 | TARGET_HIT | 0.50 | 2.81% |
| SELL | retest1 | 2024-05-21 10:05:00 | 952.95 | 2024-05-21 10:10:00 | 958.22 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-05-24 11:10:00 | 1058.85 | 2024-05-24 11:25:00 | 1066.26 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-05-24 11:10:00 | 1058.85 | 2024-05-24 15:15:00 | 1071.90 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2024-05-31 09:30:00 | 1091.30 | 2024-05-31 09:35:00 | 1085.61 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-06-11 09:50:00 | 1144.75 | 2024-06-11 10:00:00 | 1138.76 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-06-27 09:30:00 | 1503.00 | 2024-06-27 09:35:00 | 1513.91 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-06-27 09:30:00 | 1503.00 | 2024-06-27 10:15:00 | 1510.00 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-09 09:45:00 | 1411.00 | 2024-07-09 09:50:00 | 1402.31 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-07-12 10:45:00 | 1375.75 | 2024-07-12 11:15:00 | 1367.70 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-09-25 09:35:00 | 2551.70 | 2024-09-25 09:40:00 | 2572.41 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2024-09-25 09:35:00 | 2551.70 | 2024-09-25 10:45:00 | 2557.55 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-09-26 09:30:00 | 2555.60 | 2024-09-26 09:40:00 | 2541.91 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-10-15 10:00:00 | 2509.50 | 2024-10-15 11:10:00 | 2493.16 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-15 10:00:00 | 2509.50 | 2024-10-15 11:40:00 | 2509.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 09:35:00 | 2193.45 | 2024-12-02 09:50:00 | 2182.69 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-03 09:30:00 | 2165.30 | 2024-12-03 09:45:00 | 2156.21 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-12-04 11:15:00 | 2125.15 | 2024-12-04 11:50:00 | 2117.23 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-04 11:15:00 | 2125.15 | 2024-12-04 15:20:00 | 2120.15 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-12-05 09:30:00 | 2152.60 | 2024-12-05 09:35:00 | 2144.38 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-16 09:45:00 | 2227.00 | 2024-12-16 10:00:00 | 2217.06 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-31 09:40:00 | 2177.40 | 2024-12-31 10:05:00 | 2157.34 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2024-12-31 09:40:00 | 2177.40 | 2024-12-31 11:10:00 | 2177.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:30:00 | 2230.15 | 2025-01-02 09:40:00 | 2244.97 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-02 09:30:00 | 2230.15 | 2025-01-02 09:45:00 | 2230.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 10:25:00 | 2295.50 | 2025-01-03 10:30:00 | 2310.13 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-03 10:25:00 | 2295.50 | 2025-01-03 10:50:00 | 2295.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 10:30:00 | 2201.30 | 2025-01-21 10:35:00 | 2212.37 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-02-01 09:30:00 | 2081.00 | 2025-02-01 09:35:00 | 2065.22 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2025-03-20 10:05:00 | 1700.70 | 2025-03-20 10:15:00 | 1712.12 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2025-04-29 09:30:00 | 1821.50 | 2025-04-29 09:35:00 | 1806.68 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2025-04-29 09:30:00 | 1821.50 | 2025-04-29 09:40:00 | 1821.50 | STOP_HIT | 0.50 | 0.00% |
