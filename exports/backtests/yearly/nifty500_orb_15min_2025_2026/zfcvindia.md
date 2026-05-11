# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (17113 bars)
- **Last close:** 14532.00
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
| ENTRY1 | 68 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 10 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 58
- **Target hits / Stop hits / Partials:** 10 / 58 / 27
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 15.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 16 | 34.8% | 4 | 30 | 12 | 0.13% | 6.1% |
| BUY @ 2nd Alert (retest1) | 46 | 16 | 34.8% | 4 | 30 | 12 | 0.13% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 21 | 42.9% | 6 | 28 | 15 | 0.19% | 9.6% |
| SELL @ 2nd Alert (retest1) | 49 | 21 | 42.9% | 6 | 28 | 15 | 0.19% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 37 | 38.9% | 10 | 58 | 27 | 0.16% | 15.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:15:00 | 13293.00 | 13261.75 | 0.00 | ORB-long ORB[13113.00,13260.00] vol=1.5x ATR=29.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 11:35:00 | 13336.56 | 13270.27 | 0.00 | T1 1.5R @ 13336.56 |
| Stop hit — per-position SL triggered | 2025-05-14 12:40:00 | 13293.00 | 13283.62 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:15:00 | 12776.00 | 12745.87 | 0.00 | ORB-long ORB[12640.00,12775.00] vol=2.2x ATR=49.50 |
| Stop hit — per-position SL triggered | 2025-05-30 10:45:00 | 12726.50 | 12755.69 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 11:05:00 | 13650.00 | 13583.62 | 0.00 | ORB-long ORB[13448.00,13611.00] vol=2.0x ATR=48.46 |
| Stop hit — per-position SL triggered | 2025-06-04 11:20:00 | 13601.54 | 13584.90 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:20:00 | 13193.00 | 13271.34 | 0.00 | ORB-short ORB[13263.00,13398.00] vol=1.6x ATR=37.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:55:00 | 13137.33 | 13233.23 | 0.00 | T1 1.5R @ 13137.33 |
| Stop hit — per-position SL triggered | 2025-06-17 11:10:00 | 13193.00 | 13224.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:55:00 | 13317.00 | 13239.60 | 0.00 | ORB-long ORB[13130.00,13308.00] vol=2.3x ATR=47.01 |
| Stop hit — per-position SL triggered | 2025-06-18 15:20:00 | 13312.00 | 13303.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:00:00 | 13146.00 | 13249.44 | 0.00 | ORB-short ORB[13177.00,13362.00] vol=4.5x ATR=35.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:20:00 | 13092.03 | 13224.41 | 0.00 | T1 1.5R @ 13092.03 |
| Stop hit — per-position SL triggered | 2025-06-19 12:30:00 | 13146.00 | 13189.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 13272.00 | 13257.58 | 0.00 | ORB-long ORB[13104.00,13212.00] vol=1.7x ATR=29.00 |
| Stop hit — per-position SL triggered | 2025-06-25 11:15:00 | 13243.00 | 13257.89 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:55:00 | 13335.00 | 13399.65 | 0.00 | ORB-short ORB[13417.00,13584.00] vol=7.1x ATR=41.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 11:15:00 | 13273.08 | 13384.64 | 0.00 | T1 1.5R @ 13273.08 |
| Stop hit — per-position SL triggered | 2025-06-30 11:35:00 | 13335.00 | 13378.53 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 13207.00 | 13307.87 | 0.00 | ORB-short ORB[13293.00,13471.00] vol=2.2x ATR=33.34 |
| Stop hit — per-position SL triggered | 2025-07-01 11:45:00 | 13240.34 | 13286.43 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:25:00 | 13086.00 | 13131.82 | 0.00 | ORB-short ORB[13102.00,13233.00] vol=2.0x ATR=31.31 |
| Stop hit — per-position SL triggered | 2025-07-09 10:40:00 | 13117.31 | 13129.74 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:05:00 | 13029.00 | 13128.57 | 0.00 | ORB-short ORB[13098.00,13245.00] vol=4.5x ATR=26.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:20:00 | 12988.52 | 13091.53 | 0.00 | T1 1.5R @ 12988.52 |
| Stop hit — per-position SL triggered | 2025-07-10 11:30:00 | 13029.00 | 13082.55 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 12986.00 | 13022.29 | 0.00 | ORB-short ORB[13018.00,13094.00] vol=3.2x ATR=27.08 |
| Stop hit — per-position SL triggered | 2025-07-11 09:35:00 | 13013.08 | 13021.14 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:50:00 | 13027.00 | 12938.65 | 0.00 | ORB-long ORB[12800.00,12950.00] vol=1.6x ATR=37.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:20:00 | 13083.18 | 12978.38 | 0.00 | T1 1.5R @ 13083.18 |
| Stop hit — per-position SL triggered | 2025-07-14 11:00:00 | 13027.00 | 12992.43 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:15:00 | 13185.00 | 13095.67 | 0.00 | ORB-long ORB[12925.00,13122.00] vol=1.7x ATR=27.98 |
| Stop hit — per-position SL triggered | 2025-07-15 11:20:00 | 13157.02 | 13098.87 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:10:00 | 13057.00 | 13104.19 | 0.00 | ORB-short ORB[13091.00,13199.00] vol=4.2x ATR=27.82 |
| Stop hit — per-position SL triggered | 2025-07-16 11:45:00 | 13084.82 | 13098.55 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:35:00 | 13208.00 | 13138.50 | 0.00 | ORB-long ORB[13001.00,13131.00] vol=1.5x ATR=31.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:30:00 | 13255.11 | 13177.24 | 0.00 | T1 1.5R @ 13255.11 |
| Stop hit — per-position SL triggered | 2025-07-17 13:10:00 | 13208.00 | 13198.71 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:45:00 | 13249.00 | 13238.18 | 0.00 | ORB-long ORB[13150.00,13248.00] vol=2.2x ATR=42.54 |
| Stop hit — per-position SL triggered | 2025-07-22 10:45:00 | 13206.46 | 13237.58 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:50:00 | 13201.00 | 13111.80 | 0.00 | ORB-long ORB[13059.00,13189.00] vol=2.1x ATR=31.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:50:00 | 13248.24 | 13121.72 | 0.00 | T1 1.5R @ 13248.24 |
| Target hit | 2025-07-23 15:20:00 | 13500.00 | 13312.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-07-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:10:00 | 13246.00 | 13279.27 | 0.00 | ORB-short ORB[13251.00,13364.00] vol=1.8x ATR=31.26 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 13277.26 | 13277.81 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 09:30:00 | 13367.00 | 13249.36 | 0.00 | ORB-long ORB[13128.00,13279.00] vol=2.3x ATR=45.67 |
| Stop hit — per-position SL triggered | 2025-07-31 09:40:00 | 13321.33 | 13276.75 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:55:00 | 13137.00 | 13206.66 | 0.00 | ORB-short ORB[13186.00,13283.00] vol=3.6x ATR=31.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:35:00 | 13089.80 | 13191.50 | 0.00 | T1 1.5R @ 13089.80 |
| Target hit | 2025-08-01 15:20:00 | 12907.00 | 13072.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-08-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 11:10:00 | 13033.00 | 12944.75 | 0.00 | ORB-long ORB[12859.00,12981.00] vol=1.9x ATR=38.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 12:25:00 | 13090.33 | 12999.41 | 0.00 | T1 1.5R @ 13090.33 |
| Target hit | 2025-08-04 15:20:00 | 13386.00 | 13207.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-09-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:50:00 | 13538.00 | 13583.87 | 0.00 | ORB-short ORB[13552.00,13610.00] vol=1.6x ATR=22.41 |
| Stop hit — per-position SL triggered | 2025-09-12 10:55:00 | 13560.41 | 13583.72 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 13541.00 | 13597.62 | 0.00 | ORB-short ORB[13558.00,13716.00] vol=6.5x ATR=21.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:35:00 | 13508.28 | 13562.99 | 0.00 | T1 1.5R @ 13508.28 |
| Target hit | 2025-09-18 15:20:00 | 13428.00 | 13510.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 13515.00 | 13470.65 | 0.00 | ORB-long ORB[13401.00,13509.00] vol=1.7x ATR=29.58 |
| Stop hit — per-position SL triggered | 2025-09-19 09:55:00 | 13485.42 | 13469.69 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:35:00 | 13227.00 | 13262.46 | 0.00 | ORB-short ORB[13250.00,13380.00] vol=2.3x ATR=48.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:50:00 | 13154.95 | 13239.01 | 0.00 | T1 1.5R @ 13154.95 |
| Target hit | 2025-09-23 13:50:00 | 13036.00 | 13013.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-09-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:10:00 | 12540.00 | 12617.12 | 0.00 | ORB-short ORB[12562.00,12649.00] vol=1.5x ATR=30.36 |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 12570.36 | 12616.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 13389.00 | 13477.30 | 0.00 | ORB-short ORB[13425.00,13625.00] vol=1.8x ATR=34.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:30:00 | 13337.84 | 13447.12 | 0.00 | T1 1.5R @ 13337.84 |
| Target hit | 2025-10-08 15:20:00 | 13244.00 | 13297.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:45:00 | 13395.00 | 13363.97 | 0.00 | ORB-long ORB[13210.00,13387.00] vol=2.0x ATR=49.77 |
| Stop hit — per-position SL triggered | 2025-10-09 13:25:00 | 13345.23 | 13387.05 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 09:35:00 | 12944.00 | 12966.24 | 0.00 | ORB-short ORB[12960.00,13065.00] vol=2.3x ATR=23.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:00:00 | 12908.57 | 12945.65 | 0.00 | T1 1.5R @ 12908.57 |
| Stop hit — per-position SL triggered | 2025-10-24 10:05:00 | 12944.00 | 12943.05 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:30:00 | 12822.00 | 12915.19 | 0.00 | ORB-short ORB[12876.00,12959.00] vol=2.6x ATR=33.76 |
| Stop hit — per-position SL triggered | 2025-10-27 10:40:00 | 12855.76 | 12908.18 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:45:00 | 13081.00 | 13045.83 | 0.00 | ORB-long ORB[12900.00,13074.00] vol=2.1x ATR=38.61 |
| Stop hit — per-position SL triggered | 2025-10-28 09:55:00 | 13042.39 | 13047.38 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:55:00 | 12712.00 | 12760.19 | 0.00 | ORB-short ORB[12775.00,12900.00] vol=3.1x ATR=30.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:15:00 | 12666.51 | 12737.05 | 0.00 | T1 1.5R @ 12666.51 |
| Target hit | 2025-10-31 13:35:00 | 12518.00 | 12495.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:35:00 | 12754.00 | 12711.55 | 0.00 | ORB-long ORB[12602.00,12722.00] vol=2.4x ATR=30.70 |
| Stop hit — per-position SL triggered | 2025-11-04 09:50:00 | 12723.30 | 12739.85 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 12804.00 | 12796.89 | 0.00 | ORB-long ORB[12720.00,12802.00] vol=2.1x ATR=35.91 |
| Stop hit — per-position SL triggered | 2025-11-17 09:45:00 | 12768.09 | 12790.87 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:45:00 | 12592.00 | 12667.74 | 0.00 | ORB-short ORB[12677.00,12814.00] vol=3.1x ATR=26.07 |
| Stop hit — per-position SL triggered | 2025-11-18 10:50:00 | 12618.07 | 12631.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:05:00 | 13081.00 | 12772.26 | 0.00 | ORB-long ORB[12666.00,12758.00] vol=1.6x ATR=65.01 |
| Stop hit — per-position SL triggered | 2025-11-19 10:10:00 | 13015.99 | 12783.81 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:30:00 | 13043.00 | 12928.11 | 0.00 | ORB-long ORB[12893.00,13000.00] vol=2.3x ATR=43.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:45:00 | 13107.61 | 13006.07 | 0.00 | T1 1.5R @ 13107.61 |
| Stop hit — per-position SL triggered | 2025-11-20 10:50:00 | 13043.00 | 13024.42 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 12919.00 | 12865.06 | 0.00 | ORB-long ORB[12775.00,12882.00] vol=1.9x ATR=26.40 |
| Stop hit — per-position SL triggered | 2025-11-26 10:45:00 | 12892.60 | 12875.54 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:20:00 | 14422.00 | 14320.46 | 0.00 | ORB-long ORB[14183.00,14392.00] vol=2.5x ATR=65.69 |
| Stop hit — per-position SL triggered | 2025-12-03 10:30:00 | 14356.31 | 14329.88 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:10:00 | 14416.00 | 14453.80 | 0.00 | ORB-short ORB[14488.00,14699.00] vol=1.6x ATR=63.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:35:00 | 14321.12 | 14437.52 | 0.00 | T1 1.5R @ 14321.12 |
| Stop hit — per-position SL triggered | 2025-12-09 10:50:00 | 14416.00 | 14432.34 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:55:00 | 14161.00 | 14247.35 | 0.00 | ORB-short ORB[14254.00,14400.00] vol=1.7x ATR=46.90 |
| Stop hit — per-position SL triggered | 2025-12-11 10:20:00 | 14207.90 | 14227.58 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:10:00 | 14246.00 | 14214.07 | 0.00 | ORB-long ORB[14043.00,14212.00] vol=12.6x ATR=42.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 11:40:00 | 14309.89 | 14219.48 | 0.00 | T1 1.5R @ 14309.89 |
| Target hit | 2025-12-15 15:20:00 | 14631.00 | 14427.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 15164.00 | 15071.30 | 0.00 | ORB-long ORB[14840.00,15000.00] vol=1.8x ATR=74.25 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 15089.75 | 15102.41 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:20:00 | 14932.00 | 14745.30 | 0.00 | ORB-long ORB[14611.00,14755.00] vol=2.1x ATR=66.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:25:00 | 15031.69 | 14842.23 | 0.00 | T1 1.5R @ 15031.69 |
| Stop hit — per-position SL triggered | 2025-12-18 11:20:00 | 14932.00 | 14963.21 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 15202.00 | 15124.01 | 0.00 | ORB-long ORB[14922.00,15078.00] vol=1.9x ATR=53.28 |
| Stop hit — per-position SL triggered | 2025-12-19 12:00:00 | 15148.72 | 15140.65 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:45:00 | 14610.00 | 14702.52 | 0.00 | ORB-short ORB[14652.00,14825.00] vol=2.0x ATR=43.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:50:00 | 14544.08 | 14694.42 | 0.00 | T1 1.5R @ 14544.08 |
| Stop hit — per-position SL triggered | 2025-12-30 12:50:00 | 14610.00 | 14625.35 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 14938.00 | 14856.47 | 0.00 | ORB-long ORB[14729.00,14883.00] vol=2.3x ATR=44.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:50:00 | 15004.84 | 14900.11 | 0.00 | T1 1.5R @ 15004.84 |
| Stop hit — per-position SL triggered | 2026-01-02 09:55:00 | 14938.00 | 14907.97 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-01-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:10:00 | 15669.00 | 15484.29 | 0.00 | ORB-long ORB[15119.00,15354.00] vol=1.7x ATR=78.63 |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 15590.37 | 15485.99 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 15005.00 | 15122.08 | 0.00 | ORB-short ORB[15157.00,15305.00] vol=3.7x ATR=49.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:15:00 | 14930.11 | 15086.74 | 0.00 | T1 1.5R @ 14930.11 |
| Stop hit — per-position SL triggered | 2026-01-07 11:25:00 | 15005.00 | 15001.35 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:05:00 | 13935.00 | 14003.94 | 0.00 | ORB-short ORB[14000.00,14180.00] vol=2.0x ATR=47.20 |
| Stop hit — per-position SL triggered | 2026-01-13 11:25:00 | 13982.20 | 13991.95 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:55:00 | 13926.00 | 14084.82 | 0.00 | ORB-short ORB[14033.00,14224.00] vol=3.2x ATR=54.04 |
| Stop hit — per-position SL triggered | 2026-01-14 10:55:00 | 13980.04 | 14001.19 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:55:00 | 13800.00 | 13874.16 | 0.00 | ORB-short ORB[13856.00,14040.00] vol=1.9x ATR=39.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 13740.30 | 13847.76 | 0.00 | T1 1.5R @ 13740.30 |
| Stop hit — per-position SL triggered | 2026-01-20 14:10:00 | 13800.00 | 13816.76 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:55:00 | 14007.00 | 13956.65 | 0.00 | ORB-long ORB[13773.00,13942.00] vol=2.3x ATR=56.25 |
| Stop hit — per-position SL triggered | 2026-01-22 10:20:00 | 13950.75 | 13936.56 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:55:00 | 15349.00 | 15211.51 | 0.00 | ORB-long ORB[15156.00,15303.00] vol=2.9x ATR=47.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 11:00:00 | 15420.80 | 15268.93 | 0.00 | T1 1.5R @ 15420.80 |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 15349.00 | 15292.21 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:00:00 | 15093.00 | 15172.42 | 0.00 | ORB-short ORB[15132.00,15280.00] vol=1.8x ATR=36.82 |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 15129.82 | 15168.34 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 10:05:00 | 15160.00 | 15004.96 | 0.00 | ORB-long ORB[14934.00,15063.00] vol=1.9x ATR=55.45 |
| Stop hit — per-position SL triggered | 2026-02-06 10:10:00 | 15104.55 | 15030.01 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:35:00 | 14933.00 | 14905.63 | 0.00 | ORB-long ORB[14800.00,14918.00] vol=2.5x ATR=60.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:15:00 | 15023.81 | 14938.17 | 0.00 | T1 1.5R @ 15023.81 |
| Target hit | 2026-02-09 11:50:00 | 14955.00 | 14977.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 15844.00 | 15916.20 | 0.00 | ORB-short ORB[15903.00,16140.00] vol=2.7x ATR=41.95 |
| Stop hit — per-position SL triggered | 2026-02-17 11:05:00 | 15885.95 | 15915.61 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 15890.00 | 15978.11 | 0.00 | ORB-short ORB[15930.00,16118.00] vol=3.5x ATR=36.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:05:00 | 15835.32 | 15925.66 | 0.00 | T1 1.5R @ 15835.32 |
| Target hit | 2026-02-18 15:20:00 | 15585.00 | 15771.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 15645.00 | 15789.80 | 0.00 | ORB-short ORB[15798.00,15976.00] vol=1.8x ATR=40.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 15685.37 | 15778.56 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-03-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:10:00 | 14512.00 | 14388.00 | 0.00 | ORB-long ORB[14255.00,14411.00] vol=2.2x ATR=60.48 |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 14451.52 | 14391.59 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:35:00 | 13470.00 | 13536.23 | 0.00 | ORB-short ORB[13527.00,13691.00] vol=1.8x ATR=46.83 |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 13516.83 | 13516.16 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-04-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:45:00 | 13553.00 | 13613.51 | 0.00 | ORB-short ORB[13559.00,13700.00] vol=2.0x ATR=43.75 |
| Stop hit — per-position SL triggered | 2026-04-07 09:55:00 | 13596.75 | 13612.99 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 15038.00 | 14904.14 | 0.00 | ORB-long ORB[14762.00,14934.00] vol=8.5x ATR=41.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 15099.61 | 14929.28 | 0.00 | T1 1.5R @ 15099.61 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 15038.00 | 14982.04 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 14860.00 | 14941.40 | 0.00 | ORB-short ORB[14898.00,14996.00] vol=1.8x ATR=53.47 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 14913.47 | 14937.88 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 14659.00 | 14724.41 | 0.00 | ORB-short ORB[14744.00,14893.00] vol=2.0x ATR=40.00 |
| Stop hit — per-position SL triggered | 2026-05-05 10:35:00 | 14699.00 | 14704.97 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:20:00 | 14741.00 | 14658.24 | 0.00 | ORB-long ORB[14546.00,14693.00] vol=2.2x ATR=33.45 |
| Stop hit — per-position SL triggered | 2026-05-06 10:35:00 | 14707.55 | 14664.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 11:15:00 | 13293.00 | 2025-05-14 11:35:00 | 13336.56 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-14 11:15:00 | 13293.00 | 2025-05-14 12:40:00 | 13293.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-30 10:15:00 | 12776.00 | 2025-05-30 10:45:00 | 12726.50 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-04 11:05:00 | 13650.00 | 2025-06-04 11:20:00 | 13601.54 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-06-17 10:20:00 | 13193.00 | 2025-06-17 10:55:00 | 13137.33 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-17 10:20:00 | 13193.00 | 2025-06-17 11:10:00 | 13193.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 09:55:00 | 13317.00 | 2025-06-18 15:20:00 | 13312.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest1 | 2025-06-19 11:00:00 | 13146.00 | 2025-06-19 11:20:00 | 13092.03 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-19 11:00:00 | 13146.00 | 2025-06-19 12:30:00 | 13146.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 11:05:00 | 13272.00 | 2025-06-25 11:15:00 | 13243.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-30 10:55:00 | 13335.00 | 2025-06-30 11:15:00 | 13273.08 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-06-30 10:55:00 | 13335.00 | 2025-06-30 11:35:00 | 13335.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:50:00 | 13207.00 | 2025-07-01 11:45:00 | 13240.34 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-09 10:25:00 | 13086.00 | 2025-07-09 10:40:00 | 13117.31 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-10 11:05:00 | 13029.00 | 2025-07-10 11:20:00 | 12988.52 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-10 11:05:00 | 13029.00 | 2025-07-10 11:30:00 | 13029.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 09:30:00 | 12986.00 | 2025-07-11 09:35:00 | 13013.08 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-14 09:50:00 | 13027.00 | 2025-07-14 10:20:00 | 13083.18 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-07-14 09:50:00 | 13027.00 | 2025-07-14 11:00:00 | 13027.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 11:15:00 | 13185.00 | 2025-07-15 11:20:00 | 13157.02 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-16 11:10:00 | 13057.00 | 2025-07-16 11:45:00 | 13084.82 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-17 10:35:00 | 13208.00 | 2025-07-17 11:30:00 | 13255.11 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-17 10:35:00 | 13208.00 | 2025-07-17 13:10:00 | 13208.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 09:45:00 | 13249.00 | 2025-07-22 10:45:00 | 13206.46 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-23 10:50:00 | 13201.00 | 2025-07-23 11:50:00 | 13248.24 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-23 10:50:00 | 13201.00 | 2025-07-23 15:20:00 | 13500.00 | TARGET_HIT | 0.50 | 2.26% |
| SELL | retest1 | 2025-07-30 10:10:00 | 13246.00 | 2025-07-30 10:15:00 | 13277.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-31 09:30:00 | 13367.00 | 2025-07-31 09:40:00 | 13321.33 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-01 10:55:00 | 13137.00 | 2025-08-01 11:35:00 | 13089.80 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-01 10:55:00 | 13137.00 | 2025-08-01 15:20:00 | 12907.00 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2025-08-04 11:10:00 | 13033.00 | 2025-08-04 12:25:00 | 13090.33 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-04 11:10:00 | 13033.00 | 2025-08-04 15:20:00 | 13386.00 | TARGET_HIT | 0.50 | 2.71% |
| SELL | retest1 | 2025-09-12 10:50:00 | 13538.00 | 2025-09-12 10:55:00 | 13560.41 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-18 11:15:00 | 13541.00 | 2025-09-18 11:35:00 | 13508.28 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-09-18 11:15:00 | 13541.00 | 2025-09-18 15:20:00 | 13428.00 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2025-09-19 09:50:00 | 13515.00 | 2025-09-19 09:55:00 | 13485.42 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-23 09:35:00 | 13227.00 | 2025-09-23 09:50:00 | 13154.95 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-09-23 09:35:00 | 13227.00 | 2025-09-23 13:50:00 | 13036.00 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2025-09-29 11:10:00 | 12540.00 | 2025-09-29 11:15:00 | 12570.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-08 10:15:00 | 13389.00 | 2025-10-08 10:30:00 | 13337.84 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-08 10:15:00 | 13389.00 | 2025-10-08 15:20:00 | 13244.00 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-10-09 09:45:00 | 13395.00 | 2025-10-09 13:25:00 | 13345.23 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-24 09:35:00 | 12944.00 | 2025-10-24 10:00:00 | 12908.57 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-24 09:35:00 | 12944.00 | 2025-10-24 10:05:00 | 12944.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 10:30:00 | 12822.00 | 2025-10-27 10:40:00 | 12855.76 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-28 09:45:00 | 13081.00 | 2025-10-28 09:55:00 | 13042.39 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-31 09:55:00 | 12712.00 | 2025-10-31 10:15:00 | 12666.51 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-31 09:55:00 | 12712.00 | 2025-10-31 13:35:00 | 12518.00 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2025-11-04 09:35:00 | 12754.00 | 2025-11-04 09:50:00 | 12723.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-17 09:30:00 | 12804.00 | 2025-11-17 09:45:00 | 12768.09 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-18 10:45:00 | 12592.00 | 2025-11-18 10:50:00 | 12618.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-19 10:05:00 | 13081.00 | 2025-11-19 10:10:00 | 13015.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-11-20 10:30:00 | 13043.00 | 2025-11-20 10:45:00 | 13107.61 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-20 10:30:00 | 13043.00 | 2025-11-20 10:50:00 | 13043.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 10:35:00 | 12919.00 | 2025-11-26 10:45:00 | 12892.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-03 10:20:00 | 14422.00 | 2025-12-03 10:30:00 | 14356.31 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-12-09 10:10:00 | 14416.00 | 2025-12-09 10:35:00 | 14321.12 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-12-09 10:10:00 | 14416.00 | 2025-12-09 10:50:00 | 14416.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-11 09:55:00 | 14161.00 | 2025-12-11 10:20:00 | 14207.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-15 10:10:00 | 14246.00 | 2025-12-15 11:40:00 | 14309.89 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-15 10:10:00 | 14246.00 | 2025-12-15 15:20:00 | 14631.00 | TARGET_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2025-12-17 09:40:00 | 15164.00 | 2025-12-17 10:00:00 | 15089.75 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-12-18 10:20:00 | 14932.00 | 2025-12-18 10:25:00 | 15031.69 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-12-18 10:20:00 | 14932.00 | 2025-12-18 11:20:00 | 14932.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 10:45:00 | 15202.00 | 2025-12-19 12:00:00 | 15148.72 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-30 10:45:00 | 14610.00 | 2025-12-30 10:50:00 | 14544.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-30 10:45:00 | 14610.00 | 2025-12-30 12:50:00 | 14610.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:35:00 | 14938.00 | 2026-01-02 09:50:00 | 15004.84 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-01-02 09:35:00 | 14938.00 | 2026-01-02 09:55:00 | 14938.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 10:10:00 | 15669.00 | 2026-01-06 10:15:00 | 15590.37 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-01-07 10:05:00 | 15005.00 | 2026-01-07 10:15:00 | 14930.11 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-07 10:05:00 | 15005.00 | 2026-01-07 11:25:00 | 15005.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 11:05:00 | 13935.00 | 2026-01-13 11:25:00 | 13982.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-01-14 09:55:00 | 13926.00 | 2026-01-14 10:55:00 | 13980.04 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-20 10:55:00 | 13800.00 | 2026-01-20 12:15:00 | 13740.30 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-20 10:55:00 | 13800.00 | 2026-01-20 14:10:00 | 13800.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 09:55:00 | 14007.00 | 2026-01-22 10:20:00 | 13950.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-04 10:55:00 | 15349.00 | 2026-02-04 11:00:00 | 15420.80 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-04 10:55:00 | 15349.00 | 2026-02-04 11:15:00 | 15349.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 11:00:00 | 15093.00 | 2026-02-05 11:15:00 | 15129.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-06 10:05:00 | 15160.00 | 2026-02-06 10:10:00 | 15104.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-09 09:35:00 | 14933.00 | 2026-02-09 10:15:00 | 15023.81 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-09 09:35:00 | 14933.00 | 2026-02-09 11:50:00 | 14955.00 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-02-17 11:00:00 | 15844.00 | 2026-02-17 11:05:00 | 15885.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-18 11:00:00 | 15890.00 | 2026-02-18 11:05:00 | 15835.32 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-18 11:00:00 | 15890.00 | 2026-02-18 15:20:00 | 15585.00 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2026-02-25 10:40:00 | 15645.00 | 2026-02-25 10:50:00 | 15685.37 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-06 10:10:00 | 14512.00 | 2026-03-06 10:15:00 | 14451.52 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-19 10:35:00 | 13470.00 | 2026-03-19 11:15:00 | 13516.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-07 09:45:00 | 13553.00 | 2026-04-07 09:55:00 | 13596.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 11:15:00 | 15038.00 | 2026-04-28 11:20:00 | 15099.61 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-28 11:15:00 | 15038.00 | 2026-04-28 11:45:00 | 15038.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:35:00 | 14860.00 | 2026-04-29 09:45:00 | 14913.47 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-05 10:10:00 | 14659.00 | 2026-05-05 10:35:00 | 14699.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-06 10:20:00 | 14741.00 | 2026-05-06 10:35:00 | 14707.55 | STOP_HIT | 1.00 | -0.23% |
