# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36869 bars)
- **Last close:** 16790.00
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
| ENTRY1 | 34 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 26
- **Target hits / Stop hits / Partials:** 8 / 26 / 17
- **Avg / median % per leg:** 0.37% / 0.00%
- **Sum % (uncompounded):** 18.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 12 | 57.1% | 5 | 9 | 7 | 0.74% | 15.6% |
| BUY @ 2nd Alert (retest1) | 21 | 12 | 57.1% | 5 | 9 | 7 | 0.74% | 15.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 13 | 43.3% | 3 | 17 | 10 | 0.11% | 3.4% |
| SELL @ 2nd Alert (retest1) | 30 | 13 | 43.3% | 3 | 17 | 10 | 0.11% | 3.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 51 | 25 | 49.0% | 8 | 26 | 17 | 0.37% | 18.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 10:50:00 | 7467.25 | 7407.51 | 0.00 | ORB-long ORB[7341.25,7426.90] vol=2.6x ATR=21.51 |
| Stop hit — per-position SL triggered | 2024-05-15 11:10:00 | 7445.74 | 7410.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 7478.55 | 7451.07 | 0.00 | ORB-long ORB[7382.25,7444.45] vol=6.1x ATR=20.28 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 7458.27 | 7450.86 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:50:00 | 7987.65 | 8071.82 | 0.00 | ORB-short ORB[8110.00,8204.15] vol=2.6x ATR=40.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:55:00 | 7926.37 | 8042.45 | 0.00 | T1 1.5R @ 7926.37 |
| Stop hit — per-position SL triggered | 2024-05-27 10:00:00 | 7987.65 | 8038.60 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:10:00 | 13368.30 | 13521.13 | 0.00 | ORB-short ORB[13459.00,13649.65] vol=3.4x ATR=70.09 |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 13438.39 | 13519.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 14874.55 | 14799.48 | 0.00 | ORB-long ORB[14675.00,14800.00] vol=4.0x ATR=66.88 |
| Stop hit — per-position SL triggered | 2024-07-08 10:05:00 | 14807.67 | 14813.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:35:00 | 15175.55 | 15099.04 | 0.00 | ORB-long ORB[14951.65,15143.95] vol=3.5x ATR=149.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:30:00 | 15400.20 | 15264.47 | 0.00 | T1 1.5R @ 15400.20 |
| Target hit | 2024-07-11 11:10:00 | 15315.25 | 15335.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-08-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:25:00 | 13950.00 | 14006.78 | 0.00 | ORB-short ORB[14000.00,14150.00] vol=3.8x ATR=46.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:45:00 | 13879.84 | 13975.35 | 0.00 | T1 1.5R @ 13879.84 |
| Target hit | 2024-08-13 15:20:00 | 13834.10 | 13915.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:00:00 | 13550.00 | 13606.53 | 0.00 | ORB-short ORB[13582.00,13700.00] vol=1.6x ATR=40.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:20:00 | 13489.79 | 13561.77 | 0.00 | T1 1.5R @ 13489.79 |
| Stop hit — per-position SL triggered | 2024-08-14 10:25:00 | 13550.00 | 13558.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:05:00 | 13693.15 | 13789.66 | 0.00 | ORB-short ORB[13912.00,14063.00] vol=2.1x ATR=52.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:15:00 | 13614.72 | 13781.00 | 0.00 | T1 1.5R @ 13614.72 |
| Stop hit — per-position SL triggered | 2024-08-19 11:20:00 | 13693.15 | 13777.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:55:00 | 13500.00 | 13566.30 | 0.00 | ORB-short ORB[13555.00,13699.00] vol=1.6x ATR=54.33 |
| Stop hit — per-position SL triggered | 2024-08-20 10:00:00 | 13554.33 | 13562.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:35:00 | 13700.00 | 13947.44 | 0.00 | ORB-short ORB[13956.50,14100.00] vol=2.3x ATR=85.36 |
| Stop hit — per-position SL triggered | 2024-09-09 09:40:00 | 13785.36 | 13923.38 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 11:15:00 | 14400.00 | 14444.55 | 0.00 | ORB-short ORB[14410.00,14550.00] vol=6.1x ATR=46.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 14:35:00 | 14329.85 | 14411.17 | 0.00 | T1 1.5R @ 14329.85 |
| Stop hit — per-position SL triggered | 2024-09-17 15:10:00 | 14400.00 | 14399.04 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 14005.00 | 14108.81 | 0.00 | ORB-short ORB[14100.00,14300.00] vol=3.3x ATR=75.68 |
| Stop hit — per-position SL triggered | 2024-09-19 09:40:00 | 14080.68 | 14108.93 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:55:00 | 13200.00 | 13252.99 | 0.00 | ORB-short ORB[13250.00,13400.00] vol=3.5x ATR=87.14 |
| Stop hit — per-position SL triggered | 2024-09-26 10:00:00 | 13287.14 | 13256.44 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:00:00 | 13595.00 | 13678.05 | 0.00 | ORB-short ORB[13600.00,13800.00] vol=3.5x ATR=97.68 |
| Stop hit — per-position SL triggered | 2024-10-03 11:05:00 | 13692.68 | 13675.99 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 09:45:00 | 13474.70 | 13548.66 | 0.00 | ORB-short ORB[13581.10,13780.00] vol=1.6x ATR=86.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 10:05:00 | 13344.99 | 13516.03 | 0.00 | T1 1.5R @ 13344.99 |
| Target hit | 2024-10-10 15:00:00 | 13405.15 | 13393.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2024-10-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:00:00 | 13337.55 | 13409.85 | 0.00 | ORB-short ORB[13455.25,13540.00] vol=2.1x ATR=35.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 13:05:00 | 13283.60 | 13369.83 | 0.00 | T1 1.5R @ 13283.60 |
| Stop hit — per-position SL triggered | 2024-10-11 13:40:00 | 13337.55 | 13352.16 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:50:00 | 13235.70 | 13294.53 | 0.00 | ORB-short ORB[13300.45,13400.00] vol=1.9x ATR=25.64 |
| Stop hit — per-position SL triggered | 2024-10-14 11:00:00 | 13261.34 | 13293.69 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:55:00 | 13591.70 | 13526.78 | 0.00 | ORB-long ORB[13404.45,13500.00] vol=2.4x ATR=47.01 |
| Stop hit — per-position SL triggered | 2024-10-16 10:00:00 | 13544.69 | 13533.19 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:25:00 | 12290.00 | 12126.54 | 0.00 | ORB-long ORB[11947.55,12090.90] vol=2.7x ATR=59.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:30:00 | 12378.89 | 12218.42 | 0.00 | T1 1.5R @ 12378.89 |
| Target hit | 2024-10-30 11:15:00 | 12321.35 | 12336.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 12064.00 | 12006.06 | 0.00 | ORB-long ORB[11894.95,11999.00] vol=1.5x ATR=81.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:45:00 | 12186.09 | 12043.84 | 0.00 | T1 1.5R @ 12186.09 |
| Stop hit — per-position SL triggered | 2024-11-06 09:50:00 | 12064.00 | 12056.42 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:15:00 | 12123.00 | 12158.90 | 0.00 | ORB-short ORB[12130.20,12275.00] vol=4.8x ATR=23.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:25:00 | 12087.65 | 12143.55 | 0.00 | T1 1.5R @ 12087.65 |
| Target hit | 2024-11-29 15:20:00 | 11925.00 | 12001.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 11884.00 | 11906.68 | 0.00 | ORB-short ORB[11893.05,12000.00] vol=2.7x ATR=39.82 |
| Stop hit — per-position SL triggered | 2024-12-04 11:30:00 | 11923.82 | 11904.97 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:10:00 | 11795.00 | 11690.08 | 0.00 | ORB-long ORB[11608.65,11750.00] vol=2.0x ATR=33.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:15:00 | 11845.26 | 11707.83 | 0.00 | T1 1.5R @ 11845.26 |
| Stop hit — per-position SL triggered | 2024-12-09 11:20:00 | 11795.00 | 11708.72 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 11708.70 | 11722.80 | 0.00 | ORB-short ORB[11735.20,11881.80] vol=2.4x ATR=47.02 |
| Stop hit — per-position SL triggered | 2024-12-11 11:25:00 | 11755.72 | 11722.73 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:25:00 | 12498.00 | 12457.00 | 0.00 | ORB-long ORB[12208.00,12325.85] vol=14.2x ATR=56.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:40:00 | 12583.28 | 12461.54 | 0.00 | T1 1.5R @ 12583.28 |
| Target hit | 2024-12-27 15:20:00 | 13583.65 | 12750.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-01-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:45:00 | 16584.10 | 16884.89 | 0.00 | ORB-short ORB[17150.05,17400.00] vol=3.0x ATR=120.27 |
| Stop hit — per-position SL triggered | 2025-01-17 11:15:00 | 16704.37 | 16837.15 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 13766.85 | 13664.30 | 0.00 | ORB-long ORB[13421.85,13622.50] vol=3.5x ATR=102.85 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 13664.00 | 13675.86 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-02-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:50:00 | 14935.25 | 14773.06 | 0.00 | ORB-long ORB[14560.00,14734.55] vol=4.6x ATR=54.31 |
| Stop hit — per-position SL triggered | 2025-02-06 10:55:00 | 14880.94 | 14784.52 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:45:00 | 12898.85 | 12749.12 | 0.00 | ORB-long ORB[12538.55,12723.15] vol=1.6x ATR=61.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:50:00 | 12991.43 | 12835.77 | 0.00 | T1 1.5R @ 12991.43 |
| Target hit | 2025-03-21 15:20:00 | 13212.05 | 13130.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-04-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:30:00 | 13991.00 | 13677.41 | 0.00 | ORB-long ORB[13335.00,13529.00] vol=1.8x ATR=73.92 |
| Stop hit — per-position SL triggered | 2025-04-15 10:35:00 | 13917.08 | 13690.23 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:55:00 | 14280.00 | 14358.41 | 0.00 | ORB-short ORB[14301.00,14499.00] vol=2.0x ATR=59.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:10:00 | 14190.01 | 14311.72 | 0.00 | T1 1.5R @ 14190.01 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 14280.00 | 14284.56 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 13769.00 | 13602.23 | 0.00 | ORB-long ORB[13473.00,13599.00] vol=1.6x ATR=72.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 09:45:00 | 13877.04 | 13686.15 | 0.00 | T1 1.5R @ 13877.04 |
| Target hit | 2025-04-28 11:40:00 | 13828.00 | 13831.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:10:00 | 13670.00 | 13751.69 | 0.00 | ORB-short ORB[13755.00,13930.00] vol=6.8x ATR=39.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 12:30:00 | 13610.44 | 13732.29 | 0.00 | T1 1.5R @ 13610.44 |
| Stop hit — per-position SL triggered | 2025-04-29 14:45:00 | 13670.00 | 13699.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 10:50:00 | 7467.25 | 2024-05-15 11:10:00 | 7445.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-17 09:35:00 | 7478.55 | 2024-05-17 09:40:00 | 7458.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-27 09:50:00 | 7987.65 | 2024-05-27 09:55:00 | 7926.37 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-05-27 09:50:00 | 7987.65 | 2024-05-27 10:00:00 | 7987.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 10:10:00 | 13368.30 | 2024-06-26 10:15:00 | 13438.39 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-07-08 09:30:00 | 14874.55 | 2024-07-08 10:05:00 | 14807.67 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-11 09:35:00 | 15175.55 | 2024-07-11 10:30:00 | 15400.20 | PARTIAL | 0.50 | 1.48% |
| BUY | retest1 | 2024-07-11 09:35:00 | 15175.55 | 2024-07-11 11:10:00 | 15315.25 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2024-08-13 10:25:00 | 13950.00 | 2024-08-13 11:45:00 | 13879.84 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-08-13 10:25:00 | 13950.00 | 2024-08-13 15:20:00 | 13834.10 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-08-14 10:00:00 | 13550.00 | 2024-08-14 10:20:00 | 13489.79 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-14 10:00:00 | 13550.00 | 2024-08-14 10:25:00 | 13550.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-19 11:05:00 | 13693.15 | 2024-08-19 11:15:00 | 13614.72 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-19 11:05:00 | 13693.15 | 2024-08-19 11:20:00 | 13693.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 09:55:00 | 13500.00 | 2024-08-20 10:00:00 | 13554.33 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-09 09:35:00 | 13700.00 | 2024-09-09 09:40:00 | 13785.36 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-09-17 11:15:00 | 14400.00 | 2024-09-17 14:35:00 | 14329.85 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-17 11:15:00 | 14400.00 | 2024-09-17 15:10:00 | 14400.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:35:00 | 14005.00 | 2024-09-19 09:40:00 | 14080.68 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-09-26 09:55:00 | 13200.00 | 2024-09-26 10:00:00 | 13287.14 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2024-10-03 11:00:00 | 13595.00 | 2024-10-03 11:05:00 | 13692.68 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-10-10 09:45:00 | 13474.70 | 2024-10-10 10:05:00 | 13344.99 | PARTIAL | 0.50 | 0.96% |
| SELL | retest1 | 2024-10-10 09:45:00 | 13474.70 | 2024-10-10 15:00:00 | 13405.15 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-11 11:00:00 | 13337.55 | 2024-10-11 13:05:00 | 13283.60 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-11 11:00:00 | 13337.55 | 2024-10-11 13:40:00 | 13337.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 10:50:00 | 13235.70 | 2024-10-14 11:00:00 | 13261.34 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-10-16 09:55:00 | 13591.70 | 2024-10-16 10:00:00 | 13544.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-30 10:25:00 | 12290.00 | 2024-10-30 10:30:00 | 12378.89 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-30 10:25:00 | 12290.00 | 2024-10-30 11:15:00 | 12321.35 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-11-06 09:40:00 | 12064.00 | 2024-11-06 09:45:00 | 12186.09 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2024-11-06 09:40:00 | 12064.00 | 2024-11-06 09:50:00 | 12064.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 11:15:00 | 12123.00 | 2024-11-29 11:25:00 | 12087.65 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-11-29 11:15:00 | 12123.00 | 2024-11-29 15:20:00 | 11925.00 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2024-12-04 11:00:00 | 11884.00 | 2024-12-04 11:30:00 | 11923.82 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-09 11:10:00 | 11795.00 | 2024-12-09 11:15:00 | 11845.26 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-09 11:10:00 | 11795.00 | 2024-12-09 11:20:00 | 11795.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-11 11:05:00 | 11708.70 | 2024-12-11 11:25:00 | 11755.72 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-27 10:25:00 | 12498.00 | 2024-12-27 10:40:00 | 12583.28 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-27 10:25:00 | 12498.00 | 2024-12-27 15:20:00 | 13583.65 | TARGET_HIT | 0.50 | 8.69% |
| SELL | retest1 | 2025-01-17 10:45:00 | 16584.10 | 2025-01-17 11:15:00 | 16704.37 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-01-31 09:40:00 | 13766.85 | 2025-01-31 10:05:00 | 13664.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2025-02-06 10:50:00 | 14935.25 | 2025-02-06 10:55:00 | 14880.94 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-21 09:45:00 | 12898.85 | 2025-03-21 09:50:00 | 12991.43 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-03-21 09:45:00 | 12898.85 | 2025-03-21 15:20:00 | 13212.05 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2025-04-15 10:30:00 | 13991.00 | 2025-04-15 10:35:00 | 13917.08 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-04-23 09:55:00 | 14280.00 | 2025-04-23 10:10:00 | 14190.01 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-04-23 09:55:00 | 14280.00 | 2025-04-23 11:10:00 | 14280.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 09:30:00 | 13769.00 | 2025-04-28 09:45:00 | 13877.04 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-04-28 09:30:00 | 13769.00 | 2025-04-28 11:40:00 | 13828.00 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-04-29 11:10:00 | 13670.00 | 2025-04-29 12:30:00 | 13610.44 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-04-29 11:10:00 | 13670.00 | 2025-04-29 14:45:00 | 13670.00 | STOP_HIT | 0.50 | 0.00% |
