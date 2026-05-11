# Maruti Suzuki India Ltd. (MARUTI)

## Backtest Summary

- **Window:** 2024-11-07 09:15:00 → 2026-05-08 15:25:00 (27688 bars)
- **Last close:** 13733.00
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 6 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 37
- **Target hits / Stop hits / Partials:** 6 / 37 / 16
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 7.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 11 | 35.5% | 2 | 20 | 9 | 0.16% | 5.0% |
| BUY @ 2nd Alert (retest1) | 31 | 11 | 35.5% | 2 | 20 | 9 | 0.16% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 11 | 39.3% | 4 | 17 | 7 | 0.09% | 2.6% |
| SELL @ 2nd Alert (retest1) | 28 | 11 | 39.3% | 4 | 17 | 7 | 0.09% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 59 | 22 | 37.3% | 6 | 37 | 16 | 0.13% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:45:00 | 11236.90 | 11270.39 | 0.00 | ORB-short ORB[11245.55,11385.00] vol=1.7x ATR=51.78 |
| Stop hit — per-position SL triggered | 2024-11-07 11:25:00 | 11288.68 | 11262.09 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-11-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 10:55:00 | 11061.05 | 10989.54 | 0.00 | ORB-long ORB[10915.00,11006.00] vol=2.4x ATR=31.16 |
| Stop hit — per-position SL triggered | 2024-11-18 11:05:00 | 11029.89 | 10993.67 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 10942.30 | 11014.60 | 0.00 | ORB-short ORB[10989.05,11070.05] vol=2.4x ATR=31.40 |
| Stop hit — per-position SL triggered | 2024-11-28 11:00:00 | 10973.70 | 10993.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-11-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:05:00 | 11055.00 | 10977.99 | 0.00 | ORB-long ORB[10920.00,11000.00] vol=1.7x ATR=27.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:25:00 | 11095.66 | 10990.68 | 0.00 | T1 1.5R @ 11095.66 |
| Stop hit — per-position SL triggered | 2024-11-29 13:50:00 | 11055.00 | 11037.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 11:05:00 | 11280.70 | 11271.20 | 0.00 | ORB-long ORB[11203.00,11274.95] vol=1.7x ATR=17.57 |
| Stop hit — per-position SL triggered | 2024-12-03 11:10:00 | 11263.13 | 11270.12 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:50:00 | 11235.00 | 11263.09 | 0.00 | ORB-short ORB[11261.50,11300.00] vol=2.2x ATR=22.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:55:00 | 11201.25 | 11253.45 | 0.00 | T1 1.5R @ 11201.25 |
| Target hit | 2024-12-04 15:20:00 | 11107.40 | 11193.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 11200.05 | 11238.44 | 0.00 | ORB-short ORB[11234.65,11299.35] vol=2.1x ATR=18.36 |
| Stop hit — per-position SL triggered | 2024-12-12 09:40:00 | 11218.41 | 11233.34 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-12-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:45:00 | 11075.30 | 11089.81 | 0.00 | ORB-short ORB[11098.35,11162.20] vol=1.6x ATR=24.03 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 11099.33 | 11088.75 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-12-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:05:00 | 11189.00 | 11204.85 | 0.00 | ORB-short ORB[11202.15,11267.95] vol=1.6x ATR=21.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:40:00 | 11156.68 | 11201.04 | 0.00 | T1 1.5R @ 11156.68 |
| Target hit | 2024-12-17 15:20:00 | 11072.00 | 11149.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 10924.00 | 10851.11 | 0.00 | ORB-long ORB[10770.00,10869.95] vol=2.3x ATR=23.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:10:00 | 10959.98 | 10884.48 | 0.00 | T1 1.5R @ 10959.98 |
| Stop hit — per-position SL triggered | 2024-12-26 12:40:00 | 10924.00 | 10903.85 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 10885.00 | 10837.66 | 0.00 | ORB-long ORB[10826.00,10878.00] vol=2.2x ATR=20.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:15:00 | 10916.28 | 10853.30 | 0.00 | T1 1.5R @ 10916.28 |
| Stop hit — per-position SL triggered | 2025-01-01 11:40:00 | 10885.00 | 10859.36 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 11390.00 | 11297.18 | 0.00 | ORB-long ORB[11226.00,11319.85] vol=3.0x ATR=41.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:15:00 | 11451.72 | 11326.70 | 0.00 | T1 1.5R @ 11451.72 |
| Target hit | 2025-01-02 15:20:00 | 11838.75 | 11641.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-01-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 11:10:00 | 12029.30 | 11924.79 | 0.00 | ORB-long ORB[11787.95,11950.00] vol=2.1x ATR=35.94 |
| Stop hit — per-position SL triggered | 2025-01-03 11:15:00 | 11993.36 | 11928.59 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 11837.00 | 11882.88 | 0.00 | ORB-short ORB[11870.60,11979.90] vol=1.6x ATR=31.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 11790.48 | 11873.33 | 0.00 | T1 1.5R @ 11790.48 |
| Target hit | 2025-01-06 15:20:00 | 11761.05 | 11793.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 11800.90 | 11765.14 | 0.00 | ORB-long ORB[11725.30,11796.00] vol=1.8x ATR=25.91 |
| Stop hit — per-position SL triggered | 2025-01-08 11:20:00 | 11774.99 | 11765.61 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 11672.10 | 11728.32 | 0.00 | ORB-short ORB[11741.15,11850.00] vol=2.9x ATR=27.08 |
| Stop hit — per-position SL triggered | 2025-01-09 10:50:00 | 11699.18 | 11725.40 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:40:00 | 12180.85 | 12147.57 | 0.00 | ORB-long ORB[12054.60,12155.90] vol=1.8x ATR=44.38 |
| Stop hit — per-position SL triggered | 2025-01-29 09:45:00 | 12136.47 | 12148.70 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:05:00 | 12403.25 | 12336.97 | 0.00 | ORB-long ORB[12228.65,12390.05] vol=4.3x ATR=32.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:10:00 | 12451.82 | 12358.33 | 0.00 | T1 1.5R @ 12451.82 |
| Stop hit — per-position SL triggered | 2025-02-01 11:45:00 | 12403.25 | 12388.07 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 13105.00 | 13160.72 | 0.00 | ORB-short ORB[13139.95,13259.95] vol=2.3x ATR=33.35 |
| Stop hit — per-position SL triggered | 2025-02-04 09:40:00 | 13138.35 | 13154.55 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:10:00 | 13033.00 | 13057.08 | 0.00 | ORB-short ORB[13040.50,13150.00] vol=1.6x ATR=24.18 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 13057.18 | 13056.34 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-02-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:35:00 | 13103.50 | 13062.37 | 0.00 | ORB-long ORB[13021.85,13096.40] vol=3.6x ATR=26.55 |
| Stop hit — per-position SL triggered | 2025-02-06 10:55:00 | 13076.95 | 13066.32 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-02-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:00:00 | 12645.00 | 12703.90 | 0.00 | ORB-short ORB[12700.00,12780.00] vol=1.6x ATR=23.88 |
| Stop hit — per-position SL triggered | 2025-02-14 11:45:00 | 12668.88 | 12690.68 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 09:40:00 | 12876.00 | 12820.21 | 0.00 | ORB-long ORB[12735.15,12834.05] vol=1.8x ATR=24.64 |
| Stop hit — per-position SL triggered | 2025-02-18 09:50:00 | 12851.36 | 12824.83 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-02-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:00:00 | 12300.80 | 12338.16 | 0.00 | ORB-short ORB[12353.45,12438.60] vol=2.9x ATR=24.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:10:00 | 12263.69 | 12332.38 | 0.00 | T1 1.5R @ 12263.69 |
| Stop hit — per-position SL triggered | 2025-02-21 10:55:00 | 12300.80 | 12321.25 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 12459.40 | 12430.26 | 0.00 | ORB-long ORB[12350.00,12455.00] vol=2.1x ATR=25.84 |
| Stop hit — per-position SL triggered | 2025-02-25 09:40:00 | 12433.56 | 12434.75 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:50:00 | 12390.50 | 12426.43 | 0.00 | ORB-short ORB[12402.20,12550.00] vol=2.3x ATR=22.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 13:00:00 | 12356.59 | 12404.20 | 0.00 | T1 1.5R @ 12356.59 |
| Stop hit — per-position SL triggered | 2025-02-27 15:05:00 | 12390.50 | 12385.89 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-03-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 10:00:00 | 11642.90 | 11659.94 | 0.00 | ORB-short ORB[11650.00,11724.25] vol=4.7x ATR=33.20 |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 11676.10 | 11658.60 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:10:00 | 11710.35 | 11645.60 | 0.00 | ORB-long ORB[11568.35,11690.00] vol=1.6x ATR=19.87 |
| Stop hit — per-position SL triggered | 2025-03-05 11:30:00 | 11690.48 | 11653.58 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 11544.60 | 11562.81 | 0.00 | ORB-short ORB[11571.95,11729.50] vol=1.8x ATR=28.53 |
| Stop hit — per-position SL triggered | 2025-03-06 11:20:00 | 11573.13 | 11562.88 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-03-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 10:10:00 | 11625.65 | 11650.74 | 0.00 | ORB-short ORB[11632.00,11742.30] vol=2.2x ATR=30.00 |
| Stop hit — per-position SL triggered | 2025-03-07 10:35:00 | 11655.65 | 11646.24 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 11:10:00 | 11670.00 | 11634.57 | 0.00 | ORB-long ORB[11571.30,11648.05] vol=3.1x ATR=19.51 |
| Stop hit — per-position SL triggered | 2025-03-10 11:25:00 | 11650.49 | 11635.87 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 11673.95 | 11634.26 | 0.00 | ORB-long ORB[11513.80,11660.00] vol=1.7x ATR=29.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:45:00 | 11718.35 | 11651.33 | 0.00 | T1 1.5R @ 11718.35 |
| Stop hit — per-position SL triggered | 2025-03-17 09:55:00 | 11673.95 | 11662.66 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:15:00 | 11625.35 | 11604.36 | 0.00 | ORB-long ORB[11501.00,11624.85] vol=6.1x ATR=21.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:45:00 | 11656.97 | 11607.51 | 0.00 | T1 1.5R @ 11656.97 |
| Stop hit — per-position SL triggered | 2025-03-18 12:30:00 | 11625.35 | 11612.69 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 11905.10 | 11844.28 | 0.00 | ORB-long ORB[11765.00,11847.00] vol=1.7x ATR=26.14 |
| Stop hit — per-position SL triggered | 2025-03-21 11:05:00 | 11878.96 | 11849.42 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 11861.00 | 11795.20 | 0.00 | ORB-long ORB[11779.95,11826.95] vol=1.5x ATR=26.60 |
| Stop hit — per-position SL triggered | 2025-03-24 10:50:00 | 11834.40 | 11798.17 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-03-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:30:00 | 11950.00 | 12011.12 | 0.00 | ORB-short ORB[11951.40,12075.25] vol=5.9x ATR=36.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 10:55:00 | 11895.29 | 11983.29 | 0.00 | T1 1.5R @ 11895.29 |
| Target hit | 2025-03-25 15:20:00 | 11838.75 | 11936.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-04-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:45:00 | 11317.50 | 11433.44 | 0.00 | ORB-short ORB[11455.05,11550.35] vol=2.6x ATR=39.26 |
| Stop hit — per-position SL triggered | 2025-04-08 11:05:00 | 11356.76 | 11401.46 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 11:00:00 | 11880.00 | 11824.18 | 0.00 | ORB-long ORB[11680.00,11847.00] vol=3.5x ATR=27.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 12:35:00 | 11921.00 | 11852.33 | 0.00 | T1 1.5R @ 11921.00 |
| Stop hit — per-position SL triggered | 2025-04-15 13:45:00 | 11880.00 | 11864.64 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 11:00:00 | 11646.00 | 11680.63 | 0.00 | ORB-short ORB[11694.00,11788.00] vol=1.6x ATR=21.60 |
| Stop hit — per-position SL triggered | 2025-04-22 11:30:00 | 11667.60 | 11676.33 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:15:00 | 11799.00 | 11820.67 | 0.00 | ORB-short ORB[11812.00,11940.00] vol=2.9x ATR=24.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:45:00 | 11762.58 | 11815.15 | 0.00 | T1 1.5R @ 11762.58 |
| Stop hit — per-position SL triggered | 2025-04-29 12:25:00 | 11799.00 | 11804.12 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 11930.00 | 11909.40 | 0.00 | ORB-long ORB[11800.00,11922.00] vol=8.2x ATR=17.89 |
| Stop hit — per-position SL triggered | 2025-04-30 11:25:00 | 11912.11 | 11909.62 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:40:00 | 12493.00 | 12461.34 | 0.00 | ORB-long ORB[12371.00,12471.00] vol=1.7x ATR=32.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:40:00 | 12541.15 | 12487.73 | 0.00 | T1 1.5R @ 12541.15 |
| Target hit | 2025-05-06 15:20:00 | 12572.00 | 12548.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-05-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 10:55:00 | 12420.00 | 12474.15 | 0.00 | ORB-short ORB[12490.00,12597.00] vol=2.0x ATR=21.41 |
| Stop hit — per-position SL triggered | 2025-05-08 11:00:00 | 12441.41 | 12471.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-07 10:45:00 | 11236.90 | 2024-11-07 11:25:00 | 11288.68 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-11-18 10:55:00 | 11061.05 | 2024-11-18 11:05:00 | 11029.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-28 10:35:00 | 10942.30 | 2024-11-28 11:00:00 | 10973.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-29 11:05:00 | 11055.00 | 2024-11-29 11:25:00 | 11095.66 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-29 11:05:00 | 11055.00 | 2024-11-29 13:50:00 | 11055.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 11:05:00 | 11280.70 | 2024-12-03 11:10:00 | 11263.13 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-04 10:50:00 | 11235.00 | 2024-12-04 10:55:00 | 11201.25 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-04 10:50:00 | 11235.00 | 2024-12-04 15:20:00 | 11107.40 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-12-12 09:35:00 | 11200.05 | 2024-12-12 09:40:00 | 11218.41 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-13 10:45:00 | 11075.30 | 2024-12-13 11:10:00 | 11099.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-17 11:05:00 | 11189.00 | 2024-12-17 11:40:00 | 11156.68 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-17 11:05:00 | 11189.00 | 2024-12-17 15:20:00 | 11072.00 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2024-12-26 10:50:00 | 10924.00 | 2024-12-26 11:10:00 | 10959.98 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-12-26 10:50:00 | 10924.00 | 2024-12-26 12:40:00 | 10924.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:50:00 | 10885.00 | 2025-01-01 11:15:00 | 10916.28 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-01-01 10:50:00 | 10885.00 | 2025-01-01 11:40:00 | 10885.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 10:05:00 | 11390.00 | 2025-01-02 10:15:00 | 11451.72 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-02 10:05:00 | 11390.00 | 2025-01-02 15:20:00 | 11838.75 | TARGET_HIT | 0.50 | 3.94% |
| BUY | retest1 | 2025-01-03 11:10:00 | 12029.30 | 2025-01-03 11:15:00 | 11993.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-06 10:50:00 | 11837.00 | 2025-01-06 11:10:00 | 11790.48 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-06 10:50:00 | 11837.00 | 2025-01-06 15:20:00 | 11761.05 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-08 11:15:00 | 11800.90 | 2025-01-08 11:20:00 | 11774.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-09 10:45:00 | 11672.10 | 2025-01-09 10:50:00 | 11699.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-29 09:40:00 | 12180.85 | 2025-01-29 09:45:00 | 12136.47 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-02-01 11:05:00 | 12403.25 | 2025-02-01 11:10:00 | 12451.82 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-02-01 11:05:00 | 12403.25 | 2025-02-01 11:45:00 | 12403.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 09:35:00 | 13105.00 | 2025-02-04 09:40:00 | 13138.35 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-05 11:10:00 | 13033.00 | 2025-02-05 11:20:00 | 13057.18 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-02-06 10:35:00 | 13103.50 | 2025-02-06 10:55:00 | 13076.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-02-14 11:00:00 | 12645.00 | 2025-02-14 11:45:00 | 12668.88 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-02-18 09:40:00 | 12876.00 | 2025-02-18 09:50:00 | 12851.36 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-02-21 10:00:00 | 12300.80 | 2025-02-21 10:10:00 | 12263.69 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-02-21 10:00:00 | 12300.80 | 2025-02-21 10:55:00 | 12300.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:30:00 | 12459.40 | 2025-02-25 09:40:00 | 12433.56 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-02-27 10:50:00 | 12390.50 | 2025-02-27 13:00:00 | 12356.59 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-02-27 10:50:00 | 12390.50 | 2025-02-27 15:05:00 | 12390.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-04 10:00:00 | 11642.90 | 2025-03-04 10:15:00 | 11676.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-05 11:10:00 | 11710.35 | 2025-03-05 11:30:00 | 11690.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-03-06 11:05:00 | 11544.60 | 2025-03-06 11:20:00 | 11573.13 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-03-07 10:10:00 | 11625.65 | 2025-03-07 10:35:00 | 11655.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-10 11:10:00 | 11670.00 | 2025-03-10 11:25:00 | 11650.49 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-03-17 09:35:00 | 11673.95 | 2025-03-17 09:45:00 | 11718.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-17 09:35:00 | 11673.95 | 2025-03-17 09:55:00 | 11673.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 11:15:00 | 11625.35 | 2025-03-18 11:45:00 | 11656.97 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-03-18 11:15:00 | 11625.35 | 2025-03-18 12:30:00 | 11625.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:50:00 | 11905.10 | 2025-03-21 11:05:00 | 11878.96 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-24 10:40:00 | 11861.00 | 2025-03-24 10:50:00 | 11834.40 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-03-25 10:30:00 | 11950.00 | 2025-03-25 10:55:00 | 11895.29 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-03-25 10:30:00 | 11950.00 | 2025-03-25 15:20:00 | 11838.75 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2025-04-08 10:45:00 | 11317.50 | 2025-04-08 11:05:00 | 11356.76 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-15 11:00:00 | 11880.00 | 2025-04-15 12:35:00 | 11921.00 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-04-15 11:00:00 | 11880.00 | 2025-04-15 13:45:00 | 11880.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-22 11:00:00 | 11646.00 | 2025-04-22 11:30:00 | 11667.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-04-29 11:15:00 | 11799.00 | 2025-04-29 11:45:00 | 11762.58 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-04-29 11:15:00 | 11799.00 | 2025-04-29 12:25:00 | 11799.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 11:15:00 | 11930.00 | 2025-04-30 11:25:00 | 11912.11 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-05-06 09:40:00 | 12493.00 | 2025-05-06 10:40:00 | 12541.15 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-06 09:40:00 | 12493.00 | 2025-05-06 15:20:00 | 12572.00 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-05-08 10:55:00 | 12420.00 | 2025-05-08 11:00:00 | 12441.41 | STOP_HIT | 1.00 | -0.17% |
