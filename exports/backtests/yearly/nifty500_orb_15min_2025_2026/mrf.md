# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 130490.00
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 19 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 68
- **Target hits / Stop hits / Partials:** 19 / 68 / 33
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 20.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 18 | 36.7% | 7 | 31 | 11 | 0.14% | 6.7% |
| BUY @ 2nd Alert (retest1) | 49 | 18 | 36.7% | 7 | 31 | 11 | 0.14% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 71 | 34 | 47.9% | 12 | 37 | 22 | 0.19% | 13.5% |
| SELL @ 2nd Alert (retest1) | 71 | 34 | 47.9% | 12 | 37 | 22 | 0.19% | 13.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 52 | 43.3% | 19 | 68 | 33 | 0.17% | 20.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:25:00 | 142155.00 | 141719.34 | 0.00 | ORB-long ORB[141415.00,142000.00] vol=2.2x ATR=249.78 |
| Stop hit — per-position SL triggered | 2025-05-16 10:30:00 | 141905.22 | 141705.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:35:00 | 144835.00 | 144384.51 | 0.00 | ORB-long ORB[143705.00,144730.00] vol=2.0x ATR=255.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:45:00 | 145218.14 | 144504.00 | 0.00 | T1 1.5R @ 145218.14 |
| Target hit | 2025-05-26 15:20:00 | 146685.00 | 146140.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 141990.00 | 142909.90 | 0.00 | ORB-short ORB[142250.00,143920.00] vol=2.1x ATR=377.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 12:10:00 | 141423.24 | 142673.39 | 0.00 | T1 1.5R @ 141423.24 |
| Target hit | 2025-05-29 15:20:00 | 140865.00 | 141918.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:55:00 | 138800.00 | 139896.50 | 0.00 | ORB-short ORB[140000.00,141900.00] vol=2.0x ATR=469.25 |
| Stop hit — per-position SL triggered | 2025-06-03 10:20:00 | 139269.25 | 139665.35 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:40:00 | 136840.00 | 137317.14 | 0.00 | ORB-short ORB[137335.00,139000.00] vol=1.7x ATR=410.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 15:05:00 | 136224.00 | 136947.71 | 0.00 | T1 1.5R @ 136224.00 |
| Target hit | 2025-06-04 15:20:00 | 136000.00 | 136788.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:45:00 | 138040.00 | 137449.81 | 0.00 | ORB-long ORB[136535.00,137705.00] vol=1.7x ATR=336.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 10:05:00 | 138544.17 | 137787.16 | 0.00 | T1 1.5R @ 138544.17 |
| Target hit | 2025-06-05 15:20:00 | 140400.00 | 139260.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-06-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:50:00 | 139320.00 | 140403.91 | 0.00 | ORB-short ORB[139540.00,141335.00] vol=1.5x ATR=426.72 |
| Stop hit — per-position SL triggered | 2025-06-09 10:00:00 | 139746.72 | 140307.71 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:45:00 | 136880.00 | 136450.89 | 0.00 | ORB-long ORB[136100.00,136675.00] vol=1.5x ATR=194.68 |
| Stop hit — per-position SL triggered | 2025-06-18 09:50:00 | 136685.32 | 136479.64 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:35:00 | 137150.00 | 138017.53 | 0.00 | ORB-short ORB[138185.00,138875.00] vol=3.5x ATR=329.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:55:00 | 136655.37 | 137486.37 | 0.00 | T1 1.5R @ 136655.37 |
| Target hit | 2025-06-20 11:00:00 | 137085.00 | 137054.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2025-06-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:00:00 | 136810.00 | 136219.52 | 0.00 | ORB-long ORB[135830.00,136660.00] vol=3.6x ATR=272.24 |
| Stop hit — per-position SL triggered | 2025-06-24 13:00:00 | 136537.76 | 136408.12 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 10:45:00 | 139895.00 | 139384.98 | 0.00 | ORB-long ORB[139240.00,139880.00] vol=2.2x ATR=321.92 |
| Stop hit — per-position SL triggered | 2025-06-26 10:50:00 | 139573.08 | 139391.24 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 141925.00 | 142954.90 | 0.00 | ORB-short ORB[142705.00,143695.00] vol=2.0x ATR=280.61 |
| Stop hit — per-position SL triggered | 2025-07-01 11:35:00 | 142205.61 | 142763.56 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 143650.00 | 143324.63 | 0.00 | ORB-long ORB[142800.00,143465.00] vol=2.5x ATR=401.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:35:00 | 144252.15 | 143838.19 | 0.00 | T1 1.5R @ 144252.15 |
| Target hit | 2025-07-02 09:55:00 | 143950.00 | 143975.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2025-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:00:00 | 143925.00 | 144549.51 | 0.00 | ORB-short ORB[144050.00,146000.00] vol=5.5x ATR=266.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:45:00 | 143525.53 | 144401.90 | 0.00 | T1 1.5R @ 143525.53 |
| Stop hit — per-position SL triggered | 2025-07-04 13:10:00 | 143925.00 | 144316.50 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:05:00 | 142685.00 | 143201.31 | 0.00 | ORB-short ORB[143005.00,144660.00] vol=1.7x ATR=220.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 12:50:00 | 142354.47 | 143010.16 | 0.00 | T1 1.5R @ 142354.47 |
| Stop hit — per-position SL triggered | 2025-07-07 14:40:00 | 142685.00 | 142902.56 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 146140.00 | 145355.36 | 0.00 | ORB-long ORB[144000.00,145700.00] vol=2.6x ATR=448.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:30:00 | 146813.27 | 146309.97 | 0.00 | T1 1.5R @ 146813.27 |
| Target hit | 2025-07-09 15:20:00 | 150215.00 | 149551.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:50:00 | 147955.00 | 149138.38 | 0.00 | ORB-short ORB[149070.00,150700.00] vol=1.5x ATR=355.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:15:00 | 147421.66 | 148842.02 | 0.00 | T1 1.5R @ 147421.66 |
| Stop hit — per-position SL triggered | 2025-07-10 11:40:00 | 147955.00 | 148595.58 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 148480.00 | 149723.30 | 0.00 | ORB-short ORB[150025.00,151500.00] vol=2.0x ATR=309.44 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 148789.44 | 149687.09 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:40:00 | 150800.00 | 149896.53 | 0.00 | ORB-long ORB[148500.00,150195.00] vol=2.8x ATR=448.55 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 150351.45 | 149950.55 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:40:00 | 151230.00 | 150601.71 | 0.00 | ORB-long ORB[149900.00,150800.00] vol=1.7x ATR=349.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:50:00 | 151754.26 | 150843.73 | 0.00 | T1 1.5R @ 151754.26 |
| Stop hit — per-position SL triggered | 2025-07-24 11:45:00 | 151230.00 | 151372.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:45:00 | 150580.00 | 149582.72 | 0.00 | ORB-long ORB[148400.00,150290.00] vol=1.5x ATR=529.17 |
| Stop hit — per-position SL triggered | 2025-07-28 10:05:00 | 150050.83 | 149689.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:00:00 | 149655.00 | 148939.54 | 0.00 | ORB-long ORB[148015.00,148805.00] vol=3.5x ATR=411.82 |
| Stop hit — per-position SL triggered | 2025-07-29 10:25:00 | 149243.18 | 148983.18 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:00:00 | 147045.00 | 148014.33 | 0.00 | ORB-short ORB[147850.00,149180.00] vol=1.6x ATR=385.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:30:00 | 146467.04 | 147665.99 | 0.00 | T1 1.5R @ 146467.04 |
| Stop hit — per-position SL triggered | 2025-08-01 15:05:00 | 147045.00 | 146642.54 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:55:00 | 139905.00 | 140376.54 | 0.00 | ORB-short ORB[140110.00,141700.00] vol=1.6x ATR=299.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:35:00 | 139455.07 | 140202.66 | 0.00 | T1 1.5R @ 139455.07 |
| Target hit | 2025-08-13 15:20:00 | 138250.00 | 139604.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-08-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:55:00 | 140450.00 | 139616.93 | 0.00 | ORB-long ORB[138755.00,139990.00] vol=2.7x ATR=412.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:00:00 | 141068.37 | 139880.58 | 0.00 | T1 1.5R @ 141068.37 |
| Stop hit — per-position SL triggered | 2025-08-18 10:25:00 | 140450.00 | 140173.77 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:05:00 | 147175.00 | 147481.48 | 0.00 | ORB-short ORB[147235.00,148750.00] vol=1.7x ATR=206.22 |
| Stop hit — per-position SL triggered | 2025-08-22 13:45:00 | 147381.22 | 147407.20 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 11:15:00 | 141000.00 | 141229.93 | 0.00 | ORB-short ORB[141045.00,142605.00] vol=2.3x ATR=254.60 |
| Stop hit — per-position SL triggered | 2025-08-29 12:10:00 | 141254.60 | 141182.77 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 146850.00 | 147383.59 | 0.00 | ORB-short ORB[147130.00,148050.00] vol=1.6x ATR=338.46 |
| Stop hit — per-position SL triggered | 2025-09-09 09:40:00 | 147188.46 | 147364.25 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:55:00 | 147000.00 | 146189.37 | 0.00 | ORB-long ORB[145330.00,146900.00] vol=1.9x ATR=362.46 |
| Stop hit — per-position SL triggered | 2025-09-11 11:30:00 | 146637.54 | 146289.75 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:55:00 | 147390.00 | 147017.55 | 0.00 | ORB-long ORB[146500.00,147195.00] vol=1.6x ATR=317.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 10:35:00 | 147866.95 | 147206.07 | 0.00 | T1 1.5R @ 147866.95 |
| Stop hit — per-position SL triggered | 2025-09-12 11:20:00 | 147390.00 | 147533.14 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:50:00 | 148700.00 | 147949.11 | 0.00 | ORB-long ORB[146885.00,148500.00] vol=1.6x ATR=373.54 |
| Stop hit — per-position SL triggered | 2025-09-15 09:55:00 | 148326.46 | 147989.54 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:40:00 | 149195.00 | 148779.91 | 0.00 | ORB-long ORB[147800.00,149175.00] vol=2.7x ATR=326.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:50:00 | 149685.20 | 149020.99 | 0.00 | T1 1.5R @ 149685.20 |
| Target hit | 2025-09-16 11:10:00 | 150330.00 | 150478.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:40:00 | 150390.00 | 151447.31 | 0.00 | ORB-short ORB[150950.00,152355.00] vol=2.8x ATR=373.43 |
| Stop hit — per-position SL triggered | 2025-09-17 10:45:00 | 150763.43 | 151403.23 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:45:00 | 151350.00 | 150743.15 | 0.00 | ORB-long ORB[150240.00,151110.00] vol=2.0x ATR=320.70 |
| Stop hit — per-position SL triggered | 2025-09-18 09:50:00 | 151029.30 | 150755.50 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:05:00 | 148975.00 | 149436.56 | 0.00 | ORB-short ORB[149275.00,151165.00] vol=2.8x ATR=361.57 |
| Stop hit — per-position SL triggered | 2025-09-19 11:20:00 | 149336.57 | 149413.51 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:55:00 | 151400.00 | 150489.55 | 0.00 | ORB-long ORB[149125.00,151200.00] vol=2.4x ATR=322.53 |
| Stop hit — per-position SL triggered | 2025-09-22 11:05:00 | 151077.47 | 150554.45 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:40:00 | 155005.00 | 154343.73 | 0.00 | ORB-long ORB[152995.00,154830.00] vol=2.1x ATR=540.67 |
| Stop hit — per-position SL triggered | 2025-09-23 09:50:00 | 154464.33 | 154386.57 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:05:00 | 151325.00 | 152148.21 | 0.00 | ORB-short ORB[151555.00,153710.00] vol=3.2x ATR=301.47 |
| Stop hit — per-position SL triggered | 2025-09-25 11:10:00 | 151626.47 | 152114.93 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 150435.00 | 149263.61 | 0.00 | ORB-long ORB[147405.00,148880.00] vol=3.2x ATR=425.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:20:00 | 151073.21 | 149569.46 | 0.00 | T1 1.5R @ 151073.21 |
| Target hit | 2025-10-03 15:20:00 | 152720.00 | 152117.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-10-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:00:00 | 157245.00 | 156345.35 | 0.00 | ORB-long ORB[155215.00,156945.00] vol=3.2x ATR=344.24 |
| Stop hit — per-position SL triggered | 2025-10-09 11:10:00 | 156900.76 | 156508.53 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 155650.00 | 156578.54 | 0.00 | ORB-short ORB[156400.00,157340.00] vol=2.9x ATR=360.03 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 156010.03 | 156496.94 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:35:00 | 156265.00 | 155275.22 | 0.00 | ORB-long ORB[154600.00,155415.00] vol=1.6x ATR=502.39 |
| Stop hit — per-position SL triggered | 2025-10-13 09:50:00 | 155762.61 | 155423.32 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 155700.00 | 156776.44 | 0.00 | ORB-short ORB[156595.00,157770.00] vol=2.1x ATR=335.82 |
| Stop hit — per-position SL triggered | 2025-10-15 11:25:00 | 156035.82 | 156599.54 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:45:00 | 158240.00 | 157541.61 | 0.00 | ORB-long ORB[156535.00,158195.00] vol=4.8x ATR=409.45 |
| Stop hit — per-position SL triggered | 2025-10-20 10:55:00 | 157830.55 | 157559.75 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:25:00 | 162660.00 | 161956.82 | 0.00 | ORB-long ORB[160285.00,162230.00] vol=1.6x ATR=459.21 |
| Stop hit — per-position SL triggered | 2025-10-23 11:35:00 | 162200.79 | 162032.47 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 162835.00 | 162339.09 | 0.00 | ORB-long ORB[161055.00,162800.00] vol=1.9x ATR=394.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:20:00 | 163426.10 | 162741.07 | 0.00 | T1 1.5R @ 163426.10 |
| Stop hit — per-position SL triggered | 2025-10-24 10:40:00 | 162835.00 | 162767.50 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 159805.00 | 160880.66 | 0.00 | ORB-short ORB[160200.00,162065.00] vol=1.5x ATR=334.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 159303.87 | 160672.62 | 0.00 | T1 1.5R @ 159303.87 |
| Target hit | 2025-10-28 15:20:00 | 158300.00 | 159618.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:55:00 | 158500.00 | 159203.28 | 0.00 | ORB-short ORB[159000.00,160560.00] vol=2.4x ATR=238.20 |
| Stop hit — per-position SL triggered | 2025-10-30 11:05:00 | 158738.20 | 159157.85 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:25:00 | 159285.00 | 158544.45 | 0.00 | ORB-long ORB[157800.00,158875.00] vol=2.3x ATR=310.47 |
| Stop hit — per-position SL triggered | 2025-11-03 10:30:00 | 158974.53 | 158576.68 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:55:00 | 159495.00 | 158778.04 | 0.00 | ORB-long ORB[158400.00,159150.00] vol=2.6x ATR=297.16 |
| Stop hit — per-position SL triggered | 2025-11-04 10:00:00 | 159197.84 | 158824.01 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 158625.00 | 159025.77 | 0.00 | ORB-short ORB[158925.00,159505.00] vol=2.1x ATR=267.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:30:00 | 158223.07 | 158899.24 | 0.00 | T1 1.5R @ 158223.07 |
| Stop hit — per-position SL triggered | 2025-11-10 12:10:00 | 158625.00 | 158782.63 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 157815.00 | 158098.49 | 0.00 | ORB-short ORB[158000.00,158500.00] vol=2.1x ATR=239.55 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 158054.55 | 158074.40 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:30:00 | 155650.00 | 156232.63 | 0.00 | ORB-short ORB[155965.00,157510.00] vol=1.7x ATR=329.80 |
| Stop hit — per-position SL triggered | 2025-11-18 11:40:00 | 155979.80 | 155831.00 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 153800.00 | 154484.80 | 0.00 | ORB-short ORB[154600.00,155655.00] vol=2.4x ATR=214.31 |
| Stop hit — per-position SL triggered | 2025-11-19 11:20:00 | 154014.31 | 154430.26 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:35:00 | 154405.00 | 153459.90 | 0.00 | ORB-long ORB[152000.00,154000.00] vol=2.7x ATR=419.91 |
| Stop hit — per-position SL triggered | 2025-11-21 09:40:00 | 153985.09 | 153608.49 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:30:00 | 152600.00 | 152959.84 | 0.00 | ORB-short ORB[152825.00,153905.00] vol=1.7x ATR=234.99 |
| Stop hit — per-position SL triggered | 2025-11-28 11:35:00 | 152834.99 | 152893.10 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 11:15:00 | 153155.00 | 153663.39 | 0.00 | ORB-short ORB[153650.00,154300.00] vol=1.8x ATR=203.52 |
| Stop hit — per-position SL triggered | 2025-12-05 11:45:00 | 153358.52 | 153610.33 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:05:00 | 152000.00 | 152528.50 | 0.00 | ORB-short ORB[152295.00,153135.00] vol=2.0x ATR=221.66 |
| Stop hit — per-position SL triggered | 2025-12-10 11:20:00 | 152221.66 | 152506.71 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:40:00 | 150800.00 | 151347.42 | 0.00 | ORB-short ORB[151020.00,152300.00] vol=1.6x ATR=332.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:55:00 | 150300.63 | 151127.22 | 0.00 | T1 1.5R @ 150300.63 |
| Stop hit — per-position SL triggered | 2025-12-11 10:10:00 | 150800.00 | 151089.56 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:10:00 | 152695.00 | 151822.87 | 0.00 | ORB-long ORB[151275.00,152550.00] vol=2.7x ATR=240.25 |
| Stop hit — per-position SL triggered | 2025-12-15 13:20:00 | 152454.75 | 152020.62 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:05:00 | 152450.00 | 152809.38 | 0.00 | ORB-short ORB[152700.00,153295.00] vol=2.7x ATR=197.57 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 152647.57 | 152769.26 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:50:00 | 155030.00 | 154006.44 | 0.00 | ORB-long ORB[152260.00,154200.00] vol=3.6x ATR=458.56 |
| Stop hit — per-position SL triggered | 2025-12-19 09:55:00 | 154571.44 | 154071.25 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:35:00 | 153200.00 | 153494.94 | 0.00 | ORB-short ORB[153400.00,153885.00] vol=1.7x ATR=242.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 12:00:00 | 152836.83 | 153238.72 | 0.00 | T1 1.5R @ 152836.83 |
| Target hit | 2025-12-24 15:20:00 | 151315.00 | 152003.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-12-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:00:00 | 149125.00 | 149968.57 | 0.00 | ORB-short ORB[149710.00,151000.00] vol=2.0x ATR=332.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:55:00 | 148625.67 | 149724.57 | 0.00 | T1 1.5R @ 148625.67 |
| Stop hit — per-position SL triggered | 2025-12-29 11:00:00 | 149125.00 | 149715.52 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 153350.00 | 152581.93 | 0.00 | ORB-long ORB[151500.00,152305.00] vol=1.5x ATR=377.27 |
| Stop hit — per-position SL triggered | 2025-12-31 11:45:00 | 152972.73 | 152650.15 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:45:00 | 151680.00 | 151305.71 | 0.00 | ORB-long ORB[150800.00,151605.00] vol=1.9x ATR=267.60 |
| Stop hit — per-position SL triggered | 2026-01-02 10:20:00 | 151412.40 | 151444.42 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:55:00 | 150405.00 | 151028.46 | 0.00 | ORB-short ORB[150700.00,151665.00] vol=1.8x ATR=232.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 12:15:00 | 150055.94 | 150757.55 | 0.00 | T1 1.5R @ 150055.94 |
| Stop hit — per-position SL triggered | 2026-01-05 12:35:00 | 150405.00 | 150709.01 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:45:00 | 149615.00 | 149870.06 | 0.00 | ORB-short ORB[149705.00,150550.00] vol=2.0x ATR=285.77 |
| Stop hit — per-position SL triggered | 2026-01-07 10:00:00 | 149900.77 | 149845.43 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 149100.00 | 149520.31 | 0.00 | ORB-short ORB[149305.00,150255.00] vol=1.7x ATR=233.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:05:00 | 148750.19 | 149417.87 | 0.00 | T1 1.5R @ 148750.19 |
| Stop hit — per-position SL triggered | 2026-01-08 12:25:00 | 149100.00 | 149312.05 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 10:35:00 | 147810.00 | 148379.66 | 0.00 | ORB-short ORB[148100.00,149725.00] vol=2.1x ATR=277.54 |
| Stop hit — per-position SL triggered | 2026-01-09 10:45:00 | 148087.54 | 148366.81 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 145975.00 | 146524.46 | 0.00 | ORB-short ORB[146210.00,148005.00] vol=1.6x ATR=340.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 15:15:00 | 145463.69 | 146097.09 | 0.00 | T1 1.5R @ 145463.69 |
| Target hit | 2026-01-14 15:20:00 | 145500.00 | 146050.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2026-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:00:00 | 144010.00 | 144540.47 | 0.00 | ORB-short ORB[144565.00,145665.00] vol=2.4x ATR=249.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:45:00 | 143635.73 | 144346.47 | 0.00 | T1 1.5R @ 143635.73 |
| Target hit | 2026-01-16 15:20:00 | 142800.00 | 143384.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-01-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:10:00 | 142800.00 | 143834.36 | 0.00 | ORB-short ORB[143000.00,145135.00] vol=4.4x ATR=292.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 142361.64 | 143713.06 | 0.00 | T1 1.5R @ 142361.64 |
| Stop hit — per-position SL triggered | 2026-01-19 12:55:00 | 142800.00 | 142841.70 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:30:00 | 133220.00 | 134370.06 | 0.00 | ORB-short ORB[133835.00,135500.00] vol=1.8x ATR=390.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:00:00 | 132634.11 | 133909.79 | 0.00 | T1 1.5R @ 132634.11 |
| Target hit | 2026-01-29 15:20:00 | 130750.00 | 131911.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 133200.00 | 132330.95 | 0.00 | ORB-long ORB[131200.00,132875.00] vol=3.8x ATR=473.90 |
| Stop hit — per-position SL triggered | 2026-02-02 10:25:00 | 132726.10 | 132699.23 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 152400.00 | 150951.05 | 0.00 | ORB-long ORB[149800.00,152000.00] vol=2.6x ATR=504.09 |
| Stop hit — per-position SL triggered | 2026-02-12 10:00:00 | 151895.91 | 151129.01 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 148535.00 | 149752.31 | 0.00 | ORB-short ORB[149735.00,150990.00] vol=1.9x ATR=310.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:55:00 | 148069.83 | 149539.87 | 0.00 | T1 1.5R @ 148069.83 |
| Target hit | 2026-02-16 15:20:00 | 146800.00 | 148490.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 148585.00 | 148111.13 | 0.00 | ORB-long ORB[147750.00,148500.00] vol=2.5x ATR=343.67 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 148241.33 | 148275.26 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 143215.00 | 144258.00 | 0.00 | ORB-short ORB[144205.00,146200.00] vol=1.5x ATR=299.40 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 143514.40 | 144065.57 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 146345.00 | 145614.36 | 0.00 | ORB-long ORB[144245.00,145975.00] vol=3.0x ATR=320.26 |
| Stop hit — per-position SL triggered | 2026-02-25 12:30:00 | 146024.74 | 145729.94 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 144800.00 | 146199.60 | 0.00 | ORB-short ORB[146605.00,147815.00] vol=2.3x ATR=332.39 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 145132.39 | 145906.91 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 143080.00 | 143718.04 | 0.00 | ORB-short ORB[143520.00,144700.00] vol=2.3x ATR=329.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:00:00 | 142586.31 | 143620.20 | 0.00 | T1 1.5R @ 142586.31 |
| Target hit | 2026-02-27 15:20:00 | 140770.00 | 141669.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2026-03-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:30:00 | 134855.00 | 136584.12 | 0.00 | ORB-short ORB[136500.00,138490.00] vol=2.4x ATR=515.45 |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 135370.45 | 136251.58 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 136600.00 | 135684.24 | 0.00 | ORB-long ORB[134435.00,135900.00] vol=2.0x ATR=353.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:50:00 | 137130.20 | 135821.58 | 0.00 | T1 1.5R @ 137130.20 |
| Target hit | 2026-03-12 15:20:00 | 137955.00 | 136799.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2026-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:50:00 | 138480.00 | 139148.60 | 0.00 | ORB-short ORB[138965.00,140095.00] vol=1.8x ATR=243.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 138115.06 | 138842.08 | 0.00 | T1 1.5R @ 138115.06 |
| Target hit | 2026-04-22 15:20:00 | 136500.00 | 137801.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 131440.00 | 132006.76 | 0.00 | ORB-short ORB[131900.00,132695.00] vol=1.8x ATR=314.24 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 131754.24 | 131975.61 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 128480.00 | 128960.56 | 0.00 | ORB-short ORB[128600.00,130095.00] vol=2.3x ATR=200.32 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 128680.32 | 128944.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 10:25:00 | 142155.00 | 2025-05-16 10:30:00 | 141905.22 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-05-26 09:35:00 | 144835.00 | 2025-05-26 09:45:00 | 145218.14 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-05-26 09:35:00 | 144835.00 | 2025-05-26 15:20:00 | 146685.00 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2025-05-29 11:10:00 | 141990.00 | 2025-05-29 12:10:00 | 141423.24 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-05-29 11:10:00 | 141990.00 | 2025-05-29 15:20:00 | 140865.00 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-06-03 09:55:00 | 138800.00 | 2025-06-03 10:20:00 | 139269.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-04 10:40:00 | 136840.00 | 2025-06-04 15:05:00 | 136224.00 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-04 10:40:00 | 136840.00 | 2025-06-04 15:20:00 | 136000.00 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-05 09:45:00 | 138040.00 | 2025-06-05 10:05:00 | 138544.17 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-06-05 09:45:00 | 138040.00 | 2025-06-05 15:20:00 | 140400.00 | TARGET_HIT | 0.50 | 1.71% |
| SELL | retest1 | 2025-06-09 09:50:00 | 139320.00 | 2025-06-09 10:00:00 | 139746.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-18 09:45:00 | 136880.00 | 2025-06-18 09:50:00 | 136685.32 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-06-20 09:35:00 | 137150.00 | 2025-06-20 09:55:00 | 136655.37 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-20 09:35:00 | 137150.00 | 2025-06-20 11:00:00 | 137085.00 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-06-24 11:00:00 | 136810.00 | 2025-06-24 13:00:00 | 136537.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-26 10:45:00 | 139895.00 | 2025-06-26 10:50:00 | 139573.08 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-01 10:50:00 | 141925.00 | 2025-07-01 11:35:00 | 142205.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-02 09:30:00 | 143650.00 | 2025-07-02 09:35:00 | 144252.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-02 09:30:00 | 143650.00 | 2025-07-02 09:55:00 | 143950.00 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-07-04 11:00:00 | 143925.00 | 2025-07-04 11:45:00 | 143525.53 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-04 11:00:00 | 143925.00 | 2025-07-04 13:10:00 | 143925.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 11:05:00 | 142685.00 | 2025-07-07 12:50:00 | 142354.47 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-07 11:05:00 | 142685.00 | 2025-07-07 14:40:00 | 142685.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 09:30:00 | 146140.00 | 2025-07-09 11:30:00 | 146813.27 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-09 09:30:00 | 146140.00 | 2025-07-09 15:20:00 | 150215.00 | TARGET_HIT | 0.50 | 2.79% |
| SELL | retest1 | 2025-07-10 10:50:00 | 147955.00 | 2025-07-10 11:15:00 | 147421.66 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-10 10:50:00 | 147955.00 | 2025-07-10 11:40:00 | 147955.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 11:10:00 | 148480.00 | 2025-07-18 11:15:00 | 148789.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-22 09:40:00 | 150800.00 | 2025-07-22 09:45:00 | 150351.45 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-24 09:40:00 | 151230.00 | 2025-07-24 09:50:00 | 151754.26 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-24 09:40:00 | 151230.00 | 2025-07-24 11:45:00 | 151230.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-28 09:45:00 | 150580.00 | 2025-07-28 10:05:00 | 150050.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-29 10:00:00 | 149655.00 | 2025-07-29 10:25:00 | 149243.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-01 10:00:00 | 147045.00 | 2025-08-01 10:30:00 | 146467.04 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-01 10:00:00 | 147045.00 | 2025-08-01 15:05:00 | 147045.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 10:55:00 | 139905.00 | 2025-08-13 11:35:00 | 139455.07 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-13 10:55:00 | 139905.00 | 2025-08-13 15:20:00 | 138250.00 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2025-08-18 09:55:00 | 140450.00 | 2025-08-18 10:00:00 | 141068.37 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-18 09:55:00 | 140450.00 | 2025-08-18 10:25:00 | 140450.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 11:05:00 | 147175.00 | 2025-08-22 13:45:00 | 147381.22 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-29 11:15:00 | 141000.00 | 2025-08-29 12:10:00 | 141254.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-09 09:30:00 | 146850.00 | 2025-09-09 09:40:00 | 147188.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-11 10:55:00 | 147000.00 | 2025-09-11 11:30:00 | 146637.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-12 09:55:00 | 147390.00 | 2025-09-12 10:35:00 | 147866.95 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-12 09:55:00 | 147390.00 | 2025-09-12 11:20:00 | 147390.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 09:50:00 | 148700.00 | 2025-09-15 09:55:00 | 148326.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-16 09:40:00 | 149195.00 | 2025-09-16 09:50:00 | 149685.20 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-16 09:40:00 | 149195.00 | 2025-09-16 11:10:00 | 150330.00 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2025-09-17 10:40:00 | 150390.00 | 2025-09-17 10:45:00 | 150763.43 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-18 09:45:00 | 151350.00 | 2025-09-18 09:50:00 | 151029.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-19 11:05:00 | 148975.00 | 2025-09-19 11:20:00 | 149336.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-22 10:55:00 | 151400.00 | 2025-09-22 11:05:00 | 151077.47 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-23 09:40:00 | 155005.00 | 2025-09-23 09:50:00 | 154464.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-25 11:05:00 | 151325.00 | 2025-09-25 11:10:00 | 151626.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-03 10:55:00 | 150435.00 | 2025-10-03 11:20:00 | 151073.21 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-03 10:55:00 | 150435.00 | 2025-10-03 15:20:00 | 152720.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2025-10-09 11:00:00 | 157245.00 | 2025-10-09 11:10:00 | 156900.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-10 09:30:00 | 155650.00 | 2025-10-10 09:35:00 | 156010.03 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-13 09:35:00 | 156265.00 | 2025-10-13 09:50:00 | 155762.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-15 10:55:00 | 155700.00 | 2025-10-15 11:25:00 | 156035.82 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-20 10:45:00 | 158240.00 | 2025-10-20 10:55:00 | 157830.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-23 10:25:00 | 162660.00 | 2025-10-23 11:35:00 | 162200.79 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-24 09:30:00 | 162835.00 | 2025-10-24 10:20:00 | 163426.10 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-24 09:30:00 | 162835.00 | 2025-10-24 10:40:00 | 162835.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 11:00:00 | 159805.00 | 2025-10-28 11:15:00 | 159303.87 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-28 11:00:00 | 159805.00 | 2025-10-28 15:20:00 | 158300.00 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-10-30 10:55:00 | 158500.00 | 2025-10-30 11:05:00 | 158738.20 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-11-03 10:25:00 | 159285.00 | 2025-11-03 10:30:00 | 158974.53 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-04 09:55:00 | 159495.00 | 2025-11-04 10:00:00 | 159197.84 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-10 11:00:00 | 158625.00 | 2025-11-10 11:30:00 | 158223.07 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-10 11:00:00 | 158625.00 | 2025-11-10 12:10:00 | 158625.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 09:30:00 | 157815.00 | 2025-11-12 09:55:00 | 158054.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-18 09:30:00 | 155650.00 | 2025-11-18 11:40:00 | 155979.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-19 10:55:00 | 153800.00 | 2025-11-19 11:20:00 | 154014.31 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-11-21 09:35:00 | 154405.00 | 2025-11-21 09:40:00 | 153985.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-28 10:30:00 | 152600.00 | 2025-11-28 11:35:00 | 152834.99 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-05 11:15:00 | 153155.00 | 2025-12-05 11:45:00 | 153358.52 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-12-10 11:05:00 | 152000.00 | 2025-12-10 11:20:00 | 152221.66 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-11 09:40:00 | 150800.00 | 2025-12-11 09:55:00 | 150300.63 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-11 09:40:00 | 150800.00 | 2025-12-11 10:10:00 | 150800.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-15 11:10:00 | 152695.00 | 2025-12-15 13:20:00 | 152454.75 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-17 11:05:00 | 152450.00 | 2025-12-17 11:15:00 | 152647.57 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-19 09:50:00 | 155030.00 | 2025-12-19 09:55:00 | 154571.44 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-24 09:35:00 | 153200.00 | 2025-12-24 12:00:00 | 152836.83 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-24 09:35:00 | 153200.00 | 2025-12-24 15:20:00 | 151315.00 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2025-12-29 10:00:00 | 149125.00 | 2025-12-29 10:55:00 | 148625.67 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-29 10:00:00 | 149125.00 | 2025-12-29 11:00:00 | 149125.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:50:00 | 153350.00 | 2025-12-31 11:45:00 | 152972.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-02 09:45:00 | 151680.00 | 2026-01-02 10:20:00 | 151412.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-05 10:55:00 | 150405.00 | 2026-01-05 12:15:00 | 150055.94 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-01-05 10:55:00 | 150405.00 | 2026-01-05 12:35:00 | 150405.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:45:00 | 149615.00 | 2026-01-07 10:00:00 | 149900.77 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-08 11:00:00 | 149100.00 | 2026-01-08 12:05:00 | 148750.19 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-01-08 11:00:00 | 149100.00 | 2026-01-08 12:25:00 | 149100.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-09 10:35:00 | 147810.00 | 2026-01-09 10:45:00 | 148087.54 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-14 09:45:00 | 145975.00 | 2026-01-14 15:15:00 | 145463.69 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-14 09:45:00 | 145975.00 | 2026-01-14 15:20:00 | 145500.00 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-16 10:00:00 | 144010.00 | 2026-01-16 10:45:00 | 143635.73 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-01-16 10:00:00 | 144010.00 | 2026-01-16 15:20:00 | 142800.00 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2026-01-19 11:10:00 | 142800.00 | 2026-01-19 11:15:00 | 142361.64 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-01-19 11:10:00 | 142800.00 | 2026-01-19 12:55:00 | 142800.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 09:30:00 | 133220.00 | 2026-01-29 10:00:00 | 132634.11 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-01-29 09:30:00 | 133220.00 | 2026-01-29 15:20:00 | 130750.00 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2026-02-02 09:30:00 | 133200.00 | 2026-02-02 10:25:00 | 132726.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-12 09:50:00 | 152400.00 | 2026-02-12 10:00:00 | 151895.91 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-16 11:15:00 | 148535.00 | 2026-02-16 11:55:00 | 148069.83 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-16 11:15:00 | 148535.00 | 2026-02-16 15:20:00 | 146800.00 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2026-02-18 10:45:00 | 148585.00 | 2026-02-18 12:15:00 | 148241.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-24 11:10:00 | 143215.00 | 2026-02-24 11:45:00 | 143514.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-25 11:15:00 | 146345.00 | 2026-02-25 12:30:00 | 146024.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-26 11:00:00 | 144800.00 | 2026-02-26 11:35:00 | 145132.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-27 10:55:00 | 143080.00 | 2026-02-27 12:00:00 | 142586.31 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 10:55:00 | 143080.00 | 2026-02-27 15:20:00 | 140770.00 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2026-03-04 10:30:00 | 134855.00 | 2026-03-04 11:15:00 | 135370.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-12 11:15:00 | 136600.00 | 2026-03-12 11:50:00 | 137130.20 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-12 11:15:00 | 136600.00 | 2026-03-12 15:20:00 | 137955.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-04-22 09:50:00 | 138480.00 | 2026-04-22 10:15:00 | 138115.06 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-22 09:50:00 | 138480.00 | 2026-04-22 15:20:00 | 136500.00 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-04-28 09:35:00 | 131440.00 | 2026-04-28 09:40:00 | 131754.24 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-05 10:55:00 | 128480.00 | 2026-05-05 11:05:00 | 128680.32 | STOP_HIT | 1.00 | -0.16% |
