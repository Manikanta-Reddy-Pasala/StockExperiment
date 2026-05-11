# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-12-08 09:15:00 → 2026-05-08 15:25:00 (43192 bars)
- **Last close:** 8100.00
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
| ENTRY1 | 38 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 33
- **Target hits / Stop hits / Partials:** 5 / 33 / 11
- **Avg / median % per leg:** 0.06% / -0.21%
- **Sum % (uncompounded):** 3.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 1 | 17 | 3 | -0.09% | -1.8% |
| BUY @ 2nd Alert (retest1) | 21 | 4 | 19.0% | 1 | 17 | 3 | -0.09% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 12 | 42.9% | 4 | 16 | 8 | 0.18% | 5.0% |
| SELL @ 2nd Alert (retest1) | 28 | 12 | 42.9% | 4 | 16 | 8 | 0.18% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 49 | 16 | 32.7% | 5 | 33 | 11 | 0.06% | 3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:55:00 | 5456.90 | 5487.87 | 0.00 | ORB-short ORB[5465.40,5533.60] vol=6.8x ATR=19.57 |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 5476.47 | 5485.08 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:10:00 | 5541.95 | 5528.66 | 0.00 | ORB-long ORB[5500.00,5530.00] vol=6.2x ATR=11.80 |
| Stop hit — per-position SL triggered | 2023-12-15 10:20:00 | 5530.15 | 5529.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:30:00 | 5595.10 | 5582.63 | 0.00 | ORB-long ORB[5555.50,5594.65] vol=1.8x ATR=11.98 |
| Stop hit — per-position SL triggered | 2023-12-20 09:35:00 | 5583.12 | 5583.12 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:35:00 | 5664.40 | 5639.71 | 0.00 | ORB-long ORB[5615.70,5649.90] vol=2.7x ATR=12.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:40:00 | 5683.40 | 5645.62 | 0.00 | T1 1.5R @ 5683.40 |
| Stop hit — per-position SL triggered | 2023-12-27 10:15:00 | 5664.40 | 5674.96 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-01-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:45:00 | 5756.55 | 5802.84 | 0.00 | ORB-short ORB[5758.95,5839.95] vol=2.0x ATR=17.80 |
| Stop hit — per-position SL triggered | 2024-01-04 10:45:00 | 5774.35 | 5789.07 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:30:00 | 5683.80 | 5722.60 | 0.00 | ORB-short ORB[5715.00,5774.85] vol=3.2x ATR=18.75 |
| Stop hit — per-position SL triggered | 2024-01-08 09:55:00 | 5702.55 | 5706.81 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:05:00 | 5806.70 | 5747.72 | 0.00 | ORB-long ORB[5674.70,5747.95] vol=6.2x ATR=15.18 |
| Stop hit — per-position SL triggered | 2024-01-09 11:10:00 | 5791.52 | 5756.09 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-01-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:55:00 | 5771.00 | 5785.71 | 0.00 | ORB-short ORB[5780.05,5819.30] vol=2.1x ATR=14.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 11:00:00 | 5749.33 | 5782.04 | 0.00 | T1 1.5R @ 5749.33 |
| Target hit | 2024-01-10 14:35:00 | 5763.65 | 5762.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2024-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 09:30:00 | 5852.40 | 5828.07 | 0.00 | ORB-long ORB[5804.00,5850.00] vol=1.6x ATR=15.19 |
| Stop hit — per-position SL triggered | 2024-01-15 09:35:00 | 5837.21 | 5829.25 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:45:00 | 5954.05 | 5900.36 | 0.00 | ORB-long ORB[5829.00,5885.00] vol=5.5x ATR=15.83 |
| Stop hit — per-position SL triggered | 2024-01-16 10:55:00 | 5938.22 | 5908.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-17 10:50:00 | 5945.05 | 5921.00 | 0.00 | ORB-long ORB[5831.25,5919.00] vol=1.5x ATR=15.69 |
| Stop hit — per-position SL triggered | 2024-01-17 11:00:00 | 5929.36 | 5921.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:55:00 | 6150.00 | 6112.39 | 0.00 | ORB-long ORB[6075.00,6124.00] vol=1.9x ATR=15.33 |
| Stop hit — per-position SL triggered | 2024-01-20 10:10:00 | 6134.67 | 6125.18 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 09:30:00 | 6215.95 | 6243.11 | 0.00 | ORB-short ORB[6231.15,6297.90] vol=6.0x ATR=20.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 09:50:00 | 6184.84 | 6226.07 | 0.00 | T1 1.5R @ 6184.84 |
| Stop hit — per-position SL triggered | 2024-02-06 11:00:00 | 6215.95 | 6209.93 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-02-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:45:00 | 6196.05 | 6212.60 | 0.00 | ORB-short ORB[6198.55,6248.85] vol=5.7x ATR=16.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 10:20:00 | 6171.16 | 6201.88 | 0.00 | T1 1.5R @ 6171.16 |
| Target hit | 2024-02-07 15:20:00 | 6180.00 | 6186.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 6302.85 | 6276.09 | 0.00 | ORB-long ORB[6213.60,6269.40] vol=3.3x ATR=32.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:45:00 | 6351.06 | 6286.21 | 0.00 | T1 1.5R @ 6351.06 |
| Stop hit — per-position SL triggered | 2024-02-09 09:50:00 | 6302.85 | 6289.05 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 6670.00 | 6643.42 | 0.00 | ORB-long ORB[6598.25,6657.70] vol=2.3x ATR=22.31 |
| Stop hit — per-position SL triggered | 2024-02-13 09:50:00 | 6647.69 | 6653.85 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-02-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:00:00 | 6684.35 | 6725.96 | 0.00 | ORB-short ORB[6715.65,6764.40] vol=1.6x ATR=18.70 |
| Stop hit — per-position SL triggered | 2024-02-15 10:35:00 | 6703.05 | 6716.53 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 09:40:00 | 6542.45 | 6571.65 | 0.00 | ORB-short ORB[6568.95,6660.45] vol=2.1x ATR=22.71 |
| Stop hit — per-position SL triggered | 2024-02-16 10:00:00 | 6565.16 | 6565.70 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 09:30:00 | 6724.95 | 6701.49 | 0.00 | ORB-long ORB[6637.85,6715.00] vol=2.7x ATR=19.09 |
| Stop hit — per-position SL triggered | 2024-02-20 09:50:00 | 6705.86 | 6708.43 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:15:00 | 6794.80 | 6764.17 | 0.00 | ORB-long ORB[6717.30,6784.95] vol=2.0x ATR=15.65 |
| Stop hit — per-position SL triggered | 2024-02-21 10:20:00 | 6779.15 | 6765.90 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:10:00 | 6044.60 | 6068.59 | 0.00 | ORB-short ORB[6045.25,6092.80] vol=2.1x ATR=10.22 |
| Stop hit — per-position SL triggered | 2024-03-11 11:25:00 | 6054.82 | 6068.30 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 09:35:00 | 6127.45 | 6157.94 | 0.00 | ORB-short ORB[6145.10,6214.40] vol=1.5x ATR=16.61 |
| Stop hit — per-position SL triggered | 2024-03-21 09:50:00 | 6144.06 | 6147.22 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-03-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:30:00 | 6247.70 | 6205.60 | 0.00 | ORB-long ORB[6168.70,6209.95] vol=2.1x ATR=18.45 |
| Stop hit — per-position SL triggered | 2024-03-22 09:35:00 | 6229.25 | 6210.74 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:15:00 | 6341.00 | 6366.31 | 0.00 | ORB-short ORB[6360.95,6429.00] vol=1.9x ATR=17.14 |
| Stop hit — per-position SL triggered | 2024-04-04 11:10:00 | 6358.14 | 6358.33 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:30:00 | 6304.00 | 6331.14 | 0.00 | ORB-short ORB[6309.60,6397.75] vol=1.8x ATR=15.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:35:00 | 6281.32 | 6323.90 | 0.00 | T1 1.5R @ 6281.32 |
| Stop hit — per-position SL triggered | 2024-04-08 09:50:00 | 6304.00 | 6318.33 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:30:00 | 6402.40 | 6365.23 | 0.00 | ORB-long ORB[6330.95,6380.00] vol=1.9x ATR=15.11 |
| Stop hit — per-position SL triggered | 2024-04-09 09:35:00 | 6387.29 | 6374.65 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 10:15:00 | 6213.10 | 6254.23 | 0.00 | ORB-short ORB[6228.30,6300.00] vol=1.8x ATR=16.65 |
| Stop hit — per-position SL triggered | 2024-04-16 11:30:00 | 6229.75 | 6243.84 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-04-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:30:00 | 6287.75 | 6305.88 | 0.00 | ORB-short ORB[6291.45,6334.40] vol=1.6x ATR=17.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 09:40:00 | 6262.19 | 6297.62 | 0.00 | T1 1.5R @ 6262.19 |
| Target hit | 2024-04-18 15:20:00 | 6070.55 | 6181.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:35:00 | 6298.00 | 6266.03 | 0.00 | ORB-long ORB[6227.45,6280.95] vol=3.3x ATR=13.86 |
| Stop hit — per-position SL triggered | 2024-04-23 09:40:00 | 6284.14 | 6266.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:30:00 | 6262.20 | 6231.42 | 0.00 | ORB-long ORB[6186.15,6229.90] vol=1.7x ATR=14.14 |
| Stop hit — per-position SL triggered | 2024-04-24 10:50:00 | 6248.06 | 6234.66 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:55:00 | 6218.20 | 6234.37 | 0.00 | ORB-short ORB[6220.00,6283.35] vol=1.8x ATR=16.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:00:00 | 6193.96 | 6229.49 | 0.00 | T1 1.5R @ 6193.96 |
| Stop hit — per-position SL triggered | 2024-04-25 10:10:00 | 6218.20 | 6220.23 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 09:40:00 | 5920.15 | 5948.56 | 0.00 | ORB-short ORB[5929.85,6000.05] vol=1.7x ATR=22.98 |
| Stop hit — per-position SL triggered | 2024-04-30 09:50:00 | 5943.13 | 5945.67 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-05-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:40:00 | 5999.00 | 5974.19 | 0.00 | ORB-long ORB[5947.10,5984.85] vol=1.7x ATR=15.39 |
| Stop hit — per-position SL triggered | 2024-05-02 09:50:00 | 5983.61 | 5975.77 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 6008.00 | 6028.72 | 0.00 | ORB-short ORB[6012.00,6052.55] vol=1.9x ATR=17.24 |
| Stop hit — per-position SL triggered | 2024-05-06 09:55:00 | 6025.24 | 6023.93 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:40:00 | 5991.10 | 6022.63 | 0.00 | ORB-short ORB[6016.25,6080.00] vol=1.6x ATR=17.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:40:00 | 5965.16 | 5998.71 | 0.00 | T1 1.5R @ 5965.16 |
| Target hit | 2024-05-07 15:20:00 | 5901.50 | 5934.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 09:35:00 | 5817.00 | 5851.54 | 0.00 | ORB-short ORB[5823.15,5910.00] vol=2.1x ATR=18.52 |
| Stop hit — per-position SL triggered | 2024-05-08 09:45:00 | 5835.52 | 5845.39 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-05-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:55:00 | 5834.30 | 5846.94 | 0.00 | ORB-short ORB[5835.00,5892.00] vol=3.7x ATR=14.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 12:55:00 | 5812.94 | 5843.05 | 0.00 | T1 1.5R @ 5812.94 |
| Stop hit — per-position SL triggered | 2024-05-09 13:10:00 | 5834.30 | 5841.62 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-05-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 09:50:00 | 5826.20 | 5790.83 | 0.00 | ORB-long ORB[5753.25,5825.00] vol=1.6x ATR=23.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 10:30:00 | 5861.07 | 5800.82 | 0.00 | T1 1.5R @ 5861.07 |
| Target hit | 2024-05-10 15:20:00 | 5844.05 | 5821.60 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-12-08 10:55:00 | 5456.90 | 2023-12-08 11:15:00 | 5476.47 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-15 10:10:00 | 5541.95 | 2023-12-15 10:20:00 | 5530.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-20 09:30:00 | 5595.10 | 2023-12-20 09:35:00 | 5583.12 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-27 09:35:00 | 5664.40 | 2023-12-27 09:40:00 | 5683.40 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-12-27 09:35:00 | 5664.40 | 2023-12-27 10:15:00 | 5664.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-04 09:45:00 | 5756.55 | 2024-01-04 10:45:00 | 5774.35 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-08 09:30:00 | 5683.80 | 2024-01-08 09:55:00 | 5702.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-01-09 11:05:00 | 5806.70 | 2024-01-09 11:10:00 | 5791.52 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-10 10:55:00 | 5771.00 | 2024-01-10 11:00:00 | 5749.33 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-01-10 10:55:00 | 5771.00 | 2024-01-10 14:35:00 | 5763.65 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-01-15 09:30:00 | 5852.40 | 2024-01-15 09:35:00 | 5837.21 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-01-16 10:45:00 | 5954.05 | 2024-01-16 10:55:00 | 5938.22 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-01-17 10:50:00 | 5945.05 | 2024-01-17 11:00:00 | 5929.36 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-01-20 09:55:00 | 6150.00 | 2024-01-20 10:10:00 | 6134.67 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-06 09:30:00 | 6215.95 | 2024-02-06 09:50:00 | 6184.84 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-02-06 09:30:00 | 6215.95 | 2024-02-06 11:00:00 | 6215.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-07 09:45:00 | 6196.05 | 2024-02-07 10:20:00 | 6171.16 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-02-07 09:45:00 | 6196.05 | 2024-02-07 15:20:00 | 6180.00 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-02-09 09:40:00 | 6302.85 | 2024-02-09 09:45:00 | 6351.06 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-02-09 09:40:00 | 6302.85 | 2024-02-09 09:50:00 | 6302.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-13 09:35:00 | 6670.00 | 2024-02-13 09:50:00 | 6647.69 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-15 10:00:00 | 6684.35 | 2024-02-15 10:35:00 | 6703.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-16 09:40:00 | 6542.45 | 2024-02-16 10:00:00 | 6565.16 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-02-20 09:30:00 | 6724.95 | 2024-02-20 09:50:00 | 6705.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-02-21 10:15:00 | 6794.80 | 2024-02-21 10:20:00 | 6779.15 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-11 11:10:00 | 6044.60 | 2024-03-11 11:25:00 | 6054.82 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-03-21 09:35:00 | 6127.45 | 2024-03-21 09:50:00 | 6144.06 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-03-22 09:30:00 | 6247.70 | 2024-03-22 09:35:00 | 6229.25 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-04 10:15:00 | 6341.00 | 2024-04-04 11:10:00 | 6358.14 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-08 09:30:00 | 6304.00 | 2024-04-08 09:35:00 | 6281.32 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-04-08 09:30:00 | 6304.00 | 2024-04-08 09:50:00 | 6304.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-09 09:30:00 | 6402.40 | 2024-04-09 09:35:00 | 6387.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-16 10:15:00 | 6213.10 | 2024-04-16 11:30:00 | 6229.75 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-18 09:30:00 | 6287.75 | 2024-04-18 09:40:00 | 6262.19 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-04-18 09:30:00 | 6287.75 | 2024-04-18 15:20:00 | 6070.55 | TARGET_HIT | 0.50 | 3.45% |
| BUY | retest1 | 2024-04-23 09:35:00 | 6298.00 | 2024-04-23 09:40:00 | 6284.14 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-24 10:30:00 | 6262.20 | 2024-04-24 10:50:00 | 6248.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-25 09:55:00 | 6218.20 | 2024-04-25 10:00:00 | 6193.96 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-25 09:55:00 | 6218.20 | 2024-04-25 10:10:00 | 6218.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-30 09:40:00 | 5920.15 | 2024-04-30 09:50:00 | 5943.13 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-02 09:40:00 | 5999.00 | 2024-05-02 09:50:00 | 5983.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-06 09:35:00 | 6008.00 | 2024-05-06 09:55:00 | 6025.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-07 09:40:00 | 5991.10 | 2024-05-07 11:40:00 | 5965.16 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-07 09:40:00 | 5991.10 | 2024-05-07 15:20:00 | 5901.50 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-05-08 09:35:00 | 5817.00 | 2024-05-08 09:45:00 | 5835.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-09 10:55:00 | 5834.30 | 2024-05-09 12:55:00 | 5812.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-05-09 10:55:00 | 5834.30 | 2024-05-09 13:10:00 | 5834.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-10 09:50:00 | 5826.20 | 2024-05-10 10:30:00 | 5861.07 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-05-10 09:50:00 | 5826.20 | 2024-05-10 15:20:00 | 5844.05 | TARGET_HIT | 0.50 | 0.31% |
