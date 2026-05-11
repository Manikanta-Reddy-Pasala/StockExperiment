# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 4313.00
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
| PARTIAL | 32 |
| TARGET_HIT | 13 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 74
- **Target hits / Stop hits / Partials:** 13 / 74 / 32
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 6.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 18 | 34.0% | 4 | 35 | 14 | 0.02% | 1.0% |
| BUY @ 2nd Alert (retest1) | 53 | 18 | 34.0% | 4 | 35 | 14 | 0.02% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 27 | 40.9% | 9 | 39 | 18 | 0.08% | 5.2% |
| SELL @ 2nd Alert (retest1) | 66 | 27 | 40.9% | 9 | 39 | 18 | 0.08% | 5.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 119 | 45 | 37.8% | 13 | 74 | 32 | 0.05% | 6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 6076.00 | 6034.89 | 0.00 | ORB-long ORB[5981.00,6050.00] vol=5.5x ATR=21.75 |
| Stop hit — per-position SL triggered | 2025-05-14 09:35:00 | 6054.25 | 6055.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 6169.50 | 6131.99 | 0.00 | ORB-long ORB[6061.50,6143.50] vol=4.1x ATR=21.90 |
| Stop hit — per-position SL triggered | 2025-05-15 09:40:00 | 6147.60 | 6138.41 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 09:30:00 | 6230.50 | 6264.37 | 0.00 | ORB-short ORB[6238.00,6319.50] vol=2.2x ATR=21.74 |
| Stop hit — per-position SL triggered | 2025-05-20 09:35:00 | 6252.24 | 6263.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 6252.00 | 6232.54 | 0.00 | ORB-long ORB[6202.50,6246.00] vol=2.3x ATR=15.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:55:00 | 6274.57 | 6243.77 | 0.00 | T1 1.5R @ 6274.57 |
| Target hit | 2025-05-23 15:20:00 | 6291.50 | 6275.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-05-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:10:00 | 6358.50 | 6329.03 | 0.00 | ORB-long ORB[6278.00,6337.00] vol=1.6x ATR=14.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:15:00 | 6380.34 | 6337.86 | 0.00 | T1 1.5R @ 6380.34 |
| Stop hit — per-position SL triggered | 2025-05-26 10:25:00 | 6358.50 | 6340.53 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:25:00 | 6363.00 | 6394.50 | 0.00 | ORB-short ORB[6402.50,6438.00] vol=1.5x ATR=15.53 |
| Stop hit — per-position SL triggered | 2025-05-28 10:30:00 | 6378.53 | 6393.06 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:15:00 | 6456.50 | 6465.76 | 0.00 | ORB-short ORB[6465.00,6519.00] vol=1.9x ATR=14.56 |
| Stop hit — per-position SL triggered | 2025-05-29 12:55:00 | 6471.06 | 6463.42 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:10:00 | 6369.00 | 6402.68 | 0.00 | ORB-short ORB[6397.00,6436.50] vol=1.8x ATR=12.30 |
| Stop hit — per-position SL triggered | 2025-06-03 11:55:00 | 6381.30 | 6395.11 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 11:00:00 | 6481.00 | 6462.33 | 0.00 | ORB-long ORB[6446.00,6475.50] vol=3.9x ATR=10.46 |
| Stop hit — per-position SL triggered | 2025-06-05 11:05:00 | 6470.54 | 6462.98 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:05:00 | 6478.50 | 6494.38 | 0.00 | ORB-short ORB[6482.50,6520.00] vol=2.7x ATR=11.47 |
| Stop hit — per-position SL triggered | 2025-06-06 11:10:00 | 6489.97 | 6494.21 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 6322.00 | 6343.83 | 0.00 | ORB-short ORB[6335.00,6383.50] vol=2.7x ATR=16.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:35:00 | 6297.97 | 6332.12 | 0.00 | T1 1.5R @ 6297.97 |
| Target hit | 2025-06-16 10:20:00 | 6301.00 | 6297.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2025-06-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 11:05:00 | 6444.50 | 6412.77 | 0.00 | ORB-long ORB[6362.00,6425.00] vol=2.4x ATR=13.47 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 6431.03 | 6414.45 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:10:00 | 6211.50 | 6236.62 | 0.00 | ORB-short ORB[6225.00,6282.00] vol=2.3x ATR=12.69 |
| Stop hit — per-position SL triggered | 2025-06-26 11:45:00 | 6224.19 | 6233.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:50:00 | 6363.00 | 6335.46 | 0.00 | ORB-long ORB[6282.00,6342.00] vol=4.3x ATR=15.69 |
| Stop hit — per-position SL triggered | 2025-06-27 09:55:00 | 6347.31 | 6336.18 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:45:00 | 6244.00 | 6290.57 | 0.00 | ORB-short ORB[6289.00,6330.00] vol=3.0x ATR=13.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:50:00 | 6223.79 | 6280.43 | 0.00 | T1 1.5R @ 6223.79 |
| Target hit | 2025-07-01 12:30:00 | 6225.00 | 6215.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2025-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:35:00 | 6103.50 | 6149.05 | 0.00 | ORB-short ORB[6148.00,6208.00] vol=2.8x ATR=16.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:55:00 | 6079.06 | 6136.10 | 0.00 | T1 1.5R @ 6079.06 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 6103.50 | 6125.63 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:40:00 | 6232.00 | 6190.33 | 0.00 | ORB-long ORB[6151.00,6230.00] vol=1.7x ATR=13.68 |
| Stop hit — per-position SL triggered | 2025-07-04 10:50:00 | 6218.32 | 6191.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:45:00 | 6168.50 | 6183.41 | 0.00 | ORB-short ORB[6169.50,6210.00] vol=2.5x ATR=13.58 |
| Stop hit — per-position SL triggered | 2025-07-07 09:50:00 | 6182.08 | 6183.21 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:15:00 | 6118.00 | 6130.08 | 0.00 | ORB-short ORB[6164.00,6195.50] vol=2.0x ATR=12.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:00:00 | 6099.03 | 6126.31 | 0.00 | T1 1.5R @ 6099.03 |
| Stop hit — per-position SL triggered | 2025-07-08 12:20:00 | 6118.00 | 6124.05 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:55:00 | 6140.00 | 6155.08 | 0.00 | ORB-short ORB[6145.00,6187.50] vol=2.7x ATR=13.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 12:05:00 | 6119.92 | 6144.57 | 0.00 | T1 1.5R @ 6119.92 |
| Target hit | 2025-07-09 13:25:00 | 6137.50 | 6136.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:15:00 | 6108.50 | 6130.21 | 0.00 | ORB-short ORB[6113.50,6150.00] vol=1.5x ATR=14.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:35:00 | 6087.26 | 6119.43 | 0.00 | T1 1.5R @ 6087.26 |
| Target hit | 2025-07-10 12:20:00 | 6103.50 | 6102.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — SELL (started 2025-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:00:00 | 6270.50 | 6294.02 | 0.00 | ORB-short ORB[6277.50,6319.50] vol=1.5x ATR=13.83 |
| Stop hit — per-position SL triggered | 2025-07-17 11:30:00 | 6284.33 | 6292.01 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 09:35:00 | 5999.50 | 6010.75 | 0.00 | ORB-short ORB[6000.50,6032.50] vol=1.8x ATR=16.27 |
| Stop hit — per-position SL triggered | 2025-07-29 09:40:00 | 6015.77 | 6011.19 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:50:00 | 6108.00 | 6087.06 | 0.00 | ORB-long ORB[6062.00,6090.00] vol=1.7x ATR=15.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 09:55:00 | 6131.90 | 6105.45 | 0.00 | T1 1.5R @ 6131.90 |
| Stop hit — per-position SL triggered | 2025-07-30 10:35:00 | 6108.00 | 6114.26 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:50:00 | 5979.00 | 6012.89 | 0.00 | ORB-short ORB[5988.00,6050.00] vol=4.2x ATR=16.95 |
| Stop hit — per-position SL triggered | 2025-08-04 11:40:00 | 5995.95 | 6006.88 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 11:00:00 | 5800.50 | 5834.06 | 0.00 | ORB-short ORB[5815.00,5871.00] vol=3.6x ATR=14.04 |
| Stop hit — per-position SL triggered | 2025-08-08 11:20:00 | 5814.54 | 5832.28 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:15:00 | 5695.00 | 5707.33 | 0.00 | ORB-short ORB[5698.50,5769.50] vol=1.6x ATR=13.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:50:00 | 5674.50 | 5704.20 | 0.00 | T1 1.5R @ 5674.50 |
| Stop hit — per-position SL triggered | 2025-08-11 12:00:00 | 5695.00 | 5703.77 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 11:05:00 | 5754.00 | 5711.79 | 0.00 | ORB-long ORB[5674.00,5717.00] vol=4.8x ATR=15.52 |
| Stop hit — per-position SL triggered | 2025-08-14 11:25:00 | 5738.48 | 5717.56 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 5750.00 | 5720.75 | 0.00 | ORB-long ORB[5696.50,5731.00] vol=2.4x ATR=15.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:45:00 | 5772.89 | 5736.63 | 0.00 | T1 1.5R @ 5772.89 |
| Target hit | 2025-08-18 10:45:00 | 5753.50 | 5756.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2025-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:30:00 | 5647.50 | 5675.70 | 0.00 | ORB-short ORB[5658.50,5712.00] vol=2.0x ATR=15.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 09:40:00 | 5623.60 | 5664.67 | 0.00 | T1 1.5R @ 5623.60 |
| Stop hit — per-position SL triggered | 2025-08-19 09:50:00 | 5647.50 | 5662.01 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 5710.50 | 5725.20 | 0.00 | ORB-short ORB[5712.00,5770.00] vol=1.9x ATR=12.55 |
| Stop hit — per-position SL triggered | 2025-08-21 09:55:00 | 5723.05 | 5723.18 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:15:00 | 5574.00 | 5535.86 | 0.00 | ORB-long ORB[5513.00,5565.00] vol=1.8x ATR=16.81 |
| Stop hit — per-position SL triggered | 2025-08-26 10:25:00 | 5557.19 | 5539.31 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:35:00 | 5465.50 | 5453.97 | 0.00 | ORB-long ORB[5428.00,5458.50] vol=1.5x ATR=12.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:55:00 | 5484.50 | 5460.21 | 0.00 | T1 1.5R @ 5484.50 |
| Stop hit — per-position SL triggered | 2025-09-05 10:05:00 | 5465.50 | 5459.61 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 11:15:00 | 5495.50 | 5479.14 | 0.00 | ORB-long ORB[5448.50,5489.50] vol=4.4x ATR=10.14 |
| Stop hit — per-position SL triggered | 2025-09-08 11:25:00 | 5485.36 | 5479.36 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:35:00 | 5528.00 | 5506.58 | 0.00 | ORB-long ORB[5475.50,5519.00] vol=3.4x ATR=13.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:40:00 | 5548.03 | 5533.18 | 0.00 | T1 1.5R @ 5548.03 |
| Stop hit — per-position SL triggered | 2025-09-09 10:50:00 | 5528.00 | 5617.15 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:40:00 | 5691.50 | 5729.77 | 0.00 | ORB-short ORB[5707.50,5765.00] vol=1.6x ATR=15.04 |
| Stop hit — per-position SL triggered | 2025-09-17 11:30:00 | 5706.54 | 5718.40 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:35:00 | 5380.00 | 5402.30 | 0.00 | ORB-short ORB[5391.50,5460.00] vol=2.0x ATR=15.91 |
| Stop hit — per-position SL triggered | 2025-09-26 09:50:00 | 5395.91 | 5395.76 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:35:00 | 5243.50 | 5222.79 | 0.00 | ORB-long ORB[5190.00,5229.00] vol=2.9x ATR=17.05 |
| Stop hit — per-position SL triggered | 2025-10-03 09:45:00 | 5226.45 | 5226.39 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 5330.50 | 5368.04 | 0.00 | ORB-short ORB[5372.00,5427.50] vol=1.6x ATR=10.96 |
| Stop hit — per-position SL triggered | 2025-10-07 12:00:00 | 5341.46 | 5362.16 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:55:00 | 5517.50 | 5479.93 | 0.00 | ORB-long ORB[5436.00,5498.00] vol=2.6x ATR=22.66 |
| Stop hit — per-position SL triggered | 2025-10-09 10:10:00 | 5494.84 | 5486.24 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 5336.00 | 5361.41 | 0.00 | ORB-short ORB[5370.00,5400.00] vol=2.0x ATR=11.46 |
| Stop hit — per-position SL triggered | 2025-10-14 11:30:00 | 5347.46 | 5358.94 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 5396.00 | 5393.63 | 0.00 | ORB-long ORB[5355.00,5388.00] vol=3.2x ATR=14.17 |
| Stop hit — per-position SL triggered | 2025-10-16 09:40:00 | 5381.83 | 5392.65 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 5368.00 | 5387.00 | 0.00 | ORB-short ORB[5381.00,5420.00] vol=1.6x ATR=11.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:50:00 | 5350.02 | 5377.19 | 0.00 | T1 1.5R @ 5350.02 |
| Stop hit — per-position SL triggered | 2025-10-17 12:00:00 | 5368.00 | 5375.67 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:45:00 | 5555.00 | 5521.20 | 0.00 | ORB-long ORB[5462.00,5526.00] vol=1.5x ATR=21.08 |
| Stop hit — per-position SL triggered | 2025-10-24 10:10:00 | 5533.92 | 5527.61 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:10:00 | 5591.50 | 5559.73 | 0.00 | ORB-long ORB[5520.00,5566.50] vol=1.8x ATR=12.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:25:00 | 5609.81 | 5565.87 | 0.00 | T1 1.5R @ 5609.81 |
| Stop hit — per-position SL triggered | 2025-10-27 10:30:00 | 5591.50 | 5566.58 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 09:55:00 | 5497.50 | 5531.16 | 0.00 | ORB-short ORB[5530.00,5574.50] vol=1.5x ATR=12.20 |
| Stop hit — per-position SL triggered | 2025-10-29 10:25:00 | 5509.70 | 5525.72 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 10:55:00 | 5400.00 | 5422.35 | 0.00 | ORB-short ORB[5422.00,5455.00] vol=4.9x ATR=10.89 |
| Stop hit — per-position SL triggered | 2025-11-03 11:10:00 | 5410.89 | 5418.22 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 5400.00 | 5424.60 | 0.00 | ORB-short ORB[5421.50,5464.00] vol=2.4x ATR=8.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:15:00 | 5387.30 | 5421.34 | 0.00 | T1 1.5R @ 5387.30 |
| Target hit | 2025-11-04 15:20:00 | 5389.50 | 5391.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-11-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 11:05:00 | 5135.00 | 5157.29 | 0.00 | ORB-short ORB[5140.50,5213.00] vol=2.2x ATR=11.35 |
| Stop hit — per-position SL triggered | 2025-11-07 11:20:00 | 5146.35 | 5157.03 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:20:00 | 5208.50 | 5181.95 | 0.00 | ORB-long ORB[5151.00,5195.00] vol=2.6x ATR=13.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 12:45:00 | 5228.02 | 5198.62 | 0.00 | T1 1.5R @ 5228.02 |
| Stop hit — per-position SL triggered | 2025-11-10 15:05:00 | 5208.50 | 5208.52 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:05:00 | 5235.00 | 5207.65 | 0.00 | ORB-long ORB[5181.00,5224.00] vol=5.0x ATR=12.69 |
| Stop hit — per-position SL triggered | 2025-11-11 10:10:00 | 5222.31 | 5209.12 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:40:00 | 5341.00 | 5320.57 | 0.00 | ORB-long ORB[5269.00,5335.00] vol=1.8x ATR=13.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:45:00 | 5360.99 | 5331.12 | 0.00 | T1 1.5R @ 5360.99 |
| Target hit | 2025-11-12 13:30:00 | 5417.00 | 5424.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-11-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 09:35:00 | 5235.00 | 5252.14 | 0.00 | ORB-short ORB[5240.00,5306.50] vol=2.4x ATR=13.51 |
| Stop hit — per-position SL triggered | 2025-11-14 09:40:00 | 5248.51 | 5252.71 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 5135.00 | 5158.07 | 0.00 | ORB-short ORB[5155.00,5194.00] vol=1.9x ATR=9.46 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 5144.46 | 5156.20 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:35:00 | 5143.50 | 5120.59 | 0.00 | ORB-long ORB[5097.00,5126.00] vol=1.9x ATR=10.84 |
| Stop hit — per-position SL triggered | 2025-12-02 10:00:00 | 5132.66 | 5125.96 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:35:00 | 5208.50 | 5199.91 | 0.00 | ORB-long ORB[5170.50,5204.50] vol=1.7x ATR=11.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:50:00 | 5225.22 | 5208.67 | 0.00 | T1 1.5R @ 5225.22 |
| Stop hit — per-position SL triggered | 2025-12-04 11:40:00 | 5208.50 | 5225.24 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:45:00 | 5249.50 | 5228.85 | 0.00 | ORB-long ORB[5202.50,5248.00] vol=3.4x ATR=13.60 |
| Stop hit — per-position SL triggered | 2025-12-05 10:55:00 | 5235.90 | 5230.39 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 4918.00 | 4963.77 | 0.00 | ORB-short ORB[4963.50,5003.00] vol=1.6x ATR=12.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 12:35:00 | 4899.62 | 4937.84 | 0.00 | T1 1.5R @ 4899.62 |
| Target hit | 2025-12-10 15:20:00 | 4862.00 | 4911.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-12-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:05:00 | 5005.00 | 5016.94 | 0.00 | ORB-short ORB[5011.50,5040.00] vol=2.2x ATR=11.76 |
| Stop hit — per-position SL triggered | 2025-12-12 14:00:00 | 5016.76 | 5009.83 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:25:00 | 5120.00 | 5058.83 | 0.00 | ORB-long ORB[5028.50,5079.50] vol=3.8x ATR=16.89 |
| Stop hit — per-position SL triggered | 2025-12-15 10:30:00 | 5103.11 | 5063.56 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 5023.00 | 5009.45 | 0.00 | ORB-long ORB[4985.00,5014.00] vol=2.8x ATR=9.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:35:00 | 5037.67 | 5019.43 | 0.00 | T1 1.5R @ 5037.67 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 5023.00 | 5020.01 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:55:00 | 5465.00 | 5437.55 | 0.00 | ORB-long ORB[5402.50,5450.00] vol=1.6x ATR=20.30 |
| Stop hit — per-position SL triggered | 2025-12-23 10:20:00 | 5444.70 | 5441.68 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 5360.00 | 5341.26 | 0.00 | ORB-long ORB[5316.00,5358.00] vol=2.9x ATR=13.94 |
| Stop hit — per-position SL triggered | 2025-12-29 09:55:00 | 5346.06 | 5344.04 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:25:00 | 5277.50 | 5299.66 | 0.00 | ORB-short ORB[5296.50,5342.00] vol=2.9x ATR=10.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:00:00 | 5262.20 | 5289.90 | 0.00 | T1 1.5R @ 5262.20 |
| Target hit | 2025-12-30 15:20:00 | 5194.00 | 5246.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2026-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:05:00 | 5206.50 | 5215.11 | 0.00 | ORB-short ORB[5221.00,5253.00] vol=2.2x ATR=11.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:20:00 | 5189.42 | 5212.24 | 0.00 | T1 1.5R @ 5189.42 |
| Stop hit — per-position SL triggered | 2026-01-01 10:25:00 | 5206.50 | 5211.98 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 5240.00 | 5217.51 | 0.00 | ORB-long ORB[5197.50,5219.00] vol=1.6x ATR=10.51 |
| Stop hit — per-position SL triggered | 2026-01-02 10:05:00 | 5229.49 | 5218.97 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 5321.50 | 5341.49 | 0.00 | ORB-short ORB[5326.50,5373.00] vol=2.7x ATR=16.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:55:00 | 5296.90 | 5331.42 | 0.00 | T1 1.5R @ 5296.90 |
| Stop hit — per-position SL triggered | 2026-01-05 10:05:00 | 5321.50 | 5330.75 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:30:00 | 5527.50 | 5553.42 | 0.00 | ORB-short ORB[5532.50,5614.50] vol=2.7x ATR=26.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:00:00 | 5487.70 | 5536.69 | 0.00 | T1 1.5R @ 5487.70 |
| Stop hit — per-position SL triggered | 2026-01-19 10:10:00 | 5527.50 | 5534.47 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 5460.50 | 5513.32 | 0.00 | ORB-short ORB[5491.50,5566.00] vol=1.8x ATR=23.04 |
| Stop hit — per-position SL triggered | 2026-01-20 10:05:00 | 5483.54 | 5485.39 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:45:00 | 5381.00 | 5407.89 | 0.00 | ORB-short ORB[5405.50,5453.50] vol=1.6x ATR=14.83 |
| Stop hit — per-position SL triggered | 2026-01-28 12:00:00 | 5395.83 | 5402.74 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:45:00 | 5336.50 | 5362.86 | 0.00 | ORB-short ORB[5342.50,5405.50] vol=1.7x ATR=13.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:55:00 | 5315.72 | 5357.57 | 0.00 | T1 1.5R @ 5315.72 |
| Target hit | 2026-01-29 13:55:00 | 5322.00 | 5320.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2026-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:40:00 | 5372.00 | 5335.41 | 0.00 | ORB-long ORB[5285.50,5319.50] vol=1.5x ATR=11.25 |
| Stop hit — per-position SL triggered | 2026-02-01 10:50:00 | 5360.75 | 5339.25 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 5289.00 | 5262.26 | 0.00 | ORB-long ORB[5228.00,5266.00] vol=2.0x ATR=12.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 5308.12 | 5283.63 | 0.00 | T1 1.5R @ 5308.12 |
| Target hit | 2026-02-10 11:45:00 | 5325.00 | 5328.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2026-02-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:55:00 | 4773.00 | 4802.11 | 0.00 | ORB-short ORB[4778.50,4836.50] vol=2.5x ATR=15.93 |
| Stop hit — per-position SL triggered | 2026-02-16 12:10:00 | 4788.93 | 4792.51 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:55:00 | 4800.00 | 4832.06 | 0.00 | ORB-short ORB[4802.00,4873.00] vol=1.7x ATR=19.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 4771.21 | 4813.57 | 0.00 | T1 1.5R @ 4771.21 |
| Target hit | 2026-02-23 15:20:00 | 4703.50 | 4744.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 4271.00 | 4275.67 | 0.00 | ORB-short ORB[4272.00,4327.00] vol=1.6x ATR=12.96 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 4283.96 | 4275.62 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 4198.70 | 4223.20 | 0.00 | ORB-short ORB[4215.00,4255.00] vol=3.5x ATR=15.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:30:00 | 4175.79 | 4214.12 | 0.00 | T1 1.5R @ 4175.79 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 4198.70 | 4193.47 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 4119.20 | 4156.44 | 0.00 | ORB-short ORB[4135.80,4190.00] vol=1.6x ATR=17.20 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 4136.40 | 4155.96 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 4246.00 | 4216.61 | 0.00 | ORB-long ORB[4187.50,4228.60] vol=1.7x ATR=18.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:20:00 | 4274.22 | 4243.64 | 0.00 | T1 1.5R @ 4274.22 |
| Stop hit — per-position SL triggered | 2026-03-25 11:35:00 | 4246.00 | 4243.97 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 4055.60 | 4080.11 | 0.00 | ORB-short ORB[4073.60,4130.00] vol=1.8x ATR=13.48 |
| Stop hit — per-position SL triggered | 2026-03-30 11:05:00 | 4069.08 | 4078.93 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 4399.40 | 4437.45 | 0.00 | ORB-short ORB[4425.20,4490.00] vol=1.7x ATR=16.53 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 4415.93 | 4424.94 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 4598.00 | 4554.98 | 0.00 | ORB-long ORB[4516.60,4561.70] vol=2.8x ATR=14.95 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 4583.05 | 4557.18 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 4258.90 | 4225.20 | 0.00 | ORB-long ORB[4198.60,4244.70] vol=1.5x ATR=19.26 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 4239.64 | 4233.51 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 4181.50 | 4158.48 | 0.00 | ORB-long ORB[4138.50,4180.00] vol=2.0x ATR=12.61 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 4168.89 | 4166.08 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 4208.00 | 4187.90 | 0.00 | ORB-long ORB[4151.00,4205.80] vol=1.8x ATR=12.38 |
| Stop hit — per-position SL triggered | 2026-05-05 09:55:00 | 4195.62 | 4189.81 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 4295.20 | 4278.23 | 0.00 | ORB-long ORB[4245.00,4290.00] vol=3.7x ATR=12.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:55:00 | 4314.56 | 4290.09 | 0.00 | T1 1.5R @ 4314.56 |
| Stop hit — per-position SL triggered | 2026-05-06 10:35:00 | 4295.20 | 4293.88 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 4276.60 | 4296.62 | 0.00 | ORB-short ORB[4284.00,4328.70] vol=1.6x ATR=13.30 |
| Stop hit — per-position SL triggered | 2026-05-07 12:20:00 | 4289.90 | 4288.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:30:00 | 6076.00 | 2025-05-14 09:35:00 | 6054.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-15 09:30:00 | 6169.50 | 2025-05-15 09:40:00 | 6147.60 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-05-20 09:30:00 | 6230.50 | 2025-05-20 09:35:00 | 6252.24 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-23 09:30:00 | 6252.00 | 2025-05-23 09:55:00 | 6274.57 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-05-23 09:30:00 | 6252.00 | 2025-05-23 15:20:00 | 6291.50 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-05-26 10:10:00 | 6358.50 | 2025-05-26 10:15:00 | 6380.34 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-26 10:10:00 | 6358.50 | 2025-05-26 10:25:00 | 6358.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-28 10:25:00 | 6363.00 | 2025-05-28 10:30:00 | 6378.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-29 11:15:00 | 6456.50 | 2025-05-29 12:55:00 | 6471.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-03 11:10:00 | 6369.00 | 2025-06-03 11:55:00 | 6381.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-05 11:00:00 | 6481.00 | 2025-06-05 11:05:00 | 6470.54 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-06-06 11:05:00 | 6478.50 | 2025-06-06 11:10:00 | 6489.97 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-06-16 09:30:00 | 6322.00 | 2025-06-16 09:35:00 | 6297.97 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-16 09:30:00 | 6322.00 | 2025-06-16 10:20:00 | 6301.00 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-06-17 11:05:00 | 6444.50 | 2025-06-17 11:15:00 | 6431.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-26 11:10:00 | 6211.50 | 2025-06-26 11:45:00 | 6224.19 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-27 09:50:00 | 6363.00 | 2025-06-27 09:55:00 | 6347.31 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-01 10:45:00 | 6244.00 | 2025-07-01 10:50:00 | 6223.79 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-01 10:45:00 | 6244.00 | 2025-07-01 12:30:00 | 6225.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-02 10:35:00 | 6103.50 | 2025-07-02 10:55:00 | 6079.06 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-02 10:35:00 | 6103.50 | 2025-07-02 11:20:00 | 6103.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 10:40:00 | 6232.00 | 2025-07-04 10:50:00 | 6218.32 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-07 09:45:00 | 6168.50 | 2025-07-07 09:50:00 | 6182.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-08 11:15:00 | 6118.00 | 2025-07-08 12:00:00 | 6099.03 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-08 11:15:00 | 6118.00 | 2025-07-08 12:20:00 | 6118.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-09 09:55:00 | 6140.00 | 2025-07-09 12:05:00 | 6119.92 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-09 09:55:00 | 6140.00 | 2025-07-09 13:25:00 | 6137.50 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-07-10 10:15:00 | 6108.50 | 2025-07-10 11:35:00 | 6087.26 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-10 10:15:00 | 6108.50 | 2025-07-10 12:20:00 | 6103.50 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-07-17 11:00:00 | 6270.50 | 2025-07-17 11:30:00 | 6284.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-29 09:35:00 | 5999.50 | 2025-07-29 09:40:00 | 6015.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-30 09:50:00 | 6108.00 | 2025-07-30 09:55:00 | 6131.90 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-30 09:50:00 | 6108.00 | 2025-07-30 10:35:00 | 6108.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-04 10:50:00 | 5979.00 | 2025-08-04 11:40:00 | 5995.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-08 11:00:00 | 5800.50 | 2025-08-08 11:20:00 | 5814.54 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-11 11:15:00 | 5695.00 | 2025-08-11 11:50:00 | 5674.50 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-11 11:15:00 | 5695.00 | 2025-08-11 12:00:00 | 5695.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-14 11:05:00 | 5754.00 | 2025-08-14 11:25:00 | 5738.48 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-18 09:30:00 | 5750.00 | 2025-08-18 09:45:00 | 5772.89 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-18 09:30:00 | 5750.00 | 2025-08-18 10:45:00 | 5753.50 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-08-19 09:30:00 | 5647.50 | 2025-08-19 09:40:00 | 5623.60 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-19 09:30:00 | 5647.50 | 2025-08-19 09:50:00 | 5647.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 09:40:00 | 5710.50 | 2025-08-21 09:55:00 | 5723.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-08-26 10:15:00 | 5574.00 | 2025-08-26 10:25:00 | 5557.19 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-05 09:35:00 | 5465.50 | 2025-09-05 09:55:00 | 5484.50 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-05 09:35:00 | 5465.50 | 2025-09-05 10:05:00 | 5465.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 11:15:00 | 5495.50 | 2025-09-08 11:25:00 | 5485.36 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-09 09:35:00 | 5528.00 | 2025-09-09 09:40:00 | 5548.03 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-09 09:35:00 | 5528.00 | 2025-09-09 10:50:00 | 5528.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-17 10:40:00 | 5691.50 | 2025-09-17 11:30:00 | 5706.54 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-26 09:35:00 | 5380.00 | 2025-09-26 09:50:00 | 5395.91 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-03 09:35:00 | 5243.50 | 2025-10-03 09:45:00 | 5226.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-07 11:10:00 | 5330.50 | 2025-10-07 12:00:00 | 5341.46 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-09 09:55:00 | 5517.50 | 2025-10-09 10:10:00 | 5494.84 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-14 11:15:00 | 5336.00 | 2025-10-14 11:30:00 | 5347.46 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-16 09:35:00 | 5396.00 | 2025-10-16 09:40:00 | 5381.83 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-17 10:20:00 | 5368.00 | 2025-10-17 11:50:00 | 5350.02 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-17 10:20:00 | 5368.00 | 2025-10-17 12:00:00 | 5368.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:45:00 | 5555.00 | 2025-10-24 10:10:00 | 5533.92 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-27 10:10:00 | 5591.50 | 2025-10-27 10:25:00 | 5609.81 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-27 10:10:00 | 5591.50 | 2025-10-27 10:30:00 | 5591.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 09:55:00 | 5497.50 | 2025-10-29 10:25:00 | 5509.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-03 10:55:00 | 5400.00 | 2025-11-03 11:10:00 | 5410.89 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-04 11:00:00 | 5400.00 | 2025-11-04 11:15:00 | 5387.30 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-04 11:00:00 | 5400.00 | 2025-11-04 15:20:00 | 5389.50 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-11-07 11:05:00 | 5135.00 | 2025-11-07 11:20:00 | 5146.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-10 10:20:00 | 5208.50 | 2025-11-10 12:45:00 | 5228.02 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-10 10:20:00 | 5208.50 | 2025-11-10 15:05:00 | 5208.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-11 10:05:00 | 5235.00 | 2025-11-11 10:10:00 | 5222.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-12 09:40:00 | 5341.00 | 2025-11-12 09:45:00 | 5360.99 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-12 09:40:00 | 5341.00 | 2025-11-12 13:30:00 | 5417.00 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2025-11-14 09:35:00 | 5235.00 | 2025-11-14 09:40:00 | 5248.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-01 10:50:00 | 5135.00 | 2025-12-01 11:15:00 | 5144.46 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-02 09:35:00 | 5143.50 | 2025-12-02 10:00:00 | 5132.66 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-04 09:35:00 | 5208.50 | 2025-12-04 09:50:00 | 5225.22 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-04 09:35:00 | 5208.50 | 2025-12-04 11:40:00 | 5208.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:45:00 | 5249.50 | 2025-12-05 10:55:00 | 5235.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-10 10:45:00 | 4918.00 | 2025-12-10 12:35:00 | 4899.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-10 10:45:00 | 4918.00 | 2025-12-10 15:20:00 | 4862.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-12-12 10:05:00 | 5005.00 | 2025-12-12 14:00:00 | 5016.76 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-15 10:25:00 | 5120.00 | 2025-12-15 10:30:00 | 5103.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-17 09:30:00 | 5023.00 | 2025-12-17 09:35:00 | 5037.67 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-17 09:30:00 | 5023.00 | 2025-12-17 09:40:00 | 5023.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 09:55:00 | 5465.00 | 2025-12-23 10:20:00 | 5444.70 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-29 09:30:00 | 5360.00 | 2025-12-29 09:55:00 | 5346.06 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-30 10:25:00 | 5277.50 | 2025-12-30 11:00:00 | 5262.20 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-30 10:25:00 | 5277.50 | 2025-12-30 15:20:00 | 5194.00 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2026-01-01 10:05:00 | 5206.50 | 2026-01-01 10:20:00 | 5189.42 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-01 10:05:00 | 5206.50 | 2026-01-01 10:25:00 | 5206.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:00:00 | 5240.00 | 2026-01-02 10:05:00 | 5229.49 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-05 09:30:00 | 5321.50 | 2026-01-05 09:55:00 | 5296.90 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-01-05 09:30:00 | 5321.50 | 2026-01-05 10:05:00 | 5321.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 09:30:00 | 5527.50 | 2026-01-19 10:00:00 | 5487.70 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-01-19 09:30:00 | 5527.50 | 2026-01-19 10:10:00 | 5527.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 09:35:00 | 5460.50 | 2026-01-20 10:05:00 | 5483.54 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-01-28 10:45:00 | 5381.00 | 2026-01-28 12:00:00 | 5395.83 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-29 09:45:00 | 5336.50 | 2026-01-29 09:55:00 | 5315.72 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-29 09:45:00 | 5336.50 | 2026-01-29 13:55:00 | 5322.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-01 10:40:00 | 5372.00 | 2026-02-01 10:50:00 | 5360.75 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-10 09:35:00 | 5289.00 | 2026-02-10 09:40:00 | 5308.12 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-10 09:35:00 | 5289.00 | 2026-02-10 11:45:00 | 5325.00 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-02-16 10:55:00 | 4773.00 | 2026-02-16 12:10:00 | 4788.93 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-23 09:55:00 | 4800.00 | 2026-02-23 10:40:00 | 4771.21 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-23 09:55:00 | 4800.00 | 2026-02-23 15:20:00 | 4703.50 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2026-03-13 10:35:00 | 4271.00 | 2026-03-13 10:50:00 | 4283.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-16 10:20:00 | 4198.70 | 2026-03-16 10:30:00 | 4175.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-16 10:20:00 | 4198.70 | 2026-03-16 11:15:00 | 4198.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-23 11:00:00 | 4119.20 | 2026-03-23 11:05:00 | 4136.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 09:45:00 | 4246.00 | 2026-03-25 11:20:00 | 4274.22 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-25 09:45:00 | 4246.00 | 2026-03-25 11:35:00 | 4246.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 10:55:00 | 4055.60 | 2026-03-30 11:05:00 | 4069.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-10 09:40:00 | 4399.40 | 2026-04-10 09:55:00 | 4415.93 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-21 10:10:00 | 4598.00 | 2026-04-21 10:15:00 | 4583.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 09:45:00 | 4258.90 | 2026-04-27 10:05:00 | 4239.64 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-29 09:40:00 | 4181.50 | 2026-04-29 09:55:00 | 4168.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:50:00 | 4208.00 | 2026-05-05 09:55:00 | 4195.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-06 09:30:00 | 4295.20 | 2026-05-06 09:55:00 | 4314.56 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-06 09:30:00 | 4295.20 | 2026-05-06 10:35:00 | 4295.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 09:50:00 | 4276.60 | 2026-05-07 12:20:00 | 4289.90 | STOP_HIT | 1.00 | -0.31% |
