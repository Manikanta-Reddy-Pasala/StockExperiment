# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 4600.10
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
| ENTRY1 | 80 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 19 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 61
- **Target hits / Stop hits / Partials:** 19 / 61 / 33
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 25.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 28 | 50.9% | 11 | 27 | 17 | 0.31% | 16.9% |
| BUY @ 2nd Alert (retest1) | 55 | 28 | 50.9% | 11 | 27 | 17 | 0.31% | 16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 24 | 41.4% | 8 | 34 | 16 | 0.15% | 8.8% |
| SELL @ 2nd Alert (retest1) | 58 | 24 | 41.4% | 8 | 34 | 16 | 0.15% | 8.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 52 | 46.0% | 19 | 61 | 33 | 0.23% | 25.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 5175.05 | 5195.00 | 0.00 | ORB-short ORB[5239.55,5269.70] vol=1.8x ATR=20.08 |
| Stop hit — per-position SL triggered | 2024-05-13 11:25:00 | 5195.13 | 5194.36 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:30:00 | 5478.80 | 5448.62 | 0.00 | ORB-long ORB[5400.00,5444.75] vol=1.6x ATR=23.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:45:00 | 5514.59 | 5492.91 | 0.00 | T1 1.5R @ 5514.59 |
| Target hit | 2024-05-17 10:50:00 | 5501.05 | 5501.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2024-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 11:05:00 | 5583.60 | 5614.77 | 0.00 | ORB-short ORB[5627.00,5673.60] vol=2.5x ATR=14.16 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 5597.76 | 5612.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:55:00 | 4959.70 | 4993.53 | 0.00 | ORB-short ORB[5041.00,5075.95] vol=3.6x ATR=16.26 |
| Stop hit — per-position SL triggered | 2024-05-31 11:10:00 | 4975.96 | 4991.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 10:20:00 | 5161.75 | 5152.11 | 0.00 | ORB-long ORB[5085.55,5158.95] vol=4.6x ATR=25.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 10:55:00 | 5199.25 | 5161.76 | 0.00 | T1 1.5R @ 5199.25 |
| Target hit | 2024-06-05 12:25:00 | 5210.00 | 5210.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 5377.90 | 5355.33 | 0.00 | ORB-long ORB[5302.00,5360.00] vol=2.5x ATR=18.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 10:30:00 | 5406.37 | 5365.96 | 0.00 | T1 1.5R @ 5406.37 |
| Target hit | 2024-06-06 15:20:00 | 5538.55 | 5487.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:15:00 | 5585.00 | 5532.52 | 0.00 | ORB-long ORB[5501.20,5560.00] vol=4.6x ATR=27.39 |
| Stop hit — per-position SL triggered | 2024-06-07 10:25:00 | 5557.61 | 5542.90 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:10:00 | 5810.00 | 5711.39 | 0.00 | ORB-long ORB[5675.60,5760.00] vol=8.1x ATR=27.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:15:00 | 5851.05 | 5734.81 | 0.00 | T1 1.5R @ 5851.05 |
| Target hit | 2024-06-10 15:20:00 | 5902.00 | 5883.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 5999.00 | 5950.30 | 0.00 | ORB-long ORB[5890.00,5976.80] vol=2.7x ATR=31.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:05:00 | 6046.08 | 5987.68 | 0.00 | T1 1.5R @ 6046.08 |
| Target hit | 2024-06-11 15:20:00 | 6044.35 | 6049.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:05:00 | 6123.15 | 6102.21 | 0.00 | ORB-long ORB[6066.00,6120.10] vol=2.5x ATR=18.96 |
| Stop hit — per-position SL triggered | 2024-06-13 10:20:00 | 6104.19 | 6102.82 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 6050.00 | 6070.92 | 0.00 | ORB-short ORB[6083.00,6145.00] vol=7.9x ATR=21.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:45:00 | 6017.84 | 6047.61 | 0.00 | T1 1.5R @ 6017.84 |
| Target hit | 2024-06-14 10:00:00 | 6048.25 | 6036.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2024-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:35:00 | 6610.00 | 6569.53 | 0.00 | ORB-long ORB[6502.20,6598.45] vol=1.6x ATR=27.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:40:00 | 6651.59 | 6697.93 | 0.00 | T1 1.5R @ 6651.59 |
| Target hit | 2024-06-27 09:45:00 | 6667.40 | 6703.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-07-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:40:00 | 6591.55 | 6629.55 | 0.00 | ORB-short ORB[6675.70,6730.75] vol=1.7x ATR=27.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:40:00 | 6549.90 | 6615.45 | 0.00 | T1 1.5R @ 6549.90 |
| Stop hit — per-position SL triggered | 2024-07-02 13:05:00 | 6591.55 | 6586.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:05:00 | 6594.00 | 6638.24 | 0.00 | ORB-short ORB[6600.05,6698.00] vol=1.7x ATR=25.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:10:00 | 6556.38 | 6633.79 | 0.00 | T1 1.5R @ 6556.38 |
| Target hit | 2024-07-03 15:20:00 | 6445.40 | 6507.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:35:00 | 6521.95 | 6540.21 | 0.00 | ORB-short ORB[6533.15,6570.00] vol=2.1x ATR=15.16 |
| Stop hit — per-position SL triggered | 2024-07-05 11:30:00 | 6537.11 | 6534.57 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:45:00 | 6660.00 | 6621.08 | 0.00 | ORB-long ORB[6568.00,6616.25] vol=4.7x ATR=18.17 |
| Stop hit — per-position SL triggered | 2024-07-08 15:15:00 | 6641.83 | 6642.19 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:10:00 | 6700.00 | 6680.63 | 0.00 | ORB-long ORB[6648.15,6695.00] vol=6.1x ATR=16.87 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 6683.13 | 6680.75 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 6562.45 | 6587.92 | 0.00 | ORB-short ORB[6580.20,6655.95] vol=1.6x ATR=19.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:45:00 | 6533.37 | 6575.93 | 0.00 | T1 1.5R @ 6533.37 |
| Stop hit — per-position SL triggered | 2024-07-10 09:50:00 | 6562.45 | 6572.26 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:40:00 | 6604.00 | 6582.10 | 0.00 | ORB-long ORB[6557.00,6589.35] vol=3.7x ATR=18.47 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 6585.53 | 6589.78 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:45:00 | 6600.00 | 6598.29 | 0.00 | ORB-long ORB[6575.00,6598.90] vol=2.4x ATR=14.19 |
| Stop hit — per-position SL triggered | 2024-07-16 11:25:00 | 6585.81 | 6590.60 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 6551.90 | 6575.93 | 0.00 | ORB-short ORB[6569.70,6612.40] vol=1.6x ATR=21.16 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 6573.06 | 6572.59 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 09:50:00 | 6661.35 | 6686.82 | 0.00 | ORB-short ORB[6676.55,6740.00] vol=1.5x ATR=26.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:25:00 | 6622.25 | 6669.61 | 0.00 | T1 1.5R @ 6622.25 |
| Target hit | 2024-07-24 11:50:00 | 6659.65 | 6651.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 6670.00 | 6646.18 | 0.00 | ORB-long ORB[6575.00,6634.45] vol=7.1x ATR=23.01 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 6646.99 | 6646.78 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 7000.05 | 6956.36 | 0.00 | ORB-long ORB[6909.10,6973.30] vol=2.1x ATR=20.84 |
| Stop hit — per-position SL triggered | 2024-07-31 10:10:00 | 6979.21 | 6958.85 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:45:00 | 6739.50 | 6727.33 | 0.00 | ORB-long ORB[6637.00,6738.05] vol=1.5x ATR=22.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 12:30:00 | 6773.08 | 6735.12 | 0.00 | T1 1.5R @ 6773.08 |
| Target hit | 2024-08-06 15:20:00 | 6775.05 | 6765.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 6348.25 | 6365.88 | 0.00 | ORB-short ORB[6379.90,6456.00] vol=2.1x ATR=34.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:55:00 | 6295.85 | 6347.08 | 0.00 | T1 1.5R @ 6295.85 |
| Target hit | 2024-08-13 13:10:00 | 6317.40 | 6311.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2024-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:10:00 | 6187.50 | 6192.73 | 0.00 | ORB-short ORB[6200.00,6252.50] vol=17.7x ATR=18.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 12:40:00 | 6160.02 | 6192.30 | 0.00 | T1 1.5R @ 6160.02 |
| Stop hit — per-position SL triggered | 2024-08-19 12:55:00 | 6187.50 | 6192.18 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:40:00 | 6249.95 | 6233.51 | 0.00 | ORB-long ORB[6188.45,6215.20] vol=17.3x ATR=16.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:55:00 | 6274.98 | 6242.08 | 0.00 | T1 1.5R @ 6274.98 |
| Stop hit — per-position SL triggered | 2024-08-20 10:55:00 | 6249.95 | 6248.29 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 6390.80 | 6355.20 | 0.00 | ORB-long ORB[6250.05,6322.00] vol=16.1x ATR=18.46 |
| Stop hit — per-position SL triggered | 2024-08-21 11:40:00 | 6372.34 | 6356.08 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 6337.15 | 6327.66 | 0.00 | ORB-long ORB[6285.00,6330.00] vol=15.1x ATR=21.36 |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 6315.79 | 6327.36 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:40:00 | 6356.40 | 6379.01 | 0.00 | ORB-short ORB[6364.80,6460.00] vol=2.0x ATR=22.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 12:45:00 | 6322.36 | 6365.71 | 0.00 | T1 1.5R @ 6322.36 |
| Stop hit — per-position SL triggered | 2024-08-28 13:35:00 | 6356.40 | 6362.92 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:40:00 | 6339.70 | 6327.67 | 0.00 | ORB-long ORB[6289.05,6325.00] vol=16.7x ATR=20.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 09:45:00 | 6370.20 | 6328.52 | 0.00 | T1 1.5R @ 6370.20 |
| Stop hit — per-position SL triggered | 2024-09-02 09:50:00 | 6339.70 | 6328.66 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:50:00 | 6410.45 | 6387.64 | 0.00 | ORB-long ORB[6365.00,6401.80] vol=1.8x ATR=15.90 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 6394.55 | 6395.52 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 6352.10 | 6423.22 | 0.00 | ORB-short ORB[6439.15,6506.45] vol=3.7x ATR=19.82 |
| Stop hit — per-position SL triggered | 2024-09-06 10:25:00 | 6371.92 | 6413.70 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:45:00 | 6245.00 | 6262.28 | 0.00 | ORB-short ORB[6250.00,6343.20] vol=1.5x ATR=18.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 10:35:00 | 6217.71 | 6247.58 | 0.00 | T1 1.5R @ 6217.71 |
| Stop hit — per-position SL triggered | 2024-09-09 12:45:00 | 6245.00 | 6240.45 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:35:00 | 6328.00 | 6378.16 | 0.00 | ORB-short ORB[6413.20,6495.80] vol=2.7x ATR=20.78 |
| Stop hit — per-position SL triggered | 2024-09-19 10:45:00 | 6348.78 | 6365.07 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:15:00 | 6271.40 | 6329.40 | 0.00 | ORB-short ORB[6341.30,6424.95] vol=1.7x ATR=26.98 |
| Stop hit — per-position SL triggered | 2024-09-20 10:25:00 | 6298.38 | 6318.84 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:45:00 | 6746.25 | 6768.25 | 0.00 | ORB-short ORB[6755.05,6793.20] vol=1.5x ATR=16.08 |
| Stop hit — per-position SL triggered | 2024-10-11 09:50:00 | 6762.33 | 6767.78 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:55:00 | 6624.90 | 6676.95 | 0.00 | ORB-short ORB[6700.00,6764.50] vol=2.2x ATR=15.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:50:00 | 6600.91 | 6657.66 | 0.00 | T1 1.5R @ 6600.91 |
| Stop hit — per-position SL triggered | 2024-10-14 13:50:00 | 6624.90 | 6639.38 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:40:00 | 6632.45 | 6599.48 | 0.00 | ORB-long ORB[6555.00,6614.70] vol=4.0x ATR=20.18 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 6612.27 | 6605.75 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:30:00 | 6524.40 | 6539.62 | 0.00 | ORB-short ORB[6566.00,6612.90] vol=1.6x ATR=18.24 |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 6542.64 | 6535.46 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 6520.00 | 6555.25 | 0.00 | ORB-short ORB[6540.30,6607.35] vol=2.3x ATR=20.53 |
| Stop hit — per-position SL triggered | 2024-10-21 09:45:00 | 6540.53 | 6542.93 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:30:00 | 6301.90 | 6364.33 | 0.00 | ORB-short ORB[6365.00,6430.45] vol=1.8x ATR=38.12 |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 6340.02 | 6319.44 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 11:15:00 | 6499.95 | 6394.98 | 0.00 | ORB-long ORB[6302.00,6398.95] vol=1.6x ATR=23.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 11:20:00 | 6534.68 | 6406.43 | 0.00 | T1 1.5R @ 6534.68 |
| Stop hit — per-position SL triggered | 2024-10-28 11:35:00 | 6499.95 | 6434.04 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:40:00 | 6453.20 | 6461.94 | 0.00 | ORB-short ORB[6460.25,6517.95] vol=7.0x ATR=21.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:25:00 | 6421.40 | 6451.61 | 0.00 | T1 1.5R @ 6421.40 |
| Stop hit — per-position SL triggered | 2024-10-29 10:45:00 | 6453.20 | 6449.38 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 6205.00 | 6233.45 | 0.00 | ORB-short ORB[6220.65,6282.00] vol=2.6x ATR=22.98 |
| Stop hit — per-position SL triggered | 2024-11-13 09:35:00 | 6227.98 | 6230.54 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:40:00 | 5775.95 | 5789.93 | 0.00 | ORB-short ORB[5780.00,5851.95] vol=3.0x ATR=27.80 |
| Stop hit — per-position SL triggered | 2024-11-22 10:20:00 | 5803.75 | 5787.53 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:55:00 | 5847.60 | 5813.54 | 0.00 | ORB-long ORB[5751.15,5837.40] vol=1.9x ATR=26.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 10:15:00 | 5887.69 | 5823.31 | 0.00 | T1 1.5R @ 5887.69 |
| Stop hit — per-position SL triggered | 2024-11-25 10:30:00 | 5847.60 | 5828.32 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 5869.00 | 5840.69 | 0.00 | ORB-long ORB[5795.00,5858.00] vol=2.7x ATR=17.59 |
| Stop hit — per-position SL triggered | 2024-11-27 10:10:00 | 5851.41 | 5850.28 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:10:00 | 5760.00 | 5705.70 | 0.00 | ORB-long ORB[5660.10,5737.35] vol=3.3x ATR=13.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:25:00 | 5779.54 | 5725.34 | 0.00 | T1 1.5R @ 5779.54 |
| Target hit | 2024-12-02 15:20:00 | 5904.70 | 5798.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-12-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:50:00 | 6031.20 | 6092.35 | 0.00 | ORB-short ORB[6077.60,6130.35] vol=2.0x ATR=16.61 |
| Stop hit — per-position SL triggered | 2024-12-06 10:55:00 | 6047.81 | 6088.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:25:00 | 6090.00 | 6068.84 | 0.00 | ORB-long ORB[6037.95,6065.75] vol=4.7x ATR=13.46 |
| Stop hit — per-position SL triggered | 2024-12-09 12:35:00 | 6076.54 | 6084.99 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 6270.00 | 6275.51 | 0.00 | ORB-short ORB[6278.65,6330.20] vol=3.8x ATR=13.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:40:00 | 6249.71 | 6260.21 | 0.00 | T1 1.5R @ 6249.71 |
| Target hit | 2024-12-12 14:30:00 | 6260.05 | 6254.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — SELL (started 2024-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:45:00 | 6174.25 | 6190.80 | 0.00 | ORB-short ORB[6177.55,6231.00] vol=3.3x ATR=15.67 |
| Stop hit — per-position SL triggered | 2024-12-13 10:00:00 | 6189.92 | 6190.52 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:10:00 | 5857.90 | 5928.39 | 0.00 | ORB-short ORB[5970.15,6034.70] vol=2.0x ATR=29.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 5813.45 | 5875.58 | 0.00 | T1 1.5R @ 5813.45 |
| Target hit | 2024-12-18 15:20:00 | 5675.35 | 5812.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 5762.55 | 5776.87 | 0.00 | ORB-short ORB[5770.05,5820.00] vol=1.8x ATR=10.33 |
| Stop hit — per-position SL triggered | 2024-12-27 11:35:00 | 5772.88 | 5776.56 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 10:55:00 | 5447.05 | 5461.65 | 0.00 | ORB-short ORB[5450.00,5519.95] vol=1.7x ATR=24.92 |
| Stop hit — per-position SL triggered | 2024-12-31 11:00:00 | 5471.97 | 5461.32 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:30:00 | 5519.00 | 5546.77 | 0.00 | ORB-short ORB[5520.85,5594.65] vol=1.6x ATR=24.49 |
| Stop hit — per-position SL triggered | 2025-01-01 10:45:00 | 5543.49 | 5542.12 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 5504.00 | 5522.72 | 0.00 | ORB-short ORB[5517.60,5552.10] vol=4.1x ATR=10.96 |
| Stop hit — per-position SL triggered | 2025-01-02 11:25:00 | 5514.96 | 5512.80 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 5497.50 | 5563.32 | 0.00 | ORB-short ORB[5580.10,5648.15] vol=5.9x ATR=14.08 |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 5511.58 | 5554.18 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 5620.00 | 5550.14 | 0.00 | ORB-long ORB[5480.00,5555.75] vol=1.7x ATR=19.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 09:35:00 | 5649.97 | 5589.98 | 0.00 | T1 1.5R @ 5649.97 |
| Target hit | 2025-01-09 11:55:00 | 5685.00 | 5700.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:35:00 | 5200.00 | 5185.88 | 0.00 | ORB-long ORB[5147.90,5190.00] vol=1.9x ATR=11.17 |
| Stop hit — per-position SL triggered | 2025-01-17 10:45:00 | 5188.83 | 5185.82 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:00:00 | 5006.15 | 5059.42 | 0.00 | ORB-short ORB[5031.05,5097.40] vol=2.0x ATR=15.74 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 5021.89 | 5048.90 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:55:00 | 5037.65 | 5043.69 | 0.00 | ORB-short ORB[5083.40,5135.00] vol=2.3x ATR=14.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:40:00 | 5015.46 | 5033.52 | 0.00 | T1 1.5R @ 5015.46 |
| Target hit | 2025-01-24 15:20:00 | 4964.55 | 5002.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-01-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:35:00 | 5090.95 | 5064.69 | 0.00 | ORB-long ORB[5008.30,5049.90] vol=1.7x ATR=14.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:10:00 | 5113.26 | 5078.07 | 0.00 | T1 1.5R @ 5113.26 |
| Stop hit — per-position SL triggered | 2025-01-30 13:55:00 | 5090.95 | 5091.20 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:15:00 | 5078.40 | 5089.64 | 0.00 | ORB-short ORB[5091.60,5147.95] vol=2.0x ATR=13.38 |
| Stop hit — per-position SL triggered | 2025-01-31 10:25:00 | 5091.78 | 5089.34 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:35:00 | 5155.75 | 5151.14 | 0.00 | ORB-long ORB[5121.55,5153.75] vol=2.7x ATR=15.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 09:45:00 | 5178.86 | 5156.91 | 0.00 | T1 1.5R @ 5178.86 |
| Target hit | 2025-02-01 10:00:00 | 5160.15 | 5161.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 10:15:00 | 5175.00 | 5152.51 | 0.00 | ORB-long ORB[5101.00,5173.90] vol=1.5x ATR=19.76 |
| Stop hit — per-position SL triggered | 2025-02-04 11:40:00 | 5155.24 | 5164.86 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 4890.05 | 4901.98 | 0.00 | ORB-short ORB[4909.35,4951.05] vol=6.1x ATR=17.05 |
| Stop hit — per-position SL triggered | 2025-02-10 09:35:00 | 4907.10 | 4901.91 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 11:05:00 | 4625.15 | 4563.29 | 0.00 | ORB-long ORB[4542.50,4585.60] vol=3.5x ATR=23.41 |
| Stop hit — per-position SL triggered | 2025-02-18 11:10:00 | 4601.74 | 4565.24 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:10:00 | 4731.65 | 4762.59 | 0.00 | ORB-short ORB[4743.10,4791.40] vol=2.8x ATR=15.29 |
| Stop hit — per-position SL triggered | 2025-02-21 10:30:00 | 4746.94 | 4758.48 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:25:00 | 4750.40 | 4729.69 | 0.00 | ORB-long ORB[4690.00,4749.40] vol=2.5x ATR=15.81 |
| Stop hit — per-position SL triggered | 2025-02-25 11:00:00 | 4734.59 | 4731.56 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 09:50:00 | 4739.50 | 4679.16 | 0.00 | ORB-long ORB[4600.40,4669.20] vol=1.6x ATR=29.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 10:20:00 | 4783.96 | 4693.60 | 0.00 | T1 1.5R @ 4783.96 |
| Target hit | 2025-03-04 15:20:00 | 4840.65 | 4809.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2025-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:50:00 | 4811.30 | 4818.75 | 0.00 | ORB-short ORB[4832.20,4878.85] vol=3.5x ATR=17.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:00:00 | 4784.81 | 4816.02 | 0.00 | T1 1.5R @ 4784.81 |
| Target hit | 2025-03-12 12:10:00 | 4738.65 | 4737.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2025-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:25:00 | 4784.00 | 4742.38 | 0.00 | ORB-long ORB[4722.20,4760.00] vol=1.8x ATR=20.56 |
| Stop hit — per-position SL triggered | 2025-03-13 10:40:00 | 4763.44 | 4746.86 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:35:00 | 4984.00 | 4931.80 | 0.00 | ORB-long ORB[4883.50,4942.25] vol=1.9x ATR=20.36 |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 4963.64 | 4960.26 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:50:00 | 4995.40 | 4943.79 | 0.00 | ORB-long ORB[4900.55,4961.95] vol=1.7x ATR=24.14 |
| Stop hit — per-position SL triggered | 2025-03-24 11:25:00 | 4971.26 | 4952.57 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:30:00 | 4879.80 | 4885.25 | 0.00 | ORB-short ORB[4889.80,4932.90] vol=12.6x ATR=11.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:45:00 | 4862.15 | 4883.75 | 0.00 | T1 1.5R @ 4862.15 |
| Stop hit — per-position SL triggered | 2025-04-23 10:50:00 | 4879.80 | 4883.72 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:20:00 | 4731.50 | 4701.98 | 0.00 | ORB-long ORB[4657.80,4703.40] vol=1.5x ATR=17.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:35:00 | 4758.07 | 4718.75 | 0.00 | T1 1.5R @ 4758.07 |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 4731.50 | 4730.87 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 09:35:00 | 4610.80 | 4640.62 | 0.00 | ORB-short ORB[4630.00,4690.00] vol=2.4x ATR=16.51 |
| Stop hit — per-position SL triggered | 2025-05-07 09:40:00 | 4627.31 | 4639.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:15:00 | 5175.05 | 2024-05-13 11:25:00 | 5195.13 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-17 10:30:00 | 5478.80 | 2024-05-17 10:45:00 | 5514.59 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-05-17 10:30:00 | 5478.80 | 2024-05-17 10:50:00 | 5501.05 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-23 11:05:00 | 5583.60 | 2024-05-23 11:20:00 | 5597.76 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-31 10:55:00 | 4959.70 | 2024-05-31 11:10:00 | 4975.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-05 10:20:00 | 5161.75 | 2024-06-05 10:55:00 | 5199.25 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-06-05 10:20:00 | 5161.75 | 2024-06-05 12:25:00 | 5210.00 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-06-06 10:05:00 | 5377.90 | 2024-06-06 10:30:00 | 5406.37 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-06 10:05:00 | 5377.90 | 2024-06-06 15:20:00 | 5538.55 | TARGET_HIT | 0.50 | 2.99% |
| BUY | retest1 | 2024-06-07 10:15:00 | 5585.00 | 2024-06-07 10:25:00 | 5557.61 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-10 11:10:00 | 5810.00 | 2024-06-10 11:15:00 | 5851.05 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-06-10 11:10:00 | 5810.00 | 2024-06-10 15:20:00 | 5902.00 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2024-06-11 09:50:00 | 5999.00 | 2024-06-11 11:05:00 | 6046.08 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-06-11 09:50:00 | 5999.00 | 2024-06-11 15:20:00 | 6044.35 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-06-13 10:05:00 | 6123.15 | 2024-06-13 10:20:00 | 6104.19 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-14 09:30:00 | 6050.00 | 2024-06-14 09:45:00 | 6017.84 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-14 09:30:00 | 6050.00 | 2024-06-14 10:00:00 | 6048.25 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2024-06-27 09:35:00 | 6610.00 | 2024-06-27 09:40:00 | 6651.59 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-27 09:35:00 | 6610.00 | 2024-06-27 09:45:00 | 6667.40 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-07-02 10:40:00 | 6591.55 | 2024-07-02 11:40:00 | 6549.90 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-02 10:40:00 | 6591.55 | 2024-07-02 13:05:00 | 6591.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-03 11:05:00 | 6594.00 | 2024-07-03 11:10:00 | 6556.38 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-07-03 11:05:00 | 6594.00 | 2024-07-03 15:20:00 | 6445.40 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2024-07-05 10:35:00 | 6521.95 | 2024-07-05 11:30:00 | 6537.11 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-08 10:45:00 | 6660.00 | 2024-07-08 15:15:00 | 6641.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-09 10:10:00 | 6700.00 | 2024-07-09 10:15:00 | 6683.13 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 09:30:00 | 6562.45 | 2024-07-10 09:45:00 | 6533.37 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-10 09:30:00 | 6562.45 | 2024-07-10 09:50:00 | 6562.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:40:00 | 6604.00 | 2024-07-11 10:15:00 | 6585.53 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-16 10:45:00 | 6600.00 | 2024-07-16 11:25:00 | 6585.81 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-18 09:30:00 | 6551.90 | 2024-07-18 09:40:00 | 6573.06 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-24 09:50:00 | 6661.35 | 2024-07-24 10:25:00 | 6622.25 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-07-24 09:50:00 | 6661.35 | 2024-07-24 11:50:00 | 6659.65 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2024-07-26 09:35:00 | 6670.00 | 2024-07-26 09:40:00 | 6646.99 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-31 10:00:00 | 7000.05 | 2024-07-31 10:10:00 | 6979.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-06 10:45:00 | 6739.50 | 2024-08-06 12:30:00 | 6773.08 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-06 10:45:00 | 6739.50 | 2024-08-06 15:20:00 | 6775.05 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-13 09:35:00 | 6348.25 | 2024-08-13 09:55:00 | 6295.85 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-08-13 09:35:00 | 6348.25 | 2024-08-13 13:10:00 | 6317.40 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-08-19 11:10:00 | 6187.50 | 2024-08-19 12:40:00 | 6160.02 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-19 11:10:00 | 6187.50 | 2024-08-19 12:55:00 | 6187.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 09:40:00 | 6249.95 | 2024-08-20 09:55:00 | 6274.98 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-20 09:40:00 | 6249.95 | 2024-08-20 10:55:00 | 6249.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 11:10:00 | 6390.80 | 2024-08-21 11:40:00 | 6372.34 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-26 09:35:00 | 6337.15 | 2024-08-26 10:15:00 | 6315.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-28 10:40:00 | 6356.40 | 2024-08-28 12:45:00 | 6322.36 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-28 10:40:00 | 6356.40 | 2024-08-28 13:35:00 | 6356.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 09:40:00 | 6339.70 | 2024-09-02 09:45:00 | 6370.20 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-02 09:40:00 | 6339.70 | 2024-09-02 09:50:00 | 6339.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:50:00 | 6410.45 | 2024-09-04 10:15:00 | 6394.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-06 10:05:00 | 6352.10 | 2024-09-06 10:25:00 | 6371.92 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-09 09:45:00 | 6245.00 | 2024-09-09 10:35:00 | 6217.71 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-09 09:45:00 | 6245.00 | 2024-09-09 12:45:00 | 6245.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:35:00 | 6328.00 | 2024-09-19 10:45:00 | 6348.78 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-20 10:15:00 | 6271.40 | 2024-09-20 10:25:00 | 6298.38 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-11 09:45:00 | 6746.25 | 2024-10-11 09:50:00 | 6762.33 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-14 10:55:00 | 6624.90 | 2024-10-14 11:50:00 | 6600.91 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-14 10:55:00 | 6624.90 | 2024-10-14 13:50:00 | 6624.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 09:40:00 | 6632.45 | 2024-10-15 09:50:00 | 6612.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-16 10:30:00 | 6524.40 | 2024-10-16 12:15:00 | 6542.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-21 09:30:00 | 6520.00 | 2024-10-21 09:45:00 | 6540.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-23 09:30:00 | 6301.90 | 2024-10-23 11:15:00 | 6340.02 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-10-28 11:15:00 | 6499.95 | 2024-10-28 11:20:00 | 6534.68 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-28 11:15:00 | 6499.95 | 2024-10-28 11:35:00 | 6499.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:40:00 | 6453.20 | 2024-10-29 10:25:00 | 6421.40 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-29 09:40:00 | 6453.20 | 2024-10-29 10:45:00 | 6453.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 6205.00 | 2024-11-13 09:35:00 | 6227.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-22 09:40:00 | 5775.95 | 2024-11-22 10:20:00 | 5803.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-11-25 09:55:00 | 5847.60 | 2024-11-25 10:15:00 | 5887.69 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-11-25 09:55:00 | 5847.60 | 2024-11-25 10:30:00 | 5847.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:40:00 | 5869.00 | 2024-11-27 10:10:00 | 5851.41 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-02 11:10:00 | 5760.00 | 2024-12-02 11:25:00 | 5779.54 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-02 11:10:00 | 5760.00 | 2024-12-02 15:20:00 | 5904.70 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2024-12-06 10:50:00 | 6031.20 | 2024-12-06 10:55:00 | 6047.81 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-09 10:25:00 | 6090.00 | 2024-12-09 12:35:00 | 6076.54 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-12 11:00:00 | 6270.00 | 2024-12-12 11:40:00 | 6249.71 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-12 11:00:00 | 6270.00 | 2024-12-12 14:30:00 | 6260.05 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2024-12-13 09:45:00 | 6174.25 | 2024-12-13 10:00:00 | 6189.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-18 10:10:00 | 5857.90 | 2024-12-18 12:15:00 | 5813.45 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-12-18 10:10:00 | 5857.90 | 2024-12-18 15:20:00 | 5675.35 | TARGET_HIT | 0.50 | 3.12% |
| SELL | retest1 | 2024-12-27 11:05:00 | 5762.55 | 2024-12-27 11:35:00 | 5772.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-31 10:55:00 | 5447.05 | 2024-12-31 11:00:00 | 5471.97 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-01 10:30:00 | 5519.00 | 2025-01-01 10:45:00 | 5543.49 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-02 10:55:00 | 5504.00 | 2025-01-02 11:25:00 | 5514.96 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-06 11:10:00 | 5497.50 | 2025-01-06 11:15:00 | 5511.58 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-09 09:30:00 | 5620.00 | 2025-01-09 09:35:00 | 5649.97 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-09 09:30:00 | 5620.00 | 2025-01-09 11:55:00 | 5685.00 | TARGET_HIT | 0.50 | 1.16% |
| BUY | retest1 | 2025-01-17 10:35:00 | 5200.00 | 2025-01-17 10:45:00 | 5188.83 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-21 11:00:00 | 5006.15 | 2025-01-21 11:40:00 | 5021.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-24 10:55:00 | 5037.65 | 2025-01-24 12:40:00 | 5015.46 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-24 10:55:00 | 5037.65 | 2025-01-24 15:20:00 | 4964.55 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-01-30 10:35:00 | 5090.95 | 2025-01-30 11:10:00 | 5113.26 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-30 10:35:00 | 5090.95 | 2025-01-30 13:55:00 | 5090.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-31 10:15:00 | 5078.40 | 2025-01-31 10:25:00 | 5091.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-01 09:35:00 | 5155.75 | 2025-02-01 09:45:00 | 5178.86 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-02-01 09:35:00 | 5155.75 | 2025-02-01 10:00:00 | 5160.15 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-02-04 10:15:00 | 5175.00 | 2025-02-04 11:40:00 | 5155.24 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-10 09:30:00 | 4890.05 | 2025-02-10 09:35:00 | 4907.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-18 11:05:00 | 4625.15 | 2025-02-18 11:10:00 | 4601.74 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-02-21 10:10:00 | 4731.65 | 2025-02-21 10:30:00 | 4746.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-25 10:25:00 | 4750.40 | 2025-02-25 11:00:00 | 4734.59 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-04 09:50:00 | 4739.50 | 2025-03-04 10:20:00 | 4783.96 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2025-03-04 09:50:00 | 4739.50 | 2025-03-04 15:20:00 | 4840.65 | TARGET_HIT | 0.50 | 2.13% |
| SELL | retest1 | 2025-03-12 09:50:00 | 4811.30 | 2025-03-12 10:00:00 | 4784.81 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-03-12 09:50:00 | 4811.30 | 2025-03-12 12:10:00 | 4738.65 | TARGET_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2025-03-13 10:25:00 | 4784.00 | 2025-03-13 10:40:00 | 4763.44 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-19 10:35:00 | 4984.00 | 2025-03-19 11:15:00 | 4963.64 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-24 10:50:00 | 4995.40 | 2025-03-24 11:25:00 | 4971.26 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-04-23 10:30:00 | 4879.80 | 2025-04-23 10:45:00 | 4862.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-04-23 10:30:00 | 4879.80 | 2025-04-23 10:50:00 | 4879.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-06 10:20:00 | 4731.50 | 2025-05-06 10:35:00 | 4758.07 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-06 10:20:00 | 4731.50 | 2025-05-06 11:15:00 | 4731.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-07 09:35:00 | 4610.80 | 2025-05-07 09:40:00 | 4627.31 | STOP_HIT | 1.00 | -0.36% |
