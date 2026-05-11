# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 13 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 64
- **Target hits / Stop hits / Partials:** 13 / 64 / 38
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 16.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 30 | 41.7% | 6 | 42 | 24 | 0.13% | 9.7% |
| BUY @ 2nd Alert (retest1) | 72 | 30 | 41.7% | 6 | 42 | 24 | 0.13% | 9.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 21 | 48.8% | 7 | 22 | 14 | 0.16% | 6.9% |
| SELL @ 2nd Alert (retest1) | 43 | 21 | 48.8% | 7 | 22 | 14 | 0.16% | 6.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 115 | 51 | 44.3% | 13 | 64 | 38 | 0.14% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 5898.00 | 5892.43 | 0.00 | ORB-long ORB[5850.00,5893.05] vol=1.7x ATR=10.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 12:25:00 | 5914.01 | 5897.33 | 0.00 | T1 1.5R @ 5914.01 |
| Stop hit — per-position SL triggered | 2024-05-16 12:30:00 | 5898.00 | 5897.36 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:15:00 | 5875.00 | 5899.65 | 0.00 | ORB-short ORB[5903.15,5958.60] vol=3.1x ATR=14.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:55:00 | 5853.75 | 5885.10 | 0.00 | T1 1.5R @ 5853.75 |
| Target hit | 2024-05-22 12:45:00 | 5869.35 | 5868.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-05-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:00:00 | 5903.00 | 5913.71 | 0.00 | ORB-short ORB[5908.05,5945.60] vol=1.6x ATR=15.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:05:00 | 5879.86 | 5907.75 | 0.00 | T1 1.5R @ 5879.86 |
| Target hit | 2024-05-24 11:50:00 | 5898.95 | 5891.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:15:00 | 5900.30 | 5929.23 | 0.00 | ORB-short ORB[5922.00,5971.45] vol=2.0x ATR=10.90 |
| Stop hit — per-position SL triggered | 2024-05-28 11:25:00 | 5911.20 | 5927.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:15:00 | 5815.00 | 5840.00 | 0.00 | ORB-short ORB[5854.90,5901.75] vol=1.9x ATR=13.57 |
| Stop hit — per-position SL triggered | 2024-05-30 11:45:00 | 5828.57 | 5837.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:05:00 | 6069.80 | 6026.91 | 0.00 | ORB-long ORB[5992.05,6049.00] vol=3.5x ATR=13.88 |
| Stop hit — per-position SL triggered | 2024-06-10 11:35:00 | 6055.92 | 6042.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 11:15:00 | 6163.15 | 6188.41 | 0.00 | ORB-short ORB[6218.05,6269.00] vol=1.6x ATR=15.74 |
| Stop hit — per-position SL triggered | 2024-06-19 12:10:00 | 6178.89 | 6181.63 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 6222.55 | 6203.38 | 0.00 | ORB-long ORB[6145.00,6210.00] vol=2.6x ATR=14.12 |
| Stop hit — per-position SL triggered | 2024-06-21 10:00:00 | 6208.43 | 6209.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:30:00 | 6210.10 | 6198.70 | 0.00 | ORB-long ORB[6156.15,6205.95] vol=1.5x ATR=12.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 10:40:00 | 6228.22 | 6203.24 | 0.00 | T1 1.5R @ 6228.22 |
| Stop hit — per-position SL triggered | 2024-06-24 11:20:00 | 6210.10 | 6221.12 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:45:00 | 6108.05 | 6126.22 | 0.00 | ORB-short ORB[6121.00,6163.75] vol=1.5x ATR=11.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:55:00 | 6090.89 | 6111.03 | 0.00 | T1 1.5R @ 6090.89 |
| Stop hit — per-position SL triggered | 2024-07-02 11:10:00 | 6108.05 | 6108.53 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:50:00 | 6240.00 | 6195.41 | 0.00 | ORB-long ORB[6161.00,6210.00] vol=2.3x ATR=12.94 |
| Stop hit — per-position SL triggered | 2024-07-04 10:55:00 | 6227.06 | 6197.37 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 6262.20 | 6231.94 | 0.00 | ORB-long ORB[6159.00,6245.00] vol=2.2x ATR=18.20 |
| Stop hit — per-position SL triggered | 2024-07-05 09:45:00 | 6244.00 | 6240.24 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:55:00 | 6314.00 | 6340.15 | 0.00 | ORB-short ORB[6332.90,6387.95] vol=5.3x ATR=15.14 |
| Stop hit — per-position SL triggered | 2024-07-08 15:20:00 | 6326.45 | 6320.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 6368.95 | 6350.62 | 0.00 | ORB-long ORB[6320.30,6365.40] vol=1.8x ATR=11.93 |
| Stop hit — per-position SL triggered | 2024-07-10 10:00:00 | 6357.02 | 6352.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:50:00 | 6404.45 | 6452.32 | 0.00 | ORB-short ORB[6472.90,6525.25] vol=2.0x ATR=13.24 |
| Stop hit — per-position SL triggered | 2024-07-19 11:00:00 | 6417.69 | 6448.82 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 6402.30 | 6429.39 | 0.00 | ORB-short ORB[6403.25,6455.00] vol=1.7x ATR=15.82 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 6418.12 | 6429.04 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 11:10:00 | 6401.00 | 6385.35 | 0.00 | ORB-long ORB[6355.05,6400.00] vol=5.9x ATR=12.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 11:15:00 | 6419.66 | 6387.06 | 0.00 | T1 1.5R @ 6419.66 |
| Stop hit — per-position SL triggered | 2024-07-24 11:25:00 | 6401.00 | 6387.77 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:25:00 | 6473.95 | 6456.79 | 0.00 | ORB-long ORB[6393.55,6472.25] vol=1.6x ATR=15.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:30:00 | 6497.25 | 6459.35 | 0.00 | T1 1.5R @ 6497.25 |
| Target hit | 2024-07-26 15:20:00 | 6663.20 | 6589.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 6705.90 | 6649.41 | 0.00 | ORB-long ORB[6606.60,6655.40] vol=1.6x ATR=15.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:20:00 | 6729.29 | 6659.83 | 0.00 | T1 1.5R @ 6729.29 |
| Stop hit — per-position SL triggered | 2024-08-01 11:45:00 | 6705.90 | 6669.42 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:10:00 | 6661.25 | 6692.85 | 0.00 | ORB-short ORB[6667.80,6744.35] vol=1.6x ATR=17.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:25:00 | 6635.36 | 6686.98 | 0.00 | T1 1.5R @ 6635.36 |
| Target hit | 2024-08-08 15:20:00 | 6528.90 | 6615.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:05:00 | 6530.95 | 6557.71 | 0.00 | ORB-short ORB[6535.00,6582.25] vol=2.4x ATR=12.57 |
| Stop hit — per-position SL triggered | 2024-08-09 11:10:00 | 6543.52 | 6557.27 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:25:00 | 6484.85 | 6531.14 | 0.00 | ORB-short ORB[6540.00,6591.90] vol=1.7x ATR=15.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:40:00 | 6461.07 | 6518.78 | 0.00 | T1 1.5R @ 6461.07 |
| Stop hit — per-position SL triggered | 2024-08-12 11:25:00 | 6484.85 | 6484.14 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:00:00 | 6729.15 | 6649.30 | 0.00 | ORB-long ORB[6589.40,6675.00] vol=3.8x ATR=31.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:15:00 | 6776.35 | 6670.26 | 0.00 | T1 1.5R @ 6776.35 |
| Stop hit — per-position SL triggered | 2024-08-14 11:50:00 | 6729.15 | 6735.51 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:35:00 | 6717.05 | 6683.45 | 0.00 | ORB-long ORB[6627.10,6698.60] vol=1.8x ATR=16.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 12:25:00 | 6741.93 | 6696.47 | 0.00 | T1 1.5R @ 6741.93 |
| Target hit | 2024-08-21 15:20:00 | 6765.30 | 6727.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-08-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:10:00 | 6922.90 | 6887.50 | 0.00 | ORB-long ORB[6854.05,6889.00] vol=1.6x ATR=19.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:25:00 | 6952.19 | 6904.53 | 0.00 | T1 1.5R @ 6952.19 |
| Stop hit — per-position SL triggered | 2024-08-30 10:50:00 | 6922.90 | 6911.62 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 6918.05 | 6887.19 | 0.00 | ORB-long ORB[6854.05,6906.70] vol=1.9x ATR=15.61 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 6902.44 | 6893.47 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:55:00 | 7019.10 | 6982.33 | 0.00 | ORB-long ORB[6900.20,6985.00] vol=1.9x ATR=15.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:05:00 | 7042.62 | 6987.57 | 0.00 | T1 1.5R @ 7042.62 |
| Stop hit — per-position SL triggered | 2024-09-05 11:10:00 | 7019.10 | 6989.49 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:50:00 | 6900.00 | 6911.64 | 0.00 | ORB-short ORB[6920.70,6962.45] vol=1.5x ATR=16.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:10:00 | 6875.18 | 6909.76 | 0.00 | T1 1.5R @ 6875.18 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 6900.00 | 6905.68 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 11:05:00 | 6879.95 | 6898.91 | 0.00 | ORB-short ORB[6880.05,6928.00] vol=2.4x ATR=14.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 11:55:00 | 6858.13 | 6891.22 | 0.00 | T1 1.5R @ 6858.13 |
| Target hit | 2024-09-09 15:15:00 | 6866.00 | 6864.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:45:00 | 6950.50 | 6932.01 | 0.00 | ORB-long ORB[6900.00,6934.75] vol=1.8x ATR=14.33 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 6936.17 | 6933.17 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 6987.75 | 6955.69 | 0.00 | ORB-long ORB[6915.00,6972.95] vol=2.4x ATR=20.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:40:00 | 7018.81 | 6968.76 | 0.00 | T1 1.5R @ 7018.81 |
| Stop hit — per-position SL triggered | 2024-09-12 09:45:00 | 6987.75 | 6971.69 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:55:00 | 7056.85 | 7048.31 | 0.00 | ORB-long ORB[7015.00,7047.55] vol=3.6x ATR=14.16 |
| Stop hit — per-position SL triggered | 2024-09-18 10:00:00 | 7042.69 | 7048.36 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:30:00 | 7093.60 | 7066.29 | 0.00 | ORB-long ORB[7038.20,7075.90] vol=1.8x ATR=17.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:55:00 | 7120.50 | 7075.88 | 0.00 | T1 1.5R @ 7120.50 |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 7093.60 | 7080.96 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:05:00 | 7184.00 | 7158.19 | 0.00 | ORB-long ORB[7076.20,7151.20] vol=2.4x ATR=13.14 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 7170.86 | 7159.00 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 7171.95 | 7136.07 | 0.00 | ORB-long ORB[7117.60,7160.00] vol=2.4x ATR=16.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:55:00 | 7196.18 | 7143.00 | 0.00 | T1 1.5R @ 7196.18 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 7171.95 | 7156.06 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:55:00 | 6798.30 | 6822.21 | 0.00 | ORB-short ORB[6801.00,6857.50] vol=2.1x ATR=18.66 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 6816.96 | 6816.29 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 6895.00 | 6812.58 | 0.00 | ORB-long ORB[6748.70,6825.95] vol=2.7x ATR=22.00 |
| Stop hit — per-position SL triggered | 2024-10-08 11:35:00 | 6873.00 | 6819.57 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 7006.90 | 6985.30 | 0.00 | ORB-long ORB[6961.75,7000.00] vol=3.6x ATR=14.11 |
| Stop hit — per-position SL triggered | 2024-10-14 09:35:00 | 6992.79 | 6984.81 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:50:00 | 7029.75 | 7001.56 | 0.00 | ORB-long ORB[6950.00,7002.65] vol=2.7x ATR=20.08 |
| Stop hit — per-position SL triggered | 2024-10-18 09:55:00 | 7009.67 | 7002.06 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 11:05:00 | 7019.85 | 6992.77 | 0.00 | ORB-long ORB[6972.25,7015.85] vol=2.6x ATR=14.88 |
| Stop hit — per-position SL triggered | 2024-10-21 12:05:00 | 7004.97 | 6999.76 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:45:00 | 6926.00 | 6973.93 | 0.00 | ORB-short ORB[6992.85,7037.25] vol=4.1x ATR=15.72 |
| Stop hit — per-position SL triggered | 2024-10-22 10:50:00 | 6941.72 | 6972.74 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:45:00 | 6943.15 | 6908.63 | 0.00 | ORB-long ORB[6870.65,6937.25] vol=2.1x ATR=17.52 |
| Stop hit — per-position SL triggered | 2024-10-24 11:00:00 | 6925.63 | 6912.39 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:30:00 | 6830.00 | 6898.93 | 0.00 | ORB-short ORB[6885.80,6944.80] vol=1.9x ATR=21.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:35:00 | 6797.68 | 6870.84 | 0.00 | T1 1.5R @ 6797.68 |
| Stop hit — per-position SL triggered | 2024-10-28 09:40:00 | 6830.00 | 6867.24 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:15:00 | 6885.00 | 6919.33 | 0.00 | ORB-short ORB[6924.35,6966.75] vol=3.6x ATR=16.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:50:00 | 6859.66 | 6907.77 | 0.00 | T1 1.5R @ 6859.66 |
| Stop hit — per-position SL triggered | 2024-10-29 12:00:00 | 6885.00 | 6905.97 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:20:00 | 6991.15 | 6968.62 | 0.00 | ORB-long ORB[6946.65,6987.45] vol=1.9x ATR=19.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:25:00 | 7019.72 | 6975.12 | 0.00 | T1 1.5R @ 7019.72 |
| Stop hit — per-position SL triggered | 2024-10-30 10:40:00 | 6991.15 | 6981.00 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:00:00 | 6908.05 | 6952.11 | 0.00 | ORB-short ORB[6930.00,6998.20] vol=4.0x ATR=28.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 11:30:00 | 6864.58 | 6929.16 | 0.00 | T1 1.5R @ 6864.58 |
| Stop hit — per-position SL triggered | 2024-11-13 12:30:00 | 6908.05 | 6918.22 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:45:00 | 6905.45 | 6868.27 | 0.00 | ORB-long ORB[6827.05,6891.85] vol=1.6x ATR=18.54 |
| Stop hit — per-position SL triggered | 2024-11-14 10:50:00 | 6886.91 | 6870.09 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:55:00 | 6909.00 | 6848.30 | 0.00 | ORB-long ORB[6751.60,6845.00] vol=1.6x ATR=21.78 |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 6887.22 | 6863.66 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 6933.55 | 6992.71 | 0.00 | ORB-short ORB[6970.05,7030.00] vol=1.5x ATR=16.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 6908.92 | 6986.13 | 0.00 | T1 1.5R @ 6908.92 |
| Target hit | 2024-11-28 15:20:00 | 6850.05 | 6897.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 11:15:00 | 7163.45 | 7248.54 | 0.00 | ORB-short ORB[7244.25,7334.95] vol=2.3x ATR=19.87 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 7183.32 | 7237.02 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:45:00 | 7296.00 | 7257.48 | 0.00 | ORB-long ORB[7234.30,7281.65] vol=2.7x ATR=19.37 |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 7276.63 | 7262.39 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:45:00 | 7327.60 | 7282.67 | 0.00 | ORB-long ORB[7216.05,7273.95] vol=1.7x ATR=15.90 |
| Stop hit — per-position SL triggered | 2024-12-11 11:00:00 | 7311.70 | 7286.87 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:15:00 | 7202.65 | 7243.83 | 0.00 | ORB-short ORB[7241.00,7318.65] vol=2.1x ATR=13.06 |
| Stop hit — per-position SL triggered | 2024-12-12 11:35:00 | 7215.71 | 7238.99 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 7279.90 | 7238.19 | 0.00 | ORB-long ORB[7195.10,7273.80] vol=1.8x ATR=19.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:25:00 | 7308.45 | 7242.92 | 0.00 | T1 1.5R @ 7308.45 |
| Stop hit — per-position SL triggered | 2024-12-23 12:30:00 | 7279.90 | 7257.59 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:15:00 | 7319.50 | 7284.56 | 0.00 | ORB-long ORB[7220.05,7311.60] vol=1.6x ATR=19.72 |
| Stop hit — per-position SL triggered | 2024-12-30 13:25:00 | 7299.78 | 7301.52 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 7405.00 | 7374.27 | 0.00 | ORB-long ORB[7300.40,7391.50] vol=2.7x ATR=19.51 |
| Stop hit — per-position SL triggered | 2025-01-01 09:35:00 | 7385.49 | 7377.27 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 10:45:00 | 7356.40 | 7336.50 | 0.00 | ORB-long ORB[7290.75,7355.00] vol=1.8x ATR=15.88 |
| Stop hit — per-position SL triggered | 2025-01-06 11:10:00 | 7340.52 | 7340.23 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:55:00 | 7073.20 | 7085.46 | 0.00 | ORB-short ORB[7083.70,7139.00] vol=1.7x ATR=22.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 12:05:00 | 7039.53 | 7080.92 | 0.00 | T1 1.5R @ 7039.53 |
| Target hit | 2025-01-10 15:20:00 | 7034.90 | 7055.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:15:00 | 6731.35 | 6793.35 | 0.00 | ORB-short ORB[6831.10,6899.00] vol=3.0x ATR=25.71 |
| Stop hit — per-position SL triggered | 2025-01-14 10:50:00 | 6757.06 | 6775.45 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:10:00 | 6843.70 | 6824.29 | 0.00 | ORB-long ORB[6770.35,6835.00] vol=1.9x ATR=14.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:30:00 | 6865.00 | 6829.37 | 0.00 | T1 1.5R @ 6865.00 |
| Stop hit — per-position SL triggered | 2025-01-30 12:45:00 | 6843.70 | 6845.70 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 11:10:00 | 6904.15 | 6874.45 | 0.00 | ORB-long ORB[6796.45,6889.90] vol=1.8x ATR=17.91 |
| Stop hit — per-position SL triggered | 2025-02-05 11:15:00 | 6886.24 | 6875.11 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:55:00 | 6838.95 | 6865.45 | 0.00 | ORB-short ORB[6862.55,6929.85] vol=1.6x ATR=23.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 12:10:00 | 6803.92 | 6854.09 | 0.00 | T1 1.5R @ 6803.92 |
| Target hit | 2025-02-07 15:20:00 | 6791.60 | 6828.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2025-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:10:00 | 6751.70 | 6776.22 | 0.00 | ORB-short ORB[6764.20,6834.35] vol=1.5x ATR=19.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:00:00 | 6722.06 | 6762.53 | 0.00 | T1 1.5R @ 6722.06 |
| Stop hit — per-position SL triggered | 2025-02-10 14:35:00 | 6751.70 | 6731.31 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:50:00 | 6171.45 | 6213.59 | 0.00 | ORB-short ORB[6210.10,6282.25] vol=1.5x ATR=16.49 |
| Stop hit — per-position SL triggered | 2025-02-27 09:55:00 | 6187.94 | 6210.93 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:10:00 | 6189.70 | 6173.52 | 0.00 | ORB-long ORB[6112.30,6169.95] vol=7.3x ATR=16.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 10:30:00 | 6214.07 | 6176.59 | 0.00 | T1 1.5R @ 6214.07 |
| Stop hit — per-position SL triggered | 2025-03-04 13:40:00 | 6189.70 | 6186.52 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-11 09:35:00 | 6073.25 | 6099.50 | 0.00 | ORB-short ORB[6080.35,6135.95] vol=1.6x ATR=16.93 |
| Stop hit — per-position SL triggered | 2025-03-11 09:45:00 | 6090.18 | 6091.70 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:45:00 | 6212.25 | 6198.30 | 0.00 | ORB-long ORB[6162.50,6209.90] vol=5.7x ATR=12.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:50:00 | 6230.78 | 6203.00 | 0.00 | T1 1.5R @ 6230.78 |
| Target hit | 2025-03-18 13:00:00 | 6226.20 | 6226.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:45:00 | 6335.00 | 6292.85 | 0.00 | ORB-long ORB[6227.10,6314.00] vol=1.7x ATR=18.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:25:00 | 6362.84 | 6313.18 | 0.00 | T1 1.5R @ 6362.84 |
| Target hit | 2025-03-19 15:20:00 | 6426.30 | 6386.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-03-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:10:00 | 6567.85 | 6530.06 | 0.00 | ORB-long ORB[6470.05,6518.20] vol=2.9x ATR=15.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 6591.20 | 6533.91 | 0.00 | T1 1.5R @ 6591.20 |
| Stop hit — per-position SL triggered | 2025-03-21 10:25:00 | 6567.85 | 6541.67 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 10:20:00 | 6637.90 | 6611.40 | 0.00 | ORB-long ORB[6562.05,6635.25] vol=2.1x ATR=15.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 10:30:00 | 6661.38 | 6615.09 | 0.00 | T1 1.5R @ 6661.38 |
| Stop hit — per-position SL triggered | 2025-03-25 10:40:00 | 6637.90 | 6617.35 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 11:00:00 | 6546.05 | 6479.49 | 0.00 | ORB-long ORB[6435.00,6523.00] vol=1.9x ATR=19.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 13:35:00 | 6575.35 | 6515.94 | 0.00 | T1 1.5R @ 6575.35 |
| Target hit | 2025-03-28 15:20:00 | 6610.60 | 6567.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2025-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:55:00 | 6761.80 | 6737.93 | 0.00 | ORB-long ORB[6675.00,6758.60] vol=4.6x ATR=15.87 |
| Stop hit — per-position SL triggered | 2025-04-03 11:50:00 | 6745.93 | 6744.47 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 11:15:00 | 6819.35 | 6792.44 | 0.00 | ORB-long ORB[6665.00,6755.70] vol=10.0x ATR=18.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 12:10:00 | 6847.01 | 6802.39 | 0.00 | T1 1.5R @ 6847.01 |
| Target hit | 2025-04-09 15:20:00 | 6830.40 | 6816.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2025-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:00:00 | 7156.00 | 7121.19 | 0.00 | ORB-long ORB[7058.00,7150.00] vol=1.9x ATR=13.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 12:45:00 | 7176.03 | 7131.74 | 0.00 | T1 1.5R @ 7176.03 |
| Stop hit — per-position SL triggered | 2025-04-21 13:55:00 | 7156.00 | 7142.98 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:50:00 | 7146.00 | 7123.17 | 0.00 | ORB-long ORB[7100.00,7145.00] vol=1.6x ATR=15.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:05:00 | 7168.98 | 7130.88 | 0.00 | T1 1.5R @ 7168.98 |
| Stop hit — per-position SL triggered | 2025-04-22 10:15:00 | 7146.00 | 7133.06 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:40:00 | 7021.00 | 7000.87 | 0.00 | ORB-long ORB[6950.00,6997.00] vol=1.7x ATR=12.83 |
| Stop hit — per-position SL triggered | 2025-05-05 10:55:00 | 7008.17 | 7003.02 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 11:15:00 | 6892.50 | 6917.10 | 0.00 | ORB-short ORB[6901.00,6979.00] vol=3.2x ATR=13.75 |
| Stop hit — per-position SL triggered | 2025-05-08 11:55:00 | 6906.25 | 6913.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 11:15:00 | 5898.00 | 2024-05-16 12:25:00 | 5914.01 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-05-16 11:15:00 | 5898.00 | 2024-05-16 12:30:00 | 5898.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 10:15:00 | 5875.00 | 2024-05-22 10:55:00 | 5853.75 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-22 10:15:00 | 5875.00 | 2024-05-22 12:45:00 | 5869.35 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-05-24 10:00:00 | 5903.00 | 2024-05-24 10:05:00 | 5879.86 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-24 10:00:00 | 5903.00 | 2024-05-24 11:50:00 | 5898.95 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-05-28 11:15:00 | 5900.30 | 2024-05-28 11:25:00 | 5911.20 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-05-30 11:15:00 | 5815.00 | 2024-05-30 11:45:00 | 5828.57 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-10 11:05:00 | 6069.80 | 2024-06-10 11:35:00 | 6055.92 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-06-19 11:15:00 | 6163.15 | 2024-06-19 12:10:00 | 6178.89 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-21 09:35:00 | 6222.55 | 2024-06-21 10:00:00 | 6208.43 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-24 10:30:00 | 6210.10 | 2024-06-24 10:40:00 | 6228.22 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-24 10:30:00 | 6210.10 | 2024-06-24 11:20:00 | 6210.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:45:00 | 6108.05 | 2024-07-02 10:55:00 | 6090.89 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-07-02 09:45:00 | 6108.05 | 2024-07-02 11:10:00 | 6108.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:50:00 | 6240.00 | 2024-07-04 10:55:00 | 6227.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-05 09:30:00 | 6262.20 | 2024-07-05 09:45:00 | 6244.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-08 10:55:00 | 6314.00 | 2024-07-08 15:20:00 | 6326.45 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-10 09:45:00 | 6368.95 | 2024-07-10 10:00:00 | 6357.02 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-19 10:50:00 | 6404.45 | 2024-07-19 11:00:00 | 6417.69 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-23 11:15:00 | 6402.30 | 2024-07-23 11:20:00 | 6418.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-24 11:10:00 | 6401.00 | 2024-07-24 11:15:00 | 6419.66 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-07-24 11:10:00 | 6401.00 | 2024-07-24 11:25:00 | 6401.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:25:00 | 6473.95 | 2024-07-26 10:30:00 | 6497.25 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-26 10:25:00 | 6473.95 | 2024-07-26 15:20:00 | 6663.20 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2024-08-01 11:10:00 | 6705.90 | 2024-08-01 11:20:00 | 6729.29 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-01 11:10:00 | 6705.90 | 2024-08-01 11:45:00 | 6705.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 11:10:00 | 6661.25 | 2024-08-08 11:25:00 | 6635.36 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-08 11:10:00 | 6661.25 | 2024-08-08 15:20:00 | 6528.90 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2024-08-09 11:05:00 | 6530.95 | 2024-08-09 11:10:00 | 6543.52 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-12 10:25:00 | 6484.85 | 2024-08-12 10:40:00 | 6461.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-12 10:25:00 | 6484.85 | 2024-08-12 11:25:00 | 6484.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-14 10:00:00 | 6729.15 | 2024-08-14 10:15:00 | 6776.35 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-08-14 10:00:00 | 6729.15 | 2024-08-14 11:50:00 | 6729.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:35:00 | 6717.05 | 2024-08-21 12:25:00 | 6741.93 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-21 10:35:00 | 6717.05 | 2024-08-21 15:20:00 | 6765.30 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-08-30 10:10:00 | 6922.90 | 2024-08-30 10:25:00 | 6952.19 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-30 10:10:00 | 6922.90 | 2024-08-30 10:50:00 | 6922.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:30:00 | 6918.05 | 2024-09-03 09:40:00 | 6902.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-05 10:55:00 | 7019.10 | 2024-09-05 11:05:00 | 7042.62 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-05 10:55:00 | 7019.10 | 2024-09-05 11:10:00 | 7019.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:50:00 | 6900.00 | 2024-09-06 11:10:00 | 6875.18 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-06 10:50:00 | 6900.00 | 2024-09-06 11:40:00 | 6900.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-09 11:05:00 | 6879.95 | 2024-09-09 11:55:00 | 6858.13 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-09-09 11:05:00 | 6879.95 | 2024-09-09 15:15:00 | 6866.00 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-09-11 09:45:00 | 6950.50 | 2024-09-11 09:55:00 | 6936.17 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-12 09:35:00 | 6987.75 | 2024-09-12 09:40:00 | 7018.81 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-12 09:35:00 | 6987.75 | 2024-09-12 09:45:00 | 6987.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 09:55:00 | 7056.85 | 2024-09-18 10:00:00 | 7042.69 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-19 10:30:00 | 7093.60 | 2024-09-19 10:55:00 | 7120.50 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-19 10:30:00 | 7093.60 | 2024-09-19 11:15:00 | 7093.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 11:05:00 | 7184.00 | 2024-09-24 11:15:00 | 7170.86 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-27 10:50:00 | 7171.95 | 2024-09-27 10:55:00 | 7196.18 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-27 10:50:00 | 7171.95 | 2024-09-27 11:05:00 | 7171.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:55:00 | 6798.30 | 2024-10-07 11:20:00 | 6816.96 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-08 11:05:00 | 6895.00 | 2024-10-08 11:35:00 | 6873.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-14 09:30:00 | 7006.90 | 2024-10-14 09:35:00 | 6992.79 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-10-18 09:50:00 | 7029.75 | 2024-10-18 09:55:00 | 7009.67 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-21 11:05:00 | 7019.85 | 2024-10-21 12:05:00 | 7004.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-22 10:45:00 | 6926.00 | 2024-10-22 10:50:00 | 6941.72 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-24 10:45:00 | 6943.15 | 2024-10-24 11:00:00 | 6925.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-28 09:30:00 | 6830.00 | 2024-10-28 09:35:00 | 6797.68 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-28 09:30:00 | 6830.00 | 2024-10-28 09:40:00 | 6830.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 11:15:00 | 6885.00 | 2024-10-29 11:50:00 | 6859.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-29 11:15:00 | 6885.00 | 2024-10-29 12:00:00 | 6885.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 10:20:00 | 6991.15 | 2024-10-30 10:25:00 | 7019.72 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-30 10:20:00 | 6991.15 | 2024-10-30 10:40:00 | 6991.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 10:00:00 | 6908.05 | 2024-11-13 11:30:00 | 6864.58 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-11-13 10:00:00 | 6908.05 | 2024-11-13 12:30:00 | 6908.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-14 10:45:00 | 6905.45 | 2024-11-14 10:50:00 | 6886.91 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-22 09:55:00 | 6909.00 | 2024-11-22 10:15:00 | 6887.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-28 10:35:00 | 6933.55 | 2024-11-28 10:45:00 | 6908.92 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-11-28 10:35:00 | 6933.55 | 2024-11-28 15:20:00 | 6850.05 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2024-12-05 11:15:00 | 7163.45 | 2024-12-05 12:05:00 | 7183.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-06 10:45:00 | 7296.00 | 2024-12-06 11:15:00 | 7276.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-11 10:45:00 | 7327.60 | 2024-12-11 11:00:00 | 7311.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-12 11:15:00 | 7202.65 | 2024-12-12 11:35:00 | 7215.71 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-23 11:15:00 | 7279.90 | 2024-12-23 11:25:00 | 7308.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-23 11:15:00 | 7279.90 | 2024-12-23 12:30:00 | 7279.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:15:00 | 7319.50 | 2024-12-30 13:25:00 | 7299.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-01 09:30:00 | 7405.00 | 2025-01-01 09:35:00 | 7385.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-06 10:45:00 | 7356.40 | 2025-01-06 11:10:00 | 7340.52 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-10 10:55:00 | 7073.20 | 2025-01-10 12:05:00 | 7039.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-10 10:55:00 | 7073.20 | 2025-01-10 15:20:00 | 7034.90 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-14 10:15:00 | 6731.35 | 2025-01-14 10:50:00 | 6757.06 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-30 11:10:00 | 6843.70 | 2025-01-30 11:30:00 | 6865.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-01-30 11:10:00 | 6843.70 | 2025-01-30 12:45:00 | 6843.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 11:10:00 | 6904.15 | 2025-02-05 11:15:00 | 6886.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-07 09:55:00 | 6838.95 | 2025-02-07 12:10:00 | 6803.92 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-02-07 09:55:00 | 6838.95 | 2025-02-07 15:20:00 | 6791.60 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-02-10 10:10:00 | 6751.70 | 2025-02-10 11:00:00 | 6722.06 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-02-10 10:10:00 | 6751.70 | 2025-02-10 14:35:00 | 6751.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-27 09:50:00 | 6171.45 | 2025-02-27 09:55:00 | 6187.94 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-03-04 10:10:00 | 6189.70 | 2025-03-04 10:30:00 | 6214.07 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-04 10:10:00 | 6189.70 | 2025-03-04 13:40:00 | 6189.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-11 09:35:00 | 6073.25 | 2025-03-11 09:45:00 | 6090.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-18 09:45:00 | 6212.25 | 2025-03-18 09:50:00 | 6230.78 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-18 09:45:00 | 6212.25 | 2025-03-18 13:00:00 | 6226.20 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-03-19 09:45:00 | 6335.00 | 2025-03-19 10:25:00 | 6362.84 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-19 09:45:00 | 6335.00 | 2025-03-19 15:20:00 | 6426.30 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2025-03-21 10:10:00 | 6567.85 | 2025-03-21 10:15:00 | 6591.20 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-03-21 10:10:00 | 6567.85 | 2025-03-21 10:25:00 | 6567.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 10:20:00 | 6637.90 | 2025-03-25 10:30:00 | 6661.38 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-03-25 10:20:00 | 6637.90 | 2025-03-25 10:40:00 | 6637.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 11:00:00 | 6546.05 | 2025-03-28 13:35:00 | 6575.35 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-28 11:00:00 | 6546.05 | 2025-03-28 15:20:00 | 6610.60 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2025-04-03 10:55:00 | 6761.80 | 2025-04-03 11:50:00 | 6745.93 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-04-09 11:15:00 | 6819.35 | 2025-04-09 12:10:00 | 6847.01 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-09 11:15:00 | 6819.35 | 2025-04-09 15:20:00 | 6830.40 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-04-21 11:00:00 | 7156.00 | 2025-04-21 12:45:00 | 7176.03 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-04-21 11:00:00 | 7156.00 | 2025-04-21 13:55:00 | 7156.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 09:50:00 | 7146.00 | 2025-04-22 10:05:00 | 7168.98 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-04-22 09:50:00 | 7146.00 | 2025-04-22 10:15:00 | 7146.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:40:00 | 7021.00 | 2025-05-05 10:55:00 | 7008.17 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-05-08 11:15:00 | 6892.50 | 2025-05-08 11:55:00 | 6906.25 | STOP_HIT | 1.00 | -0.20% |
