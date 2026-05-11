# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (52267 bars)
- **Last close:** 1479.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 16 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 73
- **Target hits / Stop hits / Partials:** 16 / 73 / 32
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 13.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 24 | 45.3% | 8 | 29 | 16 | 0.18% | 9.3% |
| BUY @ 2nd Alert (retest1) | 53 | 24 | 45.3% | 8 | 29 | 16 | 0.18% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 24 | 35.3% | 8 | 44 | 16 | 0.06% | 3.8% |
| SELL @ 2nd Alert (retest1) | 68 | 24 | 35.3% | 8 | 44 | 16 | 0.06% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 121 | 48 | 39.7% | 16 | 73 | 32 | 0.11% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:00:00 | 784.05 | 778.53 | 0.00 | ORB-long ORB[773.90,780.00] vol=2.5x ATR=2.61 |
| Stop hit — per-position SL triggered | 2023-05-12 11:15:00 | 781.44 | 778.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 10:15:00 | 824.90 | 818.16 | 0.00 | ORB-long ORB[810.00,819.80] vol=3.1x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-05-18 10:30:00 | 822.56 | 819.24 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:50:00 | 803.00 | 814.00 | 0.00 | ORB-short ORB[818.45,827.15] vol=1.6x ATR=4.00 |
| Stop hit — per-position SL triggered | 2023-05-19 11:10:00 | 807.00 | 810.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 10:55:00 | 807.00 | 803.04 | 0.00 | ORB-long ORB[797.55,806.40] vol=2.0x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 11:20:00 | 810.30 | 804.09 | 0.00 | T1 1.5R @ 810.30 |
| Stop hit — per-position SL triggered | 2023-05-23 12:05:00 | 807.00 | 806.16 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 11:00:00 | 793.00 | 795.77 | 0.00 | ORB-short ORB[796.20,801.25] vol=2.0x ATR=1.50 |
| Stop hit — per-position SL triggered | 2023-05-26 11:05:00 | 794.50 | 795.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:55:00 | 801.55 | 795.49 | 0.00 | ORB-long ORB[784.35,796.30] vol=2.4x ATR=2.69 |
| Stop hit — per-position SL triggered | 2023-05-29 10:00:00 | 798.86 | 795.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:35:00 | 792.80 | 795.41 | 0.00 | ORB-short ORB[795.45,799.80] vol=2.4x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-05-30 09:45:00 | 795.59 | 795.13 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 10:35:00 | 790.75 | 795.79 | 0.00 | ORB-short ORB[793.55,799.90] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2023-05-31 10:45:00 | 792.38 | 795.36 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 09:45:00 | 793.65 | 797.92 | 0.00 | ORB-short ORB[798.20,803.15] vol=4.4x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 09:50:00 | 790.43 | 795.97 | 0.00 | T1 1.5R @ 790.43 |
| Stop hit — per-position SL triggered | 2023-06-02 11:20:00 | 793.65 | 794.30 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:20:00 | 800.00 | 796.94 | 0.00 | ORB-long ORB[792.95,798.00] vol=4.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2023-06-05 10:25:00 | 798.08 | 797.01 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:25:00 | 792.10 | 795.22 | 0.00 | ORB-short ORB[793.40,798.30] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2023-06-06 11:45:00 | 793.74 | 794.40 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:00:00 | 814.80 | 806.99 | 0.00 | ORB-long ORB[800.00,810.60] vol=3.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-06-07 10:10:00 | 812.01 | 807.49 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 803.85 | 805.95 | 0.00 | ORB-short ORB[805.05,810.60] vol=2.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:35:00 | 800.32 | 804.27 | 0.00 | T1 1.5R @ 800.32 |
| Stop hit — per-position SL triggered | 2023-06-09 09:55:00 | 803.85 | 802.97 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 09:55:00 | 800.10 | 801.73 | 0.00 | ORB-short ORB[800.35,804.95] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-06-12 10:05:00 | 801.32 | 802.06 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:40:00 | 807.25 | 810.64 | 0.00 | ORB-short ORB[807.45,814.95] vol=3.2x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 11:15:00 | 804.09 | 809.10 | 0.00 | T1 1.5R @ 804.09 |
| Stop hit — per-position SL triggered | 2023-06-13 11:30:00 | 807.25 | 809.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:05:00 | 800.50 | 804.52 | 0.00 | ORB-short ORB[803.85,808.25] vol=3.9x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-06-14 11:15:00 | 802.06 | 804.17 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 10:50:00 | 791.90 | 796.34 | 0.00 | ORB-short ORB[797.20,802.05] vol=4.4x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:55:00 | 789.38 | 795.42 | 0.00 | T1 1.5R @ 789.38 |
| Target hit | 2023-06-15 15:20:00 | 784.60 | 791.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2023-06-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:40:00 | 798.05 | 796.52 | 0.00 | ORB-long ORB[791.70,795.55] vol=2.3x ATR=2.33 |
| Stop hit — per-position SL triggered | 2023-06-16 10:15:00 | 795.72 | 796.77 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:35:00 | 829.75 | 822.88 | 0.00 | ORB-long ORB[813.65,820.00] vol=5.2x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 09:45:00 | 835.36 | 826.01 | 0.00 | T1 1.5R @ 835.36 |
| Stop hit — per-position SL triggered | 2023-06-20 10:30:00 | 829.75 | 830.07 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:30:00 | 796.55 | 799.67 | 0.00 | ORB-short ORB[796.95,802.95] vol=1.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-06-27 10:45:00 | 798.50 | 799.21 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 891.80 | 884.54 | 0.00 | ORB-long ORB[874.20,882.40] vol=4.1x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 09:45:00 | 897.61 | 888.36 | 0.00 | T1 1.5R @ 897.61 |
| Stop hit — per-position SL triggered | 2023-07-05 10:35:00 | 891.80 | 892.81 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:25:00 | 897.50 | 905.34 | 0.00 | ORB-short ORB[905.55,913.00] vol=2.8x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:40:00 | 893.40 | 903.58 | 0.00 | T1 1.5R @ 893.40 |
| Stop hit — per-position SL triggered | 2023-07-07 11:00:00 | 897.50 | 900.40 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 11:15:00 | 903.05 | 916.89 | 0.00 | ORB-short ORB[916.00,927.95] vol=2.3x ATR=2.71 |
| Stop hit — per-position SL triggered | 2023-07-10 11:20:00 | 905.76 | 916.63 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:15:00 | 913.00 | 906.89 | 0.00 | ORB-long ORB[896.45,905.00] vol=3.4x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:30:00 | 919.14 | 909.64 | 0.00 | T1 1.5R @ 919.14 |
| Target hit | 2023-07-14 14:20:00 | 918.20 | 918.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2023-07-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 10:30:00 | 944.85 | 938.60 | 0.00 | ORB-long ORB[929.05,938.00] vol=2.5x ATR=3.21 |
| Stop hit — per-position SL triggered | 2023-07-19 11:50:00 | 941.64 | 941.16 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 11:00:00 | 964.70 | 969.65 | 0.00 | ORB-short ORB[968.05,975.50] vol=3.7x ATR=2.64 |
| Stop hit — per-position SL triggered | 2023-07-25 11:45:00 | 967.34 | 968.73 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:45:00 | 994.05 | 987.11 | 0.00 | ORB-long ORB[979.05,992.00] vol=2.2x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:55:00 | 998.86 | 995.42 | 0.00 | T1 1.5R @ 998.86 |
| Target hit | 2023-07-26 13:10:00 | 998.50 | 999.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:15:00 | 1020.05 | 1015.71 | 0.00 | ORB-long ORB[1008.15,1018.00] vol=2.8x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-07-28 10:20:00 | 1016.63 | 1015.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:20:00 | 1025.00 | 1027.37 | 0.00 | ORB-short ORB[1029.85,1036.45] vol=2.4x ATR=3.70 |
| Stop hit — per-position SL triggered | 2023-08-03 10:50:00 | 1028.70 | 1027.31 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:35:00 | 1045.15 | 1043.10 | 0.00 | ORB-long ORB[1036.05,1044.45] vol=5.2x ATR=4.03 |
| Stop hit — per-position SL triggered | 2023-08-04 10:05:00 | 1041.12 | 1044.38 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:50:00 | 1010.00 | 1022.46 | 0.00 | ORB-short ORB[1027.00,1038.40] vol=11.6x ATR=4.22 |
| Stop hit — per-position SL triggered | 2023-08-07 10:55:00 | 1014.22 | 1021.77 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 09:35:00 | 1032.00 | 1038.26 | 0.00 | ORB-short ORB[1038.05,1044.40] vol=1.6x ATR=3.59 |
| Stop hit — per-position SL triggered | 2023-08-09 09:50:00 | 1035.59 | 1036.26 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 09:45:00 | 1020.20 | 1024.68 | 0.00 | ORB-short ORB[1026.45,1036.55] vol=5.3x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:25:00 | 1015.35 | 1022.36 | 0.00 | T1 1.5R @ 1015.35 |
| Target hit | 2023-08-10 15:20:00 | 1004.30 | 1013.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2023-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:40:00 | 1016.85 | 1011.78 | 0.00 | ORB-long ORB[1005.05,1014.00] vol=1.8x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 09:50:00 | 1021.98 | 1013.45 | 0.00 | T1 1.5R @ 1021.98 |
| Stop hit — per-position SL triggered | 2023-08-16 10:00:00 | 1016.85 | 1014.48 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:40:00 | 1014.05 | 1008.79 | 0.00 | ORB-long ORB[1002.00,1010.40] vol=2.1x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 10:10:00 | 1018.70 | 1012.92 | 0.00 | T1 1.5R @ 1018.70 |
| Target hit | 2023-08-17 11:55:00 | 1016.00 | 1021.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2023-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 09:30:00 | 993.30 | 998.06 | 0.00 | ORB-short ORB[995.10,1007.45] vol=2.7x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-08-21 10:05:00 | 996.97 | 997.41 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 11:00:00 | 1011.60 | 1006.77 | 0.00 | ORB-long ORB[999.75,1011.15] vol=3.9x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 11:05:00 | 1015.27 | 1007.76 | 0.00 | T1 1.5R @ 1015.27 |
| Stop hit — per-position SL triggered | 2023-08-23 12:15:00 | 1011.60 | 1010.53 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-08-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:50:00 | 1002.05 | 1011.11 | 0.00 | ORB-short ORB[1010.00,1024.15] vol=4.1x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-08-24 10:00:00 | 1005.85 | 1010.01 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-08-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:00:00 | 1029.00 | 1034.43 | 0.00 | ORB-short ORB[1032.85,1047.75] vol=5.6x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:10:00 | 1023.73 | 1033.06 | 0.00 | T1 1.5R @ 1023.73 |
| Target hit | 2023-08-31 12:45:00 | 1028.15 | 1027.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2023-09-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 09:50:00 | 1044.00 | 1040.79 | 0.00 | ORB-long ORB[1029.75,1043.60] vol=2.0x ATR=3.75 |
| Stop hit — per-position SL triggered | 2023-09-01 09:55:00 | 1040.25 | 1040.98 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:55:00 | 1088.00 | 1080.64 | 0.00 | ORB-long ORB[1074.90,1084.80] vol=2.1x ATR=4.55 |
| Stop hit — per-position SL triggered | 2023-09-04 10:00:00 | 1083.45 | 1081.21 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:50:00 | 1070.95 | 1063.59 | 0.00 | ORB-long ORB[1055.05,1065.00] vol=2.3x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:10:00 | 1077.08 | 1067.04 | 0.00 | T1 1.5R @ 1077.08 |
| Stop hit — per-position SL triggered | 2023-09-05 10:15:00 | 1070.95 | 1067.40 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 11:10:00 | 1078.35 | 1085.49 | 0.00 | ORB-short ORB[1079.10,1094.40] vol=3.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2023-09-11 11:40:00 | 1081.03 | 1084.51 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 1056.00 | 1063.02 | 0.00 | ORB-short ORB[1060.50,1070.40] vol=2.1x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 1050.13 | 1056.02 | 0.00 | T1 1.5R @ 1050.13 |
| Target hit | 2023-09-12 10:10:00 | 1038.20 | 1037.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2023-09-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:50:00 | 1019.90 | 1024.19 | 0.00 | ORB-short ORB[1024.25,1034.65] vol=3.3x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 12:20:00 | 1014.50 | 1020.87 | 0.00 | T1 1.5R @ 1014.50 |
| Stop hit — per-position SL triggered | 2023-09-21 15:10:00 | 1019.90 | 1016.95 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:30:00 | 1124.00 | 1119.54 | 0.00 | ORB-long ORB[1100.10,1114.40] vol=6.7x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 09:45:00 | 1130.72 | 1123.11 | 0.00 | T1 1.5R @ 1130.72 |
| Target hit | 2023-10-10 12:25:00 | 1148.60 | 1148.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 1161.75 | 1156.71 | 0.00 | ORB-long ORB[1145.40,1158.90] vol=2.6x ATR=5.40 |
| Stop hit — per-position SL triggered | 2023-10-11 09:55:00 | 1156.35 | 1157.82 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 10:20:00 | 1180.85 | 1171.66 | 0.00 | ORB-long ORB[1161.00,1178.40] vol=3.0x ATR=5.50 |
| Stop hit — per-position SL triggered | 2023-10-12 12:05:00 | 1175.35 | 1174.73 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 11:15:00 | 1162.85 | 1163.18 | 0.00 | ORB-short ORB[1168.60,1178.65] vol=3.5x ATR=3.43 |
| Stop hit — per-position SL triggered | 2023-10-13 11:20:00 | 1166.28 | 1163.68 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:20:00 | 1205.80 | 1194.69 | 0.00 | ORB-long ORB[1182.30,1196.95] vol=1.7x ATR=6.03 |
| Stop hit — per-position SL triggered | 2023-10-18 10:50:00 | 1199.77 | 1198.22 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:50:00 | 1166.60 | 1168.33 | 0.00 | ORB-short ORB[1175.15,1184.40] vol=5.6x ATR=6.55 |
| Stop hit — per-position SL triggered | 2023-10-19 10:30:00 | 1173.15 | 1168.35 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 09:45:00 | 1074.00 | 1080.18 | 0.00 | ORB-short ORB[1079.00,1086.20] vol=2.0x ATR=3.55 |
| Stop hit — per-position SL triggered | 2023-11-01 09:50:00 | 1077.55 | 1080.10 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:05:00 | 1090.30 | 1087.46 | 0.00 | ORB-long ORB[1080.05,1089.90] vol=4.4x ATR=3.25 |
| Stop hit — per-position SL triggered | 2023-11-02 10:50:00 | 1087.05 | 1087.52 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 09:45:00 | 1077.25 | 1082.14 | 0.00 | ORB-short ORB[1080.05,1089.65] vol=4.0x ATR=4.16 |
| Stop hit — per-position SL triggered | 2023-11-06 09:55:00 | 1081.41 | 1081.92 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:10:00 | 1102.25 | 1096.59 | 0.00 | ORB-long ORB[1089.40,1101.70] vol=6.7x ATR=2.66 |
| Stop hit — per-position SL triggered | 2023-11-07 11:20:00 | 1099.59 | 1096.80 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:40:00 | 1345.00 | 1337.21 | 0.00 | ORB-long ORB[1330.70,1344.45] vol=2.2x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:45:00 | 1358.33 | 1347.95 | 0.00 | T1 1.5R @ 1358.33 |
| Stop hit — per-position SL triggered | 2023-12-08 10:50:00 | 1345.00 | 1349.01 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 10:20:00 | 1295.05 | 1301.37 | 0.00 | ORB-short ORB[1298.00,1306.00] vol=3.2x ATR=3.59 |
| Stop hit — per-position SL triggered | 2023-12-14 10:50:00 | 1298.64 | 1301.99 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 11:15:00 | 1196.50 | 1209.45 | 0.00 | ORB-short ORB[1201.05,1218.00] vol=7.1x ATR=3.84 |
| Stop hit — per-position SL triggered | 2023-12-19 11:25:00 | 1200.34 | 1208.32 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:00:00 | 1248.00 | 1240.03 | 0.00 | ORB-long ORB[1226.40,1244.65] vol=2.1x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 10:40:00 | 1257.63 | 1243.96 | 0.00 | T1 1.5R @ 1257.63 |
| Stop hit — per-position SL triggered | 2023-12-20 11:10:00 | 1248.00 | 1245.02 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-12-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 11:10:00 | 1230.25 | 1239.98 | 0.00 | ORB-short ORB[1238.95,1254.40] vol=4.2x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 12:30:00 | 1224.62 | 1237.32 | 0.00 | T1 1.5R @ 1224.62 |
| Stop hit — per-position SL triggered | 2023-12-22 15:00:00 | 1230.25 | 1230.79 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-12-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:25:00 | 1218.05 | 1226.30 | 0.00 | ORB-short ORB[1219.95,1236.35] vol=3.0x ATR=4.94 |
| Stop hit — per-position SL triggered | 2023-12-26 10:40:00 | 1222.99 | 1224.57 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 09:40:00 | 1211.95 | 1217.10 | 0.00 | ORB-short ORB[1214.05,1230.45] vol=4.4x ATR=4.88 |
| Stop hit — per-position SL triggered | 2023-12-29 10:35:00 | 1216.83 | 1214.29 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:45:00 | 1255.60 | 1259.39 | 0.00 | ORB-short ORB[1259.15,1269.65] vol=2.4x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 09:55:00 | 1248.37 | 1258.13 | 0.00 | T1 1.5R @ 1248.37 |
| Target hit | 2024-01-03 10:50:00 | 1247.85 | 1247.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2024-01-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:45:00 | 1559.40 | 1574.30 | 0.00 | ORB-short ORB[1563.00,1586.25] vol=2.0x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 11:15:00 | 1547.76 | 1569.28 | 0.00 | T1 1.5R @ 1547.76 |
| Stop hit — per-position SL triggered | 2024-01-11 11:50:00 | 1559.40 | 1565.99 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:55:00 | 1371.00 | 1380.31 | 0.00 | ORB-short ORB[1398.95,1414.55] vol=5.6x ATR=8.49 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 1379.49 | 1379.57 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:40:00 | 1429.70 | 1423.89 | 0.00 | ORB-long ORB[1416.90,1425.95] vol=1.6x ATR=8.70 |
| Stop hit — per-position SL triggered | 2024-01-20 09:55:00 | 1421.00 | 1423.87 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-24 10:45:00 | 1334.10 | 1351.24 | 0.00 | ORB-short ORB[1348.00,1368.05] vol=2.8x ATR=7.17 |
| Stop hit — per-position SL triggered | 2024-01-24 11:25:00 | 1341.27 | 1344.48 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 09:30:00 | 1408.25 | 1416.72 | 0.00 | ORB-short ORB[1412.60,1425.00] vol=1.6x ATR=5.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:05:00 | 1399.56 | 1410.61 | 0.00 | T1 1.5R @ 1399.56 |
| Target hit | 2024-01-30 13:45:00 | 1406.40 | 1404.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2024-01-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:35:00 | 1400.00 | 1395.40 | 0.00 | ORB-long ORB[1385.05,1396.85] vol=1.9x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 09:45:00 | 1408.94 | 1399.07 | 0.00 | T1 1.5R @ 1408.94 |
| Target hit | 2024-01-31 10:45:00 | 1419.55 | 1422.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — SELL (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 1443.15 | 1454.16 | 0.00 | ORB-short ORB[1450.65,1469.15] vol=1.7x ATR=7.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:45:00 | 1432.31 | 1449.57 | 0.00 | T1 1.5R @ 1432.31 |
| Target hit | 2024-02-09 13:20:00 | 1428.00 | 1427.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 11:15:00 | 1433.00 | 1441.95 | 0.00 | ORB-short ORB[1439.95,1454.95] vol=1.6x ATR=4.63 |
| Stop hit — per-position SL triggered | 2024-02-16 11:20:00 | 1437.63 | 1441.73 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 11:15:00 | 1410.70 | 1411.76 | 0.00 | ORB-short ORB[1413.10,1432.20] vol=2.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-02-19 12:40:00 | 1414.76 | 1411.40 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 1444.00 | 1451.48 | 0.00 | ORB-short ORB[1450.00,1467.20] vol=3.8x ATR=7.53 |
| Stop hit — per-position SL triggered | 2024-03-04 10:00:00 | 1451.53 | 1445.07 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 09:40:00 | 1464.85 | 1460.80 | 0.00 | ORB-long ORB[1450.05,1462.75] vol=2.1x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 09:50:00 | 1472.74 | 1475.61 | 0.00 | T1 1.5R @ 1472.74 |
| Target hit | 2024-03-05 10:10:00 | 1478.40 | 1479.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 75 — BUY (started 2024-03-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:50:00 | 1455.00 | 1450.32 | 0.00 | ORB-long ORB[1443.00,1452.20] vol=1.6x ATR=7.73 |
| Stop hit — per-position SL triggered | 2024-03-07 11:05:00 | 1447.27 | 1450.22 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:15:00 | 1395.00 | 1402.15 | 0.00 | ORB-short ORB[1404.05,1424.90] vol=3.1x ATR=7.82 |
| Stop hit — per-position SL triggered | 2024-03-15 11:20:00 | 1402.82 | 1401.77 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:00:00 | 1413.00 | 1424.25 | 0.00 | ORB-short ORB[1414.10,1434.00] vol=1.7x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:05:00 | 1403.55 | 1423.15 | 0.00 | T1 1.5R @ 1403.55 |
| Stop hit — per-position SL triggered | 2024-03-19 10:10:00 | 1413.00 | 1422.68 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:00:00 | 1392.00 | 1405.12 | 0.00 | ORB-short ORB[1404.15,1421.95] vol=2.8x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-03-20 10:05:00 | 1397.11 | 1403.57 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-03-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:40:00 | 1475.95 | 1470.43 | 0.00 | ORB-long ORB[1454.60,1472.00] vol=4.0x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-03-22 09:45:00 | 1471.44 | 1470.58 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-03-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:10:00 | 1443.25 | 1453.91 | 0.00 | ORB-short ORB[1450.00,1468.20] vol=2.2x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-03-26 10:45:00 | 1450.37 | 1451.28 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-04-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:45:00 | 1672.80 | 1684.37 | 0.00 | ORB-short ORB[1684.00,1695.00] vol=1.6x ATR=8.02 |
| Stop hit — per-position SL triggered | 2024-04-04 10:45:00 | 1680.82 | 1678.23 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 09:45:00 | 1713.95 | 1694.62 | 0.00 | ORB-long ORB[1675.00,1698.00] vol=6.0x ATR=9.79 |
| Stop hit — per-position SL triggered | 2024-04-08 09:50:00 | 1704.16 | 1700.75 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:00:00 | 1780.00 | 1764.46 | 0.00 | ORB-long ORB[1739.55,1757.20] vol=2.3x ATR=10.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:35:00 | 1796.34 | 1776.29 | 0.00 | T1 1.5R @ 1796.34 |
| Target hit | 2024-04-09 13:40:00 | 1809.45 | 1813.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 84 — BUY (started 2024-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:35:00 | 1890.00 | 1879.58 | 0.00 | ORB-long ORB[1869.10,1886.20] vol=2.2x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-04-23 10:40:00 | 1882.22 | 1879.68 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 10:15:00 | 1902.90 | 1922.83 | 0.00 | ORB-short ORB[1920.80,1939.00] vol=1.9x ATR=8.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 12:15:00 | 1890.24 | 1916.05 | 0.00 | T1 1.5R @ 1890.24 |
| Target hit | 2024-04-24 15:20:00 | 1859.30 | 1901.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2024-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 11:05:00 | 1998.10 | 2015.17 | 0.00 | ORB-short ORB[2008.00,2032.95] vol=2.6x ATR=7.47 |
| Stop hit — per-position SL triggered | 2024-04-29 11:35:00 | 2005.57 | 2010.42 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 09:55:00 | 1998.30 | 2002.82 | 0.00 | ORB-short ORB[2002.80,2020.45] vol=2.9x ATR=8.28 |
| Stop hit — per-position SL triggered | 2024-04-30 10:25:00 | 2006.58 | 2001.67 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-05-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:45:00 | 2024.35 | 2010.91 | 0.00 | ORB-long ORB[2002.65,2012.90] vol=1.5x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 09:50:00 | 2034.03 | 2027.24 | 0.00 | T1 1.5R @ 2034.03 |
| Target hit | 2024-05-02 10:05:00 | 2029.30 | 2031.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 89 — SELL (started 2024-05-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 11:00:00 | 1994.90 | 2011.75 | 0.00 | ORB-short ORB[2000.00,2021.55] vol=3.8x ATR=7.92 |
| Stop hit — per-position SL triggered | 2024-05-03 11:10:00 | 2002.82 | 2011.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:00:00 | 784.05 | 2023-05-12 11:15:00 | 781.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-05-18 10:15:00 | 824.90 | 2023-05-18 10:30:00 | 822.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-19 09:50:00 | 803.00 | 2023-05-19 11:10:00 | 807.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-05-23 10:55:00 | 807.00 | 2023-05-23 11:20:00 | 810.30 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-05-23 10:55:00 | 807.00 | 2023-05-23 12:05:00 | 807.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-26 11:00:00 | 793.00 | 2023-05-26 11:05:00 | 794.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-05-29 09:55:00 | 801.55 | 2023-05-29 10:00:00 | 798.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-05-30 09:35:00 | 792.80 | 2023-05-30 09:45:00 | 795.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-05-31 10:35:00 | 790.75 | 2023-05-31 10:45:00 | 792.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-02 09:45:00 | 793.65 | 2023-06-02 09:50:00 | 790.43 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-06-02 09:45:00 | 793.65 | 2023-06-02 11:20:00 | 793.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-05 10:20:00 | 800.00 | 2023-06-05 10:25:00 | 798.08 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-06 10:25:00 | 792.10 | 2023-06-06 11:45:00 | 793.74 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-07 10:00:00 | 814.80 | 2023-06-07 10:10:00 | 812.01 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-09 09:30:00 | 803.85 | 2023-06-09 09:35:00 | 800.32 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-06-09 09:30:00 | 803.85 | 2023-06-09 09:55:00 | 803.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-12 09:55:00 | 800.10 | 2023-06-12 10:05:00 | 801.32 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-13 10:40:00 | 807.25 | 2023-06-13 11:15:00 | 804.09 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-06-13 10:40:00 | 807.25 | 2023-06-13 11:30:00 | 807.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 11:05:00 | 800.50 | 2023-06-14 11:15:00 | 802.06 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-15 10:50:00 | 791.90 | 2023-06-15 10:55:00 | 789.38 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-06-15 10:50:00 | 791.90 | 2023-06-15 15:20:00 | 784.60 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2023-06-16 09:40:00 | 798.05 | 2023-06-16 10:15:00 | 795.72 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-20 09:35:00 | 829.75 | 2023-06-20 09:45:00 | 835.36 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-06-20 09:35:00 | 829.75 | 2023-06-20 10:30:00 | 829.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-27 10:30:00 | 796.55 | 2023-06-27 10:45:00 | 798.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-05 09:30:00 | 891.80 | 2023-07-05 09:45:00 | 897.61 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2023-07-05 09:30:00 | 891.80 | 2023-07-05 10:35:00 | 891.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-07 10:25:00 | 897.50 | 2023-07-07 10:40:00 | 893.40 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-07-07 10:25:00 | 897.50 | 2023-07-07 11:00:00 | 897.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-10 11:15:00 | 903.05 | 2023-07-10 11:20:00 | 905.76 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-14 10:15:00 | 913.00 | 2023-07-14 10:30:00 | 919.14 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-07-14 10:15:00 | 913.00 | 2023-07-14 14:20:00 | 918.20 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2023-07-19 10:30:00 | 944.85 | 2023-07-19 11:50:00 | 941.64 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-25 11:00:00 | 964.70 | 2023-07-25 11:45:00 | 967.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-26 10:45:00 | 994.05 | 2023-07-26 10:55:00 | 998.86 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-26 10:45:00 | 994.05 | 2023-07-26 13:10:00 | 998.50 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-28 10:15:00 | 1020.05 | 2023-07-28 10:20:00 | 1016.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-03 10:20:00 | 1025.00 | 2023-08-03 10:50:00 | 1028.70 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-04 09:35:00 | 1045.15 | 2023-08-04 10:05:00 | 1041.12 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-08-07 10:50:00 | 1010.00 | 2023-08-07 10:55:00 | 1014.22 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-08-09 09:35:00 | 1032.00 | 2023-08-09 09:50:00 | 1035.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-08-10 09:45:00 | 1020.20 | 2023-08-10 10:25:00 | 1015.35 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-08-10 09:45:00 | 1020.20 | 2023-08-10 15:20:00 | 1004.30 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2023-08-16 09:40:00 | 1016.85 | 2023-08-16 09:50:00 | 1021.98 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-08-16 09:40:00 | 1016.85 | 2023-08-16 10:00:00 | 1016.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 09:40:00 | 1014.05 | 2023-08-17 10:10:00 | 1018.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-08-17 09:40:00 | 1014.05 | 2023-08-17 11:55:00 | 1016.00 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-08-21 09:30:00 | 993.30 | 2023-08-21 10:05:00 | 996.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-08-23 11:00:00 | 1011.60 | 2023-08-23 11:05:00 | 1015.27 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-23 11:00:00 | 1011.60 | 2023-08-23 12:15:00 | 1011.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-24 09:50:00 | 1002.05 | 2023-08-24 10:00:00 | 1005.85 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-08-31 11:00:00 | 1029.00 | 2023-08-31 11:10:00 | 1023.73 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-08-31 11:00:00 | 1029.00 | 2023-08-31 12:45:00 | 1028.15 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2023-09-01 09:50:00 | 1044.00 | 2023-09-01 09:55:00 | 1040.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-09-04 09:55:00 | 1088.00 | 2023-09-04 10:00:00 | 1083.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-09-05 09:50:00 | 1070.95 | 2023-09-05 10:10:00 | 1077.08 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-09-05 09:50:00 | 1070.95 | 2023-09-05 10:15:00 | 1070.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-11 11:10:00 | 1078.35 | 2023-09-11 11:40:00 | 1081.03 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-12 09:30:00 | 1056.00 | 2023-09-12 09:35:00 | 1050.13 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-09-12 09:30:00 | 1056.00 | 2023-09-12 10:10:00 | 1038.20 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2023-09-21 09:50:00 | 1019.90 | 2023-09-21 12:20:00 | 1014.50 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-09-21 09:50:00 | 1019.90 | 2023-09-21 15:10:00 | 1019.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-10 09:30:00 | 1124.00 | 2023-10-10 09:45:00 | 1130.72 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-10-10 09:30:00 | 1124.00 | 2023-10-10 12:25:00 | 1148.60 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2023-10-11 09:35:00 | 1161.75 | 2023-10-11 09:55:00 | 1156.35 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-10-12 10:20:00 | 1180.85 | 2023-10-12 12:05:00 | 1175.35 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-10-13 11:15:00 | 1162.85 | 2023-10-13 11:20:00 | 1166.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-10-18 10:20:00 | 1205.80 | 2023-10-18 10:50:00 | 1199.77 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-10-19 09:50:00 | 1166.60 | 2023-10-19 10:30:00 | 1173.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2023-11-01 09:45:00 | 1074.00 | 2023-11-01 09:50:00 | 1077.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-02 10:05:00 | 1090.30 | 2023-11-02 10:50:00 | 1087.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-11-06 09:45:00 | 1077.25 | 2023-11-06 09:55:00 | 1081.41 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-11-07 11:10:00 | 1102.25 | 2023-11-07 11:20:00 | 1099.59 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-08 10:40:00 | 1345.00 | 2023-12-08 10:45:00 | 1358.33 | PARTIAL | 0.50 | 0.99% |
| BUY | retest1 | 2023-12-08 10:40:00 | 1345.00 | 2023-12-08 10:50:00 | 1345.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-14 10:20:00 | 1295.05 | 2023-12-14 10:50:00 | 1298.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-19 11:15:00 | 1196.50 | 2023-12-19 11:25:00 | 1200.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-20 10:00:00 | 1248.00 | 2023-12-20 10:40:00 | 1257.63 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2023-12-20 10:00:00 | 1248.00 | 2023-12-20 11:10:00 | 1248.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-22 11:10:00 | 1230.25 | 2023-12-22 12:30:00 | 1224.62 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-12-22 11:10:00 | 1230.25 | 2023-12-22 15:00:00 | 1230.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-26 10:25:00 | 1218.05 | 2023-12-26 10:40:00 | 1222.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-12-29 09:40:00 | 1211.95 | 2023-12-29 10:35:00 | 1216.83 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-01-03 09:45:00 | 1255.60 | 2024-01-03 09:55:00 | 1248.37 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-01-03 09:45:00 | 1255.60 | 2024-01-03 10:50:00 | 1247.85 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-01-11 10:45:00 | 1559.40 | 2024-01-11 11:15:00 | 1547.76 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-01-11 10:45:00 | 1559.40 | 2024-01-11 11:50:00 | 1559.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:55:00 | 1371.00 | 2024-01-18 10:05:00 | 1379.49 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-01-20 09:40:00 | 1429.70 | 2024-01-20 09:55:00 | 1421.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-01-24 10:45:00 | 1334.10 | 2024-01-24 11:25:00 | 1341.27 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-01-30 09:30:00 | 1408.25 | 2024-01-30 10:05:00 | 1399.56 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-01-30 09:30:00 | 1408.25 | 2024-01-30 13:45:00 | 1406.40 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-01-31 09:35:00 | 1400.00 | 2024-01-31 09:45:00 | 1408.94 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-01-31 09:35:00 | 1400.00 | 2024-01-31 10:45:00 | 1419.55 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-02-09 09:40:00 | 1443.15 | 2024-02-09 09:45:00 | 1432.31 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-02-09 09:40:00 | 1443.15 | 2024-02-09 13:20:00 | 1428.00 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2024-02-16 11:15:00 | 1433.00 | 2024-02-16 11:20:00 | 1437.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-19 11:15:00 | 1410.70 | 2024-02-19 12:40:00 | 1414.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-04 09:30:00 | 1444.00 | 2024-03-04 10:00:00 | 1451.53 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-03-05 09:40:00 | 1464.85 | 2024-03-05 09:50:00 | 1472.74 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-03-05 09:40:00 | 1464.85 | 2024-03-05 10:10:00 | 1478.40 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-03-07 09:50:00 | 1455.00 | 2024-03-07 11:05:00 | 1447.27 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-03-15 11:15:00 | 1395.00 | 2024-03-15 11:20:00 | 1402.82 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-03-19 10:00:00 | 1413.00 | 2024-03-19 10:05:00 | 1403.55 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-03-19 10:00:00 | 1413.00 | 2024-03-19 10:10:00 | 1413.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 10:00:00 | 1392.00 | 2024-03-20 10:05:00 | 1397.11 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-03-22 09:40:00 | 1475.95 | 2024-03-22 09:45:00 | 1471.44 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-26 10:10:00 | 1443.25 | 2024-03-26 10:45:00 | 1450.37 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-04-04 09:45:00 | 1672.80 | 2024-04-04 10:45:00 | 1680.82 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-04-08 09:45:00 | 1713.95 | 2024-04-08 09:50:00 | 1704.16 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-04-09 10:00:00 | 1780.00 | 2024-04-09 10:35:00 | 1796.34 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2024-04-09 10:00:00 | 1780.00 | 2024-04-09 13:40:00 | 1809.45 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2024-04-23 10:35:00 | 1890.00 | 2024-04-23 10:40:00 | 1882.22 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-04-24 10:15:00 | 1902.90 | 2024-04-24 12:15:00 | 1890.24 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-04-24 10:15:00 | 1902.90 | 2024-04-24 15:20:00 | 1859.30 | TARGET_HIT | 0.50 | 2.29% |
| SELL | retest1 | 2024-04-29 11:05:00 | 1998.10 | 2024-04-29 11:35:00 | 2005.57 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-04-30 09:55:00 | 1998.30 | 2024-04-30 10:25:00 | 2006.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-02 09:45:00 | 2024.35 | 2024-05-02 09:50:00 | 2034.03 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-02 09:45:00 | 2024.35 | 2024-05-02 10:05:00 | 2029.30 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-05-03 11:00:00 | 1994.90 | 2024-05-03 11:10:00 | 2002.82 | STOP_HIT | 1.00 | -0.40% |
