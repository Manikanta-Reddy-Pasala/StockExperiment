# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 1949.90
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 11 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 21
- **Target hits / Stop hits / Partials:** 11 / 21 / 16
- **Avg / median % per leg:** 0.50% / 0.42%
- **Sum % (uncompounded):** 23.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 18 | 62.1% | 8 | 11 | 10 | 0.50% | 14.4% |
| BUY @ 2nd Alert (retest1) | 29 | 18 | 62.1% | 8 | 11 | 10 | 0.50% | 14.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.50% | 9.6% |
| SELL @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.50% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 48 | 27 | 56.2% | 11 | 21 | 16 | 0.50% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 919.55 | 928.38 | 0.00 | ORB-short ORB[935.00,940.00] vol=2.7x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-05-17 10:55:00 | 923.08 | 928.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:20:00 | 941.90 | 934.31 | 0.00 | ORB-long ORB[918.00,931.00] vol=8.5x ATR=5.78 |
| Stop hit — per-position SL triggered | 2024-05-22 10:25:00 | 936.12 | 934.58 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:50:00 | 800.75 | 791.04 | 0.00 | ORB-long ORB[781.00,792.50] vol=2.0x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-06-06 09:55:00 | 795.16 | 791.98 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:45:00 | 789.00 | 781.15 | 0.00 | ORB-long ORB[772.00,782.00] vol=1.7x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:05:00 | 795.31 | 784.78 | 0.00 | T1 1.5R @ 795.31 |
| Target hit | 2024-06-07 12:30:00 | 807.05 | 808.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 820.00 | 812.60 | 0.00 | ORB-long ORB[805.00,816.85] vol=3.2x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-06-13 09:40:00 | 816.18 | 813.90 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 831.00 | 834.45 | 0.00 | ORB-short ORB[831.15,840.00] vol=2.4x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-06-21 10:50:00 | 833.75 | 834.38 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 830.10 | 833.25 | 0.00 | ORB-short ORB[830.80,843.00] vol=1.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:20:00 | 827.04 | 832.99 | 0.00 | T1 1.5R @ 827.04 |
| Stop hit — per-position SL triggered | 2024-06-25 11:45:00 | 830.10 | 832.75 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:35:00 | 782.35 | 788.12 | 0.00 | ORB-short ORB[786.70,797.20] vol=2.2x ATR=3.95 |
| Stop hit — per-position SL triggered | 2024-06-27 10:40:00 | 786.30 | 787.96 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:50:00 | 815.00 | 810.93 | 0.00 | ORB-long ORB[803.80,814.80] vol=1.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:20:00 | 819.65 | 814.13 | 0.00 | T1 1.5R @ 819.65 |
| Target hit | 2024-07-04 15:20:00 | 845.00 | 828.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-07-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:00:00 | 830.00 | 833.44 | 0.00 | ORB-short ORB[830.75,843.15] vol=2.2x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:15:00 | 825.97 | 832.35 | 0.00 | T1 1.5R @ 825.97 |
| Target hit | 2024-07-09 15:20:00 | 826.50 | 827.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 833.50 | 828.50 | 0.00 | ORB-long ORB[823.05,829.90] vol=2.4x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:35:00 | 837.34 | 832.84 | 0.00 | T1 1.5R @ 837.34 |
| Target hit | 2024-07-10 09:55:00 | 836.70 | 837.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:50:00 | 830.00 | 835.04 | 0.00 | ORB-short ORB[834.55,841.75] vol=1.8x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-07-12 11:10:00 | 833.23 | 833.60 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:55:00 | 894.30 | 900.98 | 0.00 | ORB-short ORB[900.15,912.00] vol=3.2x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 10:05:00 | 889.19 | 894.43 | 0.00 | T1 1.5R @ 889.19 |
| Target hit | 2024-08-13 15:20:00 | 831.10 | 852.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 846.05 | 842.60 | 0.00 | ORB-long ORB[832.05,844.20] vol=11.3x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:35:00 | 853.40 | 843.43 | 0.00 | T1 1.5R @ 853.40 |
| Target hit | 2024-08-16 10:40:00 | 852.20 | 852.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 888.30 | 883.38 | 0.00 | ORB-long ORB[880.00,886.45] vol=2.7x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-08-19 09:45:00 | 884.15 | 883.51 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 889.00 | 882.24 | 0.00 | ORB-long ORB[877.00,883.20] vol=2.1x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:35:00 | 894.46 | 884.31 | 0.00 | T1 1.5R @ 894.46 |
| Target hit | 2024-08-20 13:25:00 | 903.05 | 905.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:55:00 | 919.15 | 901.31 | 0.00 | ORB-long ORB[886.00,899.00] vol=3.2x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-08-21 11:00:00 | 915.02 | 903.16 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:35:00 | 868.40 | 849.85 | 0.00 | ORB-long ORB[836.00,846.45] vol=3.4x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:45:00 | 875.44 | 852.67 | 0.00 | T1 1.5R @ 875.44 |
| Target hit | 2024-09-03 15:20:00 | 888.35 | 864.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:35:00 | 890.45 | 884.14 | 0.00 | ORB-long ORB[875.30,886.00] vol=3.2x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:40:00 | 896.15 | 888.45 | 0.00 | T1 1.5R @ 896.15 |
| Target hit | 2024-09-04 11:50:00 | 891.40 | 896.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 928.10 | 939.51 | 0.00 | ORB-short ORB[935.05,948.15] vol=2.0x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 921.22 | 936.70 | 0.00 | T1 1.5R @ 921.22 |
| Stop hit — per-position SL triggered | 2024-09-06 10:30:00 | 928.10 | 932.69 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 963.50 | 976.22 | 0.00 | ORB-short ORB[980.40,991.30] vol=3.2x ATR=6.02 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 969.52 | 974.70 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 1026.00 | 1009.95 | 0.00 | ORB-long ORB[1006.25,1021.40] vol=3.7x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:00:00 | 1032.72 | 1013.29 | 0.00 | T1 1.5R @ 1032.72 |
| Stop hit — per-position SL triggered | 2024-09-26 11:10:00 | 1026.00 | 1015.26 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:45:00 | 976.20 | 972.92 | 0.00 | ORB-long ORB[965.10,975.40] vol=1.7x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-09-30 09:50:00 | 971.35 | 972.88 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:35:00 | 911.00 | 905.10 | 0.00 | ORB-long ORB[897.55,910.05] vol=1.7x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-10-10 09:40:00 | 906.90 | 905.26 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 921.00 | 914.35 | 0.00 | ORB-long ORB[906.10,918.75] vol=4.6x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:25:00 | 925.43 | 918.64 | 0.00 | T1 1.5R @ 925.43 |
| Stop hit — per-position SL triggered | 2024-10-11 11:40:00 | 921.00 | 919.74 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:50:00 | 980.80 | 984.19 | 0.00 | ORB-short ORB[985.90,994.20] vol=2.6x ATR=5.66 |
| Stop hit — per-position SL triggered | 2024-10-28 10:00:00 | 986.46 | 984.19 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1170.15 | 1179.76 | 0.00 | ORB-short ORB[1180.95,1190.00] vol=3.5x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:45:00 | 1162.02 | 1175.40 | 0.00 | T1 1.5R @ 1162.02 |
| Stop hit — per-position SL triggered | 2024-11-18 09:50:00 | 1170.15 | 1175.24 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:50:00 | 1488.00 | 1459.07 | 0.00 | ORB-long ORB[1433.80,1451.40] vol=1.9x ATR=8.42 |
| Stop hit — per-position SL triggered | 2024-12-09 11:00:00 | 1479.58 | 1460.16 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 1664.00 | 1639.70 | 0.00 | ORB-long ORB[1622.65,1645.80] vol=3.2x ATR=13.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:40:00 | 1683.76 | 1653.99 | 0.00 | T1 1.5R @ 1683.76 |
| Target hit | 2024-12-26 10:20:00 | 1679.90 | 1681.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2025-01-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:50:00 | 1463.25 | 1471.72 | 0.00 | ORB-short ORB[1472.65,1493.75] vol=2.2x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:05:00 | 1451.86 | 1465.34 | 0.00 | T1 1.5R @ 1451.86 |
| Target hit | 2025-01-21 14:40:00 | 1436.00 | 1434.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 1517.70 | 1542.24 | 0.00 | ORB-short ORB[1540.05,1562.40] vol=2.4x ATR=9.05 |
| Stop hit — per-position SL triggered | 2025-02-21 09:45:00 | 1526.75 | 1539.31 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 1591.70 | 1581.19 | 0.00 | ORB-long ORB[1566.55,1584.90] vol=3.9x ATR=8.40 |
| Stop hit — per-position SL triggered | 2025-03-12 09:40:00 | 1583.30 | 1582.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 10:50:00 | 919.55 | 2024-05-17 10:55:00 | 923.08 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-22 10:20:00 | 941.90 | 2024-05-22 10:25:00 | 936.12 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-06-06 09:50:00 | 800.75 | 2024-06-06 09:55:00 | 795.16 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2024-06-07 09:45:00 | 789.00 | 2024-06-07 10:05:00 | 795.31 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-06-07 09:45:00 | 789.00 | 2024-06-07 12:30:00 | 807.05 | TARGET_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2024-06-13 09:35:00 | 820.00 | 2024-06-13 09:40:00 | 816.18 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-06-21 10:45:00 | 831.00 | 2024-06-21 10:50:00 | 833.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-25 11:15:00 | 830.10 | 2024-06-25 11:20:00 | 827.04 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-06-25 11:15:00 | 830.10 | 2024-06-25 11:45:00 | 830.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:35:00 | 782.35 | 2024-06-27 10:40:00 | 786.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-07-04 09:50:00 | 815.00 | 2024-07-04 10:20:00 | 819.65 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-04 09:50:00 | 815.00 | 2024-07-04 15:20:00 | 845.00 | TARGET_HIT | 0.50 | 3.68% |
| SELL | retest1 | 2024-07-09 10:00:00 | 830.00 | 2024-07-09 10:15:00 | 825.97 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-09 10:00:00 | 830.00 | 2024-07-09 15:20:00 | 826.50 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-10 09:30:00 | 833.50 | 2024-07-10 09:35:00 | 837.34 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-10 09:30:00 | 833.50 | 2024-07-10 09:55:00 | 836.70 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-12 09:50:00 | 830.00 | 2024-07-12 11:10:00 | 833.23 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-13 09:55:00 | 894.30 | 2024-08-13 10:05:00 | 889.19 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-13 09:55:00 | 894.30 | 2024-08-13 15:20:00 | 831.10 | TARGET_HIT | 0.50 | 7.07% |
| BUY | retest1 | 2024-08-16 09:30:00 | 846.05 | 2024-08-16 09:35:00 | 853.40 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-08-16 09:30:00 | 846.05 | 2024-08-16 10:40:00 | 852.20 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2024-08-19 09:35:00 | 888.30 | 2024-08-19 09:45:00 | 884.15 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-20 09:30:00 | 889.00 | 2024-08-20 09:35:00 | 894.46 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-20 09:30:00 | 889.00 | 2024-08-20 13:25:00 | 903.05 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2024-08-21 10:55:00 | 919.15 | 2024-08-21 11:00:00 | 915.02 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-03 10:35:00 | 868.40 | 2024-09-03 10:45:00 | 875.44 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2024-09-03 10:35:00 | 868.40 | 2024-09-03 15:20:00 | 888.35 | TARGET_HIT | 0.50 | 2.30% |
| BUY | retest1 | 2024-09-04 09:35:00 | 890.45 | 2024-09-04 09:40:00 | 896.15 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-09-04 09:35:00 | 890.45 | 2024-09-04 11:50:00 | 891.40 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-09-06 09:45:00 | 928.10 | 2024-09-06 10:05:00 | 921.22 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-09-06 09:45:00 | 928.10 | 2024-09-06 10:30:00 | 928.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 09:50:00 | 963.50 | 2024-09-17 10:00:00 | 969.52 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-09-26 10:55:00 | 1026.00 | 2024-09-26 11:00:00 | 1032.72 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-09-26 10:55:00 | 1026.00 | 2024-09-26 11:10:00 | 1026.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-30 09:45:00 | 976.20 | 2024-09-30 09:50:00 | 971.35 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-10 09:35:00 | 911.00 | 2024-10-10 09:40:00 | 906.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-11 10:55:00 | 921.00 | 2024-10-11 11:25:00 | 925.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-11 10:55:00 | 921.00 | 2024-10-11 11:40:00 | 921.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-28 09:50:00 | 980.80 | 2024-10-28 10:00:00 | 986.46 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-11-18 09:30:00 | 1170.15 | 2024-11-18 09:45:00 | 1162.02 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-11-18 09:30:00 | 1170.15 | 2024-11-18 09:50:00 | 1170.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 10:50:00 | 1488.00 | 2024-12-09 11:00:00 | 1479.58 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-12-26 09:35:00 | 1664.00 | 2024-12-26 09:40:00 | 1683.76 | PARTIAL | 0.50 | 1.19% |
| BUY | retest1 | 2024-12-26 09:35:00 | 1664.00 | 2024-12-26 10:20:00 | 1679.90 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-01-21 09:50:00 | 1463.25 | 2025-01-21 10:05:00 | 1451.86 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-01-21 09:50:00 | 1463.25 | 2025-01-21 14:40:00 | 1436.00 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2025-02-21 09:40:00 | 1517.70 | 2025-02-21 09:45:00 | 1526.75 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-03-12 09:30:00 | 1591.70 | 2025-03-12 09:40:00 | 1583.30 | STOP_HIT | 1.00 | -0.53% |
