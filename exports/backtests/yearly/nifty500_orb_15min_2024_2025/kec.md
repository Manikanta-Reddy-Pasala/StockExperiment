# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35284 bars)
- **Last close:** 597.80
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
| PARTIAL | 16 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 26
- **Target hits / Stop hits / Partials:** 8 / 26 / 16
- **Avg / median % per leg:** 0.32% / 0.00%
- **Sum % (uncompounded):** 16.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 15 | 48.4% | 4 | 16 | 11 | 0.33% | 10.3% |
| BUY @ 2nd Alert (retest1) | 31 | 15 | 48.4% | 4 | 16 | 11 | 0.33% | 10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 9 | 47.4% | 4 | 10 | 5 | 0.30% | 5.8% |
| SELL @ 2nd Alert (retest1) | 19 | 9 | 47.4% | 4 | 10 | 5 | 0.30% | 5.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 50 | 24 | 48.0% | 8 | 26 | 16 | 0.32% | 16.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:40:00 | 728.85 | 722.43 | 0.00 | ORB-long ORB[717.00,723.85] vol=3.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-05-15 09:45:00 | 726.89 | 724.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 11:05:00 | 779.00 | 772.38 | 0.00 | ORB-long ORB[761.35,768.35] vol=7.1x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 12:00:00 | 784.97 | 775.01 | 0.00 | T1 1.5R @ 784.97 |
| Target hit | 2024-05-16 13:25:00 | 786.95 | 787.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-05-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:20:00 | 797.30 | 792.61 | 0.00 | ORB-long ORB[787.00,796.00] vol=2.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-05-21 10:25:00 | 794.01 | 792.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:25:00 | 790.70 | 794.58 | 0.00 | ORB-short ORB[793.00,799.60] vol=3.3x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:40:00 | 786.89 | 793.71 | 0.00 | T1 1.5R @ 786.89 |
| Stop hit — per-position SL triggered | 2024-05-23 10:45:00 | 790.70 | 793.61 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:50:00 | 769.55 | 771.94 | 0.00 | ORB-short ORB[771.00,779.00] vol=1.8x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:30:00 | 765.93 | 771.37 | 0.00 | T1 1.5R @ 765.93 |
| Target hit | 2024-05-28 13:05:00 | 763.80 | 763.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 758.25 | 761.53 | 0.00 | ORB-short ORB[760.00,765.90] vol=1.7x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 761.22 | 758.63 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:20:00 | 747.00 | 752.30 | 0.00 | ORB-short ORB[753.15,761.35] vol=1.7x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-05-31 11:40:00 | 750.12 | 751.05 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 865.40 | 873.47 | 0.00 | ORB-short ORB[870.15,878.35] vol=1.7x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 12:30:00 | 861.86 | 871.20 | 0.00 | T1 1.5R @ 861.86 |
| Target hit | 2024-06-25 15:20:00 | 859.25 | 865.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:30:00 | 865.95 | 858.86 | 0.00 | ORB-long ORB[853.65,861.80] vol=2.6x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-06-26 10:45:00 | 862.61 | 859.54 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 937.00 | 932.87 | 0.00 | ORB-long ORB[921.40,934.90] vol=3.3x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 932.25 | 934.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:30:00 | 885.00 | 878.08 | 0.00 | ORB-long ORB[872.40,880.00] vol=3.4x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-07-15 10:35:00 | 882.25 | 878.64 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:05:00 | 891.95 | 886.05 | 0.00 | ORB-long ORB[882.35,887.80] vol=3.1x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:15:00 | 895.75 | 888.88 | 0.00 | T1 1.5R @ 895.75 |
| Stop hit — per-position SL triggered | 2024-07-16 10:20:00 | 891.95 | 889.06 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 872.00 | 874.84 | 0.00 | ORB-short ORB[872.65,884.45] vol=2.4x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-07-23 11:45:00 | 874.46 | 874.63 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 895.00 | 891.56 | 0.00 | ORB-long ORB[885.00,893.90] vol=3.9x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:40:00 | 898.94 | 897.06 | 0.00 | T1 1.5R @ 898.94 |
| Target hit | 2024-07-31 15:20:00 | 926.55 | 917.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 847.70 | 839.40 | 0.00 | ORB-long ORB[830.50,841.00] vol=1.9x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 12:30:00 | 853.04 | 843.11 | 0.00 | T1 1.5R @ 853.04 |
| Stop hit — per-position SL triggered | 2024-08-07 13:00:00 | 847.70 | 843.90 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:15:00 | 856.25 | 847.11 | 0.00 | ORB-long ORB[842.50,849.95] vol=1.7x ATR=4.16 |
| Stop hit — per-position SL triggered | 2024-08-08 13:00:00 | 852.09 | 853.74 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 812.00 | 819.37 | 0.00 | ORB-short ORB[821.20,832.25] vol=1.9x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-08-14 11:10:00 | 814.97 | 818.34 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 11:00:00 | 866.85 | 860.94 | 0.00 | ORB-long ORB[859.30,864.95] vol=1.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-08-28 11:10:00 | 864.49 | 861.16 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:50:00 | 975.65 | 979.24 | 0.00 | ORB-short ORB[975.95,989.00] vol=1.6x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-09-10 10:00:00 | 979.36 | 978.98 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 11:15:00 | 956.50 | 964.73 | 0.00 | ORB-short ORB[965.15,978.70] vol=2.7x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-09-11 13:10:00 | 959.00 | 961.96 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:05:00 | 979.25 | 984.36 | 0.00 | ORB-short ORB[981.75,995.65] vol=1.7x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 981.83 | 984.08 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:50:00 | 994.90 | 1011.12 | 0.00 | ORB-short ORB[1014.75,1026.65] vol=1.6x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 986.06 | 1002.36 | 0.00 | T1 1.5R @ 986.06 |
| Target hit | 2024-10-07 15:00:00 | 993.25 | 992.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-11-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:35:00 | 1024.55 | 1013.99 | 0.00 | ORB-long ORB[1005.00,1015.95] vol=4.6x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:45:00 | 1031.18 | 1020.23 | 0.00 | T1 1.5R @ 1031.18 |
| Stop hit — per-position SL triggered | 2024-11-25 13:00:00 | 1024.55 | 1022.00 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:05:00 | 1039.30 | 1024.49 | 0.00 | ORB-long ORB[1013.40,1028.10] vol=1.9x ATR=6.07 |
| Stop hit — per-position SL triggered | 2024-11-27 10:10:00 | 1033.23 | 1026.99 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 1082.85 | 1076.47 | 0.00 | ORB-long ORB[1065.00,1080.70] vol=1.8x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:10:00 | 1089.62 | 1080.65 | 0.00 | T1 1.5R @ 1089.62 |
| Target hit | 2024-12-03 11:10:00 | 1087.85 | 1091.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2024-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:50:00 | 1199.30 | 1211.30 | 0.00 | ORB-short ORB[1209.00,1223.70] vol=2.1x ATR=5.80 |
| Stop hit — per-position SL triggered | 2024-12-11 12:50:00 | 1205.10 | 1207.92 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-12-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:25:00 | 1176.95 | 1185.55 | 0.00 | ORB-short ORB[1181.55,1195.90] vol=2.0x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-12-17 10:50:00 | 1180.47 | 1183.94 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:10:00 | 1215.00 | 1204.17 | 0.00 | ORB-long ORB[1194.20,1212.00] vol=1.5x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:10:00 | 1223.95 | 1210.09 | 0.00 | T1 1.5R @ 1223.95 |
| Stop hit — per-position SL triggered | 2025-01-01 11:55:00 | 1215.00 | 1212.35 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:45:00 | 819.10 | 818.94 | 0.00 | ORB-long ORB[807.50,816.95] vol=1.6x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:10:00 | 825.28 | 819.24 | 0.00 | T1 1.5R @ 825.28 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 819.10 | 819.60 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 821.40 | 812.17 | 0.00 | ORB-long ORB[806.00,815.30] vol=1.7x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:10:00 | 827.42 | 814.65 | 0.00 | T1 1.5R @ 827.42 |
| Stop hit — per-position SL triggered | 2025-02-07 12:10:00 | 821.40 | 818.78 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:45:00 | 746.80 | 741.84 | 0.00 | ORB-long ORB[735.90,745.00] vol=1.5x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 11:35:00 | 751.17 | 743.76 | 0.00 | T1 1.5R @ 751.17 |
| Target hit | 2025-03-19 15:20:00 | 763.70 | 754.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 10:55:00 | 760.55 | 767.51 | 0.00 | ORB-short ORB[770.00,778.95] vol=1.8x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 11:15:00 | 755.65 | 766.84 | 0.00 | T1 1.5R @ 755.65 |
| Target hit | 2025-04-03 15:20:00 | 727.00 | 745.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:40:00 | 727.25 | 722.57 | 0.00 | ORB-long ORB[717.00,726.25] vol=1.5x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:50:00 | 730.76 | 723.46 | 0.00 | T1 1.5R @ 730.76 |
| Stop hit — per-position SL triggered | 2025-04-22 11:35:00 | 727.25 | 726.73 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:35:00 | 728.35 | 720.47 | 0.00 | ORB-long ORB[713.00,723.20] vol=1.6x ATR=4.35 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 724.00 | 720.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:40:00 | 728.85 | 2024-05-15 09:45:00 | 726.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-16 11:05:00 | 779.00 | 2024-05-16 12:00:00 | 784.97 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-05-16 11:05:00 | 779.00 | 2024-05-16 13:25:00 | 786.95 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-05-21 10:20:00 | 797.30 | 2024-05-21 10:25:00 | 794.01 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-23 10:25:00 | 790.70 | 2024-05-23 10:40:00 | 786.89 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-23 10:25:00 | 790.70 | 2024-05-23 10:45:00 | 790.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 10:50:00 | 769.55 | 2024-05-28 11:30:00 | 765.93 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-28 10:50:00 | 769.55 | 2024-05-28 13:05:00 | 763.80 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-05-30 09:30:00 | 758.25 | 2024-05-30 10:35:00 | 761.22 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-31 10:20:00 | 747.00 | 2024-05-31 11:40:00 | 750.12 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-25 11:15:00 | 865.40 | 2024-06-25 12:30:00 | 861.86 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-25 11:15:00 | 865.40 | 2024-06-25 15:20:00 | 859.25 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2024-06-26 10:30:00 | 865.95 | 2024-06-26 10:45:00 | 862.61 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-04 09:45:00 | 937.00 | 2024-07-04 10:10:00 | 932.25 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-07-15 10:30:00 | 885.00 | 2024-07-15 10:35:00 | 882.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-16 10:05:00 | 891.95 | 2024-07-16 10:15:00 | 895.75 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-16 10:05:00 | 891.95 | 2024-07-16 10:20:00 | 891.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 872.00 | 2024-07-23 11:45:00 | 874.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-31 09:30:00 | 895.00 | 2024-07-31 09:40:00 | 898.94 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-31 09:30:00 | 895.00 | 2024-07-31 15:20:00 | 926.55 | TARGET_HIT | 0.50 | 3.53% |
| BUY | retest1 | 2024-08-07 10:45:00 | 847.70 | 2024-08-07 12:30:00 | 853.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-07 10:45:00 | 847.70 | 2024-08-07 13:00:00 | 847.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:15:00 | 856.25 | 2024-08-08 13:00:00 | 852.09 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-14 10:50:00 | 812.00 | 2024-08-14 11:10:00 | 814.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-28 11:00:00 | 866.85 | 2024-08-28 11:10:00 | 864.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-10 09:50:00 | 975.65 | 2024-09-10 10:00:00 | 979.36 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-11 11:15:00 | 956.50 | 2024-09-11 13:10:00 | 959.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-13 11:05:00 | 979.25 | 2024-09-13 11:20:00 | 981.83 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-07 09:50:00 | 994.90 | 2024-10-07 10:15:00 | 986.06 | PARTIAL | 0.50 | 0.89% |
| SELL | retest1 | 2024-10-07 09:50:00 | 994.90 | 2024-10-07 15:00:00 | 993.25 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-11-25 10:35:00 | 1024.55 | 2024-11-25 11:45:00 | 1031.18 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-25 10:35:00 | 1024.55 | 2024-11-25 13:00:00 | 1024.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 10:05:00 | 1039.30 | 2024-11-27 10:10:00 | 1033.23 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-12-03 09:30:00 | 1082.85 | 2024-12-03 10:10:00 | 1089.62 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-03 09:30:00 | 1082.85 | 2024-12-03 11:10:00 | 1087.85 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-11 10:50:00 | 1199.30 | 2024-12-11 12:50:00 | 1205.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-17 10:25:00 | 1176.95 | 2024-12-17 10:50:00 | 1180.47 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-01 10:10:00 | 1215.00 | 2025-01-01 11:10:00 | 1223.95 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-01-01 10:10:00 | 1215.00 | 2025-01-01 11:55:00 | 1215.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 10:45:00 | 819.10 | 2025-01-31 11:10:00 | 825.28 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-01-31 10:45:00 | 819.10 | 2025-01-31 11:30:00 | 819.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 11:00:00 | 821.40 | 2025-02-07 11:10:00 | 827.42 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-02-07 11:00:00 | 821.40 | 2025-02-07 12:10:00 | 821.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 10:45:00 | 746.80 | 2025-03-19 11:35:00 | 751.17 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-19 10:45:00 | 746.80 | 2025-03-19 15:20:00 | 763.70 | TARGET_HIT | 0.50 | 2.26% |
| SELL | retest1 | 2025-04-03 10:55:00 | 760.55 | 2025-04-03 11:15:00 | 755.65 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-04-03 10:55:00 | 760.55 | 2025-04-03 15:20:00 | 727.00 | TARGET_HIT | 0.50 | 4.41% |
| BUY | retest1 | 2025-04-22 10:40:00 | 727.25 | 2025-04-22 10:50:00 | 730.76 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-22 10:40:00 | 727.25 | 2025-04-22 11:35:00 | 727.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 09:35:00 | 728.35 | 2025-04-28 09:40:00 | 724.00 | STOP_HIT | 1.00 | -0.60% |
