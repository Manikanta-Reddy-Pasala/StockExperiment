# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 487.90
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
| ENTRY1 | 45 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 16 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 29
- **Target hits / Stop hits / Partials:** 16 / 29 / 24
- **Avg / median % per leg:** 0.29% / 0.41%
- **Sum % (uncompounded):** 19.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 18 | 56.2% | 6 | 14 | 12 | 0.35% | 11.1% |
| BUY @ 2nd Alert (retest1) | 32 | 18 | 56.2% | 6 | 14 | 12 | 0.35% | 11.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 22 | 59.5% | 10 | 15 | 12 | 0.24% | 8.9% |
| SELL @ 2nd Alert (retest1) | 37 | 22 | 59.5% | 10 | 15 | 12 | 0.24% | 8.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 69 | 40 | 58.0% | 16 | 29 | 24 | 0.29% | 19.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 655.00 | 651.55 | 0.00 | ORB-long ORB[648.40,654.80] vol=1.8x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:40:00 | 659.18 | 655.16 | 0.00 | T1 1.5R @ 659.18 |
| Stop hit — per-position SL triggered | 2024-05-15 13:00:00 | 655.00 | 659.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:45:00 | 642.05 | 643.53 | 0.00 | ORB-short ORB[643.80,647.05] vol=3.6x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 12:30:00 | 639.24 | 642.02 | 0.00 | T1 1.5R @ 639.24 |
| Target hit | 2024-05-17 13:45:00 | 639.50 | 639.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 639.45 | 643.11 | 0.00 | ORB-short ORB[641.05,646.55] vol=3.8x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:55:00 | 634.93 | 639.27 | 0.00 | T1 1.5R @ 634.93 |
| Target hit | 2024-05-21 11:25:00 | 635.65 | 635.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2024-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:05:00 | 626.55 | 626.50 | 0.00 | ORB-long ORB[622.35,626.50] vol=19.0x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:15:00 | 629.37 | 626.60 | 0.00 | T1 1.5R @ 629.37 |
| Target hit | 2024-05-23 15:20:00 | 627.25 | 628.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:15:00 | 634.00 | 633.71 | 0.00 | ORB-long ORB[629.05,631.50] vol=12.1x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-05-24 10:20:00 | 632.42 | 633.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:50:00 | 642.20 | 640.37 | 0.00 | ORB-long ORB[634.00,640.95] vol=1.8x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 639.57 | 640.46 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:55:00 | 646.00 | 641.80 | 0.00 | ORB-long ORB[636.55,642.80] vol=2.5x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:00:00 | 649.11 | 644.04 | 0.00 | T1 1.5R @ 649.11 |
| Target hit | 2024-05-29 15:20:00 | 650.70 | 649.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:40:00 | 645.35 | 649.22 | 0.00 | ORB-short ORB[645.85,654.45] vol=2.6x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:05:00 | 641.21 | 648.15 | 0.00 | T1 1.5R @ 641.21 |
| Stop hit — per-position SL triggered | 2024-05-30 10:30:00 | 645.35 | 647.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 626.95 | 628.67 | 0.00 | ORB-short ORB[630.05,633.90] vol=3.3x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-06-06 10:20:00 | 629.77 | 628.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:45:00 | 644.90 | 643.34 | 0.00 | ORB-long ORB[637.10,643.15] vol=1.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-06-07 09:55:00 | 642.29 | 643.50 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 676.00 | 680.67 | 0.00 | ORB-short ORB[680.00,683.05] vol=1.9x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:05:00 | 673.23 | 677.50 | 0.00 | T1 1.5R @ 673.23 |
| Target hit | 2024-06-12 11:25:00 | 672.00 | 670.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2024-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:50:00 | 733.85 | 728.04 | 0.00 | ORB-long ORB[722.05,728.15] vol=1.8x ATR=3.03 |
| Stop hit — per-position SL triggered | 2024-06-26 09:55:00 | 730.82 | 728.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 813.95 | 817.46 | 0.00 | ORB-short ORB[816.30,826.70] vol=1.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-07-04 10:20:00 | 817.37 | 817.17 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 804.10 | 811.17 | 0.00 | ORB-short ORB[811.55,817.85] vol=1.6x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:55:00 | 799.70 | 807.36 | 0.00 | T1 1.5R @ 799.70 |
| Target hit | 2024-07-08 14:20:00 | 799.05 | 798.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 809.20 | 816.59 | 0.00 | ORB-short ORB[814.55,823.20] vol=2.0x ATR=4.16 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 813.36 | 816.40 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:10:00 | 823.70 | 825.74 | 0.00 | ORB-short ORB[824.10,834.00] vol=2.0x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-07-11 10:55:00 | 827.35 | 826.07 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 869.40 | 859.05 | 0.00 | ORB-long ORB[838.80,850.45] vol=2.9x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:40:00 | 875.75 | 864.12 | 0.00 | T1 1.5R @ 875.75 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 869.40 | 864.48 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:30:00 | 887.00 | 882.94 | 0.00 | ORB-long ORB[872.30,884.60] vol=9.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-07-25 10:35:00 | 883.12 | 882.95 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1007.50 | 1005.28 | 0.00 | ORB-long ORB[998.55,1006.65] vol=1.6x ATR=6.04 |
| Stop hit — per-position SL triggered | 2024-08-16 09:55:00 | 1001.46 | 1004.84 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 983.70 | 975.67 | 0.00 | ORB-long ORB[968.80,979.50] vol=2.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-08-20 11:05:00 | 979.61 | 976.08 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 986.40 | 979.99 | 0.00 | ORB-long ORB[975.50,981.75] vol=2.4x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 09:35:00 | 991.94 | 983.71 | 0.00 | T1 1.5R @ 991.94 |
| Stop hit — per-position SL triggered | 2024-08-21 09:40:00 | 986.40 | 984.14 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:05:00 | 1199.25 | 1214.23 | 0.00 | ORB-short ORB[1210.95,1227.95] vol=1.7x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:25:00 | 1192.29 | 1210.47 | 0.00 | T1 1.5R @ 1192.29 |
| Target hit | 2024-09-16 15:20:00 | 1170.75 | 1184.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 1151.10 | 1156.69 | 0.00 | ORB-short ORB[1157.10,1167.50] vol=2.0x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:25:00 | 1143.42 | 1147.27 | 0.00 | T1 1.5R @ 1143.42 |
| Target hit | 2024-09-18 11:10:00 | 1147.85 | 1146.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-09-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:50:00 | 1168.60 | 1157.84 | 0.00 | ORB-long ORB[1149.55,1163.70] vol=1.8x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:55:00 | 1176.35 | 1161.02 | 0.00 | T1 1.5R @ 1176.35 |
| Target hit | 2024-09-19 11:50:00 | 1212.45 | 1221.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 1174.05 | 1183.76 | 0.00 | ORB-short ORB[1186.00,1203.00] vol=3.3x ATR=6.58 |
| Stop hit — per-position SL triggered | 2024-10-07 10:05:00 | 1180.63 | 1182.68 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 1185.20 | 1184.96 | 0.00 | ORB-long ORB[1162.25,1175.00] vol=1.7x ATR=6.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:10:00 | 1194.44 | 1190.16 | 0.00 | T1 1.5R @ 1194.44 |
| Target hit | 2024-10-08 11:15:00 | 1190.00 | 1190.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2024-10-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:25:00 | 1175.35 | 1178.68 | 0.00 | ORB-short ORB[1187.20,1197.00] vol=3.6x ATR=5.17 |
| Stop hit — per-position SL triggered | 2024-10-09 12:25:00 | 1180.52 | 1174.35 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:40:00 | 1198.15 | 1200.09 | 0.00 | ORB-short ORB[1200.10,1215.00] vol=6.2x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-10-15 11:10:00 | 1202.08 | 1199.90 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:35:00 | 1229.95 | 1246.99 | 0.00 | ORB-short ORB[1255.10,1270.45] vol=1.9x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-10-25 10:45:00 | 1235.32 | 1242.95 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:55:00 | 1255.80 | 1263.67 | 0.00 | ORB-short ORB[1262.00,1277.95] vol=2.2x ATR=6.88 |
| Stop hit — per-position SL triggered | 2024-10-29 12:35:00 | 1262.68 | 1258.59 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-11-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 10:10:00 | 1334.95 | 1330.96 | 0.00 | ORB-long ORB[1315.00,1331.00] vol=8.9x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:20:00 | 1343.85 | 1336.03 | 0.00 | T1 1.5R @ 1343.85 |
| Target hit | 2024-11-05 10:30:00 | 1336.00 | 1336.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2024-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:55:00 | 1310.05 | 1315.42 | 0.00 | ORB-short ORB[1312.05,1325.00] vol=1.6x ATR=5.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:10:00 | 1301.61 | 1313.46 | 0.00 | T1 1.5R @ 1301.61 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 1310.05 | 1313.41 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 10:05:00 | 1263.15 | 1272.22 | 0.00 | ORB-short ORB[1269.35,1285.90] vol=3.0x ATR=6.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 10:35:00 | 1253.24 | 1265.93 | 0.00 | T1 1.5R @ 1253.24 |
| Target hit | 2024-11-11 12:45:00 | 1257.40 | 1257.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 1267.90 | 1278.15 | 0.00 | ORB-short ORB[1278.00,1292.20] vol=2.5x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:15:00 | 1262.73 | 1276.65 | 0.00 | T1 1.5R @ 1262.73 |
| Target hit | 2024-11-27 15:20:00 | 1258.25 | 1262.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:45:00 | 1252.70 | 1255.19 | 0.00 | ORB-short ORB[1254.05,1270.00] vol=1.8x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-12-18 10:50:00 | 1256.32 | 1255.46 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:50:00 | 1154.30 | 1144.96 | 0.00 | ORB-long ORB[1137.80,1152.70] vol=2.5x ATR=5.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:00:00 | 1162.15 | 1146.63 | 0.00 | T1 1.5R @ 1162.15 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 1154.30 | 1147.70 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:10:00 | 1103.85 | 1110.21 | 0.00 | ORB-short ORB[1106.20,1119.20] vol=3.3x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-01-03 12:10:00 | 1107.62 | 1107.49 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:45:00 | 1070.80 | 1079.90 | 0.00 | ORB-short ORB[1092.40,1107.00] vol=1.6x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 12:25:00 | 1064.24 | 1075.24 | 0.00 | T1 1.5R @ 1064.24 |
| Target hit | 2025-01-10 15:20:00 | 1062.50 | 1066.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-01-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:40:00 | 1062.40 | 1073.37 | 0.00 | ORB-short ORB[1077.60,1091.35] vol=6.2x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:55:00 | 1056.34 | 1059.80 | 0.00 | T1 1.5R @ 1056.34 |
| Target hit | 2025-01-21 12:35:00 | 1055.45 | 1054.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1020.05 | 1008.78 | 0.00 | ORB-long ORB[1000.10,1014.65] vol=1.6x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:40:00 | 1028.59 | 1011.24 | 0.00 | T1 1.5R @ 1028.59 |
| Target hit | 2025-01-30 14:45:00 | 1037.85 | 1037.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:45:00 | 1150.15 | 1156.36 | 0.00 | ORB-short ORB[1155.40,1171.10] vol=6.7x ATR=5.58 |
| Stop hit — per-position SL triggered | 2025-02-07 10:55:00 | 1155.73 | 1156.32 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 11:00:00 | 1135.80 | 1108.87 | 0.00 | ORB-long ORB[1103.00,1116.35] vol=3.7x ATR=8.33 |
| Stop hit — per-position SL triggered | 2025-02-18 11:05:00 | 1127.47 | 1111.74 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:50:00 | 1163.15 | 1158.34 | 0.00 | ORB-long ORB[1147.00,1161.30] vol=2.5x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:10:00 | 1171.49 | 1159.34 | 0.00 | T1 1.5R @ 1171.49 |
| Stop hit — per-position SL triggered | 2025-03-07 12:00:00 | 1163.15 | 1161.25 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:20:00 | 1138.20 | 1147.06 | 0.00 | ORB-short ORB[1150.00,1163.00] vol=1.5x ATR=4.56 |
| Stop hit — per-position SL triggered | 2025-03-12 10:35:00 | 1142.76 | 1146.07 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-03-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:10:00 | 1194.85 | 1181.00 | 0.00 | ORB-long ORB[1177.35,1192.50] vol=1.8x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 11:15:00 | 1203.13 | 1184.17 | 0.00 | T1 1.5R @ 1203.13 |
| Stop hit — per-position SL triggered | 2025-03-20 11:40:00 | 1194.85 | 1191.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:35:00 | 655.00 | 2024-05-15 09:40:00 | 659.18 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-05-15 09:35:00 | 655.00 | 2024-05-15 13:00:00 | 655.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-17 10:45:00 | 642.05 | 2024-05-17 12:30:00 | 639.24 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-17 10:45:00 | 642.05 | 2024-05-17 13:45:00 | 639.50 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-21 09:30:00 | 639.45 | 2024-05-21 09:55:00 | 634.93 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-05-21 09:30:00 | 639.45 | 2024-05-21 11:25:00 | 635.65 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2024-05-23 11:05:00 | 626.55 | 2024-05-23 11:15:00 | 629.37 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-23 11:05:00 | 626.55 | 2024-05-23 15:20:00 | 627.25 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-05-24 10:15:00 | 634.00 | 2024-05-24 10:20:00 | 632.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-28 09:50:00 | 642.20 | 2024-05-28 10:15:00 | 639.57 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-29 09:55:00 | 646.00 | 2024-05-29 10:00:00 | 649.11 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-29 09:55:00 | 646.00 | 2024-05-29 15:20:00 | 650.70 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-05-30 09:40:00 | 645.35 | 2024-05-30 10:05:00 | 641.21 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-30 09:40:00 | 645.35 | 2024-05-30 10:30:00 | 645.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-06 10:05:00 | 626.95 | 2024-06-06 10:20:00 | 629.77 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-07 09:45:00 | 644.90 | 2024-06-07 09:55:00 | 642.29 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-12 09:50:00 | 676.00 | 2024-06-12 10:05:00 | 673.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-12 09:50:00 | 676.00 | 2024-06-12 11:25:00 | 672.00 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-26 09:50:00 | 733.85 | 2024-06-26 09:55:00 | 730.82 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-04 09:45:00 | 813.95 | 2024-07-04 10:20:00 | 817.37 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-08 09:40:00 | 804.10 | 2024-07-08 09:55:00 | 799.70 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-08 09:40:00 | 804.10 | 2024-07-08 14:20:00 | 799.05 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-10 10:35:00 | 809.20 | 2024-07-10 10:40:00 | 813.36 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-07-11 10:10:00 | 823.70 | 2024-07-11 10:55:00 | 827.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-16 10:30:00 | 869.40 | 2024-07-16 10:40:00 | 875.75 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-16 10:30:00 | 869.40 | 2024-07-16 10:45:00 | 869.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 10:30:00 | 887.00 | 2024-07-25 10:35:00 | 883.12 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-16 09:30:00 | 1007.50 | 2024-08-16 09:55:00 | 1001.46 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-08-20 10:55:00 | 983.70 | 2024-08-20 11:05:00 | 979.61 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-21 09:30:00 | 986.40 | 2024-08-21 09:35:00 | 991.94 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-21 09:30:00 | 986.40 | 2024-08-21 09:40:00 | 986.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 11:05:00 | 1199.25 | 2024-09-16 11:25:00 | 1192.29 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-16 11:05:00 | 1199.25 | 2024-09-16 15:20:00 | 1170.75 | TARGET_HIT | 0.50 | 2.38% |
| SELL | retest1 | 2024-09-18 09:35:00 | 1151.10 | 2024-09-18 10:25:00 | 1143.42 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-09-18 09:35:00 | 1151.10 | 2024-09-18 11:10:00 | 1147.85 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-09-19 10:50:00 | 1168.60 | 2024-09-19 10:55:00 | 1176.35 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-09-19 10:50:00 | 1168.60 | 2024-09-19 11:50:00 | 1212.45 | TARGET_HIT | 0.50 | 3.75% |
| SELL | retest1 | 2024-10-07 09:55:00 | 1174.05 | 2024-10-07 10:05:00 | 1180.63 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-10-08 11:05:00 | 1185.20 | 2024-10-08 11:10:00 | 1194.44 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-10-08 11:05:00 | 1185.20 | 2024-10-08 11:15:00 | 1190.00 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-09 10:25:00 | 1175.35 | 2024-10-09 12:25:00 | 1180.52 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-15 10:40:00 | 1198.15 | 2024-10-15 11:10:00 | 1202.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-25 10:35:00 | 1229.95 | 2024-10-25 10:45:00 | 1235.32 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-29 09:55:00 | 1255.80 | 2024-10-29 12:35:00 | 1262.68 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-11-05 10:10:00 | 1334.95 | 2024-11-05 10:20:00 | 1343.85 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-11-05 10:10:00 | 1334.95 | 2024-11-05 10:30:00 | 1336.00 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-11-07 09:55:00 | 1310.05 | 2024-11-07 10:10:00 | 1301.61 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-11-07 09:55:00 | 1310.05 | 2024-11-07 10:15:00 | 1310.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-11 10:05:00 | 1263.15 | 2024-11-11 10:35:00 | 1253.24 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-11-11 10:05:00 | 1263.15 | 2024-11-11 12:45:00 | 1257.40 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-27 11:00:00 | 1267.90 | 2024-11-27 11:15:00 | 1262.73 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-11-27 11:00:00 | 1267.90 | 2024-11-27 15:20:00 | 1258.25 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2024-12-18 10:45:00 | 1252.70 | 2024-12-18 10:50:00 | 1256.32 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-27 10:50:00 | 1154.30 | 2024-12-27 11:00:00 | 1162.15 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-27 10:50:00 | 1154.30 | 2024-12-27 11:10:00 | 1154.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 11:10:00 | 1103.85 | 2025-01-03 12:10:00 | 1107.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-10 10:45:00 | 1070.80 | 2025-01-10 12:25:00 | 1064.24 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-10 10:45:00 | 1070.80 | 2025-01-10 15:20:00 | 1062.50 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-01-21 10:40:00 | 1062.40 | 2025-01-21 10:55:00 | 1056.34 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-21 10:40:00 | 1062.40 | 2025-01-21 12:35:00 | 1055.45 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1020.05 | 2025-01-30 09:40:00 | 1028.59 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1020.05 | 2025-01-30 14:45:00 | 1037.85 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2025-02-07 10:45:00 | 1150.15 | 2025-02-07 10:55:00 | 1155.73 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-02-18 11:00:00 | 1135.80 | 2025-02-18 11:05:00 | 1127.47 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-03-07 10:50:00 | 1163.15 | 2025-03-07 11:10:00 | 1171.49 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-03-07 10:50:00 | 1163.15 | 2025-03-07 12:00:00 | 1163.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 10:20:00 | 1138.20 | 2025-03-12 10:35:00 | 1142.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-20 11:10:00 | 1194.85 | 2025-03-20 11:15:00 | 1203.13 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-03-20 11:10:00 | 1194.85 | 2025-03-20 11:40:00 | 1194.85 | STOP_HIT | 0.50 | 0.00% |
