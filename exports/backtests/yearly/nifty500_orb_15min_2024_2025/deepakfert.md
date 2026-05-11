# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35446 bars)
- **Last close:** 1342.00
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
| ENTRY1 | 44 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 36
- **Target hits / Stop hits / Partials:** 8 / 36 / 17
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 8.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 19 | 47.5% | 7 | 21 | 12 | 0.26% | 10.2% |
| BUY @ 2nd Alert (retest1) | 40 | 19 | 47.5% | 7 | 21 | 12 | 0.26% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 6 | 28.6% | 1 | 15 | 5 | -0.10% | -2.1% |
| SELL @ 2nd Alert (retest1) | 21 | 6 | 28.6% | 1 | 15 | 5 | -0.10% | -2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 25 | 41.0% | 8 | 36 | 17 | 0.13% | 8.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:30:00 | 514.80 | 522.52 | 0.00 | ORB-short ORB[530.00,536.60] vol=2.5x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 518.25 | 520.26 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:05:00 | 554.10 | 554.68 | 0.00 | ORB-short ORB[554.45,559.45] vol=2.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-05-16 11:10:00 | 555.89 | 554.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-18 09:50:00 | 554.65 | 554.93 | 0.00 | ORB-short ORB[556.15,560.95] vol=3.0x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 555.95 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 4 — SELL (started 2024-05-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:30:00 | 544.25 | 549.71 | 0.00 | ORB-short ORB[549.05,555.95] vol=1.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-05-21 11:10:00 | 546.44 | 547.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 551.10 | 554.72 | 0.00 | ORB-short ORB[551.95,556.95] vol=2.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 547.28 | 553.72 | 0.00 | T1 1.5R @ 547.28 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 551.10 | 553.05 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 549.30 | 555.02 | 0.00 | ORB-short ORB[556.00,559.50] vol=2.3x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-05-23 10:45:00 | 550.87 | 554.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:40:00 | 552.80 | 550.23 | 0.00 | ORB-long ORB[547.05,551.40] vol=1.8x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 09:45:00 | 555.62 | 551.42 | 0.00 | T1 1.5R @ 555.62 |
| Stop hit — per-position SL triggered | 2024-05-24 09:50:00 | 552.80 | 551.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 547.30 | 550.90 | 0.00 | ORB-short ORB[552.00,556.65] vol=4.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 12:05:00 | 544.42 | 548.21 | 0.00 | T1 1.5R @ 544.42 |
| Target hit | 2024-05-28 15:20:00 | 546.90 | 547.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:30:00 | 561.90 | 559.69 | 0.00 | ORB-long ORB[556.10,560.40] vol=1.8x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 565.55 | 563.59 | 0.00 | T1 1.5R @ 565.55 |
| Target hit | 2024-06-06 11:30:00 | 569.45 | 570.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-06-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:50:00 | 578.55 | 575.89 | 0.00 | ORB-long ORB[571.80,576.85] vol=3.2x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:35:00 | 581.73 | 578.47 | 0.00 | T1 1.5R @ 581.73 |
| Target hit | 2024-06-07 13:05:00 | 579.05 | 579.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-06-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:05:00 | 592.50 | 587.75 | 0.00 | ORB-long ORB[581.60,589.50] vol=4.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:10:00 | 596.33 | 588.56 | 0.00 | T1 1.5R @ 596.33 |
| Stop hit — per-position SL triggered | 2024-06-10 10:35:00 | 592.50 | 590.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 11:10:00 | 601.95 | 599.04 | 0.00 | ORB-long ORB[593.60,599.90] vol=1.8x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:15:00 | 605.93 | 599.64 | 0.00 | T1 1.5R @ 605.93 |
| Stop hit — per-position SL triggered | 2024-06-18 11:45:00 | 601.95 | 601.10 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:35:00 | 680.00 | 674.78 | 0.00 | ORB-long ORB[667.25,676.90] vol=3.9x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:45:00 | 686.50 | 677.24 | 0.00 | T1 1.5R @ 686.50 |
| Target hit | 2024-06-26 13:15:00 | 686.35 | 686.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:45:00 | 690.35 | 683.50 | 0.00 | ORB-long ORB[679.65,685.45] vol=1.7x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-07-01 13:40:00 | 686.06 | 688.83 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 722.75 | 718.37 | 0.00 | ORB-long ORB[711.40,720.35] vol=1.5x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:40:00 | 728.38 | 722.27 | 0.00 | T1 1.5R @ 728.38 |
| Target hit | 2024-07-03 11:00:00 | 741.70 | 741.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:45:00 | 772.75 | 769.33 | 0.00 | ORB-long ORB[765.00,772.70] vol=2.3x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-07-11 09:50:00 | 768.85 | 769.41 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 779.30 | 769.89 | 0.00 | ORB-long ORB[754.25,765.55] vol=6.6x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 775.07 | 771.43 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:35:00 | 779.80 | 772.34 | 0.00 | ORB-long ORB[768.00,779.55] vol=2.5x ATR=5.09 |
| Stop hit — per-position SL triggered | 2024-07-23 10:45:00 | 774.71 | 773.05 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:40:00 | 947.95 | 963.61 | 0.00 | ORB-short ORB[960.00,974.00] vol=1.9x ATR=7.50 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 955.45 | 962.76 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:50:00 | 949.65 | 956.74 | 0.00 | ORB-short ORB[957.35,965.00] vol=2.3x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:30:00 | 945.29 | 955.02 | 0.00 | T1 1.5R @ 945.29 |
| Stop hit — per-position SL triggered | 2024-08-20 12:15:00 | 949.65 | 953.92 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1073.15 | 1080.17 | 0.00 | ORB-short ORB[1074.05,1089.00] vol=2.0x ATR=6.01 |
| Stop hit — per-position SL triggered | 2024-08-28 10:30:00 | 1079.16 | 1078.42 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:50:00 | 1078.30 | 1070.94 | 0.00 | ORB-long ORB[1063.05,1074.90] vol=2.0x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:55:00 | 1084.69 | 1073.16 | 0.00 | T1 1.5R @ 1084.69 |
| Target hit | 2024-08-29 10:20:00 | 1082.25 | 1083.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:15:00 | 1067.25 | 1074.60 | 0.00 | ORB-short ORB[1075.00,1086.40] vol=3.3x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 12:15:00 | 1061.81 | 1073.30 | 0.00 | T1 1.5R @ 1061.81 |
| Stop hit — per-position SL triggered | 2024-09-10 13:30:00 | 1067.25 | 1072.05 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:30:00 | 1048.10 | 1055.80 | 0.00 | ORB-short ORB[1053.30,1067.75] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-09-11 11:30:00 | 1051.43 | 1053.39 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:20:00 | 987.50 | 983.47 | 0.00 | ORB-long ORB[975.00,986.40] vol=1.7x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:50:00 | 992.98 | 985.68 | 0.00 | T1 1.5R @ 992.98 |
| Target hit | 2024-09-23 12:20:00 | 1018.95 | 1025.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2024-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:10:00 | 1069.00 | 1058.26 | 0.00 | ORB-long ORB[1054.05,1066.05] vol=2.3x ATR=5.59 |
| Stop hit — per-position SL triggered | 2024-09-26 10:25:00 | 1063.41 | 1059.00 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:40:00 | 1078.00 | 1072.09 | 0.00 | ORB-long ORB[1063.35,1076.95] vol=1.6x ATR=5.69 |
| Stop hit — per-position SL triggered | 2024-09-27 09:55:00 | 1072.31 | 1073.94 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 1337.85 | 1346.32 | 0.00 | ORB-short ORB[1347.00,1366.50] vol=3.5x ATR=5.25 |
| Stop hit — per-position SL triggered | 2024-12-12 09:55:00 | 1343.10 | 1343.31 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:05:00 | 1302.55 | 1313.25 | 0.00 | ORB-short ORB[1311.90,1331.35] vol=2.2x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:30:00 | 1295.65 | 1311.64 | 0.00 | T1 1.5R @ 1295.65 |
| Stop hit — per-position SL triggered | 2024-12-16 13:05:00 | 1302.55 | 1307.28 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 1311.15 | 1298.42 | 0.00 | ORB-long ORB[1285.65,1301.45] vol=3.2x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:05:00 | 1322.18 | 1305.44 | 0.00 | T1 1.5R @ 1322.18 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 1311.15 | 1307.46 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 1186.25 | 1172.49 | 0.00 | ORB-long ORB[1160.20,1174.65] vol=2.9x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-12-24 10:00:00 | 1179.38 | 1173.41 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 1171.80 | 1185.60 | 0.00 | ORB-short ORB[1190.70,1201.65] vol=7.0x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-12-27 10:35:00 | 1176.67 | 1166.17 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:05:00 | 1196.35 | 1177.93 | 0.00 | ORB-long ORB[1158.90,1174.00] vol=2.5x ATR=5.67 |
| Stop hit — per-position SL triggered | 2024-12-30 12:20:00 | 1190.68 | 1183.99 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:05:00 | 1206.85 | 1197.47 | 0.00 | ORB-long ORB[1190.00,1200.15] vol=1.9x ATR=6.67 |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 1200.18 | 1198.31 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 1204.35 | 1182.39 | 0.00 | ORB-long ORB[1159.80,1173.00] vol=2.0x ATR=6.58 |
| Stop hit — per-position SL triggered | 2025-01-07 11:05:00 | 1197.77 | 1184.31 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-02-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:50:00 | 1109.90 | 1131.16 | 0.00 | ORB-short ORB[1140.90,1149.80] vol=3.1x ATR=5.63 |
| Stop hit — per-position SL triggered | 2025-02-04 10:55:00 | 1115.53 | 1128.49 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-02-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:50:00 | 1161.95 | 1149.43 | 0.00 | ORB-long ORB[1135.35,1152.00] vol=5.0x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-02-06 09:55:00 | 1155.58 | 1150.42 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 1190.85 | 1169.41 | 0.00 | ORB-long ORB[1163.15,1179.45] vol=4.4x ATR=5.90 |
| Stop hit — per-position SL triggered | 2025-02-07 11:20:00 | 1184.95 | 1173.65 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-03-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:40:00 | 1110.95 | 1095.88 | 0.00 | ORB-long ORB[1072.10,1088.65] vol=2.6x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:10:00 | 1121.37 | 1102.63 | 0.00 | T1 1.5R @ 1121.37 |
| Stop hit — per-position SL triggered | 2025-03-07 10:35:00 | 1110.95 | 1105.26 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:20:00 | 1127.00 | 1117.50 | 0.00 | ORB-long ORB[1108.75,1118.40] vol=1.9x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-03-21 10:35:00 | 1122.43 | 1118.52 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 1170.00 | 1157.98 | 0.00 | ORB-long ORB[1137.00,1152.10] vol=9.3x ATR=6.25 |
| Stop hit — per-position SL triggered | 2025-03-26 09:45:00 | 1163.75 | 1161.73 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:05:00 | 1296.00 | 1285.15 | 0.00 | ORB-long ORB[1276.00,1292.00] vol=1.7x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:35:00 | 1303.66 | 1288.28 | 0.00 | T1 1.5R @ 1303.66 |
| Target hit | 2025-04-24 15:20:00 | 1319.90 | 1304.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 1293.30 | 1286.85 | 0.00 | ORB-long ORB[1261.70,1280.00] vol=2.2x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-05-05 11:05:00 | 1288.34 | 1287.05 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 1290.30 | 1279.84 | 0.00 | ORB-long ORB[1267.80,1285.50] vol=1.8x ATR=6.83 |
| Stop hit — per-position SL triggered | 2025-05-08 09:40:00 | 1283.47 | 1281.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:30:00 | 514.80 | 2024-05-13 11:15:00 | 518.25 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2024-05-16 11:05:00 | 554.10 | 2024-05-16 11:10:00 | 555.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-18 09:50:00 | 554.65 | 2024-05-21 09:15:00 | 555.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-21 10:30:00 | 544.25 | 2024-05-21 11:10:00 | 546.44 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-22 09:35:00 | 551.10 | 2024-05-22 09:40:00 | 547.28 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-05-22 09:35:00 | 551.10 | 2024-05-22 09:55:00 | 551.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:35:00 | 549.30 | 2024-05-23 10:45:00 | 550.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-24 09:40:00 | 552.80 | 2024-05-24 09:45:00 | 555.62 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-24 09:40:00 | 552.80 | 2024-05-24 09:50:00 | 552.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 09:35:00 | 547.30 | 2024-05-28 12:05:00 | 544.42 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-05-28 09:35:00 | 547.30 | 2024-05-28 15:20:00 | 546.90 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-06-06 09:30:00 | 561.90 | 2024-06-06 09:45:00 | 565.55 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-06 09:30:00 | 561.90 | 2024-06-06 11:30:00 | 569.45 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2024-06-07 09:50:00 | 578.55 | 2024-06-07 10:35:00 | 581.73 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-07 09:50:00 | 578.55 | 2024-06-07 13:05:00 | 579.05 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-06-10 10:05:00 | 592.50 | 2024-06-10 10:10:00 | 596.33 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-10 10:05:00 | 592.50 | 2024-06-10 10:35:00 | 592.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 11:10:00 | 601.95 | 2024-06-18 11:15:00 | 605.93 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-18 11:10:00 | 601.95 | 2024-06-18 11:45:00 | 601.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:35:00 | 680.00 | 2024-06-26 09:45:00 | 686.50 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2024-06-26 09:35:00 | 680.00 | 2024-06-26 13:15:00 | 686.35 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-07-01 09:45:00 | 690.35 | 2024-07-01 13:40:00 | 686.06 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-07-03 09:30:00 | 722.75 | 2024-07-03 09:40:00 | 728.38 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-07-03 09:30:00 | 722.75 | 2024-07-03 11:00:00 | 741.70 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2024-07-11 09:45:00 | 772.75 | 2024-07-11 09:50:00 | 768.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-07-12 09:30:00 | 779.30 | 2024-07-12 09:35:00 | 775.07 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-23 10:35:00 | 779.80 | 2024-07-23 10:45:00 | 774.71 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2024-08-14 09:40:00 | 947.95 | 2024-08-14 09:45:00 | 955.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest1 | 2024-08-20 10:50:00 | 949.65 | 2024-08-20 11:30:00 | 945.29 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-20 10:50:00 | 949.65 | 2024-08-20 12:15:00 | 949.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1073.15 | 2024-08-28 10:30:00 | 1079.16 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-08-29 09:50:00 | 1078.30 | 2024-08-29 09:55:00 | 1084.69 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-29 09:50:00 | 1078.30 | 2024-08-29 10:20:00 | 1082.25 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-10 11:15:00 | 1067.25 | 2024-09-10 12:15:00 | 1061.81 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-10 11:15:00 | 1067.25 | 2024-09-10 13:30:00 | 1067.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 10:30:00 | 1048.10 | 2024-09-11 11:30:00 | 1051.43 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-23 10:20:00 | 987.50 | 2024-09-23 10:50:00 | 992.98 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-23 10:20:00 | 987.50 | 2024-09-23 12:20:00 | 1018.95 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2024-09-26 10:10:00 | 1069.00 | 2024-09-26 10:25:00 | 1063.41 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-09-27 09:40:00 | 1078.00 | 2024-09-27 09:55:00 | 1072.31 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-12-12 09:35:00 | 1337.85 | 2024-12-12 09:55:00 | 1343.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-16 11:05:00 | 1302.55 | 2024-12-16 11:30:00 | 1295.65 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-16 11:05:00 | 1302.55 | 2024-12-16 13:05:00 | 1302.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 09:40:00 | 1311.15 | 2024-12-17 10:05:00 | 1322.18 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2024-12-17 09:40:00 | 1311.15 | 2024-12-17 10:20:00 | 1311.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:50:00 | 1186.25 | 2024-12-24 10:00:00 | 1179.38 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-12-27 10:30:00 | 1171.80 | 2024-12-27 10:35:00 | 1176.67 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-30 11:05:00 | 1196.35 | 2024-12-30 12:20:00 | 1190.68 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-01 10:05:00 | 1206.85 | 2025-01-01 10:15:00 | 1200.18 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-01-07 10:55:00 | 1204.35 | 2025-01-07 11:05:00 | 1197.77 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-02-04 10:50:00 | 1109.90 | 2025-02-04 10:55:00 | 1115.53 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-02-06 09:50:00 | 1161.95 | 2025-02-06 09:55:00 | 1155.58 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-02-07 11:10:00 | 1190.85 | 2025-02-07 11:20:00 | 1184.95 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-03-07 09:40:00 | 1110.95 | 2025-03-07 10:10:00 | 1121.37 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2025-03-07 09:40:00 | 1110.95 | 2025-03-07 10:35:00 | 1110.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:20:00 | 1127.00 | 2025-03-21 10:35:00 | 1122.43 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-26 09:30:00 | 1170.00 | 2025-03-26 09:45:00 | 1163.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-04-24 10:05:00 | 1296.00 | 2025-04-24 10:35:00 | 1303.66 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-24 10:05:00 | 1296.00 | 2025-04-24 15:20:00 | 1319.90 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2025-05-05 11:00:00 | 1293.30 | 2025-05-05 11:05:00 | 1288.34 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-08 09:35:00 | 1290.30 | 2025-05-08 09:40:00 | 1283.47 | STOP_HIT | 1.00 | -0.53% |
