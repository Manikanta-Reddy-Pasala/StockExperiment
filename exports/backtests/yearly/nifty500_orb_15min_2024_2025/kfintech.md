# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 917.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 17
- **Target hits / Stop hits / Partials:** 4 / 17 / 7
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 0.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 3 | 8 | 3 | 0.09% | 1.3% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 3 | 8 | 3 | 0.09% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 5 | 35.7% | 1 | 9 | 4 | -0.04% | -0.5% |
| SELL @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 1 | 9 | 4 | -0.04% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 11 | 39.3% | 4 | 17 | 7 | 0.03% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 744.60 | 747.94 | 0.00 | ORB-short ORB[754.00,763.45] vol=2.1x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-05-13 11:35:00 | 748.69 | 747.73 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:45:00 | 768.65 | 774.04 | 0.00 | ORB-short ORB[772.80,779.40] vol=2.0x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:05:00 | 763.96 | 772.47 | 0.00 | T1 1.5R @ 763.96 |
| Stop hit — per-position SL triggered | 2024-05-15 10:45:00 | 768.65 | 770.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 751.00 | 754.86 | 0.00 | ORB-short ORB[755.70,763.95] vol=4.4x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-05-16 09:45:00 | 753.90 | 754.78 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:45:00 | 721.05 | 717.63 | 0.00 | ORB-long ORB[711.50,719.10] vol=1.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 12:00:00 | 724.88 | 721.51 | 0.00 | T1 1.5R @ 724.88 |
| Target hit | 2024-06-11 15:20:00 | 727.00 | 724.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-06-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:00:00 | 691.00 | 683.99 | 0.00 | ORB-long ORB[680.00,686.65] vol=2.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-06-21 10:05:00 | 688.38 | 684.40 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 689.85 | 694.72 | 0.00 | ORB-short ORB[692.00,699.40] vol=3.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-06-25 12:25:00 | 692.03 | 693.58 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 754.30 | 758.41 | 0.00 | ORB-short ORB[755.65,763.65] vol=1.7x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:50:00 | 751.46 | 756.67 | 0.00 | T1 1.5R @ 751.46 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 754.30 | 756.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 762.70 | 759.11 | 0.00 | ORB-long ORB[754.80,762.00] vol=2.2x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 759.45 | 759.10 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 746.15 | 751.99 | 0.00 | ORB-short ORB[748.00,757.15] vol=1.5x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 749.03 | 751.56 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 791.70 | 779.85 | 0.00 | ORB-long ORB[766.05,777.25] vol=2.8x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 785.71 | 781.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:15:00 | 995.25 | 1003.82 | 0.00 | ORB-short ORB[1005.00,1019.15] vol=3.0x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:40:00 | 990.74 | 1001.58 | 0.00 | T1 1.5R @ 990.74 |
| Target hit | 2024-08-23 14:20:00 | 994.05 | 994.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2024-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:05:00 | 997.45 | 1000.97 | 0.00 | ORB-short ORB[998.05,1009.00] vol=2.3x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 993.24 | 1000.00 | 0.00 | T1 1.5R @ 993.24 |
| Stop hit — per-position SL triggered | 2024-09-10 11:30:00 | 997.45 | 999.56 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1050.55 | 1045.44 | 0.00 | ORB-long ORB[1036.70,1047.05] vol=2.7x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-10-10 09:35:00 | 1045.42 | 1045.53 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:10:00 | 1048.50 | 1040.28 | 0.00 | ORB-long ORB[1033.20,1044.40] vol=1.7x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-10-11 12:20:00 | 1043.29 | 1045.73 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 1114.05 | 1097.31 | 0.00 | ORB-long ORB[1087.00,1098.95] vol=3.8x ATR=6.29 |
| Stop hit — per-position SL triggered | 2024-10-16 10:05:00 | 1107.76 | 1098.70 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 1051.45 | 1063.11 | 0.00 | ORB-short ORB[1061.50,1076.95] vol=1.9x ATR=4.58 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 1056.03 | 1060.28 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-11-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:40:00 | 1129.75 | 1118.65 | 0.00 | ORB-long ORB[1105.00,1118.00] vol=1.8x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:30:00 | 1138.83 | 1125.06 | 0.00 | T1 1.5R @ 1138.83 |
| Target hit | 2024-11-26 15:20:00 | 1150.75 | 1148.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 1213.55 | 1195.38 | 0.00 | ORB-long ORB[1178.55,1194.45] vol=5.8x ATR=8.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:35:00 | 1225.84 | 1207.95 | 0.00 | T1 1.5R @ 1225.84 |
| Target hit | 2024-12-03 09:50:00 | 1216.05 | 1216.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2024-12-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:55:00 | 1276.35 | 1266.46 | 0.00 | ORB-long ORB[1254.15,1271.95] vol=3.2x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-12-12 10:30:00 | 1271.08 | 1268.80 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 1233.10 | 1242.22 | 0.00 | ORB-short ORB[1238.00,1253.25] vol=2.7x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-12-13 10:00:00 | 1238.48 | 1238.53 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:10:00 | 1036.00 | 1021.48 | 0.00 | ORB-long ORB[1010.30,1023.00] vol=3.5x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 1031.66 | 1022.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:15:00 | 744.60 | 2024-05-13 11:35:00 | 748.69 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-05-15 09:45:00 | 768.65 | 2024-05-15 10:05:00 | 763.96 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-05-15 09:45:00 | 768.65 | 2024-05-15 10:45:00 | 768.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 09:40:00 | 751.00 | 2024-05-16 09:45:00 | 753.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-11 09:45:00 | 721.05 | 2024-06-11 12:00:00 | 724.88 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-11 09:45:00 | 721.05 | 2024-06-11 15:20:00 | 727.00 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-21 10:00:00 | 691.00 | 2024-06-21 10:05:00 | 688.38 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-25 11:15:00 | 689.85 | 2024-06-25 12:25:00 | 692.03 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-12 09:30:00 | 754.30 | 2024-07-12 09:50:00 | 751.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-12 09:30:00 | 754.30 | 2024-07-12 09:55:00 | 754.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:30:00 | 762.70 | 2024-07-16 09:35:00 | 759.45 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-18 09:35:00 | 746.15 | 2024-07-18 09:40:00 | 749.03 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-26 09:35:00 | 791.70 | 2024-07-26 09:40:00 | 785.71 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2024-08-23 11:15:00 | 995.25 | 2024-08-23 11:40:00 | 990.74 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-08-23 11:15:00 | 995.25 | 2024-08-23 14:20:00 | 994.05 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2024-09-10 11:05:00 | 997.45 | 2024-09-10 11:15:00 | 993.24 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-10 11:05:00 | 997.45 | 2024-09-10 11:30:00 | 997.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 09:30:00 | 1050.55 | 2024-10-10 09:35:00 | 1045.42 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-10-11 10:10:00 | 1048.50 | 2024-10-11 12:20:00 | 1043.29 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-16 10:00:00 | 1114.05 | 2024-10-16 10:05:00 | 1107.76 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-10-21 09:45:00 | 1051.45 | 2024-10-21 10:00:00 | 1056.03 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-26 10:40:00 | 1129.75 | 2024-11-26 11:30:00 | 1138.83 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-11-26 10:40:00 | 1129.75 | 2024-11-26 15:20:00 | 1150.75 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2024-12-03 09:30:00 | 1213.55 | 2024-12-03 09:35:00 | 1225.84 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2024-12-03 09:30:00 | 1213.55 | 2024-12-03 09:50:00 | 1216.05 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-12-12 09:55:00 | 1276.35 | 2024-12-12 10:30:00 | 1271.08 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-13 09:30:00 | 1233.10 | 2024-12-13 10:00:00 | 1238.48 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-16 10:10:00 | 1036.00 | 2025-04-16 10:15:00 | 1031.66 | STOP_HIT | 1.00 | -0.42% |
