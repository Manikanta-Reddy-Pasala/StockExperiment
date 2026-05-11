# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 953.00
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 6 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 116 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 76
- **Target hits / Stop hits / Partials:** 6 / 76 / 34
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 7.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 24 | 35.8% | 3 | 43 | 21 | 0.06% | 4.0% |
| BUY @ 2nd Alert (retest1) | 67 | 24 | 35.8% | 3 | 43 | 21 | 0.06% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 16 | 32.7% | 3 | 33 | 13 | 0.08% | 3.9% |
| SELL @ 2nd Alert (retest1) | 49 | 16 | 32.7% | 3 | 33 | 13 | 0.08% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 116 | 40 | 34.5% | 6 | 76 | 34 | 0.07% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:50:00 | 974.55 | 973.49 | 0.00 | ORB-long ORB[960.00,973.50] vol=2.7x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:05:00 | 979.26 | 973.70 | 0.00 | T1 1.5R @ 979.26 |
| Stop hit — per-position SL triggered | 2025-05-13 12:35:00 | 974.55 | 975.00 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:15:00 | 984.90 | 978.12 | 0.00 | ORB-long ORB[975.60,980.00] vol=3.8x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-05-14 10:20:00 | 982.39 | 978.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1002.00 | 998.04 | 0.00 | ORB-long ORB[986.85,999.90] vol=4.7x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:45:00 | 1007.25 | 1000.10 | 0.00 | T1 1.5R @ 1007.25 |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 1002.00 | 1001.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:35:00 | 984.95 | 978.15 | 0.00 | ORB-long ORB[974.00,978.80] vol=2.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-05-22 09:45:00 | 982.27 | 979.58 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:55:00 | 971.95 | 978.60 | 0.00 | ORB-short ORB[975.05,987.95] vol=2.8x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:00:00 | 967.37 | 977.73 | 0.00 | T1 1.5R @ 967.37 |
| Stop hit — per-position SL triggered | 2025-05-27 10:05:00 | 971.95 | 977.60 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:30:00 | 963.25 | 965.57 | 0.00 | ORB-short ORB[968.90,980.30] vol=2.6x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:05:00 | 958.33 | 964.61 | 0.00 | T1 1.5R @ 958.33 |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 963.25 | 964.52 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 969.00 | 965.42 | 0.00 | ORB-long ORB[959.30,967.00] vol=1.9x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:45:00 | 973.19 | 966.53 | 0.00 | T1 1.5R @ 973.19 |
| Stop hit — per-position SL triggered | 2025-05-29 09:50:00 | 969.00 | 967.07 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 947.95 | 951.68 | 0.00 | ORB-short ORB[952.50,962.00] vol=3.3x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-05-30 11:00:00 | 950.48 | 951.49 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:20:00 | 980.00 | 972.58 | 0.00 | ORB-long ORB[964.00,977.00] vol=2.3x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:15:00 | 986.23 | 975.55 | 0.00 | T1 1.5R @ 986.23 |
| Target hit | 2025-06-02 15:15:00 | 984.55 | 985.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2025-06-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 11:00:00 | 1001.00 | 987.83 | 0.00 | ORB-long ORB[980.05,993.25] vol=2.9x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 997.12 | 991.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 1004.65 | 1001.08 | 0.00 | ORB-long ORB[993.95,1003.25] vol=6.0x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:00:00 | 1008.28 | 1004.18 | 0.00 | T1 1.5R @ 1008.28 |
| Target hit | 2025-06-05 11:55:00 | 1005.15 | 1005.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2025-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:55:00 | 998.40 | 1003.27 | 0.00 | ORB-short ORB[1000.40,1008.95] vol=2.5x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:35:00 | 995.14 | 1002.39 | 0.00 | T1 1.5R @ 995.14 |
| Stop hit — per-position SL triggered | 2025-06-06 10:55:00 | 998.40 | 1002.10 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:30:00 | 1052.20 | 1044.85 | 0.00 | ORB-long ORB[1035.50,1045.60] vol=3.8x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:05:00 | 1058.61 | 1047.90 | 0.00 | T1 1.5R @ 1058.61 |
| Stop hit — per-position SL triggered | 2025-06-10 11:10:00 | 1052.20 | 1048.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 1068.40 | 1067.22 | 0.00 | ORB-long ORB[1058.30,1067.00] vol=4.6x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:35:00 | 1073.83 | 1067.90 | 0.00 | T1 1.5R @ 1073.83 |
| Stop hit — per-position SL triggered | 2025-06-16 09:45:00 | 1068.40 | 1068.43 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:50:00 | 1013.30 | 1017.54 | 0.00 | ORB-short ORB[1015.00,1023.95] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-06-20 09:55:00 | 1016.23 | 1016.87 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:55:00 | 1068.05 | 1061.02 | 0.00 | ORB-long ORB[1052.70,1063.50] vol=5.1x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-06-30 11:00:00 | 1064.47 | 1061.19 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 11:00:00 | 1081.70 | 1075.09 | 0.00 | ORB-long ORB[1071.00,1080.90] vol=2.6x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:05:00 | 1086.98 | 1077.21 | 0.00 | T1 1.5R @ 1086.98 |
| Stop hit — per-position SL triggered | 2025-07-02 11:10:00 | 1081.70 | 1077.56 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:10:00 | 1083.00 | 1079.03 | 0.00 | ORB-long ORB[1072.90,1081.30] vol=6.8x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 10:25:00 | 1087.58 | 1081.72 | 0.00 | T1 1.5R @ 1087.58 |
| Target hit | 2025-07-04 12:15:00 | 1087.50 | 1087.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:45:00 | 1084.70 | 1086.77 | 0.00 | ORB-short ORB[1086.10,1095.00] vol=1.5x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-07-07 10:50:00 | 1087.36 | 1087.29 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:50:00 | 1075.00 | 1082.82 | 0.00 | ORB-short ORB[1076.70,1090.80] vol=1.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-07-08 11:35:00 | 1078.17 | 1081.35 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:05:00 | 1088.50 | 1090.17 | 0.00 | ORB-short ORB[1092.90,1105.00] vol=5.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:15:00 | 1082.30 | 1089.80 | 0.00 | T1 1.5R @ 1082.30 |
| Stop hit — per-position SL triggered | 2025-07-10 11:45:00 | 1088.50 | 1088.13 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:55:00 | 1135.00 | 1134.26 | 0.00 | ORB-long ORB[1124.20,1133.70] vol=2.7x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:00:00 | 1141.26 | 1134.41 | 0.00 | T1 1.5R @ 1141.26 |
| Stop hit — per-position SL triggered | 2025-07-15 11:10:00 | 1135.00 | 1134.48 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:35:00 | 1166.40 | 1164.25 | 0.00 | ORB-long ORB[1157.40,1166.00] vol=3.0x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-07-17 09:40:00 | 1162.74 | 1164.25 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 1159.00 | 1165.37 | 0.00 | ORB-short ORB[1168.20,1179.70] vol=2.3x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:55:00 | 1152.68 | 1164.06 | 0.00 | T1 1.5R @ 1152.68 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 1159.00 | 1162.93 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 11:15:00 | 1152.80 | 1159.42 | 0.00 | ORB-short ORB[1157.20,1164.00] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-07-21 12:25:00 | 1155.85 | 1158.27 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:10:00 | 1176.80 | 1185.59 | 0.00 | ORB-short ORB[1177.10,1191.10] vol=1.5x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:20:00 | 1172.51 | 1184.34 | 0.00 | T1 1.5R @ 1172.51 |
| Stop hit — per-position SL triggered | 2025-07-24 11:30:00 | 1176.80 | 1183.74 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:55:00 | 1175.80 | 1167.82 | 0.00 | ORB-long ORB[1150.60,1164.00] vol=1.9x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 11:15:00 | 1182.08 | 1171.14 | 0.00 | T1 1.5R @ 1182.08 |
| Stop hit — per-position SL triggered | 2025-07-29 11:40:00 | 1175.80 | 1173.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:40:00 | 1152.40 | 1157.69 | 0.00 | ORB-short ORB[1161.40,1178.00] vol=2.6x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 10:50:00 | 1147.02 | 1155.18 | 0.00 | T1 1.5R @ 1147.02 |
| Stop hit — per-position SL triggered | 2025-08-05 11:25:00 | 1152.40 | 1154.61 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:50:00 | 1146.80 | 1153.62 | 0.00 | ORB-short ORB[1150.30,1165.90] vol=2.8x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-08-06 12:05:00 | 1150.39 | 1151.81 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:35:00 | 1078.30 | 1067.94 | 0.00 | ORB-long ORB[1058.00,1073.60] vol=1.6x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:45:00 | 1086.53 | 1073.78 | 0.00 | T1 1.5R @ 1086.53 |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 1078.30 | 1075.79 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:55:00 | 1060.10 | 1066.16 | 0.00 | ORB-short ORB[1063.00,1077.00] vol=1.9x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-08-14 11:30:00 | 1063.17 | 1065.24 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 1100.00 | 1089.65 | 0.00 | ORB-long ORB[1080.50,1093.00] vol=3.0x ATR=4.41 |
| Stop hit — per-position SL triggered | 2025-08-20 11:45:00 | 1095.59 | 1092.69 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 1051.40 | 1056.19 | 0.00 | ORB-short ORB[1053.60,1065.20] vol=1.7x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-08-26 09:35:00 | 1055.24 | 1055.81 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1041.10 | 1037.98 | 0.00 | ORB-long ORB[1025.00,1039.00] vol=2.5x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:50:00 | 1047.85 | 1040.29 | 0.00 | T1 1.5R @ 1047.85 |
| Stop hit — per-position SL triggered | 2025-08-29 12:25:00 | 1041.10 | 1041.35 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:15:00 | 1057.60 | 1056.72 | 0.00 | ORB-long ORB[1038.50,1053.50] vol=1.9x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:40:00 | 1062.28 | 1057.88 | 0.00 | T1 1.5R @ 1062.28 |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 1057.60 | 1058.38 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:40:00 | 1075.30 | 1069.68 | 0.00 | ORB-long ORB[1061.60,1075.00] vol=2.5x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-09-02 13:55:00 | 1071.89 | 1072.71 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1077.00 | 1083.68 | 0.00 | ORB-short ORB[1083.50,1098.00] vol=2.1x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-09-04 11:30:00 | 1080.49 | 1083.32 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:30:00 | 1065.60 | 1072.56 | 0.00 | ORB-short ORB[1067.70,1080.00] vol=1.7x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-09-08 10:40:00 | 1069.40 | 1072.07 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 09:50:00 | 1052.50 | 1057.79 | 0.00 | ORB-short ORB[1054.80,1064.00] vol=2.1x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 13:55:00 | 1044.19 | 1052.47 | 0.00 | T1 1.5R @ 1044.19 |
| Target hit | 2025-09-10 15:20:00 | 1046.00 | 1050.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:50:00 | 1044.80 | 1040.12 | 0.00 | ORB-long ORB[1035.50,1043.50] vol=3.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-09-25 10:00:00 | 1041.68 | 1040.36 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:15:00 | 1017.20 | 1017.48 | 0.00 | ORB-short ORB[1019.80,1032.30] vol=1.7x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1012.10 | 1016.11 | 0.00 | T1 1.5R @ 1012.10 |
| Target hit | 2025-09-26 15:20:00 | 998.40 | 1011.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 974.35 | 984.09 | 0.00 | ORB-short ORB[981.95,993.65] vol=4.0x ATR=3.08 |
| Stop hit — per-position SL triggered | 2025-10-01 11:30:00 | 977.43 | 979.09 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:30:00 | 999.30 | 1002.18 | 0.00 | ORB-short ORB[1003.00,1010.45] vol=2.9x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-10-08 10:45:00 | 1001.46 | 1001.72 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:10:00 | 1005.20 | 999.76 | 0.00 | ORB-long ORB[989.65,1001.85] vol=2.4x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:35:00 | 1008.37 | 1000.77 | 0.00 | T1 1.5R @ 1008.37 |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 1005.20 | 1003.46 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:15:00 | 1020.25 | 1017.58 | 0.00 | ORB-long ORB[1003.60,1016.50] vol=5.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-10-10 11:25:00 | 1017.82 | 1017.64 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1017.30 | 1015.05 | 0.00 | ORB-long ORB[1007.00,1017.00] vol=12.0x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-10-13 10:40:00 | 1014.52 | 1015.06 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:00:00 | 1005.00 | 1006.44 | 0.00 | ORB-short ORB[1005.05,1016.00] vol=5.6x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-10-14 10:25:00 | 1008.13 | 1006.51 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:45:00 | 1040.85 | 1036.66 | 0.00 | ORB-long ORB[1030.10,1038.85] vol=1.6x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:55:00 | 1045.33 | 1037.66 | 0.00 | T1 1.5R @ 1045.33 |
| Stop hit — per-position SL triggered | 2025-10-23 11:50:00 | 1040.85 | 1039.39 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 1039.80 | 1040.99 | 0.00 | ORB-short ORB[1040.40,1049.00] vol=2.6x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:45:00 | 1035.73 | 1039.63 | 0.00 | T1 1.5R @ 1035.73 |
| Stop hit — per-position SL triggered | 2025-11-03 09:50:00 | 1039.80 | 1039.65 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:55:00 | 1024.80 | 1028.28 | 0.00 | ORB-short ORB[1027.50,1036.90] vol=13.0x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-11-10 10:00:00 | 1027.93 | 1028.27 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 09:55:00 | 972.80 | 974.83 | 0.00 | ORB-short ORB[978.60,988.50] vol=5.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-11-14 10:00:00 | 976.59 | 975.03 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:00:00 | 998.00 | 989.86 | 0.00 | ORB-long ORB[980.70,989.90] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-11-17 10:35:00 | 994.86 | 992.57 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:45:00 | 992.30 | 985.95 | 0.00 | ORB-long ORB[980.80,991.30] vol=2.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-11-19 10:50:00 | 990.22 | 986.09 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:40:00 | 1015.50 | 1006.35 | 0.00 | ORB-long ORB[992.10,1007.20] vol=2.5x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:55:00 | 1020.74 | 1012.29 | 0.00 | T1 1.5R @ 1020.74 |
| Stop hit — per-position SL triggered | 2025-11-21 10:00:00 | 1015.50 | 1012.68 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 11:15:00 | 1020.60 | 1015.83 | 0.00 | ORB-long ORB[1007.00,1020.00] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-11-24 11:20:00 | 1018.37 | 1015.86 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 1013.70 | 1008.07 | 0.00 | ORB-long ORB[1003.30,1013.60] vol=2.2x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-11-26 10:55:00 | 1010.71 | 1009.08 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:35:00 | 1011.30 | 1015.25 | 0.00 | ORB-short ORB[1014.30,1022.80] vol=4.5x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-12-03 10:10:00 | 1014.02 | 1014.92 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:10:00 | 999.30 | 989.43 | 0.00 | ORB-long ORB[982.70,995.90] vol=5.5x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-12-09 11:40:00 | 996.54 | 990.56 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:15:00 | 1026.00 | 1019.37 | 0.00 | ORB-long ORB[1017.10,1021.60] vol=3.5x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:35:00 | 1029.77 | 1025.20 | 0.00 | T1 1.5R @ 1029.77 |
| Stop hit — per-position SL triggered | 2025-12-11 12:10:00 | 1026.00 | 1025.54 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:50:00 | 1046.00 | 1049.38 | 0.00 | ORB-short ORB[1047.50,1053.70] vol=2.4x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-12-18 10:45:00 | 1048.01 | 1047.23 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 11:15:00 | 1061.60 | 1057.01 | 0.00 | ORB-long ORB[1048.50,1060.00] vol=4.6x ATR=3.15 |
| Stop hit — per-position SL triggered | 2025-12-24 11:25:00 | 1058.45 | 1057.29 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:40:00 | 1080.30 | 1073.55 | 0.00 | ORB-long ORB[1055.60,1069.00] vol=1.8x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-12-26 10:00:00 | 1076.46 | 1077.85 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 10:15:00 | 1065.00 | 1061.57 | 0.00 | ORB-long ORB[1059.40,1063.20] vol=1.5x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:25:00 | 1069.74 | 1062.80 | 0.00 | T1 1.5R @ 1069.74 |
| Stop hit — per-position SL triggered | 2025-12-29 11:00:00 | 1065.00 | 1066.30 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:10:00 | 1063.50 | 1058.19 | 0.00 | ORB-long ORB[1041.90,1057.40] vol=9.7x ATR=3.02 |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 1060.48 | 1058.22 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 10:35:00 | 1082.60 | 1073.56 | 0.00 | ORB-long ORB[1067.00,1081.10] vol=1.6x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:40:00 | 1086.83 | 1074.28 | 0.00 | T1 1.5R @ 1086.83 |
| Stop hit — per-position SL triggered | 2026-01-08 11:00:00 | 1082.60 | 1076.79 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:25:00 | 1090.90 | 1084.01 | 0.00 | ORB-long ORB[1076.80,1086.80] vol=2.1x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-01-09 10:35:00 | 1087.67 | 1085.00 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:35:00 | 1076.90 | 1081.15 | 0.00 | ORB-short ORB[1078.20,1088.30] vol=5.0x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-01-14 11:00:00 | 1080.29 | 1080.95 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:45:00 | 1053.10 | 1059.18 | 0.00 | ORB-short ORB[1060.00,1068.30] vol=1.7x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-01-19 09:50:00 | 1056.23 | 1059.14 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 1070.90 | 1063.05 | 0.00 | ORB-long ORB[1055.40,1065.40] vol=2.6x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-01-20 09:35:00 | 1067.54 | 1065.49 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:00:00 | 1071.60 | 1077.64 | 0.00 | ORB-short ORB[1075.60,1087.00] vol=2.7x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-01-22 11:05:00 | 1074.19 | 1077.51 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:45:00 | 1169.90 | 1162.67 | 0.00 | ORB-long ORB[1154.70,1165.50] vol=1.7x ATR=4.07 |
| Stop hit — per-position SL triggered | 2026-02-06 09:50:00 | 1165.83 | 1162.48 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1139.40 | 1135.14 | 0.00 | ORB-long ORB[1131.10,1137.80] vol=6.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-02-13 11:20:00 | 1136.09 | 1135.79 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 1154.40 | 1156.63 | 0.00 | ORB-short ORB[1157.70,1168.00] vol=1.9x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 1147.50 | 1155.99 | 0.00 | T1 1.5R @ 1147.50 |
| Stop hit — per-position SL triggered | 2026-02-18 13:20:00 | 1154.40 | 1155.25 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1125.70 | 1117.20 | 0.00 | ORB-long ORB[1107.50,1124.00] vol=1.8x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-02-20 10:40:00 | 1121.12 | 1117.27 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 1073.40 | 1078.40 | 0.00 | ORB-short ORB[1075.90,1090.90] vol=1.8x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:10:00 | 1066.84 | 1076.06 | 0.00 | T1 1.5R @ 1066.84 |
| Target hit | 2026-03-05 13:05:00 | 1053.10 | 1052.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1087.00 | 1091.34 | 0.00 | ORB-short ORB[1087.30,1094.20] vol=2.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-03-06 11:30:00 | 1091.48 | 1091.91 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:55:00 | 1019.70 | 1022.67 | 0.00 | ORB-short ORB[1031.00,1039.40] vol=3.8x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-03-10 10:40:00 | 1026.22 | 1022.60 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 950.85 | 937.60 | 0.00 | ORB-long ORB[927.00,939.05] vol=1.7x ATR=4.94 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 945.91 | 938.70 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 978.60 | 969.87 | 0.00 | ORB-long ORB[959.00,972.90] vol=3.6x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:55:00 | 987.70 | 975.23 | 0.00 | T1 1.5R @ 987.70 |
| Stop hit — per-position SL triggered | 2026-04-08 10:00:00 | 978.60 | 975.26 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 1007.75 | 1003.80 | 0.00 | ORB-long ORB[996.45,1005.00] vol=2.4x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 1005.27 | 1004.86 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 958.00 | 960.92 | 0.00 | ORB-short ORB[959.65,968.55] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-04-29 11:35:00 | 960.44 | 959.35 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 916.00 | 922.30 | 0.00 | ORB-short ORB[918.95,927.00] vol=2.2x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:10:00 | 911.17 | 917.42 | 0.00 | T1 1.5R @ 911.17 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 916.00 | 913.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:50:00 | 974.55 | 2025-05-13 11:05:00 | 979.26 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-13 10:50:00 | 974.55 | 2025-05-13 12:35:00 | 974.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-14 10:15:00 | 984.90 | 2025-05-14 10:20:00 | 982.39 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-15 09:30:00 | 1002.00 | 2025-05-15 09:45:00 | 1007.25 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-05-15 09:30:00 | 1002.00 | 2025-05-15 10:15:00 | 1002.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-22 09:35:00 | 984.95 | 2025-05-22 09:45:00 | 982.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-27 09:55:00 | 971.95 | 2025-05-27 10:00:00 | 967.37 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-05-27 09:55:00 | 971.95 | 2025-05-27 10:05:00 | 971.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-28 10:30:00 | 963.25 | 2025-05-28 11:05:00 | 958.33 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-05-28 10:30:00 | 963.25 | 2025-05-28 11:15:00 | 963.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-29 09:35:00 | 969.00 | 2025-05-29 09:45:00 | 973.19 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-29 09:35:00 | 969.00 | 2025-05-29 09:50:00 | 969.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 10:55:00 | 947.95 | 2025-05-30 11:00:00 | 950.48 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-02 10:20:00 | 980.00 | 2025-06-02 11:15:00 | 986.23 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-06-02 10:20:00 | 980.00 | 2025-06-02 15:15:00 | 984.55 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-03 11:00:00 | 1001.00 | 2025-06-03 11:15:00 | 997.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-05 10:45:00 | 1004.65 | 2025-06-05 11:00:00 | 1008.28 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-06-05 10:45:00 | 1004.65 | 2025-06-05 11:55:00 | 1005.15 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-06-06 09:55:00 | 998.40 | 2025-06-06 10:35:00 | 995.14 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-06 09:55:00 | 998.40 | 2025-06-06 10:55:00 | 998.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:30:00 | 1052.20 | 2025-06-10 11:05:00 | 1058.61 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-10 10:30:00 | 1052.20 | 2025-06-10 11:10:00 | 1052.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-16 09:30:00 | 1068.40 | 2025-06-16 09:35:00 | 1073.83 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-16 09:30:00 | 1068.40 | 2025-06-16 09:45:00 | 1068.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-20 09:50:00 | 1013.30 | 2025-06-20 09:55:00 | 1016.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-30 10:55:00 | 1068.05 | 2025-06-30 11:00:00 | 1064.47 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-02 11:00:00 | 1081.70 | 2025-07-02 11:05:00 | 1086.98 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-02 11:00:00 | 1081.70 | 2025-07-02 11:10:00 | 1081.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 10:10:00 | 1083.00 | 2025-07-04 10:25:00 | 1087.58 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-04 10:10:00 | 1083.00 | 2025-07-04 12:15:00 | 1087.50 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-07 10:45:00 | 1084.70 | 2025-07-07 10:50:00 | 1087.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-08 10:50:00 | 1075.00 | 2025-07-08 11:35:00 | 1078.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-10 10:05:00 | 1088.50 | 2025-07-10 10:15:00 | 1082.30 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-07-10 10:05:00 | 1088.50 | 2025-07-10 11:45:00 | 1088.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:55:00 | 1135.00 | 2025-07-15 11:00:00 | 1141.26 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-15 10:55:00 | 1135.00 | 2025-07-15 11:10:00 | 1135.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 09:35:00 | 1166.40 | 2025-07-17 09:40:00 | 1162.74 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-18 10:45:00 | 1159.00 | 2025-07-18 10:55:00 | 1152.68 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-07-18 10:45:00 | 1159.00 | 2025-07-18 11:05:00 | 1159.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-21 11:15:00 | 1152.80 | 2025-07-21 12:25:00 | 1155.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-24 11:10:00 | 1176.80 | 2025-07-24 11:20:00 | 1172.51 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-24 11:10:00 | 1176.80 | 2025-07-24 11:30:00 | 1176.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 10:55:00 | 1175.80 | 2025-07-29 11:15:00 | 1182.08 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-07-29 10:55:00 | 1175.80 | 2025-07-29 11:40:00 | 1175.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-05 10:40:00 | 1152.40 | 2025-08-05 10:50:00 | 1147.02 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-08-05 10:40:00 | 1152.40 | 2025-08-05 11:25:00 | 1152.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:50:00 | 1146.80 | 2025-08-06 12:05:00 | 1150.39 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-13 09:35:00 | 1078.30 | 2025-08-13 09:45:00 | 1086.53 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-08-13 09:35:00 | 1078.30 | 2025-08-13 10:15:00 | 1078.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 10:55:00 | 1060.10 | 2025-08-14 11:30:00 | 1063.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-20 10:10:00 | 1100.00 | 2025-08-20 11:45:00 | 1095.59 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-26 09:30:00 | 1051.40 | 2025-08-26 09:35:00 | 1055.24 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-29 10:00:00 | 1041.10 | 2025-08-29 10:50:00 | 1047.85 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-08-29 10:00:00 | 1041.10 | 2025-08-29 12:25:00 | 1041.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 11:15:00 | 1057.60 | 2025-09-01 11:40:00 | 1062.28 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-01 11:15:00 | 1057.60 | 2025-09-01 12:15:00 | 1057.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 10:40:00 | 1075.30 | 2025-09-02 13:55:00 | 1071.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-04 11:15:00 | 1077.00 | 2025-09-04 11:30:00 | 1080.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-08 10:30:00 | 1065.60 | 2025-09-08 10:40:00 | 1069.40 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-10 09:50:00 | 1052.50 | 2025-09-10 13:55:00 | 1044.19 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2025-09-10 09:50:00 | 1052.50 | 2025-09-10 15:20:00 | 1046.00 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-25 09:50:00 | 1044.80 | 2025-09-25 10:00:00 | 1041.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-26 10:15:00 | 1017.20 | 2025-09-26 13:15:00 | 1012.10 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-09-26 10:15:00 | 1017.20 | 2025-09-26 15:20:00 | 998.40 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2025-10-01 11:00:00 | 974.35 | 2025-10-01 11:30:00 | 977.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-08 10:30:00 | 999.30 | 2025-10-08 10:45:00 | 1001.46 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-09 11:10:00 | 1005.20 | 2025-10-09 11:35:00 | 1008.37 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-09 11:10:00 | 1005.20 | 2025-10-09 12:15:00 | 1005.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 11:15:00 | 1020.25 | 2025-10-10 11:25:00 | 1017.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-13 10:30:00 | 1017.30 | 2025-10-13 10:40:00 | 1014.52 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-14 10:00:00 | 1005.00 | 2025-10-14 10:25:00 | 1008.13 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-23 10:45:00 | 1040.85 | 2025-10-23 10:55:00 | 1045.33 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-23 10:45:00 | 1040.85 | 2025-10-23 11:50:00 | 1040.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-03 09:30:00 | 1039.80 | 2025-11-03 09:45:00 | 1035.73 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-03 09:30:00 | 1039.80 | 2025-11-03 09:50:00 | 1039.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 09:55:00 | 1024.80 | 2025-11-10 10:00:00 | 1027.93 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-14 09:55:00 | 972.80 | 2025-11-14 10:00:00 | 976.59 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-17 10:00:00 | 998.00 | 2025-11-17 10:35:00 | 994.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-19 10:45:00 | 992.30 | 2025-11-19 10:50:00 | 990.22 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-21 09:40:00 | 1015.50 | 2025-11-21 09:55:00 | 1020.74 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-21 09:40:00 | 1015.50 | 2025-11-21 10:00:00 | 1015.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 11:15:00 | 1020.60 | 2025-11-24 11:20:00 | 1018.37 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-26 10:35:00 | 1013.70 | 2025-11-26 10:55:00 | 1010.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-03 09:35:00 | 1011.30 | 2025-12-03 10:10:00 | 1014.02 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-09 11:10:00 | 999.30 | 2025-12-09 11:40:00 | 996.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-11 11:15:00 | 1026.00 | 2025-12-11 11:35:00 | 1029.77 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-11 11:15:00 | 1026.00 | 2025-12-11 12:10:00 | 1026.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:50:00 | 1046.00 | 2025-12-18 10:45:00 | 1048.01 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-24 11:15:00 | 1061.60 | 2025-12-24 11:25:00 | 1058.45 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-26 09:40:00 | 1080.30 | 2025-12-26 10:00:00 | 1076.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-29 10:15:00 | 1065.00 | 2025-12-29 10:25:00 | 1069.74 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-29 10:15:00 | 1065.00 | 2025-12-29 11:00:00 | 1065.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:10:00 | 1063.50 | 2025-12-30 10:15:00 | 1060.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-01-08 10:35:00 | 1082.60 | 2026-01-08 10:40:00 | 1086.83 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-08 10:35:00 | 1082.60 | 2026-01-08 11:00:00 | 1082.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-09 10:25:00 | 1090.90 | 2026-01-09 10:35:00 | 1087.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-14 10:35:00 | 1076.90 | 2026-01-14 11:00:00 | 1080.29 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-19 09:45:00 | 1053.10 | 2026-01-19 09:50:00 | 1056.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-20 09:30:00 | 1070.90 | 2026-01-20 09:35:00 | 1067.54 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-22 11:00:00 | 1071.60 | 2026-01-22 11:05:00 | 1074.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-06 09:45:00 | 1169.90 | 2026-02-06 09:50:00 | 1165.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-13 10:30:00 | 1139.40 | 2026-02-13 11:20:00 | 1136.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-18 10:00:00 | 1154.40 | 2026-02-18 11:10:00 | 1147.50 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-18 10:00:00 | 1154.40 | 2026-02-18 13:20:00 | 1154.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1125.70 | 2026-02-20 10:40:00 | 1121.12 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-05 10:05:00 | 1073.40 | 2026-03-05 10:10:00 | 1066.84 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-05 10:05:00 | 1073.40 | 2026-03-05 13:05:00 | 1053.10 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1087.00 | 2026-03-06 11:30:00 | 1091.48 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-10 09:55:00 | 1019.70 | 2026-03-10 10:40:00 | 1026.22 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2026-04-01 10:55:00 | 950.85 | 2026-04-01 11:25:00 | 945.91 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-08 09:45:00 | 978.60 | 2026-04-08 09:55:00 | 987.70 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-08 09:45:00 | 978.60 | 2026-04-08 10:00:00 | 978.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:55:00 | 1007.75 | 2026-04-17 11:00:00 | 1005.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-29 10:55:00 | 958.00 | 2026-04-29 11:35:00 | 960.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-05 10:10:00 | 916.00 | 2026-05-05 11:10:00 | 911.17 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-05 10:10:00 | 916.00 | 2026-05-05 12:50:00 | 916.00 | STOP_HIT | 0.50 | 0.00% |
