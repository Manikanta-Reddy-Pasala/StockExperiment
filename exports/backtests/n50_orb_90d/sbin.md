# SBIN (SBIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1018.50
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 12
- **Target hits / Stop hits / Partials:** 6 / 12 / 9
- **Avg / median % per leg:** 0.27% / 0.31%
- **Sum % (uncompounded):** 7.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.30% | 1.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 0.30% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.26% | 5.6% |
| SELL @ 2nd Alert (retest1) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.26% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 15 | 55.6% | 6 | 12 | 9 | 0.27% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:10:00 | 1143.20 | 1144.43 | 0.00 | ORB-short ORB[1144.50,1154.00] vol=1.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-02-10 12:50:00 | 1145.48 | 1143.51 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 1161.40 | 1150.93 | 0.00 | ORB-long ORB[1142.80,1151.10] vol=2.9x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:35:00 | 1165.55 | 1153.96 | 0.00 | T1 1.5R @ 1165.55 |
| Target hit | 2026-02-11 15:20:00 | 1180.80 | 1173.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 1197.70 | 1189.64 | 0.00 | ORB-long ORB[1174.80,1188.00] vol=1.6x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-02-12 11:35:00 | 1193.40 | 1194.91 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 1215.60 | 1210.37 | 0.00 | ORB-long ORB[1204.10,1212.00] vol=1.5x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 1220.28 | 1212.56 | 0.00 | T1 1.5R @ 1220.28 |
| Target hit | 2026-02-17 14:00:00 | 1217.90 | 1217.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 1189.90 | 1198.77 | 0.00 | ORB-short ORB[1199.00,1205.60] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-02-26 10:40:00 | 1192.74 | 1197.74 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:00:00 | 1165.30 | 1167.14 | 0.00 | ORB-short ORB[1166.20,1181.00] vol=2.0x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:30:00 | 1160.11 | 1166.59 | 0.00 | T1 1.5R @ 1160.11 |
| Stop hit — per-position SL triggered | 2026-03-04 14:00:00 | 1165.30 | 1164.31 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 1168.10 | 1174.83 | 0.00 | ORB-short ORB[1174.50,1183.20] vol=2.3x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:30:00 | 1163.82 | 1169.96 | 0.00 | T1 1.5R @ 1163.82 |
| Target hit | 2026-03-05 14:35:00 | 1164.50 | 1163.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1150.00 | 1160.19 | 0.00 | ORB-short ORB[1162.60,1169.30] vol=3.4x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 1153.08 | 1158.44 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 1067.90 | 1071.55 | 0.00 | ORB-short ORB[1072.10,1081.30] vol=3.9x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:35:00 | 1061.77 | 1068.76 | 0.00 | T1 1.5R @ 1061.77 |
| Target hit | 2026-03-13 15:20:00 | 1044.80 | 1057.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:55:00 | 1064.10 | 1066.54 | 0.00 | ORB-short ORB[1064.20,1072.70] vol=2.3x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 1066.81 | 1065.78 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:35:00 | 1026.35 | 1022.32 | 0.00 | ORB-long ORB[1011.00,1023.00] vol=1.7x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-04-06 09:40:00 | 1021.56 | 1022.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:35:00 | 1055.05 | 1058.37 | 0.00 | ORB-short ORB[1056.15,1064.35] vol=1.6x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:20:00 | 1049.94 | 1057.07 | 0.00 | T1 1.5R @ 1049.94 |
| Target hit | 2026-04-09 15:20:00 | 1040.00 | 1048.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 1074.45 | 1080.20 | 0.00 | ORB-short ORB[1075.00,1088.00] vol=1.5x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:05:00 | 1070.17 | 1079.25 | 0.00 | T1 1.5R @ 1070.17 |
| Stop hit — per-position SL triggered | 2026-04-15 14:20:00 | 1074.45 | 1074.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:00:00 | 1075.00 | 1077.90 | 0.00 | ORB-short ORB[1078.00,1084.15] vol=2.1x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:35:00 | 1071.05 | 1076.21 | 0.00 | T1 1.5R @ 1071.05 |
| Target hit | 2026-04-16 14:35:00 | 1071.40 | 1071.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 1108.00 | 1109.13 | 0.00 | ORB-short ORB[1108.20,1114.80] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-04-27 13:20:00 | 1110.97 | 1108.12 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1056.00 | 1062.52 | 0.00 | ORB-short ORB[1060.00,1068.00] vol=1.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 1058.40 | 1061.24 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1065.70 | 1071.09 | 0.00 | ORB-short ORB[1072.00,1079.50] vol=1.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-05-06 11:45:00 | 1068.20 | 1070.30 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 1092.00 | 1096.40 | 0.00 | ORB-short ORB[1092.50,1108.00] vol=1.6x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:25:00 | 1087.90 | 1095.08 | 0.00 | T1 1.5R @ 1087.90 |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 1092.00 | 1093.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:10:00 | 1143.20 | 2026-02-10 12:50:00 | 1145.48 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-11 10:30:00 | 1161.40 | 2026-02-11 10:35:00 | 1165.55 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-11 10:30:00 | 1161.40 | 2026-02-11 15:20:00 | 1180.80 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-02-12 10:00:00 | 1197.70 | 2026-02-12 11:35:00 | 1193.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1215.60 | 2026-02-17 10:20:00 | 1220.28 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1215.60 | 2026-02-17 14:00:00 | 1217.90 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-02-26 10:35:00 | 1189.90 | 2026-02-26 10:40:00 | 1192.74 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1165.30 | 2026-03-04 11:30:00 | 1160.11 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1165.30 | 2026-03-04 14:00:00 | 1165.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1168.10 | 2026-03-05 11:30:00 | 1163.82 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1168.10 | 2026-03-05 14:35:00 | 1164.50 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1150.00 | 2026-03-06 11:00:00 | 1153.08 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-13 09:45:00 | 1067.90 | 2026-03-13 10:35:00 | 1061.77 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 09:45:00 | 1067.90 | 2026-03-13 15:20:00 | 1044.80 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2026-03-18 10:55:00 | 1064.10 | 2026-03-18 11:25:00 | 1066.81 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-06 09:35:00 | 1026.35 | 2026-04-06 09:40:00 | 1021.56 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-09 10:35:00 | 1055.05 | 2026-04-09 11:20:00 | 1049.94 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-09 10:35:00 | 1055.05 | 2026-04-09 15:20:00 | 1040.00 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1074.45 | 2026-04-15 11:05:00 | 1070.17 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1074.45 | 2026-04-15 14:20:00 | 1074.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 10:00:00 | 1075.00 | 2026-04-16 11:35:00 | 1071.05 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-16 10:00:00 | 1075.00 | 2026-04-16 14:35:00 | 1071.40 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-27 10:30:00 | 1108.00 | 2026-04-27 13:20:00 | 1110.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-05-05 11:00:00 | 1056.00 | 2026-05-05 11:15:00 | 1058.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-06 11:15:00 | 1065.70 | 2026-05-06 11:45:00 | 1068.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1092.00 | 2026-05-07 11:25:00 | 1087.90 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1092.00 | 2026-05-07 12:15:00 | 1092.00 | STOP_HIT | 0.50 | 0.00% |
