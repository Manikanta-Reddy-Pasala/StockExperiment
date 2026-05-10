# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1176.60
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 5
- **Avg / median % per leg:** -0.00% / -0.17%
- **Sum % (uncompounded):** -0.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 8 | 38.1% | 3 | 13 | 5 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 1163.80 | 1159.85 | 0.00 | ORB-long ORB[1157.00,1162.30] vol=2.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2026-02-09 10:40:00 | 1160.61 | 1160.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 1159.60 | 1162.06 | 0.00 | ORB-short ORB[1161.10,1167.80] vol=3.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 1156.62 | 1160.43 | 0.00 | T1 1.5R @ 1156.62 |
| Stop hit — per-position SL triggered | 2026-02-10 12:00:00 | 1159.60 | 1159.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1139.80 | 1144.02 | 0.00 | ORB-short ORB[1144.30,1154.00] vol=1.5x ATR=2.66 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 1142.46 | 1143.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1158.50 | 1152.53 | 0.00 | ORB-long ORB[1145.80,1154.50] vol=2.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 1156.24 | 1153.43 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1173.00 | 1176.29 | 0.00 | ORB-short ORB[1175.70,1185.00] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 1175.03 | 1176.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1135.30 | 1145.62 | 0.00 | ORB-short ORB[1146.50,1159.50] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 1138.10 | 1144.22 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 1075.00 | 1065.52 | 0.00 | ORB-long ORB[1048.00,1060.80] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:30:00 | 1080.49 | 1069.79 | 0.00 | T1 1.5R @ 1080.49 |
| Target hit | 2026-03-13 13:15:00 | 1078.20 | 1078.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1072.30 | 1080.61 | 0.00 | ORB-short ORB[1078.10,1093.00] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 1075.90 | 1080.34 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 1100.00 | 1098.08 | 0.00 | ORB-long ORB[1088.00,1096.20] vol=1.8x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 1096.82 | 1097.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:15:00 | 1029.90 | 1030.97 | 0.00 | ORB-short ORB[1030.90,1044.60] vol=1.6x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 1033.15 | 1030.91 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:05:00 | 1024.40 | 1019.24 | 0.00 | ORB-long ORB[1007.20,1022.00] vol=2.0x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-02 11:30:00 | 1021.23 | 1019.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1090.10 | 1087.07 | 0.00 | ORB-long ORB[1080.60,1088.50] vol=3.0x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-10 12:55:00 | 1087.17 | 1088.95 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1121.40 | 1116.05 | 0.00 | ORB-long ORB[1106.70,1119.90] vol=1.5x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:40:00 | 1126.54 | 1120.05 | 0.00 | T1 1.5R @ 1126.54 |
| Target hit | 2026-04-20 13:55:00 | 1122.10 | 1122.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 1178.10 | 1161.43 | 0.00 | ORB-long ORB[1139.80,1156.00] vol=2.0x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:10:00 | 1184.62 | 1172.03 | 0.00 | T1 1.5R @ 1184.62 |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 1178.10 | 1177.09 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 1192.00 | 1190.21 | 0.00 | ORB-long ORB[1174.00,1190.00] vol=2.4x ATR=4.95 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 1187.05 | 1190.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 1157.60 | 1159.96 | 0.00 | ORB-short ORB[1159.00,1166.90] vol=2.2x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:35:00 | 1152.91 | 1159.17 | 0.00 | T1 1.5R @ 1152.91 |
| Target hit | 2026-05-06 14:05:00 | 1152.20 | 1149.81 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 1163.80 | 2026-02-09 10:40:00 | 1160.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 11:10:00 | 1156.62 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 12:00:00 | 1159.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:30:00 | 1139.80 | 2026-02-13 10:55:00 | 1142.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-18 11:00:00 | 1158.50 | 2026-02-18 11:25:00 | 1156.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1173.00 | 2026-02-25 11:05:00 | 1175.03 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1135.30 | 2026-02-27 10:30:00 | 1138.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 11:30:00 | 1080.49 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 13:15:00 | 1078.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-16 10:50:00 | 1072.30 | 2026-03-16 11:00:00 | 1075.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 10:25:00 | 1100.00 | 2026-03-17 10:30:00 | 1096.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-24 11:15:00 | 1029.90 | 2026-03-24 11:40:00 | 1033.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-02 11:05:00 | 1024.40 | 2026-04-02 11:30:00 | 1021.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1090.10 | 2026-04-10 12:55:00 | 1087.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 11:40:00 | 1126.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 13:55:00 | 1122.10 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 12:10:00 | 1184.62 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 14:15:00 | 1178.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:40:00 | 1192.00 | 2026-04-24 09:50:00 | 1187.05 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 10:35:00 | 1152.91 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 14:05:00 | 1152.20 | TARGET_HIT | 0.50 | 0.47% |
