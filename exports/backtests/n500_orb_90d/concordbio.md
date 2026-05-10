# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1168.40
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 8
- **Avg / median % per leg:** 0.56% / 0.46%
- **Sum % (uncompounded):** 10.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.66% | 6.6% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.66% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.44% | 4.0% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.44% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 11 | 57.9% | 3 | 8 | 8 | 0.56% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 1148.90 | 1155.40 | 0.00 | ORB-short ORB[1150.00,1160.50] vol=1.6x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:25:00 | 1143.61 | 1153.02 | 0.00 | T1 1.5R @ 1143.61 |
| Stop hit — per-position SL triggered | 2026-02-19 10:30:00 | 1148.90 | 1152.39 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 1079.90 | 1085.71 | 0.00 | ORB-short ORB[1085.00,1098.30] vol=1.6x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:05:00 | 1074.98 | 1083.44 | 0.00 | T1 1.5R @ 1074.98 |
| Stop hit — per-position SL triggered | 2026-02-24 13:10:00 | 1079.90 | 1080.89 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 1090.80 | 1084.31 | 0.00 | ORB-long ORB[1072.70,1085.00] vol=1.6x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:50:00 | 1097.16 | 1085.44 | 0.00 | T1 1.5R @ 1097.16 |
| Target hit | 2026-02-25 10:25:00 | 1122.60 | 1123.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1193.10 | 1203.36 | 0.00 | ORB-short ORB[1199.00,1214.00] vol=2.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2026-03-06 11:10:00 | 1197.30 | 1202.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:00:00 | 1162.50 | 1147.56 | 0.00 | ORB-long ORB[1137.00,1153.30] vol=5.7x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:05:00 | 1169.50 | 1153.81 | 0.00 | T1 1.5R @ 1169.50 |
| Stop hit — per-position SL triggered | 2026-03-17 11:10:00 | 1162.50 | 1154.00 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1071.40 | 1067.91 | 0.00 | ORB-long ORB[1056.90,1071.00] vol=1.6x ATR=5.22 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 1066.18 | 1069.28 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 1074.00 | 1066.78 | 0.00 | ORB-long ORB[1054.90,1068.90] vol=7.7x ATR=4.86 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 1069.14 | 1067.57 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 1058.70 | 1051.98 | 0.00 | ORB-long ORB[1044.90,1055.50] vol=3.3x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:45:00 | 1064.01 | 1057.00 | 0.00 | T1 1.5R @ 1064.01 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 1058.70 | 1059.84 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:40:00 | 1036.70 | 1053.45 | 0.00 | ORB-short ORB[1055.00,1068.90] vol=2.2x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 1030.43 | 1049.95 | 0.00 | T1 1.5R @ 1030.43 |
| Target hit | 2026-04-24 15:20:00 | 1015.90 | 1033.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 1038.00 | 1032.84 | 0.00 | ORB-long ORB[1022.80,1035.90] vol=1.6x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:40:00 | 1046.02 | 1035.80 | 0.00 | T1 1.5R @ 1046.02 |
| Target hit | 2026-04-27 10:55:00 | 1060.70 | 1060.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1164.90 | 1183.36 | 0.00 | ORB-short ORB[1191.20,1208.20] vol=2.1x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:45:00 | 1155.74 | 1179.65 | 0.00 | T1 1.5R @ 1155.74 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 1164.90 | 1175.50 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 09:35:00 | 1148.90 | 2026-02-19 10:25:00 | 1143.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-19 09:35:00 | 1148.90 | 2026-02-19 10:30:00 | 1148.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1079.90 | 2026-02-24 12:05:00 | 1074.98 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-24 10:45:00 | 1079.90 | 2026-02-24 13:10:00 | 1079.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:45:00 | 1090.80 | 2026-02-25 09:50:00 | 1097.16 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-25 09:45:00 | 1090.80 | 2026-02-25 10:25:00 | 1122.60 | TARGET_HIT | 0.50 | 2.92% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1193.10 | 2026-03-06 11:10:00 | 1197.30 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-17 11:00:00 | 1162.50 | 2026-03-17 11:05:00 | 1169.50 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-17 11:00:00 | 1162.50 | 2026-03-17 11:10:00 | 1162.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 1071.40 | 2026-04-10 09:55:00 | 1066.18 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-15 09:45:00 | 1074.00 | 2026-04-15 09:55:00 | 1069.14 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-23 09:30:00 | 1058.70 | 2026-04-23 09:45:00 | 1064.01 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-23 09:30:00 | 1058.70 | 2026-04-23 10:35:00 | 1058.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:40:00 | 1036.70 | 2026-04-24 11:00:00 | 1030.43 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-24 10:40:00 | 1036.70 | 2026-04-24 15:20:00 | 1015.90 | TARGET_HIT | 0.50 | 2.01% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1038.00 | 2026-04-27 09:40:00 | 1046.02 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1038.00 | 2026-04-27 10:55:00 | 1060.70 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2026-05-05 10:15:00 | 1164.90 | 2026-05-05 10:45:00 | 1155.74 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2026-05-05 10:15:00 | 1164.90 | 2026-05-05 11:05:00 | 1164.90 | STOP_HIT | 0.50 | 0.00% |
