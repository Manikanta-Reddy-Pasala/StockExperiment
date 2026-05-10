# Godrej Consumer Products Ltd. (GODREJCP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1041.90
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 4
- **Avg / median % per leg:** -0.04% / -0.20%
- **Sum % (uncompounded):** -0.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.09% | -1.0% |
| SELL @ 2nd Alert (retest1) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.09% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 5 | 25.0% | 1 | 15 | 4 | -0.04% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 1200.50 | 1197.15 | 0.00 | ORB-long ORB[1192.30,1200.20] vol=9.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 1197.92 | 1198.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 1215.60 | 1211.64 | 0.00 | ORB-long ORB[1202.00,1211.50] vol=3.1x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-02-17 11:20:00 | 1213.39 | 1212.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1206.30 | 1206.57 | 0.00 | ORB-short ORB[1206.60,1219.40] vol=3.5x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 1208.68 | 1206.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 1198.20 | 1194.84 | 0.00 | ORB-long ORB[1184.00,1195.60] vol=2.6x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:35:00 | 1202.61 | 1196.60 | 0.00 | T1 1.5R @ 1202.61 |
| Target hit | 2026-02-20 15:20:00 | 1205.20 | 1203.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1214.20 | 1212.37 | 0.00 | ORB-long ORB[1202.10,1211.80] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-02-23 11:25:00 | 1211.73 | 1212.51 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1228.80 | 1222.93 | 0.00 | ORB-long ORB[1215.90,1226.80] vol=8.1x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-02-24 11:30:00 | 1226.36 | 1223.35 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:20:00 | 1231.90 | 1232.95 | 0.00 | ORB-short ORB[1232.10,1240.20] vol=5.0x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 1234.34 | 1233.15 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 1220.00 | 1224.31 | 0.00 | ORB-short ORB[1221.30,1238.90] vol=1.5x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:45:00 | 1215.85 | 1223.10 | 0.00 | T1 1.5R @ 1215.85 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 1220.00 | 1220.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1116.40 | 1125.65 | 0.00 | ORB-short ORB[1126.10,1132.50] vol=2.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 1119.64 | 1125.33 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 1033.10 | 1028.69 | 0.00 | ORB-long ORB[1019.50,1032.30] vol=2.3x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 1029.06 | 1029.68 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:30:00 | 1054.00 | 1042.79 | 0.00 | ORB-long ORB[1038.10,1046.40] vol=2.1x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:35:00 | 1059.51 | 1044.18 | 0.00 | T1 1.5R @ 1059.51 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 1054.00 | 1046.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 1027.60 | 1031.37 | 0.00 | ORB-short ORB[1028.50,1035.90] vol=1.5x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-03-20 10:45:00 | 1030.70 | 1030.74 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:25:00 | 1006.30 | 1008.80 | 0.00 | ORB-short ORB[1011.20,1022.30] vol=3.9x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:45:00 | 1000.52 | 1008.46 | 0.00 | T1 1.5R @ 1000.52 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 1006.30 | 1006.64 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 1123.50 | 1130.02 | 0.00 | ORB-short ORB[1126.30,1139.55] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-04-23 11:55:00 | 1126.46 | 1128.60 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 1083.85 | 1087.42 | 0.00 | ORB-short ORB[1087.05,1093.75] vol=1.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-04-28 11:10:00 | 1086.66 | 1086.81 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1090.00 | 1105.17 | 0.00 | ORB-short ORB[1109.90,1122.00] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-05-06 14:20:00 | 1094.38 | 1097.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 11:00:00 | 1200.50 | 2026-02-13 11:50:00 | 1197.92 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-17 11:15:00 | 1215.60 | 2026-02-17 11:20:00 | 1213.39 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1206.30 | 2026-02-18 11:45:00 | 1208.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-20 11:05:00 | 1198.20 | 2026-02-20 11:35:00 | 1202.61 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-20 11:05:00 | 1198.20 | 2026-02-20 15:20:00 | 1205.20 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-23 11:00:00 | 1214.20 | 2026-02-23 11:25:00 | 1211.73 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1228.80 | 2026-02-24 11:30:00 | 1226.36 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 10:20:00 | 1231.90 | 2026-02-25 11:20:00 | 1234.34 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-27 09:35:00 | 1220.00 | 2026-02-27 09:45:00 | 1215.85 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-27 09:35:00 | 1220.00 | 2026-02-27 10:30:00 | 1220.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1116.40 | 2026-03-06 10:50:00 | 1119.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-16 09:40:00 | 1033.10 | 2026-03-16 09:55:00 | 1029.06 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-18 10:30:00 | 1054.00 | 2026-03-18 10:35:00 | 1059.51 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-18 10:30:00 | 1054.00 | 2026-03-18 11:10:00 | 1054.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1027.60 | 2026-03-20 10:45:00 | 1030.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-24 10:25:00 | 1006.30 | 2026-03-24 10:45:00 | 1000.52 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-24 10:25:00 | 1006.30 | 2026-03-24 11:30:00 | 1006.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:00:00 | 1123.50 | 2026-04-23 11:55:00 | 1126.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-28 10:45:00 | 1083.85 | 2026-04-28 11:10:00 | 1086.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-06 10:45:00 | 1090.00 | 2026-05-06 14:20:00 | 1094.38 | STOP_HIT | 1.00 | -0.40% |
