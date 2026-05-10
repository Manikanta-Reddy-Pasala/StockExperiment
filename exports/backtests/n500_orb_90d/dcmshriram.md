# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1237.00
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
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 6
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.23% | 2.8% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.23% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 8 | 36.4% | 2 | 14 | 6 | 0.12% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 1148.60 | 1153.57 | 0.00 | ORB-short ORB[1150.50,1164.10] vol=3.9x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 1151.96 | 1153.27 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 1105.70 | 1108.67 | 0.00 | ORB-short ORB[1106.00,1116.30] vol=2.4x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:25:00 | 1101.52 | 1108.13 | 0.00 | T1 1.5R @ 1101.52 |
| Target hit | 2026-02-19 15:20:00 | 1078.00 | 1094.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1075.30 | 1082.41 | 0.00 | ORB-short ORB[1081.00,1091.60] vol=1.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-02-23 10:20:00 | 1079.14 | 1082.28 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1064.00 | 1071.41 | 0.00 | ORB-short ORB[1077.00,1092.80] vol=3.7x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 1069.04 | 1071.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 1072.00 | 1078.56 | 0.00 | ORB-short ORB[1074.40,1087.80] vol=3.7x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 1066.01 | 1072.58 | 0.00 | T1 1.5R @ 1066.01 |
| Stop hit — per-position SL triggered | 2026-02-25 12:20:00 | 1072.00 | 1071.33 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:15:00 | 1014.60 | 1000.54 | 0.00 | ORB-long ORB[995.60,1005.00] vol=3.3x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-03-04 11:20:00 | 1008.95 | 1005.95 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:15:00 | 1008.70 | 1001.08 | 0.00 | ORB-long ORB[987.60,1002.00] vol=4.6x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:20:00 | 1014.77 | 1001.62 | 0.00 | T1 1.5R @ 1014.77 |
| Stop hit — per-position SL triggered | 2026-03-06 10:25:00 | 1008.70 | 1001.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 1010.00 | 1005.28 | 0.00 | ORB-long ORB[995.10,1003.90] vol=1.8x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:25:00 | 1015.86 | 1007.16 | 0.00 | T1 1.5R @ 1015.86 |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1010.00 | 1009.85 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 1044.10 | 1055.55 | 0.00 | ORB-short ORB[1052.00,1063.00] vol=2.8x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:20:00 | 1036.43 | 1054.17 | 0.00 | T1 1.5R @ 1036.43 |
| Target hit | 2026-03-20 12:50:00 | 1037.80 | 1036.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 1114.00 | 1109.88 | 0.00 | ORB-long ORB[1094.80,1106.60] vol=2.8x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:30:00 | 1119.14 | 1110.54 | 0.00 | T1 1.5R @ 1119.14 |
| Stop hit — per-position SL triggered | 2026-03-27 11:50:00 | 1114.00 | 1111.93 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:20:00 | 1152.00 | 1158.82 | 0.00 | ORB-short ORB[1154.00,1167.40] vol=2.1x ATR=3.90 |
| Stop hit — per-position SL triggered | 2026-04-08 10:30:00 | 1155.90 | 1158.78 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1155.70 | 1140.83 | 0.00 | ORB-long ORB[1126.00,1135.90] vol=3.8x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 1149.92 | 1146.40 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 1202.70 | 1197.53 | 0.00 | ORB-long ORB[1192.50,1200.00] vol=3.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 1199.56 | 1197.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1197.70 | 1202.17 | 0.00 | ORB-short ORB[1199.10,1215.00] vol=2.1x ATR=3.83 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 1201.53 | 1202.18 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 1200.70 | 1206.43 | 0.00 | ORB-short ORB[1204.00,1214.40] vol=6.2x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1203.39 | 1206.24 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 1270.40 | 1264.44 | 0.00 | ORB-long ORB[1254.00,1269.90] vol=1.6x ATR=5.57 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 1264.83 | 1261.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:10:00 | 1148.60 | 2026-02-11 11:15:00 | 1151.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1105.70 | 2026-02-19 11:25:00 | 1101.52 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1105.70 | 2026-02-19 15:20:00 | 1078.00 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2026-02-23 10:15:00 | 1075.30 | 2026-02-23 10:20:00 | 1079.14 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1064.00 | 2026-02-24 09:35:00 | 1069.04 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-02-25 10:05:00 | 1072.00 | 2026-02-25 11:25:00 | 1066.01 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-25 10:05:00 | 1072.00 | 2026-02-25 12:20:00 | 1072.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 10:15:00 | 1014.60 | 2026-03-04 11:20:00 | 1008.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1008.70 | 2026-03-06 10:20:00 | 1014.77 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1008.70 | 2026-03-06 10:25:00 | 1008.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1010.00 | 2026-03-11 10:25:00 | 1015.86 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1010.00 | 2026-03-11 11:15:00 | 1010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1044.10 | 2026-03-20 10:20:00 | 1036.43 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1044.10 | 2026-03-20 12:50:00 | 1037.80 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-27 11:10:00 | 1114.00 | 2026-03-27 11:30:00 | 1119.14 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-27 11:10:00 | 1114.00 | 2026-03-27 11:50:00 | 1114.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-08 10:20:00 | 1152.00 | 2026-04-08 10:30:00 | 1155.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-15 09:35:00 | 1155.70 | 2026-04-15 09:40:00 | 1149.92 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-21 11:00:00 | 1202.70 | 2026-04-21 11:05:00 | 1199.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-23 09:35:00 | 1197.70 | 2026-04-23 09:40:00 | 1201.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-28 11:00:00 | 1200.70 | 2026-04-28 11:15:00 | 1203.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-08 09:50:00 | 1270.40 | 2026-05-08 09:55:00 | 1264.83 | STOP_HIT | 1.00 | -0.44% |
