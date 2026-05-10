# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1345.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 7
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 2.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.05% | -0.7% |
| BUY @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.05% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 6 | 66.7% | 2 | 3 | 4 | 0.41% | 3.6% |
| SELL @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 3 | 4 | 0.41% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 10 | 43.5% | 3 | 13 | 7 | 0.13% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1050.00 | 1055.13 | 0.00 | ORB-short ORB[1051.10,1062.40] vol=1.6x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 1045.61 | 1053.49 | 0.00 | T1 1.5R @ 1045.61 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 1050.00 | 1053.09 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:05:00 | 1021.00 | 1015.85 | 0.00 | ORB-long ORB[1009.50,1019.70] vol=1.9x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 12:05:00 | 1025.52 | 1016.86 | 0.00 | T1 1.5R @ 1025.52 |
| Stop hit — per-position SL triggered | 2026-02-13 15:00:00 | 1021.00 | 1019.12 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 1026.00 | 1018.40 | 0.00 | ORB-long ORB[1010.00,1020.40] vol=1.8x ATR=4.06 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 1021.94 | 1020.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 1024.40 | 1030.36 | 0.00 | ORB-short ORB[1026.00,1037.00] vol=3.3x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:20:00 | 1019.64 | 1028.98 | 0.00 | T1 1.5R @ 1019.64 |
| Target hit | 2026-02-19 15:20:00 | 1012.00 | 1018.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 1004.00 | 1013.24 | 0.00 | ORB-short ORB[1008.00,1020.60] vol=2.8x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:15:00 | 998.43 | 1010.08 | 0.00 | T1 1.5R @ 998.43 |
| Stop hit — per-position SL triggered | 2026-02-23 13:40:00 | 1004.00 | 1008.75 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 1020.00 | 1022.17 | 0.00 | ORB-short ORB[1020.70,1029.90] vol=1.8x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:45:00 | 1013.56 | 1019.96 | 0.00 | T1 1.5R @ 1013.56 |
| Target hit | 2026-02-25 15:20:00 | 1012.20 | 1016.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:10:00 | 883.00 | 890.39 | 0.00 | ORB-short ORB[889.15,899.30] vol=1.9x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-03-13 11:30:00 | 886.48 | 887.12 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 908.80 | 903.22 | 0.00 | ORB-long ORB[895.10,907.80] vol=1.6x ATR=4.63 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 904.17 | 904.15 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1151.65 | 1146.53 | 0.00 | ORB-long ORB[1134.40,1150.95] vol=2.0x ATR=6.05 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 1145.60 | 1148.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 1207.60 | 1203.22 | 0.00 | ORB-long ORB[1195.00,1207.50] vol=1.5x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:50:00 | 1212.98 | 1204.52 | 0.00 | T1 1.5R @ 1212.98 |
| Target hit | 2026-04-21 11:25:00 | 1225.00 | 1225.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 1238.45 | 1230.84 | 0.00 | ORB-long ORB[1222.05,1236.95] vol=1.7x ATR=5.11 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 1233.34 | 1231.05 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1248.80 | 1233.86 | 0.00 | ORB-long ORB[1216.70,1233.00] vol=6.2x ATR=6.32 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 1242.48 | 1235.27 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 1222.75 | 1215.22 | 0.00 | ORB-long ORB[1210.00,1222.10] vol=1.6x ATR=4.88 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 1217.87 | 1216.77 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 1261.00 | 1251.55 | 0.00 | ORB-long ORB[1245.00,1256.90] vol=1.9x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:15:00 | 1268.39 | 1256.76 | 0.00 | T1 1.5R @ 1268.39 |
| Stop hit — per-position SL triggered | 2026-05-04 12:05:00 | 1261.00 | 1257.55 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1256.50 | 1250.46 | 0.00 | ORB-long ORB[1243.10,1254.00] vol=2.1x ATR=5.36 |
| Stop hit — per-position SL triggered | 2026-05-05 09:35:00 | 1251.14 | 1251.22 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 1277.40 | 1265.69 | 0.00 | ORB-long ORB[1255.60,1273.90] vol=3.9x ATR=5.66 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 1271.74 | 1267.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 1050.00 | 2026-02-11 09:40:00 | 1045.61 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1050.00 | 2026-02-11 09:45:00 | 1050.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 11:05:00 | 1021.00 | 2026-02-13 12:05:00 | 1025.52 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-13 11:05:00 | 1021.00 | 2026-02-13 15:00:00 | 1021.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:30:00 | 1026.00 | 2026-02-16 09:40:00 | 1021.94 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-19 10:15:00 | 1024.40 | 2026-02-19 10:20:00 | 1019.64 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-19 10:15:00 | 1024.40 | 2026-02-19 15:20:00 | 1012.00 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2026-02-23 10:00:00 | 1004.00 | 2026-02-23 10:15:00 | 998.43 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-23 10:00:00 | 1004.00 | 2026-02-23 13:40:00 | 1004.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:00:00 | 1020.00 | 2026-02-25 12:45:00 | 1013.56 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-02-25 10:00:00 | 1020.00 | 2026-02-25 15:20:00 | 1012.20 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-03-13 10:10:00 | 883.00 | 2026-03-13 11:30:00 | 886.48 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-17 09:55:00 | 908.80 | 2026-03-17 10:30:00 | 904.17 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-10 09:30:00 | 1151.65 | 2026-04-10 10:05:00 | 1145.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-21 09:45:00 | 1207.60 | 2026-04-21 09:50:00 | 1212.98 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-21 09:45:00 | 1207.60 | 2026-04-21 11:25:00 | 1225.00 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2026-04-22 10:05:00 | 1238.45 | 2026-04-22 10:10:00 | 1233.34 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1248.80 | 2026-04-23 09:40:00 | 1242.48 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1222.75 | 2026-04-27 10:10:00 | 1217.87 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-04 11:10:00 | 1261.00 | 2026-05-04 11:15:00 | 1268.39 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-05-04 11:10:00 | 1261.00 | 2026-05-04 12:05:00 | 1261.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 1256.50 | 2026-05-05 09:35:00 | 1251.14 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-08 09:50:00 | 1277.40 | 2026-05-08 09:55:00 | 1271.74 | STOP_HIT | 1.00 | -0.44% |
