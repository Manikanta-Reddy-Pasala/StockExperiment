# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1248.50
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 5
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 2.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.12% | 1.6% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.12% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.28% | 1.1% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.28% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.16% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:55:00 | 1200.30 | 1193.10 | 0.00 | ORB-long ORB[1185.40,1197.50] vol=1.8x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 1196.82 | 1194.12 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:45:00 | 1201.70 | 1194.46 | 0.00 | ORB-long ORB[1189.20,1198.80] vol=2.0x ATR=3.01 |
| Stop hit — per-position SL triggered | 2026-02-11 10:50:00 | 1198.69 | 1194.76 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 11:05:00 | 1194.10 | 1190.64 | 0.00 | ORB-long ORB[1185.00,1192.80] vol=1.6x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:35:00 | 1197.76 | 1192.10 | 0.00 | T1 1.5R @ 1197.76 |
| Target hit | 2026-02-12 14:45:00 | 1195.80 | 1197.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1213.60 | 1219.92 | 0.00 | ORB-short ORB[1219.10,1230.00] vol=2.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:05:00 | 1209.45 | 1219.40 | 0.00 | T1 1.5R @ 1209.45 |
| Stop hit — per-position SL triggered | 2026-02-23 13:20:00 | 1213.60 | 1217.24 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 1228.50 | 1222.92 | 0.00 | ORB-long ORB[1211.10,1224.90] vol=2.0x ATR=2.30 |
| Stop hit — per-position SL triggered | 2026-02-24 10:30:00 | 1226.20 | 1223.71 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 1256.90 | 1247.97 | 0.00 | ORB-long ORB[1237.70,1249.90] vol=1.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 1253.11 | 1249.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1187.90 | 1179.83 | 0.00 | ORB-long ORB[1170.60,1179.20] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 1183.52 | 1182.95 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1188.40 | 1185.86 | 0.00 | ORB-long ORB[1176.60,1186.20] vol=1.7x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:35:00 | 1193.24 | 1186.50 | 0.00 | T1 1.5R @ 1193.24 |
| Stop hit — per-position SL triggered | 2026-03-06 12:00:00 | 1188.40 | 1187.07 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:35:00 | 1194.10 | 1177.40 | 0.00 | ORB-long ORB[1162.20,1180.00] vol=1.8x ATR=4.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:30:00 | 1200.87 | 1181.96 | 0.00 | T1 1.5R @ 1200.87 |
| Target hit | 2026-03-12 15:20:00 | 1222.20 | 1207.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 1155.10 | 1142.41 | 0.00 | ORB-long ORB[1128.00,1143.50] vol=1.9x ATR=5.03 |
| Stop hit — per-position SL triggered | 2026-03-17 11:30:00 | 1150.07 | 1147.43 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:30:00 | 1240.60 | 1234.71 | 0.00 | ORB-long ORB[1224.50,1237.90] vol=1.8x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-04-16 11:10:00 | 1236.56 | 1235.82 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1251.40 | 1257.01 | 0.00 | ORB-short ORB[1252.20,1264.00] vol=2.1x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:45:00 | 1245.06 | 1253.13 | 0.00 | T1 1.5R @ 1245.06 |
| Target hit | 2026-04-24 11:15:00 | 1248.00 | 1247.15 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:55:00 | 1200.30 | 2026-02-10 10:00:00 | 1196.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-11 10:45:00 | 1201.70 | 2026-02-11 10:50:00 | 1198.69 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-12 11:05:00 | 1194.10 | 2026-02-12 11:35:00 | 1197.76 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-12 11:05:00 | 1194.10 | 2026-02-12 14:45:00 | 1195.80 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1213.60 | 2026-02-23 11:05:00 | 1209.45 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-23 11:00:00 | 1213.60 | 2026-02-23 13:20:00 | 1213.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:25:00 | 1228.50 | 2026-02-24 10:30:00 | 1226.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-25 10:25:00 | 1256.90 | 2026-02-25 10:40:00 | 1253.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-05 10:15:00 | 1187.90 | 2026-03-05 11:25:00 | 1183.52 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-06 10:45:00 | 1188.40 | 2026-03-06 11:35:00 | 1193.24 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-06 10:45:00 | 1188.40 | 2026-03-06 12:00:00 | 1188.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 10:35:00 | 1194.10 | 2026-03-12 11:30:00 | 1200.87 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-12 10:35:00 | 1194.10 | 2026-03-12 15:20:00 | 1222.20 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2026-03-17 10:20:00 | 1155.10 | 2026-03-17 11:30:00 | 1150.07 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-16 10:30:00 | 1240.60 | 2026-04-16 11:10:00 | 1236.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1251.40 | 2026-04-24 09:45:00 | 1245.06 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1251.40 | 2026-04-24 11:15:00 | 1248.00 | TARGET_HIT | 0.50 | 0.27% |
