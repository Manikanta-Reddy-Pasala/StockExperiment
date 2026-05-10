# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1311.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 5
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 2.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.16% | 2.2% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.16% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.13% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1233.90 | 1225.15 | 0.00 | ORB-long ORB[1210.60,1227.60] vol=1.6x ATR=6.19 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 1227.71 | 1230.06 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 1203.30 | 1194.02 | 0.00 | ORB-long ORB[1180.00,1196.90] vol=2.1x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:15:00 | 1209.95 | 1199.01 | 0.00 | T1 1.5R @ 1209.95 |
| Target hit | 2026-02-17 11:10:00 | 1217.70 | 1218.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:20:00 | 1205.60 | 1198.96 | 0.00 | ORB-long ORB[1191.00,1200.00] vol=2.2x ATR=3.15 |
| Stop hit — per-position SL triggered | 2026-02-23 10:30:00 | 1202.45 | 1199.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1204.30 | 1206.74 | 0.00 | ORB-short ORB[1204.80,1218.90] vol=2.4x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:40:00 | 1199.47 | 1203.76 | 0.00 | T1 1.5R @ 1199.47 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 1204.30 | 1203.78 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 1210.50 | 1203.99 | 0.00 | ORB-long ORB[1199.00,1209.60] vol=1.9x ATR=4.26 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 1206.24 | 1204.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:10:00 | 1120.10 | 1123.96 | 0.00 | ORB-short ORB[1120.40,1132.00] vol=1.6x ATR=4.34 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 1124.44 | 1123.64 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 1121.70 | 1110.44 | 0.00 | ORB-long ORB[1106.00,1119.70] vol=2.0x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-03-20 11:55:00 | 1118.64 | 1111.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 1172.20 | 1164.36 | 0.00 | ORB-long ORB[1155.20,1169.60] vol=4.6x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:25:00 | 1178.10 | 1166.30 | 0.00 | T1 1.5R @ 1178.10 |
| Stop hit — per-position SL triggered | 2026-03-27 11:45:00 | 1172.20 | 1167.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:00:00 | 1260.90 | 1245.57 | 0.00 | ORB-long ORB[1227.00,1243.10] vol=2.4x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:25:00 | 1269.00 | 1250.21 | 0.00 | T1 1.5R @ 1269.00 |
| Target hit | 2026-04-13 15:20:00 | 1274.70 | 1274.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 1268.90 | 1264.43 | 0.00 | ORB-long ORB[1245.70,1258.00] vol=2.1x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:10:00 | 1273.86 | 1265.15 | 0.00 | T1 1.5R @ 1273.86 |
| Stop hit — per-position SL triggered | 2026-04-21 13:50:00 | 1268.90 | 1269.48 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 1293.30 | 1284.45 | 0.00 | ORB-long ORB[1265.80,1282.00] vol=4.1x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 1288.53 | 1288.48 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:55:00 | 1281.20 | 1269.41 | 0.00 | ORB-long ORB[1250.00,1267.80] vol=6.4x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 1276.36 | 1269.98 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:30:00 | 1233.90 | 2026-02-11 09:45:00 | 1227.71 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1203.30 | 2026-02-17 10:15:00 | 1209.95 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1203.30 | 2026-02-17 11:10:00 | 1217.70 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2026-02-23 10:20:00 | 1205.60 | 2026-02-23 10:30:00 | 1202.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1204.30 | 2026-02-24 09:40:00 | 1199.47 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1204.30 | 2026-02-24 09:45:00 | 1204.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 1210.50 | 2026-02-25 09:40:00 | 1206.24 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-18 11:10:00 | 1120.10 | 2026-03-18 11:25:00 | 1124.44 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-20 11:05:00 | 1121.70 | 2026-03-20 11:55:00 | 1118.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-27 11:00:00 | 1172.20 | 2026-03-27 11:25:00 | 1178.10 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-27 11:00:00 | 1172.20 | 2026-03-27 11:45:00 | 1172.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:00:00 | 1260.90 | 2026-04-13 10:25:00 | 1269.00 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-13 10:00:00 | 1260.90 | 2026-04-13 15:20:00 | 1274.70 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1268.90 | 2026-04-21 11:10:00 | 1273.86 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-21 10:45:00 | 1268.90 | 2026-04-21 13:50:00 | 1268.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 1293.30 | 2026-04-22 09:45:00 | 1288.53 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-04 09:55:00 | 1281.20 | 2026-05-04 10:00:00 | 1276.36 | STOP_HIT | 1.00 | -0.38% |
