# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1268.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** -0.06% / -0.41%
- **Sum % (uncompounded):** -0.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.03% | -0.2% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.03% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.06% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 1079.00 | 1061.57 | 0.00 | ORB-long ORB[1046.60,1057.85] vol=1.7x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 1073.48 | 1066.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1165.15 | 1153.24 | 0.00 | ORB-long ORB[1146.00,1162.65] vol=2.3x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 1173.30 | 1158.54 | 0.00 | T1 1.5R @ 1173.30 |
| Target hit | 2026-02-24 10:25:00 | 1182.00 | 1182.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 1142.50 | 1129.25 | 0.00 | ORB-long ORB[1121.60,1134.50] vol=4.7x ATR=4.85 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 1137.65 | 1131.63 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 1125.20 | 1134.96 | 0.00 | ORB-short ORB[1128.60,1144.40] vol=3.7x ATR=4.25 |
| Stop hit — per-position SL triggered | 2026-03-06 11:50:00 | 1129.45 | 1132.49 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 1222.30 | 1216.98 | 0.00 | ORB-long ORB[1208.40,1222.10] vol=2.3x ATR=5.68 |
| Stop hit — per-position SL triggered | 2026-04-17 10:00:00 | 1216.62 | 1217.15 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 1258.75 | 1254.68 | 0.00 | ORB-long ORB[1240.10,1258.20] vol=2.9x ATR=5.71 |
| Stop hit — per-position SL triggered | 2026-04-23 14:30:00 | 1253.04 | 1256.97 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 1278.50 | 1287.98 | 0.00 | ORB-short ORB[1283.50,1297.75] vol=1.5x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-04-28 09:35:00 | 1283.79 | 1287.41 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 1252.15 | 1267.94 | 0.00 | ORB-short ORB[1270.20,1282.25] vol=3.1x ATR=5.42 |
| Stop hit — per-position SL triggered | 2026-04-29 10:00:00 | 1257.57 | 1265.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 1313.30 | 1304.85 | 0.00 | ORB-long ORB[1291.60,1310.00] vol=2.0x ATR=6.54 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 1306.76 | 1306.25 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 1261.20 | 1271.86 | 0.00 | ORB-short ORB[1272.10,1289.30] vol=4.4x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:25:00 | 1251.72 | 1264.10 | 0.00 | T1 1.5R @ 1251.72 |
| Stop hit — per-position SL triggered | 2026-05-08 12:25:00 | 1261.20 | 1260.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:45:00 | 1079.00 | 2026-02-17 10:05:00 | 1073.48 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1165.15 | 2026-02-24 09:55:00 | 1173.30 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1165.15 | 2026-02-24 10:25:00 | 1182.00 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2026-03-05 11:00:00 | 1142.50 | 2026-03-05 11:20:00 | 1137.65 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1125.20 | 2026-03-06 11:50:00 | 1129.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1222.30 | 2026-04-17 10:00:00 | 1216.62 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-23 10:55:00 | 1258.75 | 2026-04-23 14:30:00 | 1253.04 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-28 09:30:00 | 1278.50 | 2026-04-28 09:35:00 | 1283.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-29 09:50:00 | 1252.15 | 2026-04-29 10:00:00 | 1257.57 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-05 09:55:00 | 1313.30 | 2026-05-05 10:05:00 | 1306.76 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1261.20 | 2026-05-08 11:25:00 | 1251.72 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1261.20 | 2026-05-08 12:25:00 | 1261.20 | STOP_HIT | 0.50 | 0.00% |
