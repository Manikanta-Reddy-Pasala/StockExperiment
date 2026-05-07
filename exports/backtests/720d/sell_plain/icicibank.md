# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1279.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 16 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -1.71% / -1.75%
- **Sum % (uncompounded):** -20.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 3 | 25.0% | 0 | 12 | 0 | -1.71% | -20.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 0 | 12 | 0 | -1.71% | -20.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 3 | 25.0% | 0 | 12 | 0 | -1.71% | -20.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.35 | 1279.98 | 1280.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.10 | 1275.07 | 1277.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.06 | 1264.16 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1277.30 | 1251.77 | 1261.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1277.30 | 1251.77 | 1261.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-07 11:15:00 | 1259.75 | 1255.31 | 1262.50 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 1250.80 | 1255.26 | 1262.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-10 15:15:00 | 1259.60 | 1255.37 | 1262.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1258.90 | 1255.40 | 1262.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-02-11 11:15:00 | 1253.55 | 1255.43 | 1262.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:15:00 | 1252.35 | 1255.34 | 1261.96 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-13 11:15:00 | 1258.30 | 1255.14 | 1261.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 1249.15 | 1255.05 | 1261.36 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1259.90 | 1254.98 | 1261.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-14 10:15:00 | 1250.80 | 1254.93 | 1261.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-14 12:15:00 | 1252.20 | 1254.82 | 1261.05 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-02-17 09:15:00 | 1236.00 | 1254.69 | 1260.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:15:00 | 1239.55 | 1254.38 | 1260.65 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-20 11:15:00 | 1247.60 | 1253.46 | 1259.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 13:15:00 | 1249.30 | 1253.39 | 1259.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-13 11:15:00 | 1250.45 | 1235.62 | 1245.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-13 12:15:00 | 1252.10 | 1235.79 | 1245.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-13 14:15:00 | 1251.15 | 1236.11 | 1245.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-17 09:15:00 | 1266.65 | 1236.54 | 1246.09 | ENTRY2 sustain failed after 5460m |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.54 | 1246.09 | SL hit (close>static) qty=1.00 sl=1263.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.54 | 1246.09 | SL hit (close>static) qty=1.00 sl=1263.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1302.65 | 1239.17 | 1247.10 | SL hit (close>static) qty=1.00 sl=1280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1302.65 | 1239.17 | 1247.10 | SL hit (close>static) qty=1.00 sl=1280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1302.65 | 1239.17 | 1247.10 | SL hit (close>static) qty=1.00 sl=1280.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1302.65 | 1239.17 | 1247.10 | SL hit (close>static) qty=1.00 sl=1280.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.00 | 1429.41 | 1429.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.30 | 1416.09 | 1421.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.92 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1402.60 | 1389.31 | 1401.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1402.60 | 1389.31 | 1401.74 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-20 14:15:00 | 1390.20 | 1393.74 | 1402.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 13:15:00 | 1386.00 | 1393.66 | 1402.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1380m) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 1410.30 | 1363.62 | 1370.86 | SL hit (close>static) qty=1.00 sl=1410.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 1371.20 | 1385.83 | 1381.86 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1347.60 | 1378.57 | 1378.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.57 | 1378.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.30 | 1378.23 | 1378.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 1364.90 | 1375.48 | 1376.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 1358.10 | 1375.16 | 1376.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1371.24 | 1374.55 | SL hit (close>static) qty=1.00 sl=1378.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-02 09:15:00 | 1367.10 | 1392.49 | 1387.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 11:15:00 | 1365.50 | 1391.95 | 1387.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 1350.40 | 1390.77 | 1386.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 1349.70 | 1389.98 | 1386.32 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.25 | 1382.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1290.30 | 1283.30 | 1317.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 1282.00 | 1283.32 | 1316.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1350.30 | 1287.89 | 1316.89 | SL hit (close>static) qty=1.00 sl=1333.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-28 14:15:00 | 1290.00 | 1316.06 | 1325.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1290.00 | 1315.54 | 1324.83 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1140m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-07 13:15:00 | 1250.80 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1258.90 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-02-11 13:15:00 | 1252.35 | 2025-03-18 09:15:00 | 1302.65 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-02-13 13:15:00 | 1249.15 | 2025-03-18 09:15:00 | 1302.65 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-02-17 11:15:00 | 1239.55 | 2025-03-18 09:15:00 | 1302.65 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-02-20 13:15:00 | 1249.30 | 2025-03-18 09:15:00 | 1302.65 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-10-21 13:15:00 | 1386.00 | 2026-01-06 12:15:00 | 1410.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-01-19 11:15:00 | 1371.20 | 2026-01-23 13:15:00 | 1347.60 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2026-01-30 12:15:00 | 1358.10 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-02 11:15:00 | 1365.50 | 2026-03-06 14:15:00 | 1312.00 | STOP_HIT | 1.00 | 3.92% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1349.70 | 2026-03-06 14:15:00 | 1312.00 | STOP_HIT | 1.00 | 2.79% |
| SELL | retest2 | 2026-04-09 13:15:00 | 1282.00 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -5.33% |
