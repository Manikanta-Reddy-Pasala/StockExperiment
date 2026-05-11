# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2024-07-10 09:15:00 → 2026-05-08 15:15:00 (3161 bars)
- **Last close:** 1646.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 0 |
| TARGET_HIT | 15 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 17 / 22
- **Target hits / Stop hits / Partials:** 15 / 24 / 0
- **Avg / median % per leg:** 2.29% / -0.64%
- **Sum % (uncompounded):** 89.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 17 | 44.7% | 15 | 23 | 0 | 2.63% | 100.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 38 | 17 | 44.7% | 15 | 23 | 0 | 2.63% | 100.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -10.88% | -10.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -10.88% | -10.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 17 | 43.6% | 15 | 24 | 0 | 2.29% | 89.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1395.70 | 1072.60 | 1071.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 1412.80 | 1075.98 | 1073.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1285.00 | 1288.52 | 1227.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 1285.00 | 1288.52 | 1227.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1235.00 | 1283.12 | 1233.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1237.40 | 1283.12 | 1233.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1241.30 | 1282.70 | 1233.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 1244.80 | 1282.70 | 1233.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 1246.00 | 1281.93 | 1233.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 1244.90 | 1281.23 | 1233.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:00:00 | 1247.00 | 1280.53 | 1233.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-15 09:15:00 | 1369.28 | 1282.99 | 1241.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 14:15:00 | 1278.00 | 1354.51 | 1354.74 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1409.20 | 1354.84 | 1354.71 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1321.10 | 1358.41 | 1358.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1313.40 | 1356.85 | 1357.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1373.50 | 1352.74 | 1355.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1373.50 | 1352.74 | 1355.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1373.50 | 1352.74 | 1355.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1373.50 | 1352.74 | 1355.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1381.50 | 1353.03 | 1355.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 1381.50 | 1353.03 | 1355.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1367.30 | 1353.62 | 1355.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 1367.30 | 1353.62 | 1355.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1366.60 | 1353.75 | 1355.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1366.70 | 1353.75 | 1355.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1361.80 | 1354.20 | 1356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1361.80 | 1354.20 | 1356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1426.00 | 1354.89 | 1356.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1424.90 | 1354.89 | 1356.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1431.20 | 1358.13 | 1358.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1456.90 | 1388.39 | 1377.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1391.40 | 1392.25 | 1380.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1381.60 | 1392.25 | 1380.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1382.00 | 1392.15 | 1380.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1385.20 | 1392.15 | 1380.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1384.90 | 1392.08 | 1380.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 1380.90 | 1392.08 | 1380.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1380.00 | 1391.78 | 1380.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1380.00 | 1391.78 | 1380.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1381.00 | 1391.68 | 1380.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 1383.10 | 1391.07 | 1380.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:00:00 | 1386.10 | 1391.11 | 1380.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:45:00 | 1386.60 | 1390.82 | 1380.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1383.20 | 1390.72 | 1380.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1387.00 | 1390.69 | 1380.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1393.30 | 1390.57 | 1380.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 1390.50 | 1390.44 | 1381.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 1377.20 | 1390.15 | 1381.00 | SL hit (close<static) qty=1.00 sl=1378.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 13:45:00 | 1065.00 | 2025-05-22 14:15:00 | 1180.90 | STOP_HIT | 1.00 | -10.88% |
| BUY | retest2 | 2025-07-07 13:15:00 | 1244.80 | 2025-07-15 09:15:00 | 1369.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-07 14:45:00 | 1246.00 | 2025-07-15 09:15:00 | 1370.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 09:45:00 | 1244.90 | 2025-07-15 09:15:00 | 1369.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 12:00:00 | 1247.00 | 2025-07-15 09:15:00 | 1371.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 11:15:00 | 1374.40 | 2025-08-21 09:15:00 | 1511.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:30:00 | 1400.00 | 2025-09-08 13:15:00 | 1362.30 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1379.00 | 2025-09-08 13:15:00 | 1362.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-03 09:30:00 | 1375.20 | 2025-09-08 13:15:00 | 1362.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1378.20 | 2025-09-08 13:15:00 | 1362.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-05 09:45:00 | 1381.40 | 2025-09-10 10:15:00 | 1367.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-05 11:15:00 | 1380.00 | 2025-09-10 10:15:00 | 1367.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-08 09:45:00 | 1378.70 | 2025-09-12 09:15:00 | 1360.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-09 11:00:00 | 1382.90 | 2025-09-22 14:15:00 | 1315.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2025-09-10 09:30:00 | 1381.10 | 2025-09-22 14:15:00 | 1315.00 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2025-09-11 11:15:00 | 1380.00 | 2025-09-22 14:15:00 | 1315.00 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2025-12-17 14:00:00 | 1383.10 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-18 10:00:00 | 1386.10 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-18 14:45:00 | 1386.60 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-19 12:45:00 | 1383.20 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1393.30 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-22 11:30:00 | 1390.50 | 2025-12-22 14:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-23 09:45:00 | 1390.70 | 2025-12-30 11:15:00 | 1369.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-24 12:15:00 | 1409.20 | 2025-12-30 11:15:00 | 1369.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-01-01 14:30:00 | 1399.00 | 2026-01-06 14:15:00 | 1538.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-23 10:15:00 | 1403.20 | 2026-03-02 09:15:00 | 1427.20 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2026-02-23 10:45:00 | 1399.00 | 2026-03-06 12:15:00 | 1429.60 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2026-02-23 11:15:00 | 1398.50 | 2026-03-09 14:15:00 | 1543.52 | TARGET_HIT | 1.00 | 10.37% |
| BUY | retest2 | 2026-02-27 13:45:00 | 1486.00 | 2026-03-09 14:15:00 | 1538.90 | TARGET_HIT | 1.00 | 3.56% |
| BUY | retest2 | 2026-03-04 10:00:00 | 1490.00 | 2026-03-09 14:15:00 | 1538.35 | TARGET_HIT | 1.00 | 3.24% |
| BUY | retest2 | 2026-03-09 12:30:00 | 1487.00 | 2026-03-16 10:15:00 | 1428.80 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2026-03-17 09:15:00 | 1493.00 | 2026-03-23 09:15:00 | 1453.40 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1498.20 | 2026-03-23 09:15:00 | 1453.40 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-03-20 13:15:00 | 1480.70 | 2026-03-27 09:15:00 | 1633.28 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1484.80 | 2026-03-27 10:15:00 | 1642.30 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2026-04-02 10:30:00 | 1494.40 | 2026-04-10 09:15:00 | 1643.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 09:15:00 | 1654.50 | 2026-05-05 09:15:00 | 1819.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 11:45:00 | 1644.70 | 2026-05-05 09:15:00 | 1809.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 09:15:00 | 1655.00 | 2026-05-05 09:15:00 | 1820.50 | TARGET_HIT | 1.00 | 10.00% |
