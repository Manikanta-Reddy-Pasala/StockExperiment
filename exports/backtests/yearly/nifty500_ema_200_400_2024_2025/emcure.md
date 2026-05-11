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
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 3 |
| TARGET_HIT | 16 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 40
- **Target hits / Stop hits / Partials:** 16 / 44 / 3
- **Avg / median % per leg:** 1.31% / -0.87%
- **Sum % (uncompounded):** 82.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 17 | 42.5% | 15 | 25 | 0 | 2.49% | 99.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 17 | 42.5% | 15 | 25 | 0 | 2.49% | 99.7% |
| SELL (all) | 23 | 6 | 26.1% | 1 | 19 | 3 | -0.76% | -17.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 6 | 26.1% | 1 | 19 | 3 | -0.76% | -17.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 23 | 36.5% | 16 | 44 | 3 | 1.31% | 82.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 09:15:00 | 1315.00 | 1393.67 | 1393.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 11:15:00 | 1310.00 | 1392.03 | 1393.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 1416.25 | 1388.87 | 1391.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 1416.25 | 1388.87 | 1391.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1416.25 | 1388.87 | 1391.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 1411.75 | 1388.87 | 1391.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1380.00 | 1388.78 | 1391.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 1371.60 | 1388.40 | 1391.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:00:00 | 1369.75 | 1388.21 | 1390.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:45:00 | 1371.35 | 1388.06 | 1390.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 1370.25 | 1388.06 | 1390.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1386.85 | 1384.62 | 1388.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 1386.85 | 1384.62 | 1388.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1391.20 | 1384.69 | 1388.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 1386.00 | 1384.82 | 1388.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 1399.95 | 1384.98 | 1388.87 | SL hit (close>static) qty=1.00 sl=1395.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 1427.85 | 1387.61 | 1387.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1443.00 | 1388.16 | 1387.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1396.35 | 1417.23 | 1404.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 1396.35 | 1417.23 | 1404.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1396.35 | 1417.23 | 1404.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1396.35 | 1417.23 | 1404.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1395.40 | 1417.01 | 1404.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 1393.90 | 1417.01 | 1404.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 1384.55 | 1398.63 | 1396.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 1369.05 | 1398.63 | 1396.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1371.90 | 1395.65 | 1395.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 1375.55 | 1395.65 | 1395.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:45:00 | 1374.10 | 1395.22 | 1395.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 1372.55 | 1394.79 | 1394.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1372.55 | 1394.79 | 1394.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1357.10 | 1394.41 | 1394.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 1019.50 | 1018.85 | 1116.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:00:00 | 1019.50 | 1018.85 | 1116.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1068.90 | 1032.06 | 1100.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:00:00 | 1065.15 | 1032.39 | 1099.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1011.89 | 1032.59 | 1098.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 958.64 | 1028.95 | 1093.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

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

### Cycle 5 — SELL (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 14:15:00 | 1278.00 | 1354.51 | 1354.74 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1409.20 | 1354.84 | 1354.71 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-31 13:15:00)

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

### Cycle 8 — BUY (started 2025-11-10 13:15:00)

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
| SELL | retest2 | 2024-11-27 09:15:00 | 1371.60 | 2024-12-03 09:15:00 | 1399.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-11-27 10:00:00 | 1369.75 | 2024-12-04 09:15:00 | 1399.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-27 10:45:00 | 1371.35 | 2024-12-04 09:15:00 | 1399.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-11-27 11:15:00 | 1370.25 | 2024-12-16 13:15:00 | 1397.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-12-02 15:15:00 | 1386.00 | 2024-12-16 13:15:00 | 1397.05 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-12-03 11:45:00 | 1385.45 | 2024-12-16 14:15:00 | 1412.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-12-03 12:15:00 | 1384.00 | 2024-12-16 14:15:00 | 1412.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-04 10:15:00 | 1385.90 | 2024-12-16 14:15:00 | 1412.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-04 11:15:00 | 1377.50 | 2024-12-16 14:15:00 | 1412.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-12-06 10:30:00 | 1380.00 | 2024-12-23 14:15:00 | 1387.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-06 14:00:00 | 1369.35 | 2024-12-23 14:15:00 | 1387.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-16 10:15:00 | 1375.00 | 2024-12-23 14:15:00 | 1387.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-16 12:15:00 | 1370.65 | 2024-12-24 14:15:00 | 1421.95 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-12-20 09:45:00 | 1371.00 | 2024-12-24 14:15:00 | 1421.95 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2024-12-20 14:00:00 | 1371.50 | 2024-12-24 14:15:00 | 1421.95 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2024-12-20 15:15:00 | 1358.00 | 2024-12-24 14:15:00 | 1421.95 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2025-01-20 11:15:00 | 1375.55 | 2025-01-21 09:15:00 | 1372.55 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-01-20 12:45:00 | 1374.10 | 2025-01-21 09:15:00 | 1372.55 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-04-03 11:00:00 | 1065.15 | 2025-04-04 09:15:00 | 1011.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 11:00:00 | 1065.15 | 2025-04-07 09:15:00 | 958.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-22 11:30:00 | 1064.80 | 2025-04-29 14:15:00 | 1011.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-22 11:30:00 | 1064.80 | 2025-04-30 09:15:00 | 1032.30 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-04-25 09:30:00 | 1050.00 | 2025-05-09 09:15:00 | 997.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:30:00 | 1050.00 | 2025-05-12 09:15:00 | 1037.30 | STOP_HIT | 0.50 | 1.21% |
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
