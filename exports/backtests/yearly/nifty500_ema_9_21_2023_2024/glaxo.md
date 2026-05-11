# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2480.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 236 |
| ALERT1 | 151 |
| ALERT2 | 147 |
| ALERT2_SKIP | 80 |
| ALERT3 | 393 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 212 |
| PARTIAL | 21 |
| TARGET_HIT | 17 |
| STOP_HIT | 203 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 241 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 85 / 156
- **Target hits / Stop hits / Partials:** 17 / 203 / 21
- **Avg / median % per leg:** 0.64% / -0.65%
- **Sum % (uncompounded):** 153.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 31 | 36.0% | 16 | 68 | 2 | 1.37% | 117.4% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.93% | 19.7% |
| BUY @ 3rd Alert (retest2) | 81 | 27 | 33.3% | 15 | 66 | 0 | 1.21% | 97.8% |
| SELL (all) | 155 | 54 | 34.8% | 1 | 135 | 19 | 0.23% | 36.4% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.40% | -2.0% |
| SELL @ 3rd Alert (retest2) | 150 | 53 | 35.3% | 1 | 130 | 19 | 0.26% | 38.4% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 7 | 2 | 1.77% | 17.7% |
| retest2 (combined) | 231 | 80 | 34.6% | 16 | 196 | 19 | 0.59% | 136.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 11:15:00 | 1277.40 | 1278.90 | 1279.03 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 1281.25 | 1279.23 | 1279.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 1285.00 | 1280.54 | 1279.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 1286.00 | 1289.70 | 1286.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 11:15:00 | 1286.00 | 1289.70 | 1286.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 1286.00 | 1289.70 | 1286.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:00:00 | 1286.00 | 1289.70 | 1286.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 1295.00 | 1290.76 | 1287.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 15:00:00 | 1297.15 | 1292.24 | 1288.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 10:15:00 | 1285.00 | 1290.15 | 1288.37 | SL hit (close<static) qty=1.00 sl=1286.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 1283.70 | 1287.04 | 1287.33 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 1291.55 | 1287.87 | 1287.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 1296.00 | 1292.28 | 1290.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 11:15:00 | 1309.75 | 1309.82 | 1306.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 12:00:00 | 1309.75 | 1309.82 | 1306.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 1305.25 | 1308.91 | 1306.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 13:00:00 | 1305.25 | 1308.91 | 1306.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 13:15:00 | 1305.00 | 1308.12 | 1306.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:00:00 | 1305.00 | 1308.12 | 1306.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 1303.00 | 1307.10 | 1305.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 1303.00 | 1307.10 | 1305.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 1309.90 | 1307.66 | 1306.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 15:15:00 | 1311.00 | 1306.50 | 1305.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 15:15:00 | 1300.00 | 1304.94 | 1305.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 15:15:00 | 1300.00 | 1304.94 | 1305.53 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 1316.40 | 1307.23 | 1306.52 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 1305.10 | 1306.64 | 1306.80 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 1312.00 | 1307.60 | 1307.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 1316.80 | 1310.41 | 1308.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 1306.55 | 1316.61 | 1312.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 1306.55 | 1316.61 | 1312.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 1306.55 | 1316.61 | 1312.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 12:45:00 | 1338.95 | 1323.52 | 1317.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 13:15:00 | 1347.90 | 1323.52 | 1317.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 14:15:00 | 1367.30 | 1377.63 | 1377.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 1367.30 | 1377.63 | 1377.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 1364.10 | 1373.70 | 1375.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 13:15:00 | 1371.30 | 1370.39 | 1373.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-09 13:30:00 | 1373.00 | 1370.39 | 1373.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 1365.95 | 1369.50 | 1372.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 15:00:00 | 1365.95 | 1369.50 | 1372.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 1387.00 | 1372.28 | 1373.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:00:00 | 1387.00 | 1372.28 | 1373.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 1378.00 | 1373.43 | 1373.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 11:15:00 | 1375.60 | 1373.43 | 1373.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 11:15:00 | 1379.35 | 1374.61 | 1374.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 1379.35 | 1374.61 | 1374.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 13:15:00 | 1385.00 | 1377.31 | 1375.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 12:15:00 | 1405.05 | 1405.52 | 1399.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 12:45:00 | 1405.10 | 1405.52 | 1399.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 1400.00 | 1404.43 | 1400.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 1415.70 | 1404.43 | 1400.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 12:15:00 | 1406.50 | 1427.42 | 1428.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 12:15:00 | 1406.50 | 1427.42 | 1428.99 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 1435.00 | 1424.94 | 1424.41 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 1420.50 | 1426.78 | 1427.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 09:15:00 | 1411.55 | 1422.68 | 1425.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 11:15:00 | 1425.00 | 1422.72 | 1424.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 11:15:00 | 1425.00 | 1422.72 | 1424.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 1425.00 | 1422.72 | 1424.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:45:00 | 1425.60 | 1422.72 | 1424.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 1426.00 | 1423.38 | 1424.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 13:30:00 | 1423.10 | 1422.86 | 1424.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 1401.05 | 1399.95 | 1399.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 13:15:00 | 1401.05 | 1399.95 | 1399.80 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 1395.80 | 1399.88 | 1399.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 1393.85 | 1398.08 | 1399.07 | Break + close below crossover candle low |

### Cycle 16 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 1453.80 | 1405.69 | 1401.83 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 10:15:00 | 1401.00 | 1409.96 | 1410.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 11:15:00 | 1400.00 | 1407.97 | 1409.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 14:15:00 | 1397.10 | 1390.22 | 1395.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 14:15:00 | 1397.10 | 1390.22 | 1395.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1397.10 | 1390.22 | 1395.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 1397.10 | 1390.22 | 1395.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1397.50 | 1391.67 | 1395.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 1386.05 | 1391.67 | 1395.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1401.15 | 1393.57 | 1396.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 09:30:00 | 1383.10 | 1391.26 | 1393.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 10:30:00 | 1385.10 | 1390.18 | 1392.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 13:15:00 | 1400.05 | 1394.99 | 1394.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 13:15:00 | 1400.05 | 1394.99 | 1394.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 1403.90 | 1396.78 | 1395.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 13:15:00 | 1394.00 | 1398.73 | 1397.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 13:15:00 | 1394.00 | 1398.73 | 1397.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 13:15:00 | 1394.00 | 1398.73 | 1397.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 13:45:00 | 1396.55 | 1398.73 | 1397.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 1401.25 | 1399.24 | 1397.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 09:15:00 | 1404.85 | 1400.57 | 1399.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 11:15:00 | 1395.75 | 1398.38 | 1398.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 11:15:00 | 1395.75 | 1398.38 | 1398.61 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1411.90 | 1400.10 | 1399.14 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 11:15:00 | 1392.50 | 1397.96 | 1398.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 13:15:00 | 1392.00 | 1396.00 | 1397.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 15:15:00 | 1401.05 | 1397.01 | 1397.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 15:15:00 | 1401.05 | 1397.01 | 1397.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 1401.05 | 1397.01 | 1397.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 1399.90 | 1397.01 | 1397.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 1399.85 | 1397.58 | 1397.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 10:45:00 | 1394.00 | 1396.46 | 1397.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:00:00 | 1395.05 | 1396.18 | 1397.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:30:00 | 1393.10 | 1397.45 | 1397.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 13:15:00 | 1402.25 | 1398.41 | 1397.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 13:15:00 | 1402.25 | 1398.41 | 1397.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 14:15:00 | 1404.20 | 1399.57 | 1398.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 1399.80 | 1399.85 | 1398.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1399.80 | 1399.85 | 1398.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1399.80 | 1399.85 | 1398.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 1396.60 | 1399.85 | 1398.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 1390.00 | 1397.88 | 1398.04 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 12:15:00 | 1408.35 | 1399.31 | 1398.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 10:15:00 | 1421.00 | 1405.84 | 1402.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-03 12:15:00 | 1399.00 | 1404.98 | 1402.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 12:15:00 | 1399.00 | 1404.98 | 1402.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 1399.00 | 1404.98 | 1402.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 13:00:00 | 1399.00 | 1404.98 | 1402.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 13:15:00 | 1403.35 | 1404.65 | 1402.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:30:00 | 1407.40 | 1404.48 | 1402.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 12:15:00 | 1398.45 | 1402.13 | 1402.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 12:15:00 | 1398.45 | 1402.13 | 1402.15 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 1402.40 | 1402.18 | 1402.18 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 1400.35 | 1401.82 | 1402.01 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 1408.90 | 1402.94 | 1402.47 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 1400.55 | 1402.08 | 1402.19 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 1416.95 | 1402.83 | 1401.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 10:15:00 | 1431.05 | 1408.47 | 1404.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 14:15:00 | 1438.90 | 1440.29 | 1432.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 15:00:00 | 1438.90 | 1440.29 | 1432.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1416.05 | 1435.08 | 1431.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 1418.00 | 1435.08 | 1431.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 1416.65 | 1431.39 | 1429.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:30:00 | 1417.05 | 1431.39 | 1429.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 1419.10 | 1428.93 | 1429.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 15:15:00 | 1415.00 | 1421.54 | 1425.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 1419.00 | 1418.04 | 1422.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 12:30:00 | 1419.70 | 1418.04 | 1422.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 1414.50 | 1415.97 | 1419.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 10:45:00 | 1405.95 | 1414.38 | 1418.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 13:15:00 | 1408.00 | 1413.31 | 1417.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 15:15:00 | 1407.00 | 1411.72 | 1415.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 14:30:00 | 1408.60 | 1409.84 | 1412.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 1408.00 | 1409.47 | 1412.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 1406.00 | 1409.47 | 1412.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1404.60 | 1408.50 | 1411.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:45:00 | 1399.15 | 1407.37 | 1410.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 10:15:00 | 1393.40 | 1402.58 | 1405.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 11:30:00 | 1398.75 | 1404.54 | 1405.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 12:15:00 | 1421.50 | 1407.93 | 1407.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 12:15:00 | 1421.50 | 1407.93 | 1407.16 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 1404.30 | 1410.89 | 1411.61 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-25 14:15:00 | 1420.65 | 1413.44 | 1412.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 09:15:00 | 1438.50 | 1419.02 | 1415.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 13:15:00 | 1422.75 | 1424.67 | 1419.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-28 14:00:00 | 1422.75 | 1424.67 | 1419.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 1412.65 | 1422.26 | 1419.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 15:00:00 | 1412.65 | 1422.26 | 1419.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 1414.00 | 1420.61 | 1418.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 1421.85 | 1421.59 | 1419.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 14:00:00 | 1417.70 | 1418.77 | 1418.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 14:15:00 | 1414.80 | 1417.98 | 1418.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 1414.80 | 1417.98 | 1418.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 1411.00 | 1416.58 | 1417.47 | Break + close below crossover candle low |

### Cycle 36 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 1434.25 | 1418.14 | 1417.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 10:15:00 | 1451.10 | 1432.15 | 1426.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 12:15:00 | 1426.75 | 1431.16 | 1426.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 12:15:00 | 1426.75 | 1431.16 | 1426.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 1426.75 | 1431.16 | 1426.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:00:00 | 1426.75 | 1431.16 | 1426.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 1448.00 | 1434.53 | 1428.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 09:15:00 | 1465.90 | 1435.50 | 1430.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 14:15:00 | 1425.35 | 1441.08 | 1437.17 | SL hit (close<static) qty=1.00 sl=1426.10 alert=retest2 |

### Cycle 37 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 1547.00 | 1563.28 | 1563.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 12:15:00 | 1540.20 | 1552.61 | 1556.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 10:15:00 | 1545.75 | 1544.89 | 1550.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 11:00:00 | 1545.75 | 1544.89 | 1550.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 1542.00 | 1544.30 | 1549.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 14:00:00 | 1533.10 | 1542.06 | 1547.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 11:45:00 | 1532.50 | 1536.31 | 1542.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 13:15:00 | 1531.65 | 1536.91 | 1541.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 1530.15 | 1533.06 | 1538.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 1524.70 | 1531.39 | 1537.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 13:00:00 | 1518.70 | 1526.90 | 1533.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 13:30:00 | 1519.00 | 1524.58 | 1531.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 1512.60 | 1521.22 | 1528.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 09:15:00 | 1543.10 | 1525.59 | 1530.24 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 1564.00 | 1538.92 | 1535.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 12:15:00 | 1575.00 | 1552.55 | 1543.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 1560.05 | 1562.38 | 1553.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 11:45:00 | 1560.00 | 1562.38 | 1553.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 1549.30 | 1559.76 | 1553.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 1549.30 | 1559.76 | 1553.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 1557.90 | 1559.39 | 1553.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 1566.00 | 1556.70 | 1553.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:30:00 | 1560.40 | 1557.57 | 1554.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 14:15:00 | 1539.95 | 1551.86 | 1552.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 14:15:00 | 1539.95 | 1551.86 | 1552.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 1536.90 | 1546.49 | 1549.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 10:15:00 | 1548.85 | 1546.96 | 1549.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 10:15:00 | 1548.85 | 1546.96 | 1549.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 1548.85 | 1546.96 | 1549.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:00:00 | 1548.85 | 1546.96 | 1549.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 1551.15 | 1548.12 | 1549.90 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 13:15:00 | 1566.95 | 1551.89 | 1551.45 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 1543.35 | 1550.81 | 1551.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 1531.70 | 1545.69 | 1548.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 14:15:00 | 1551.90 | 1545.26 | 1546.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 14:15:00 | 1551.90 | 1545.26 | 1546.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 1551.90 | 1545.26 | 1546.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 15:00:00 | 1551.90 | 1545.26 | 1546.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 1549.00 | 1546.01 | 1547.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 1556.25 | 1546.01 | 1547.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1555.95 | 1548.00 | 1547.85 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 12:15:00 | 1537.20 | 1545.74 | 1546.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 13:15:00 | 1533.30 | 1543.25 | 1545.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 1541.55 | 1539.75 | 1543.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 1541.55 | 1539.75 | 1543.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 1541.55 | 1539.75 | 1543.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:45:00 | 1543.80 | 1539.75 | 1543.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 1531.35 | 1538.07 | 1542.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:45:00 | 1541.90 | 1538.07 | 1542.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1543.20 | 1531.45 | 1535.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:00:00 | 1543.20 | 1531.45 | 1535.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 1533.90 | 1531.94 | 1535.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 11:30:00 | 1530.00 | 1532.18 | 1535.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 12:30:00 | 1531.80 | 1533.46 | 1535.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 13:15:00 | 1530.00 | 1533.46 | 1535.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 09:15:00 | 1572.95 | 1538.99 | 1537.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 1572.95 | 1538.99 | 1537.35 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 14:15:00 | 1533.55 | 1540.40 | 1540.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 1525.00 | 1536.10 | 1538.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 09:15:00 | 1524.50 | 1524.00 | 1530.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 09:45:00 | 1524.70 | 1524.00 | 1530.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1534.35 | 1525.57 | 1528.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 1534.35 | 1525.57 | 1528.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 1532.00 | 1526.86 | 1528.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 1528.90 | 1526.86 | 1528.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 10:15:00 | 1538.55 | 1529.80 | 1529.78 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 1524.00 | 1528.73 | 1529.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 1522.05 | 1526.73 | 1528.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 12:15:00 | 1400.00 | 1397.98 | 1405.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-03 12:45:00 | 1400.10 | 1397.98 | 1405.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 1407.55 | 1400.22 | 1405.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:45:00 | 1406.45 | 1400.22 | 1405.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 1400.00 | 1400.18 | 1405.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:15:00 | 1428.00 | 1400.18 | 1405.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 1429.00 | 1405.94 | 1407.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:30:00 | 1432.15 | 1405.94 | 1407.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 1432.10 | 1411.17 | 1409.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 1444.65 | 1417.87 | 1412.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 1426.10 | 1429.69 | 1422.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 12:00:00 | 1426.10 | 1429.69 | 1422.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 1425.95 | 1428.94 | 1423.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:30:00 | 1424.00 | 1428.94 | 1423.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 1423.85 | 1427.93 | 1423.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:00:00 | 1423.85 | 1427.93 | 1423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 1422.05 | 1426.75 | 1423.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:45:00 | 1421.25 | 1426.75 | 1423.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 1423.00 | 1426.00 | 1423.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 1427.00 | 1426.00 | 1423.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-15 15:15:00 | 1569.70 | 1528.04 | 1504.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 1639.00 | 1672.27 | 1675.03 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 10:15:00 | 1685.80 | 1678.54 | 1677.62 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 12:15:00 | 1671.65 | 1676.92 | 1677.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 1659.50 | 1673.43 | 1675.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 1674.90 | 1647.36 | 1655.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 1674.90 | 1647.36 | 1655.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1674.90 | 1647.36 | 1655.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 1674.90 | 1647.36 | 1655.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 1679.80 | 1653.85 | 1657.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 1686.85 | 1653.85 | 1657.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 1666.50 | 1660.64 | 1660.47 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 15:15:00 | 1654.50 | 1659.55 | 1660.06 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 1698.15 | 1667.27 | 1663.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1708.20 | 1685.40 | 1676.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 13:15:00 | 1695.00 | 1698.72 | 1686.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 14:00:00 | 1695.00 | 1698.72 | 1686.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 1688.50 | 1700.42 | 1694.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 15:00:00 | 1688.50 | 1700.42 | 1694.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 1692.00 | 1698.74 | 1694.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 09:30:00 | 1685.45 | 1696.39 | 1693.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 1686.90 | 1694.49 | 1693.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:45:00 | 1686.90 | 1694.49 | 1693.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 1685.65 | 1691.43 | 1691.96 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 1700.00 | 1692.33 | 1692.10 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 12:15:00 | 1685.00 | 1691.63 | 1691.91 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 09:15:00 | 1699.85 | 1691.84 | 1691.79 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1689.95 | 1691.55 | 1691.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 09:15:00 | 1678.65 | 1687.79 | 1689.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 12:15:00 | 1676.45 | 1675.31 | 1680.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 1680.55 | 1676.46 | 1679.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1680.55 | 1676.46 | 1679.51 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 13:15:00 | 1698.50 | 1682.58 | 1681.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 14:15:00 | 1730.00 | 1692.07 | 1685.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 12:15:00 | 1765.45 | 1767.49 | 1753.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 12:30:00 | 1763.55 | 1767.49 | 1753.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 1751.60 | 1764.23 | 1754.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 15:00:00 | 1751.60 | 1764.23 | 1754.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 1735.05 | 1758.40 | 1752.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:15:00 | 1755.65 | 1758.40 | 1752.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1755.35 | 1753.24 | 1751.03 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 14:15:00 | 1739.00 | 1748.81 | 1749.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 15:15:00 | 1738.95 | 1746.83 | 1748.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 1750.70 | 1747.61 | 1748.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 1750.70 | 1747.61 | 1748.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1750.70 | 1747.61 | 1748.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:30:00 | 1759.80 | 1747.61 | 1748.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 1743.95 | 1746.88 | 1748.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:30:00 | 1749.85 | 1746.88 | 1748.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1742.00 | 1745.18 | 1747.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 12:45:00 | 1748.95 | 1745.18 | 1747.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 1734.10 | 1720.51 | 1727.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:45:00 | 1730.60 | 1720.51 | 1727.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 1735.00 | 1723.41 | 1728.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 1737.15 | 1723.41 | 1728.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 1729.00 | 1727.18 | 1729.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 11:15:00 | 1725.15 | 1727.18 | 1729.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 11:45:00 | 1720.95 | 1727.26 | 1729.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 1723.05 | 1725.20 | 1727.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 14:15:00 | 1752.00 | 1733.25 | 1731.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 1752.00 | 1733.25 | 1731.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 1758.00 | 1740.72 | 1735.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 1850.00 | 1851.91 | 1820.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 09:15:00 | 1860.20 | 1851.91 | 1820.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 12:15:00 | 1953.21 | 1888.40 | 1849.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-12-29 14:15:00 | 1892.10 | 1895.41 | 1859.83 | SL hit (close<ema200) qty=0.50 sl=1895.41 alert=retest1 |

### Cycle 63 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 2083.95 | 2106.10 | 2107.15 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 09:15:00 | 2210.00 | 2123.34 | 2114.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 15:15:00 | 2264.05 | 2240.28 | 2213.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 2236.75 | 2239.57 | 2215.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 13:15:00 | 2254.00 | 2251.54 | 2240.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 2254.00 | 2251.54 | 2240.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 09:30:00 | 2318.20 | 2263.91 | 2249.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 15:00:00 | 2277.75 | 2271.69 | 2259.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:00:00 | 2285.00 | 2274.08 | 2262.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 10:15:00 | 2280.00 | 2295.46 | 2293.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 10:15:00 | 2276.00 | 2291.57 | 2292.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 10:15:00 | 2276.00 | 2291.57 | 2292.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 15:15:00 | 2265.00 | 2278.62 | 2284.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 2284.65 | 2219.69 | 2234.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 2284.65 | 2219.69 | 2234.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2284.65 | 2219.69 | 2234.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 2284.65 | 2219.69 | 2234.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 2234.25 | 2222.60 | 2234.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:45:00 | 2218.45 | 2223.99 | 2231.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 10:15:00 | 2260.00 | 2214.03 | 2208.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 10:15:00 | 2260.00 | 2214.03 | 2208.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 09:15:00 | 2267.05 | 2245.27 | 2229.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 2215.75 | 2241.80 | 2230.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 2215.75 | 2241.80 | 2230.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 2215.75 | 2241.80 | 2230.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 2215.75 | 2241.80 | 2230.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 2214.95 | 2236.43 | 2229.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:00:00 | 2238.90 | 2236.92 | 2230.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 2239.95 | 2232.81 | 2229.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-05 11:15:00 | 2462.79 | 2351.93 | 2298.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 2409.65 | 2435.17 | 2437.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 11:15:00 | 2366.55 | 2411.38 | 2425.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 2397.60 | 2392.33 | 2409.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 2397.60 | 2392.33 | 2409.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 2397.60 | 2392.33 | 2409.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:30:00 | 2430.10 | 2392.33 | 2409.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 2212.00 | 2185.76 | 2207.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:15:00 | 2219.30 | 2185.76 | 2207.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 2257.00 | 2200.01 | 2212.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:30:00 | 2261.15 | 2200.01 | 2212.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 2253.60 | 2210.72 | 2215.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 11:15:00 | 2243.85 | 2210.72 | 2215.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 11:15:00 | 2262.70 | 2221.12 | 2220.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 11:15:00 | 2262.70 | 2221.12 | 2220.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 2266.40 | 2254.89 | 2245.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 09:15:00 | 2263.45 | 2271.65 | 2260.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 2263.45 | 2271.65 | 2260.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 2263.45 | 2271.65 | 2260.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:00:00 | 2263.45 | 2271.65 | 2260.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 2264.35 | 2270.19 | 2261.05 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 2239.80 | 2257.28 | 2257.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 2188.00 | 2243.42 | 2251.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 12:15:00 | 2177.50 | 2174.65 | 2198.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 12:30:00 | 2176.25 | 2174.65 | 2198.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 2161.00 | 2168.59 | 2187.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 2147.25 | 2164.23 | 2170.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 11:15:00 | 2199.70 | 2160.88 | 2163.27 | SL hit (close>static) qty=1.00 sl=2188.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-02-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 12:15:00 | 2260.00 | 2180.70 | 2172.06 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 11:15:00 | 2154.70 | 2171.03 | 2171.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 12:15:00 | 2142.70 | 2165.36 | 2168.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 11:15:00 | 2173.95 | 2159.62 | 2163.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 2173.95 | 2159.62 | 2163.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 2173.95 | 2159.62 | 2163.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 12:00:00 | 2173.95 | 2159.62 | 2163.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 2172.00 | 2162.10 | 2164.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 2155.30 | 2162.10 | 2164.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 2144.45 | 2158.57 | 2162.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:15:00 | 2133.70 | 2155.20 | 2160.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 10:15:00 | 2129.55 | 2151.61 | 2156.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:15:00 | 2135.80 | 2140.88 | 2149.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:15:00 | 2027.01 | 2062.75 | 2092.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:15:00 | 2029.01 | 2062.75 | 2092.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 2065.20 | 2056.38 | 2077.09 | SL hit (close>ema200) qty=0.50 sl=2056.38 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 1982.00 | 1962.51 | 1961.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 12:15:00 | 1991.65 | 1968.62 | 1964.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 14:15:00 | 1946.50 | 1966.02 | 1964.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 14:15:00 | 1946.50 | 1966.02 | 1964.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 1946.50 | 1966.02 | 1964.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 15:00:00 | 1946.50 | 1966.02 | 1964.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 1978.80 | 1968.57 | 1965.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:30:00 | 1998.75 | 1972.05 | 1967.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 1992.65 | 1972.05 | 1967.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 2014.80 | 1992.63 | 1982.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 10:15:00 | 1955.05 | 1983.64 | 1986.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 10:15:00 | 1955.05 | 1983.64 | 1986.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 13:15:00 | 1941.00 | 1965.92 | 1976.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 14:15:00 | 1967.80 | 1966.30 | 1975.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 1967.80 | 1966.30 | 1975.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1951.55 | 1963.46 | 1972.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 14:45:00 | 1935.00 | 1954.79 | 1965.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 09:45:00 | 1927.55 | 1944.95 | 1959.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 11:00:00 | 1935.00 | 1923.12 | 1936.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 13:15:00 | 1937.10 | 1930.92 | 1938.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1943.00 | 1933.53 | 1938.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 1947.75 | 1933.53 | 1938.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 1950.00 | 1936.82 | 1939.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:15:00 | 1931.75 | 1936.82 | 1939.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 1972.65 | 1943.99 | 1942.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 1972.65 | 1943.99 | 1942.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 1991.25 | 1967.50 | 1955.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 13:15:00 | 1972.55 | 1972.78 | 1964.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 1973.10 | 1971.73 | 1965.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1973.10 | 1971.73 | 1965.78 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 10:15:00 | 1940.00 | 1962.32 | 1964.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 11:15:00 | 1933.30 | 1956.52 | 1961.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 13:15:00 | 1933.00 | 1932.72 | 1943.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 13:15:00 | 1933.00 | 1932.72 | 1943.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 1933.00 | 1932.72 | 1943.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 13:30:00 | 1935.75 | 1932.72 | 1943.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 1938.00 | 1933.77 | 1942.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 15:00:00 | 1925.45 | 1939.55 | 1942.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 09:30:00 | 1925.00 | 1935.35 | 1940.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 1926.60 | 1892.81 | 1888.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 1926.60 | 1892.81 | 1888.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 12:15:00 | 1957.20 | 1905.69 | 1894.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 15:15:00 | 1940.00 | 1940.11 | 1925.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:15:00 | 1956.05 | 1940.11 | 1925.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 13:15:00 | 2053.85 | 2006.03 | 1966.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-04-29 09:15:00 | 2151.66 | 2118.35 | 2078.26 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 77 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 2092.25 | 2097.79 | 2098.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 13:15:00 | 2084.00 | 2095.03 | 2097.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 14:15:00 | 2098.00 | 2095.63 | 2097.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 14:15:00 | 2098.00 | 2095.63 | 2097.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 2098.00 | 2095.63 | 2097.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 14:45:00 | 2102.60 | 2095.63 | 2097.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 2106.00 | 2097.70 | 2097.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:15:00 | 2126.60 | 2097.70 | 2097.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 2136.00 | 2105.36 | 2101.43 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 2090.25 | 2100.21 | 2100.82 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 2106.75 | 2101.48 | 2101.30 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 2086.50 | 2098.49 | 2099.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 13:15:00 | 2079.60 | 2094.71 | 2098.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 2094.00 | 2092.75 | 2096.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:15:00 | 2088.20 | 2092.75 | 2096.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2103.30 | 2094.86 | 2097.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:45:00 | 2105.90 | 2094.86 | 2097.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2098.20 | 2095.53 | 2097.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:30:00 | 2100.00 | 2095.53 | 2097.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 2045.10 | 2085.44 | 2092.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 13:00:00 | 2042.80 | 2076.91 | 2087.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:00:00 | 2040.55 | 2069.64 | 2083.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 15:15:00 | 2036.60 | 2067.33 | 2081.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 11:00:00 | 2035.45 | 2055.76 | 2071.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1983.65 | 1982.54 | 2000.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 1984.40 | 1982.54 | 2000.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 2008.00 | 1987.81 | 1999.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 2008.00 | 1987.81 | 1999.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 1998.00 | 1989.85 | 1999.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:00:00 | 1993.70 | 1995.26 | 1999.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 13:15:00 | 1994.55 | 1996.39 | 1999.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 14:15:00 | 1992.00 | 1996.71 | 1998.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 15:00:00 | 1989.95 | 1986.82 | 1991.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1983.00 | 1986.07 | 1990.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:15:00 | 1980.10 | 1986.07 | 1990.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 2037.45 | 2000.49 | 1996.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 12:15:00 | 2037.45 | 2000.49 | 1996.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 11:15:00 | 2057.00 | 2028.01 | 2013.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 10:15:00 | 2316.60 | 2331.78 | 2265.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 10:45:00 | 2300.00 | 2331.78 | 2265.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 2432.10 | 2451.35 | 2420.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 2432.85 | 2444.77 | 2420.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 2422.70 | 2440.35 | 2420.51 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 2358.25 | 2404.15 | 2408.30 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 12:15:00 | 2462.00 | 2416.94 | 2411.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 2543.50 | 2460.15 | 2435.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 11:15:00 | 2574.25 | 2576.75 | 2541.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:45:00 | 2575.00 | 2576.75 | 2541.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 2643.95 | 2587.52 | 2555.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 2520.50 | 2587.52 | 2555.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2610.80 | 2602.01 | 2568.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 2590.90 | 2602.01 | 2568.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2556.75 | 2592.96 | 2567.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:00:00 | 2556.75 | 2592.96 | 2567.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 2539.40 | 2582.25 | 2564.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 2539.40 | 2582.25 | 2564.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 13:15:00 | 2483.40 | 2551.16 | 2552.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 2409.35 | 2503.62 | 2528.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 2474.40 | 2436.63 | 2475.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 2474.40 | 2436.63 | 2475.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2474.40 | 2436.63 | 2475.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 2474.40 | 2436.63 | 2475.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 2479.65 | 2445.23 | 2475.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 2494.40 | 2445.23 | 2475.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 2452.95 | 2446.78 | 2473.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 2473.60 | 2446.78 | 2473.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 2479.95 | 2453.41 | 2474.12 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 2538.35 | 2483.85 | 2480.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 2560.00 | 2511.76 | 2497.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 2615.00 | 2630.97 | 2595.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:00:00 | 2615.00 | 2630.97 | 2595.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 2594.80 | 2619.89 | 2598.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 2594.80 | 2619.89 | 2598.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 2619.00 | 2619.71 | 2600.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 2640.15 | 2621.08 | 2604.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:00:00 | 2650.00 | 2663.37 | 2661.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 15:15:00 | 2637.25 | 2658.15 | 2659.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 15:15:00 | 2637.25 | 2658.15 | 2659.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 2583.55 | 2643.23 | 2652.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 2579.70 | 2558.15 | 2580.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 10:15:00 | 2579.70 | 2558.15 | 2580.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2579.70 | 2558.15 | 2580.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 2567.50 | 2558.15 | 2580.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 2564.25 | 2559.37 | 2579.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 2535.80 | 2554.65 | 2575.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:00:00 | 2557.30 | 2555.18 | 2573.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:30:00 | 2546.20 | 2551.34 | 2570.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 2653.15 | 2569.89 | 2575.28 | SL hit (close>static) qty=1.00 sl=2593.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 2669.90 | 2589.89 | 2583.88 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 2599.00 | 2620.55 | 2623.24 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 12:15:00 | 2656.85 | 2629.63 | 2626.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 2669.65 | 2638.33 | 2631.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 2650.80 | 2664.15 | 2651.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 14:15:00 | 2650.80 | 2664.15 | 2651.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 2650.80 | 2664.15 | 2651.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 2650.80 | 2664.15 | 2651.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 2650.00 | 2661.32 | 2651.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 2662.60 | 2661.32 | 2651.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 2632.10 | 2650.77 | 2649.14 | SL hit (close<static) qty=1.00 sl=2641.20 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 2611.65 | 2642.95 | 2645.73 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 2668.50 | 2640.20 | 2638.37 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 2621.50 | 2634.79 | 2636.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 2606.30 | 2629.09 | 2633.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 10:15:00 | 2644.90 | 2625.47 | 2630.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 2644.90 | 2625.47 | 2630.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 2644.90 | 2625.47 | 2630.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 2644.90 | 2625.47 | 2630.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 2637.90 | 2627.96 | 2630.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 2615.00 | 2623.48 | 2628.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 12:15:00 | 2583.50 | 2552.73 | 2551.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 2583.50 | 2552.73 | 2551.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 2611.40 | 2567.21 | 2559.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 2571.65 | 2573.61 | 2564.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 13:00:00 | 2571.65 | 2573.61 | 2564.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 2559.80 | 2570.85 | 2564.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 2559.80 | 2570.85 | 2564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 2579.20 | 2572.52 | 2565.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:15:00 | 2553.30 | 2572.52 | 2565.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 2553.30 | 2568.68 | 2564.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 2548.15 | 2568.68 | 2564.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2563.00 | 2567.54 | 2564.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2551.55 | 2567.54 | 2564.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2556.90 | 2565.41 | 2563.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 2556.90 | 2565.41 | 2563.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 2557.80 | 2563.89 | 2563.09 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 13:15:00 | 2560.00 | 2562.49 | 2562.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 14:15:00 | 2554.65 | 2560.92 | 2561.84 | Break + close below crossover candle low |

### Cycle 96 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 2611.75 | 2570.11 | 2565.79 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 2531.85 | 2572.24 | 2574.89 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 2600.00 | 2572.59 | 2570.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 2627.50 | 2583.58 | 2575.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 13:15:00 | 2700.00 | 2707.58 | 2675.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 14:00:00 | 2700.00 | 2707.58 | 2675.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 2669.55 | 2702.26 | 2690.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 2669.55 | 2702.26 | 2690.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 2685.85 | 2698.98 | 2690.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 2695.05 | 2698.98 | 2690.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 2695.00 | 2723.49 | 2724.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 2695.00 | 2723.49 | 2724.26 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 2768.20 | 2732.43 | 2728.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 10:15:00 | 2811.80 | 2763.54 | 2750.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 2778.00 | 2802.96 | 2781.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 2778.00 | 2802.96 | 2781.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 2778.00 | 2802.96 | 2781.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 2778.00 | 2802.96 | 2781.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 2779.15 | 2798.20 | 2781.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 2814.25 | 2779.26 | 2776.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 2844.60 | 2803.20 | 2795.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 2834.70 | 2874.24 | 2879.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 2834.70 | 2874.24 | 2879.10 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 2935.00 | 2866.18 | 2861.46 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 2880.15 | 2896.73 | 2897.96 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 2979.65 | 2913.64 | 2905.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 3012.75 | 2933.46 | 2915.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 2971.00 | 3009.78 | 2970.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 2971.00 | 3009.78 | 2970.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 2971.00 | 3009.78 | 2970.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:00:00 | 2971.00 | 3009.78 | 2970.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 2960.00 | 2999.83 | 2969.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:45:00 | 2965.65 | 2999.83 | 2969.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 2946.00 | 2989.06 | 2967.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:45:00 | 2948.05 | 2989.06 | 2967.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 2938.00 | 2978.85 | 2964.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:45:00 | 2938.80 | 2978.85 | 2964.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 2911.50 | 2949.11 | 2952.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 2905.40 | 2940.37 | 2948.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 10:15:00 | 2945.30 | 2941.35 | 2948.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 10:15:00 | 2945.30 | 2941.35 | 2948.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2945.30 | 2941.35 | 2948.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2945.30 | 2941.35 | 2948.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2944.45 | 2941.97 | 2948.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 12:15:00 | 2926.05 | 2941.97 | 2948.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 14:30:00 | 2935.15 | 2938.83 | 2944.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 15:15:00 | 2950.00 | 2941.07 | 2945.23 | SL hit (close>static) qty=1.00 sl=2949.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 2998.00 | 2952.45 | 2950.03 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 2902.50 | 2943.08 | 2947.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 2883.00 | 2926.09 | 2938.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2790.65 | 2790.14 | 2824.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 12:15:00 | 2767.65 | 2783.59 | 2815.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 13:00:00 | 2767.45 | 2780.36 | 2810.85 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 14:15:00 | 2766.00 | 2777.89 | 2806.95 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 2780.00 | 2760.24 | 2778.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-02 14:15:00 | 2780.00 | 2760.24 | 2778.36 | SL hit (close>ema400) qty=1.00 sl=2778.36 alert=retest1 |

### Cycle 108 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 2832.85 | 2794.12 | 2789.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 2871.35 | 2824.79 | 2807.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 12:15:00 | 2823.95 | 2844.83 | 2832.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 12:15:00 | 2823.95 | 2844.83 | 2832.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 2823.95 | 2844.83 | 2832.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 2823.95 | 2844.83 | 2832.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 2807.80 | 2837.42 | 2830.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 2807.80 | 2837.42 | 2830.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 2782.95 | 2819.65 | 2823.46 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 2833.25 | 2819.25 | 2818.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 2859.45 | 2827.29 | 2822.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 12:15:00 | 2839.50 | 2841.06 | 2832.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:00:00 | 2839.50 | 2841.06 | 2832.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 2834.85 | 2839.82 | 2832.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 2834.85 | 2839.82 | 2832.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 2828.50 | 2837.56 | 2832.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 2828.50 | 2837.56 | 2832.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 2815.10 | 2833.07 | 2830.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 2860.95 | 2833.07 | 2830.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 15:15:00 | 2828.00 | 2852.47 | 2855.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 2828.00 | 2852.47 | 2855.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 2790.00 | 2839.97 | 2849.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 2810.90 | 2804.80 | 2825.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:45:00 | 2796.75 | 2804.80 | 2825.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 2825.70 | 2806.95 | 2821.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 2825.70 | 2806.95 | 2821.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 2847.00 | 2814.96 | 2823.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 2847.00 | 2814.96 | 2823.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 2790.35 | 2810.04 | 2820.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 15:15:00 | 2786.00 | 2804.20 | 2816.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 2733.60 | 2719.89 | 2719.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 2733.60 | 2719.89 | 2719.19 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 2715.15 | 2722.90 | 2722.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 2700.15 | 2717.24 | 2720.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 2746.25 | 2720.28 | 2720.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 2746.25 | 2720.28 | 2720.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 2746.25 | 2720.28 | 2720.97 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 2745.60 | 2725.35 | 2723.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 12:15:00 | 2764.90 | 2737.75 | 2729.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 13:15:00 | 2721.25 | 2734.45 | 2728.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 13:15:00 | 2721.25 | 2734.45 | 2728.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 2721.25 | 2734.45 | 2728.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 2721.25 | 2734.45 | 2728.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 2747.70 | 2737.10 | 2730.47 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 2715.05 | 2728.70 | 2728.91 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 2733.05 | 2729.57 | 2729.28 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 2725.75 | 2729.05 | 2729.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 2708.05 | 2724.63 | 2727.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 2734.35 | 2720.75 | 2723.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 14:15:00 | 2734.35 | 2720.75 | 2723.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 2734.35 | 2720.75 | 2723.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 2734.35 | 2720.75 | 2723.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 2740.00 | 2724.60 | 2725.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 2712.30 | 2724.60 | 2725.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 2739.20 | 2727.52 | 2726.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 2739.20 | 2727.52 | 2726.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 2758.70 | 2733.76 | 2729.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 2708.45 | 2739.45 | 2736.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 2708.45 | 2739.45 | 2736.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 2708.45 | 2739.45 | 2736.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 2708.45 | 2739.45 | 2736.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 2625.20 | 2716.60 | 2726.08 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 2764.65 | 2697.70 | 2692.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 2851.55 | 2742.96 | 2715.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 2789.50 | 2794.30 | 2758.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:45:00 | 2791.75 | 2794.30 | 2758.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 2764.90 | 2783.06 | 2762.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 2764.90 | 2783.06 | 2762.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 2761.45 | 2778.74 | 2762.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 2761.45 | 2778.74 | 2762.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 2756.00 | 2774.19 | 2761.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 2765.00 | 2774.19 | 2761.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 2751.15 | 2767.38 | 2760.45 | SL hit (close<static) qty=1.00 sl=2756.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 2734.00 | 2763.23 | 2763.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 2719.45 | 2754.48 | 2759.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 2748.85 | 2740.46 | 2750.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 2748.85 | 2740.46 | 2750.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 2748.85 | 2740.46 | 2750.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:45:00 | 2753.10 | 2740.46 | 2750.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 2729.90 | 2738.34 | 2748.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:15:00 | 2723.95 | 2738.34 | 2748.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 2709.75 | 2688.31 | 2704.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 2694.00 | 2673.12 | 2672.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 2694.00 | 2673.12 | 2672.92 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 2666.10 | 2671.72 | 2672.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 2640.85 | 2664.29 | 2668.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 2668.85 | 2660.69 | 2666.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 2668.85 | 2660.69 | 2666.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2668.85 | 2660.69 | 2666.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:45:00 | 2628.05 | 2652.05 | 2661.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 13:45:00 | 2641.65 | 2622.16 | 2633.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 10:15:00 | 2665.35 | 2637.58 | 2637.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 10:15:00 | 2665.35 | 2637.58 | 2637.50 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 2623.20 | 2635.67 | 2636.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2574.25 | 2621.52 | 2630.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 13:15:00 | 2598.70 | 2589.90 | 2609.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 13:15:00 | 2598.70 | 2589.90 | 2609.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 2598.70 | 2589.90 | 2609.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:45:00 | 2598.30 | 2589.90 | 2609.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 2568.65 | 2582.36 | 2600.92 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 2656.20 | 2610.20 | 2608.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 2689.00 | 2647.31 | 2634.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 2667.05 | 2676.24 | 2654.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:45:00 | 2670.90 | 2676.24 | 2654.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2688.25 | 2708.55 | 2692.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 2669.35 | 2708.55 | 2692.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2678.35 | 2702.51 | 2690.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 2665.50 | 2702.51 | 2690.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2639.50 | 2689.91 | 2686.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:00:00 | 2639.50 | 2689.91 | 2686.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 2616.00 | 2675.12 | 2679.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 2608.00 | 2661.70 | 2673.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 2617.50 | 2612.59 | 2633.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 12:15:00 | 2646.95 | 2619.18 | 2631.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2646.95 | 2619.18 | 2631.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2646.95 | 2619.18 | 2631.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2632.80 | 2621.90 | 2631.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 2639.00 | 2621.90 | 2631.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 2643.80 | 2626.28 | 2632.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 2643.80 | 2626.28 | 2632.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 2625.00 | 2626.03 | 2631.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 2607.75 | 2618.70 | 2627.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:00:00 | 2617.30 | 2613.15 | 2622.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:30:00 | 2613.05 | 2613.34 | 2621.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 2614.10 | 2613.34 | 2621.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2577.40 | 2606.15 | 2616.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 10:30:00 | 2562.80 | 2596.96 | 2611.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 2559.50 | 2580.48 | 2599.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:15:00 | 2564.35 | 2566.00 | 2585.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 2564.40 | 2565.39 | 2571.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 2549.95 | 2562.30 | 2569.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 2568.95 | 2562.30 | 2569.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 2477.36 | 2539.82 | 2557.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 2486.43 | 2539.82 | 2557.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 2482.40 | 2539.82 | 2557.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 2483.39 | 2539.82 | 2557.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 2487.00 | 2484.14 | 2514.49 | SL hit (close>ema200) qty=0.50 sl=2484.14 alert=retest2 |

### Cycle 128 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 2411.00 | 2386.57 | 2384.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 2444.00 | 2419.44 | 2406.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 2425.05 | 2425.92 | 2412.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 10:00:00 | 2425.05 | 2425.92 | 2412.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 2415.95 | 2423.93 | 2412.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 2415.95 | 2423.93 | 2412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 2401.00 | 2419.34 | 2411.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 2397.00 | 2419.34 | 2411.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 2408.65 | 2417.20 | 2411.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 2415.15 | 2409.99 | 2409.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 12:15:00 | 2399.00 | 2416.81 | 2416.56 | SL hit (close<static) qty=1.00 sl=2401.00 alert=retest2 |

### Cycle 129 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 2405.00 | 2414.45 | 2415.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 15:15:00 | 2398.00 | 2409.17 | 2412.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 2384.05 | 2383.96 | 2395.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 15:00:00 | 2384.05 | 2383.96 | 2395.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 2363.60 | 2379.51 | 2391.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 2358.45 | 2375.36 | 2388.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 11:45:00 | 2360.10 | 2345.35 | 2346.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 2240.53 | 2264.33 | 2288.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 2242.09 | 2264.33 | 2288.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 2265.90 | 2257.98 | 2278.98 | SL hit (close>ema200) qty=0.50 sl=2257.98 alert=retest2 |

### Cycle 130 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 2305.80 | 2280.33 | 2278.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 11:15:00 | 2326.65 | 2289.60 | 2282.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 2292.20 | 2310.09 | 2297.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 2292.20 | 2310.09 | 2297.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 2292.20 | 2310.09 | 2297.66 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 2270.80 | 2300.18 | 2301.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 2237.70 | 2283.67 | 2293.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 2287.65 | 2277.95 | 2285.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 14:15:00 | 2287.65 | 2277.95 | 2285.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 2287.65 | 2277.95 | 2285.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:45:00 | 2287.90 | 2277.95 | 2285.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 2266.00 | 2275.56 | 2284.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:45:00 | 2261.70 | 2273.43 | 2281.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:45:00 | 2254.45 | 2268.51 | 2278.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 2265.00 | 2267.12 | 2275.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:45:00 | 2260.85 | 2264.70 | 2272.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 2278.70 | 2259.21 | 2263.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 2280.70 | 2259.21 | 2263.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2277.60 | 2262.89 | 2265.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:45:00 | 2277.05 | 2262.89 | 2265.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 2271.60 | 2267.12 | 2266.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 2271.60 | 2267.12 | 2266.90 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 2265.10 | 2266.72 | 2266.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 2250.70 | 2263.51 | 2265.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 11:15:00 | 2223.50 | 2219.29 | 2237.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 12:00:00 | 2223.50 | 2219.29 | 2237.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 2249.80 | 2227.16 | 2237.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 2249.80 | 2227.16 | 2237.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 2256.00 | 2232.93 | 2239.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 2260.00 | 2232.93 | 2239.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 2260.40 | 2245.70 | 2244.19 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 2233.95 | 2243.89 | 2244.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 12:15:00 | 2228.25 | 2236.99 | 2240.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 2232.50 | 2229.71 | 2235.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 2232.50 | 2229.71 | 2235.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 2232.50 | 2229.71 | 2235.18 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 2247.20 | 2238.40 | 2237.85 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 2235.00 | 2237.51 | 2237.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 2223.95 | 2234.79 | 2236.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 2214.05 | 2211.89 | 2220.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:45:00 | 2213.35 | 2211.89 | 2220.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2213.90 | 2212.29 | 2220.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 2218.95 | 2212.29 | 2220.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 2231.85 | 2216.20 | 2221.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 2231.85 | 2216.20 | 2221.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 2236.05 | 2220.17 | 2222.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 2236.05 | 2220.17 | 2222.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2226.95 | 2222.29 | 2223.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 2215.65 | 2222.29 | 2223.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 2220.35 | 2211.96 | 2215.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 2214.45 | 2212.57 | 2215.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:00:00 | 2221.70 | 2215.66 | 2216.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 2208.90 | 2214.31 | 2215.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 2201.20 | 2213.52 | 2215.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 2104.87 | 2150.58 | 2174.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 2109.33 | 2150.58 | 2174.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 2103.73 | 2150.58 | 2174.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 2110.61 | 2150.58 | 2174.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 2091.14 | 2130.25 | 2160.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 2087.80 | 2086.83 | 2115.19 | SL hit (close>ema200) qty=0.50 sl=2086.83 alert=retest2 |

### Cycle 138 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 2095.40 | 2060.68 | 2057.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 2110.50 | 2070.64 | 2062.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 2151.95 | 2154.12 | 2121.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 15:00:00 | 2151.95 | 2154.12 | 2121.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 2133.00 | 2149.89 | 2122.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 2108.55 | 2143.25 | 2121.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 2115.00 | 2137.60 | 2121.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:30:00 | 2117.90 | 2137.60 | 2121.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 2090.00 | 2128.08 | 2118.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 2090.00 | 2128.08 | 2118.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2128.10 | 2117.46 | 2115.02 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 14:15:00 | 2089.55 | 2114.17 | 2114.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 2077.65 | 2103.79 | 2109.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 2100.95 | 2082.95 | 2095.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 14:15:00 | 2100.95 | 2082.95 | 2095.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 2100.95 | 2082.95 | 2095.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 2100.95 | 2082.95 | 2095.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 2088.90 | 2084.14 | 2094.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 2064.00 | 2084.14 | 2094.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:15:00 | 1960.80 | 1972.73 | 1985.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-03 14:15:00 | 1983.10 | 1949.44 | 1961.02 | SL hit (close>ema200) qty=0.50 sl=1949.44 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 1988.75 | 1965.77 | 1963.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 2000.80 | 1972.78 | 1967.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 2156.30 | 2184.70 | 2139.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 10:00:00 | 2156.30 | 2184.70 | 2139.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 2144.90 | 2171.59 | 2140.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 2144.90 | 2171.59 | 2140.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 2136.60 | 2164.59 | 2140.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 2136.60 | 2164.59 | 2140.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 2139.45 | 2159.56 | 2140.25 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 2081.35 | 2128.74 | 2130.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 2049.80 | 2105.17 | 2119.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 1988.45 | 1987.96 | 2012.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 15:00:00 | 1988.45 | 1987.96 | 2012.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 2028.00 | 1995.97 | 2014.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 2250.95 | 1995.97 | 2014.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 09:15:00 | 2336.45 | 2064.07 | 2043.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 2424.60 | 2330.52 | 2232.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 2469.00 | 2492.10 | 2433.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 2469.00 | 2492.10 | 2433.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 2494.95 | 2523.76 | 2498.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 2546.60 | 2523.76 | 2498.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2578.00 | 2534.61 | 2505.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:00:00 | 2632.95 | 2567.53 | 2529.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 15:00:00 | 2624.85 | 2598.99 | 2582.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 10:15:00 | 2523.70 | 2569.93 | 2572.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 2523.70 | 2569.93 | 2572.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 2486.90 | 2542.03 | 2558.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 2488.85 | 2470.78 | 2503.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:30:00 | 2484.95 | 2470.78 | 2503.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 2523.10 | 2481.25 | 2505.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 2523.10 | 2481.25 | 2505.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 2527.00 | 2490.40 | 2507.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 2528.35 | 2490.40 | 2507.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 2582.00 | 2508.72 | 2514.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 2582.00 | 2508.72 | 2514.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 2613.60 | 2529.69 | 2523.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2651.20 | 2554.00 | 2534.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 13:15:00 | 2654.55 | 2670.47 | 2621.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 14:00:00 | 2654.55 | 2670.47 | 2621.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 2692.00 | 2725.18 | 2706.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 2692.00 | 2725.18 | 2706.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 2674.20 | 2714.98 | 2703.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 2674.20 | 2714.98 | 2703.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 2677.90 | 2703.94 | 2700.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 2677.90 | 2703.94 | 2700.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 2662.70 | 2695.69 | 2696.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 2650.10 | 2686.58 | 2692.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 2657.75 | 2653.65 | 2669.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 2657.75 | 2653.65 | 2669.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2675.00 | 2657.92 | 2670.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 2671.85 | 2657.92 | 2670.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2691.40 | 2664.62 | 2672.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2691.40 | 2664.62 | 2672.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 2697.25 | 2671.14 | 2674.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 2698.20 | 2671.14 | 2674.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 2718.30 | 2680.58 | 2678.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 12:15:00 | 2732.40 | 2690.94 | 2683.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 14:15:00 | 2680.30 | 2691.13 | 2684.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 2680.30 | 2691.13 | 2684.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 2680.30 | 2691.13 | 2684.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 2680.30 | 2691.13 | 2684.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 2690.00 | 2690.91 | 2685.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 2676.70 | 2690.91 | 2685.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 2660.65 | 2684.86 | 2683.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 2660.65 | 2684.86 | 2683.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 2646.30 | 2677.14 | 2679.84 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 2716.20 | 2682.54 | 2679.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 15:15:00 | 2720.00 | 2696.04 | 2687.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 13:15:00 | 2896.00 | 2906.91 | 2879.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 13:15:00 | 2896.00 | 2906.91 | 2879.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 2896.00 | 2906.91 | 2879.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:45:00 | 2880.00 | 2906.91 | 2879.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 2875.00 | 2897.39 | 2879.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 2814.40 | 2897.39 | 2879.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2815.70 | 2881.05 | 2873.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:45:00 | 2815.25 | 2881.05 | 2873.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 2818.00 | 2868.44 | 2868.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 11:15:00 | 2801.90 | 2855.13 | 2862.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 2842.05 | 2838.31 | 2849.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 2842.05 | 2838.31 | 2849.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2842.05 | 2838.31 | 2849.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:00:00 | 2791.60 | 2819.62 | 2834.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 2788.90 | 2813.70 | 2830.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 2838.45 | 2827.88 | 2826.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 2838.45 | 2827.88 | 2826.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 13:15:00 | 2854.60 | 2833.23 | 2829.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2851.30 | 2851.45 | 2841.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 11:15:00 | 2850.25 | 2851.21 | 2841.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 2850.25 | 2851.21 | 2841.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:00:00 | 2850.25 | 2851.21 | 2841.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 2830.70 | 2847.11 | 2840.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:00:00 | 2830.70 | 2847.11 | 2840.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 2818.20 | 2841.33 | 2838.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 2818.20 | 2841.33 | 2838.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 2792.00 | 2831.46 | 2834.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 2778.45 | 2820.86 | 2829.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 15:15:00 | 2787.00 | 2785.38 | 2803.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:15:00 | 2848.05 | 2785.38 | 2803.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 2852.95 | 2798.89 | 2807.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 2845.85 | 2798.89 | 2807.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 2862.00 | 2811.51 | 2812.62 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 2834.30 | 2816.07 | 2814.59 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 2729.70 | 2814.81 | 2816.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 15:15:00 | 2721.00 | 2759.21 | 2784.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2627.45 | 2617.28 | 2678.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2627.45 | 2617.28 | 2678.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2627.45 | 2617.28 | 2678.52 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2729.95 | 2688.38 | 2687.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 2764.00 | 2714.08 | 2699.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2853.60 | 2890.55 | 2851.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 2853.60 | 2890.55 | 2851.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 2835.70 | 2874.47 | 2850.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:30:00 | 2843.20 | 2874.47 | 2850.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 2847.50 | 2869.07 | 2850.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:45:00 | 2834.60 | 2869.07 | 2850.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 2874.00 | 2870.06 | 2852.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:15:00 | 2893.80 | 2870.06 | 2852.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 13:15:00 | 2839.60 | 2862.95 | 2859.33 | SL hit (close<static) qty=1.00 sl=2841.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 14:15:00 | 2820.00 | 2854.36 | 2855.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 15:15:00 | 2812.00 | 2845.89 | 2851.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 13:15:00 | 2829.40 | 2824.25 | 2836.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 13:15:00 | 2829.40 | 2824.25 | 2836.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 2829.40 | 2824.25 | 2836.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:00:00 | 2829.40 | 2824.25 | 2836.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 2805.00 | 2820.40 | 2834.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-22 15:15:00 | 2802.40 | 2820.40 | 2834.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 2865.30 | 2826.80 | 2832.13 | SL hit (close>static) qty=1.00 sl=2836.70 alert=retest2 |

### Cycle 156 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 2856.00 | 2838.98 | 2837.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 3139.60 | 2903.29 | 2867.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 2922.00 | 2932.36 | 2899.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 14:45:00 | 2924.10 | 2932.36 | 2899.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2838.40 | 2909.54 | 2894.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2838.40 | 2909.54 | 2894.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2846.20 | 2896.87 | 2889.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 2852.70 | 2896.87 | 2889.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 2831.00 | 2883.70 | 2884.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 2806.10 | 2848.95 | 2866.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2872.00 | 2844.77 | 2861.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 2872.00 | 2844.77 | 2861.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2872.00 | 2844.77 | 2861.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 2872.00 | 2844.77 | 2861.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 2849.50 | 2845.72 | 2860.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 2836.10 | 2845.72 | 2860.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 2845.80 | 2851.99 | 2859.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 2824.10 | 2850.20 | 2857.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 10:15:00 | 2879.80 | 2853.43 | 2857.53 | SL hit (close>static) qty=1.00 sl=2874.30 alert=retest2 |

### Cycle 158 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 2878.20 | 2863.60 | 2861.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 13:15:00 | 2921.80 | 2875.24 | 2867.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 2932.10 | 2954.56 | 2924.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:15:00 | 2857.40 | 2954.56 | 2924.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2881.00 | 2939.84 | 2920.96 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 2851.80 | 2900.25 | 2905.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 2836.40 | 2876.75 | 2885.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 2854.60 | 2839.60 | 2857.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 2854.60 | 2839.60 | 2857.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 2857.00 | 2843.08 | 2857.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 2840.60 | 2843.08 | 2857.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 2828.80 | 2840.22 | 2854.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:30:00 | 2815.70 | 2829.33 | 2845.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:30:00 | 2801.00 | 2776.50 | 2787.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 12:30:00 | 2790.00 | 2773.99 | 2776.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 15:15:00 | 2809.00 | 2782.74 | 2779.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 2809.00 | 2782.74 | 2779.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 2909.00 | 2807.99 | 2791.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 2862.00 | 2866.40 | 2840.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 2862.00 | 2866.40 | 2840.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2837.40 | 2866.26 | 2852.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 2837.40 | 2866.26 | 2852.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 2829.80 | 2858.97 | 2850.50 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 14:15:00 | 2829.80 | 2842.91 | 2844.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 2811.70 | 2822.80 | 2831.09 | Break + close below crossover candle low |

### Cycle 162 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 2995.00 | 2836.75 | 2829.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 3016.70 | 2979.29 | 2931.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 2983.20 | 2984.18 | 2942.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:15:00 | 3040.90 | 3007.29 | 2973.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2978.30 | 3004.32 | 2978.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 2978.30 | 3004.32 | 2978.45 | SL hit (close<ema400) qty=1.00 sl=2978.45 alert=retest1 |

### Cycle 163 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 3389.50 | 3413.08 | 3414.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 3370.60 | 3404.58 | 3410.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 14:15:00 | 3400.00 | 3393.79 | 3402.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:45:00 | 3395.00 | 3393.79 | 3402.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 3444.90 | 3405.01 | 3406.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 3439.70 | 3405.01 | 3406.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 3430.20 | 3410.05 | 3408.55 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 3375.20 | 3404.96 | 3407.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 3357.30 | 3386.86 | 3397.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 3316.00 | 3297.60 | 3322.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 3316.00 | 3297.60 | 3322.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 3327.90 | 3303.66 | 3322.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 3327.90 | 3303.66 | 3322.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3367.50 | 3316.43 | 3326.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 3367.50 | 3316.43 | 3326.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3345.00 | 3322.14 | 3328.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 3300.00 | 3322.14 | 3328.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 3234.70 | 3222.22 | 3221.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 3234.70 | 3222.22 | 3221.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 3260.00 | 3233.62 | 3227.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 3354.30 | 3366.00 | 3327.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:00:00 | 3354.30 | 3366.00 | 3327.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 3375.80 | 3378.82 | 3356.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 3363.80 | 3378.82 | 3356.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 3349.60 | 3368.96 | 3357.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 3349.60 | 3368.96 | 3357.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 3358.00 | 3366.76 | 3357.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 3345.10 | 3366.76 | 3357.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3325.80 | 3358.57 | 3354.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 3325.80 | 3358.57 | 3354.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 3299.90 | 3346.84 | 3349.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 3274.80 | 3324.14 | 3338.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3346.00 | 3316.18 | 3328.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3346.00 | 3316.18 | 3328.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3346.00 | 3316.18 | 3328.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 3330.00 | 3316.18 | 3328.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 3340.20 | 3320.98 | 3329.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 3346.10 | 3320.98 | 3329.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 3336.00 | 3323.99 | 3330.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 3354.20 | 3323.99 | 3330.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 3289.00 | 3316.99 | 3326.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 3284.60 | 3316.99 | 3326.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 3286.00 | 3310.52 | 3317.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 3286.00 | 3302.95 | 3312.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 3284.70 | 3302.95 | 3312.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3334.30 | 3306.30 | 3312.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 3340.00 | 3306.30 | 3312.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 3352.10 | 3315.46 | 3316.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 3352.10 | 3315.46 | 3316.10 | SL hit (close>static) qty=1.00 sl=3336.30 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 3330.00 | 3318.37 | 3317.36 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 3298.70 | 3314.43 | 3315.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 3284.60 | 3305.95 | 3311.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 3323.80 | 3284.04 | 3292.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 3323.80 | 3284.04 | 3292.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 3323.80 | 3284.04 | 3292.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 3323.80 | 3284.04 | 3292.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 3311.60 | 3289.55 | 3294.54 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 3309.90 | 3299.19 | 3298.19 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 3283.50 | 3296.65 | 3297.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 3265.90 | 3290.50 | 3294.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3249.30 | 3248.22 | 3261.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 3249.30 | 3248.22 | 3261.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 3196.40 | 3200.50 | 3223.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 3196.40 | 3200.50 | 3223.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 3217.80 | 3204.95 | 3220.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 3217.80 | 3204.95 | 3220.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 3199.90 | 3203.94 | 3218.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 3188.80 | 3202.15 | 3216.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 3188.80 | 3195.78 | 3210.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 3189.30 | 3181.30 | 3194.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3131.30 | 3126.04 | 3126.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 3131.30 | 3126.04 | 3126.00 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 3117.80 | 3124.39 | 3125.26 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 3154.20 | 3129.70 | 3127.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 13:15:00 | 3182.10 | 3154.21 | 3142.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 14:15:00 | 3138.60 | 3151.09 | 3142.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 3138.60 | 3151.09 | 3142.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 3138.60 | 3151.09 | 3142.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 3138.60 | 3151.09 | 3142.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 3152.00 | 3151.27 | 3143.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 3127.70 | 3151.27 | 3143.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 3157.00 | 3152.42 | 3144.58 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 3112.40 | 3138.98 | 3139.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 3106.50 | 3132.48 | 3136.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 3156.30 | 3123.57 | 3129.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 3156.30 | 3123.57 | 3129.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 3156.30 | 3123.57 | 3129.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 3156.30 | 3123.57 | 3129.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 3150.00 | 3128.86 | 3131.44 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 3150.60 | 3136.59 | 3134.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 3184.20 | 3155.23 | 3144.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 3158.00 | 3175.84 | 3162.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 3158.00 | 3175.84 | 3162.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 3158.00 | 3175.84 | 3162.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 3188.00 | 3177.27 | 3164.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 3191.30 | 3177.27 | 3164.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 3191.40 | 3177.86 | 3168.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 3135.20 | 3169.33 | 3165.73 | SL hit (close<static) qty=1.00 sl=3145.90 alert=retest2 |

### Cycle 177 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 3131.80 | 3157.75 | 3160.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 3112.00 | 3148.60 | 3156.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 2651.50 | 2649.40 | 2686.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:45:00 | 2652.10 | 2649.40 | 2686.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 2685.00 | 2656.52 | 2686.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 2685.30 | 2656.52 | 2686.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 2690.00 | 2663.22 | 2686.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:30:00 | 2697.20 | 2663.22 | 2686.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 2683.90 | 2667.35 | 2686.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 2691.50 | 2667.35 | 2686.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 2679.00 | 2669.68 | 2685.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 2651.30 | 2669.68 | 2685.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 2696.00 | 2650.66 | 2660.48 | SL hit (close>static) qty=1.00 sl=2695.60 alert=retest2 |

### Cycle 178 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 2750.20 | 2682.30 | 2673.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 2781.70 | 2714.66 | 2690.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2710.80 | 2722.70 | 2699.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 2710.80 | 2722.70 | 2699.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2721.90 | 2722.54 | 2701.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 2755.30 | 2731.91 | 2711.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 2741.80 | 2736.50 | 2723.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 2757.80 | 2741.38 | 2727.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 2764.00 | 2794.16 | 2795.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 2764.00 | 2794.16 | 2795.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 2740.80 | 2777.16 | 2786.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 2768.00 | 2765.14 | 2776.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 2768.00 | 2765.14 | 2776.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 2768.00 | 2765.14 | 2776.41 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 2801.20 | 2779.79 | 2779.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 2819.10 | 2796.27 | 2788.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 10:15:00 | 2789.80 | 2794.98 | 2788.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 10:15:00 | 2789.80 | 2794.98 | 2788.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 2789.80 | 2794.98 | 2788.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 2789.80 | 2794.98 | 2788.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 2788.70 | 2793.72 | 2788.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:15:00 | 2784.40 | 2793.72 | 2788.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 2773.50 | 2789.68 | 2786.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 2773.50 | 2789.68 | 2786.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 2746.90 | 2781.12 | 2783.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 09:15:00 | 2737.40 | 2763.45 | 2773.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 13:15:00 | 2749.60 | 2745.49 | 2760.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:45:00 | 2747.20 | 2745.49 | 2760.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 2776.40 | 2751.67 | 2761.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 2768.80 | 2751.67 | 2761.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2775.10 | 2756.36 | 2762.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 2757.20 | 2756.36 | 2762.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 2831.40 | 2775.18 | 2770.60 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 2785.30 | 2795.25 | 2796.07 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 2819.30 | 2800.65 | 2798.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 2842.40 | 2819.96 | 2812.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 2850.00 | 2852.61 | 2838.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 2850.00 | 2852.61 | 2838.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2850.00 | 2852.61 | 2838.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 2821.10 | 2852.61 | 2838.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2808.10 | 2843.71 | 2835.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 2814.10 | 2843.71 | 2835.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2799.90 | 2834.95 | 2832.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 2801.30 | 2834.95 | 2832.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 2788.30 | 2825.62 | 2828.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 2766.00 | 2813.70 | 2822.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 2759.40 | 2759.19 | 2777.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 14:45:00 | 2733.90 | 2755.60 | 2771.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 2751.70 | 2753.34 | 2767.23 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2764.40 | 2755.55 | 2766.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 2764.40 | 2755.55 | 2766.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 2757.60 | 2755.96 | 2766.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 2757.60 | 2755.96 | 2766.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 2757.00 | 2756.17 | 2765.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 2757.00 | 2756.17 | 2765.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 2739.90 | 2752.92 | 2762.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:45:00 | 2755.00 | 2752.92 | 2762.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2726.10 | 2742.67 | 2755.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 2724.80 | 2742.67 | 2755.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2751.00 | 2742.15 | 2749.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 2751.00 | 2742.15 | 2749.80 | SL hit (close>ema400) qty=1.00 sl=2749.80 alert=retest1 |

### Cycle 186 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 2772.60 | 2753.48 | 2751.04 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 2739.80 | 2749.65 | 2750.79 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 2767.80 | 2753.71 | 2752.34 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 2744.90 | 2751.74 | 2752.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 2735.40 | 2748.00 | 2750.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 2744.10 | 2743.27 | 2747.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 2744.10 | 2743.27 | 2747.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 2744.10 | 2743.27 | 2747.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 2744.10 | 2743.27 | 2747.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2700.00 | 2689.56 | 2708.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 2703.70 | 2689.56 | 2708.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 2685.00 | 2681.87 | 2692.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 2685.00 | 2681.87 | 2692.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 2657.00 | 2676.89 | 2689.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 2653.30 | 2673.39 | 2685.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 2697.30 | 2677.46 | 2684.19 | SL hit (close>static) qty=1.00 sl=2689.40 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 2716.10 | 2687.99 | 2687.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 2728.50 | 2703.71 | 2696.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 2725.70 | 2735.60 | 2720.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 2725.70 | 2735.60 | 2720.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 2718.70 | 2732.22 | 2720.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:00:00 | 2738.50 | 2731.09 | 2721.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 2729.00 | 2729.87 | 2722.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 2747.50 | 2732.30 | 2723.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 2734.10 | 2740.77 | 2733.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2742.00 | 2741.01 | 2734.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 2742.00 | 2741.01 | 2734.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 2735.60 | 2740.26 | 2735.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 2736.10 | 2740.26 | 2735.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 2725.80 | 2737.37 | 2734.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 2725.80 | 2737.37 | 2734.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 2725.00 | 2734.90 | 2733.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 2753.70 | 2738.00 | 2735.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 2736.00 | 2745.81 | 2740.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 2739.80 | 2754.92 | 2756.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 2739.80 | 2754.92 | 2756.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 2730.00 | 2740.50 | 2747.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 2735.50 | 2730.67 | 2738.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2735.50 | 2730.67 | 2738.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2735.50 | 2730.67 | 2738.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 2740.20 | 2730.67 | 2738.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2733.50 | 2731.24 | 2737.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 2717.10 | 2728.47 | 2734.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 2744.50 | 2733.23 | 2736.06 | SL hit (close>static) qty=1.00 sl=2738.90 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2747.80 | 2738.44 | 2737.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 2759.00 | 2743.12 | 2739.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 2758.50 | 2761.65 | 2753.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 2758.50 | 2761.65 | 2753.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2760.00 | 2760.34 | 2754.20 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 2723.50 | 2747.13 | 2749.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 2704.00 | 2732.53 | 2741.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 2714.00 | 2709.24 | 2724.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 2714.00 | 2709.24 | 2724.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 2666.60 | 2700.71 | 2719.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 2682.40 | 2700.71 | 2719.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2642.00 | 2629.70 | 2646.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 2642.00 | 2629.70 | 2646.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2657.50 | 2635.26 | 2647.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 2657.50 | 2635.26 | 2647.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 2661.50 | 2640.51 | 2649.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:15:00 | 2664.50 | 2640.51 | 2649.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 2624.00 | 2644.42 | 2649.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:15:00 | 2620.00 | 2644.42 | 2649.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 2685.70 | 2648.77 | 2650.38 | SL hit (close>static) qty=1.00 sl=2654.20 alert=retest2 |

### Cycle 194 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 2665.00 | 2652.01 | 2651.71 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2613.10 | 2650.88 | 2655.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 2594.40 | 2639.58 | 2649.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 2579.90 | 2560.50 | 2589.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 15:00:00 | 2579.90 | 2560.50 | 2589.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 2606.30 | 2569.66 | 2591.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 2562.10 | 2569.66 | 2591.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 2516.70 | 2498.74 | 2497.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 2516.70 | 2498.74 | 2497.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 2527.00 | 2510.50 | 2504.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 2510.80 | 2513.07 | 2507.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 2510.80 | 2513.07 | 2507.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 2510.80 | 2513.07 | 2507.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 2505.30 | 2513.07 | 2507.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 2505.00 | 2511.46 | 2507.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 2509.20 | 2511.46 | 2507.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2506.20 | 2510.41 | 2507.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 2522.60 | 2512.84 | 2508.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 2519.60 | 2514.20 | 2509.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 2490.00 | 2505.25 | 2506.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 2490.00 | 2505.25 | 2506.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 2438.10 | 2491.82 | 2500.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 2477.40 | 2454.39 | 2475.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 2477.40 | 2454.39 | 2475.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2477.40 | 2454.39 | 2475.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 2477.40 | 2454.39 | 2475.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2480.00 | 2459.51 | 2475.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2451.30 | 2459.51 | 2475.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 2470.00 | 2456.01 | 2469.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 2463.50 | 2457.51 | 2468.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:45:00 | 2468.50 | 2459.01 | 2468.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2496.00 | 2466.41 | 2471.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 2496.00 | 2466.41 | 2471.12 | SL hit (close>static) qty=1.00 sl=2483.20 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 2491.70 | 2474.51 | 2473.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 2493.10 | 2478.23 | 2475.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 2480.80 | 2481.42 | 2477.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 15:15:00 | 2480.80 | 2481.42 | 2477.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 2480.80 | 2481.42 | 2477.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:45:00 | 2514.90 | 2488.39 | 2482.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 2511.00 | 2493.25 | 2484.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 2518.00 | 2498.67 | 2492.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 2498.80 | 2505.72 | 2506.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 2498.80 | 2505.72 | 2506.58 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 2524.00 | 2507.76 | 2506.93 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 2470.00 | 2501.03 | 2504.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 2461.10 | 2493.05 | 2500.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 2520.10 | 2495.53 | 2498.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 2520.10 | 2495.53 | 2498.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 2520.10 | 2495.53 | 2498.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 2520.10 | 2495.53 | 2498.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 2519.20 | 2500.26 | 2500.64 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 2512.90 | 2502.79 | 2501.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 2527.70 | 2510.69 | 2505.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 15:15:00 | 2506.00 | 2514.20 | 2509.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 2506.00 | 2514.20 | 2509.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 2506.00 | 2514.20 | 2509.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 2541.10 | 2514.20 | 2509.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2551.70 | 2521.70 | 2513.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 2607.10 | 2558.86 | 2545.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 2594.30 | 2588.79 | 2570.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 2590.10 | 2598.30 | 2587.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:00:00 | 2584.00 | 2591.81 | 2586.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 2601.90 | 2593.83 | 2587.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 2577.50 | 2587.68 | 2587.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 2577.50 | 2587.68 | 2587.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 2575.00 | 2583.74 | 2585.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 12:15:00 | 2589.90 | 2582.53 | 2584.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 2589.90 | 2582.53 | 2584.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2589.90 | 2582.53 | 2584.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 2589.90 | 2582.53 | 2584.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2596.60 | 2585.34 | 2585.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:30:00 | 2596.60 | 2585.34 | 2585.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 15:15:00 | 2592.50 | 2586.72 | 2586.06 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 2578.70 | 2584.60 | 2585.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2537.40 | 2573.28 | 2579.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2471.10 | 2463.56 | 2474.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2471.10 | 2463.56 | 2474.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2471.10 | 2463.56 | 2474.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2471.10 | 2463.56 | 2474.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2478.60 | 2466.57 | 2475.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 2484.30 | 2466.57 | 2475.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2475.00 | 2468.26 | 2475.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 2455.40 | 2468.26 | 2475.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 2459.10 | 2434.89 | 2436.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 2452.40 | 2438.39 | 2437.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 2452.40 | 2438.39 | 2437.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 2477.10 | 2461.03 | 2452.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 2457.50 | 2468.56 | 2459.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 2457.50 | 2468.56 | 2459.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2457.50 | 2468.56 | 2459.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 2461.80 | 2468.56 | 2459.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2462.50 | 2467.35 | 2459.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 2473.70 | 2468.32 | 2461.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 2454.30 | 2461.39 | 2460.10 | SL hit (close<static) qty=1.00 sl=2455.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2450.50 | 2471.68 | 2472.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 2428.60 | 2458.80 | 2466.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 2384.30 | 2370.07 | 2392.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 2384.30 | 2370.07 | 2392.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 2384.30 | 2370.07 | 2392.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 2384.30 | 2370.07 | 2392.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 2398.40 | 2375.74 | 2393.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 2398.40 | 2375.74 | 2393.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 2388.00 | 2378.19 | 2392.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 2392.50 | 2378.19 | 2392.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2372.00 | 2376.95 | 2390.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 2365.80 | 2375.26 | 2387.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 2365.00 | 2373.55 | 2385.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 2366.70 | 2373.19 | 2383.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 2404.60 | 2386.95 | 2386.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 2404.60 | 2386.95 | 2386.28 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 2360.80 | 2383.01 | 2385.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 2347.80 | 2364.42 | 2374.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 2270.70 | 2265.84 | 2292.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 2270.70 | 2265.84 | 2292.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 2296.20 | 2271.91 | 2292.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 2306.60 | 2271.91 | 2292.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2312.40 | 2280.01 | 2294.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 2318.00 | 2280.01 | 2294.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2306.30 | 2285.27 | 2295.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 2310.40 | 2285.27 | 2295.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 2352.90 | 2302.46 | 2301.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 2381.20 | 2318.21 | 2308.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 2341.50 | 2347.74 | 2330.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 2341.50 | 2347.74 | 2330.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 2325.00 | 2343.20 | 2329.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 2329.20 | 2343.20 | 2329.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 2320.00 | 2338.56 | 2328.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 2320.00 | 2338.56 | 2328.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 2296.00 | 2319.42 | 2321.75 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 12:15:00 | 2353.20 | 2325.84 | 2324.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 2382.70 | 2343.88 | 2333.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 2390.00 | 2398.91 | 2370.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 2390.00 | 2398.91 | 2370.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 2368.30 | 2392.80 | 2376.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 2368.30 | 2392.80 | 2376.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2363.20 | 2386.88 | 2375.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 2356.40 | 2386.88 | 2375.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 2355.50 | 2380.60 | 2373.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 2330.00 | 2380.60 | 2373.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2348.10 | 2367.55 | 2368.88 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2388.90 | 2369.81 | 2369.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 2402.00 | 2376.25 | 2372.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 2388.90 | 2388.98 | 2381.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 2388.90 | 2388.98 | 2381.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2388.90 | 2388.98 | 2381.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2396.20 | 2388.98 | 2381.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2432.30 | 2397.64 | 2385.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 2388.10 | 2397.64 | 2385.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2369.90 | 2394.69 | 2387.70 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 2358.20 | 2379.84 | 2381.69 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2418.10 | 2383.46 | 2382.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 2432.90 | 2395.99 | 2388.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 2408.00 | 2419.97 | 2408.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 2408.00 | 2419.97 | 2408.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2408.00 | 2419.97 | 2408.03 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 2378.20 | 2399.64 | 2401.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 2372.20 | 2394.16 | 2399.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 13:15:00 | 2422.00 | 2394.83 | 2397.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 13:15:00 | 2422.00 | 2394.83 | 2397.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 2422.00 | 2394.83 | 2397.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 2422.00 | 2394.83 | 2397.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 2443.80 | 2404.63 | 2401.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 2462.50 | 2434.45 | 2419.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2658.90 | 2663.64 | 2615.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 2658.90 | 2663.64 | 2615.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2612.50 | 2655.86 | 2633.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 2612.50 | 2655.86 | 2633.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2631.20 | 2650.93 | 2633.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2634.30 | 2650.93 | 2633.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:30:00 | 2636.80 | 2642.64 | 2633.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 2596.10 | 2633.33 | 2630.07 | SL hit (close<static) qty=1.00 sl=2605.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 2590.00 | 2624.67 | 2626.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 2585.70 | 2602.81 | 2613.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 12:15:00 | 2608.10 | 2601.08 | 2610.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 12:15:00 | 2608.10 | 2601.08 | 2610.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 2608.10 | 2601.08 | 2610.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 2598.60 | 2601.08 | 2610.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 2620.40 | 2604.95 | 2611.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 2620.40 | 2604.95 | 2611.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 2604.70 | 2604.90 | 2610.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 2613.40 | 2604.90 | 2610.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2614.30 | 2605.39 | 2610.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:00:00 | 2596.70 | 2603.65 | 2608.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 2598.40 | 2593.47 | 2600.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 2595.80 | 2594.18 | 2599.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:30:00 | 2600.50 | 2590.50 | 2592.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2600.00 | 2592.40 | 2592.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 2631.50 | 2592.40 | 2592.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 2623.80 | 2598.68 | 2595.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2623.80 | 2598.68 | 2595.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 2663.40 | 2611.63 | 2601.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 2624.00 | 2629.11 | 2617.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 2624.00 | 2629.11 | 2617.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2624.00 | 2629.11 | 2617.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 2616.60 | 2629.11 | 2617.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 2598.40 | 2622.97 | 2615.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 2599.90 | 2622.97 | 2615.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2597.20 | 2617.82 | 2613.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:30:00 | 2594.70 | 2617.82 | 2613.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 2612.90 | 2618.65 | 2614.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:45:00 | 2617.50 | 2618.65 | 2614.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 2619.80 | 2618.88 | 2615.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 2617.10 | 2618.88 | 2615.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 2617.70 | 2618.65 | 2615.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 2623.10 | 2618.65 | 2615.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 2638.80 | 2622.68 | 2617.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 2649.50 | 2628.04 | 2620.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 2599.00 | 2624.53 | 2626.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 2599.00 | 2624.53 | 2626.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 2584.10 | 2616.44 | 2623.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 2497.00 | 2482.51 | 2504.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 2497.00 | 2482.51 | 2504.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2497.00 | 2482.51 | 2504.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2499.20 | 2482.51 | 2504.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 2507.00 | 2488.11 | 2502.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 2507.00 | 2488.11 | 2502.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 2516.40 | 2493.77 | 2504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 2514.80 | 2493.77 | 2504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 2499.30 | 2493.35 | 2500.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 2499.30 | 2493.35 | 2500.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 2502.10 | 2495.10 | 2500.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:45:00 | 2513.60 | 2495.10 | 2500.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 2499.00 | 2495.88 | 2500.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 14:00:00 | 2499.00 | 2495.88 | 2500.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 2515.90 | 2499.88 | 2501.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 2515.90 | 2499.88 | 2501.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 2529.50 | 2505.81 | 2504.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 2532.00 | 2511.05 | 2506.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 2508.00 | 2514.60 | 2509.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 11:15:00 | 2508.00 | 2514.60 | 2509.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 2508.00 | 2514.60 | 2509.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 2508.00 | 2514.60 | 2509.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 2498.00 | 2511.28 | 2508.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 2496.20 | 2511.28 | 2508.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 2489.90 | 2507.01 | 2506.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 2489.90 | 2507.01 | 2506.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 14:15:00 | 2487.20 | 2503.05 | 2504.96 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 2520.20 | 2505.74 | 2505.58 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 2487.20 | 2503.95 | 2505.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 2471.00 | 2497.36 | 2502.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 2503.40 | 2492.13 | 2497.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 2503.40 | 2492.13 | 2497.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 2503.40 | 2492.13 | 2497.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 2503.40 | 2492.13 | 2497.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 2512.00 | 2496.11 | 2499.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:45:00 | 2509.30 | 2496.11 | 2499.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 2504.90 | 2501.03 | 2500.97 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 2446.00 | 2490.03 | 2495.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 2433.70 | 2478.76 | 2490.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 2411.60 | 2410.01 | 2432.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 2411.60 | 2410.01 | 2432.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 2422.50 | 2413.73 | 2428.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:30:00 | 2419.90 | 2413.73 | 2428.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 2432.00 | 2417.38 | 2428.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 2432.00 | 2417.38 | 2428.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 2438.80 | 2421.67 | 2429.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 2438.80 | 2421.67 | 2429.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 2422.20 | 2421.77 | 2428.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 2450.10 | 2421.77 | 2428.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2456.00 | 2428.62 | 2431.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 2454.20 | 2428.62 | 2431.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2459.00 | 2434.69 | 2433.87 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2412.60 | 2431.35 | 2433.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2390.10 | 2418.22 | 2426.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2407.90 | 2404.30 | 2417.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:30:00 | 2399.10 | 2404.30 | 2417.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2315.30 | 2314.22 | 2338.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 2299.30 | 2327.75 | 2333.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 10:30:00 | 2307.70 | 2320.43 | 2328.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 11:00:00 | 2306.40 | 2320.43 | 2328.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 2394.60 | 2290.83 | 2292.59 | SL hit (close>static) qty=1.00 sl=2340.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2369.90 | 2306.65 | 2299.62 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 2290.00 | 2304.43 | 2305.49 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 2328.70 | 2307.16 | 2305.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2371.40 | 2338.27 | 2329.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 14:15:00 | 2355.80 | 2356.98 | 2343.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 15:00:00 | 2355.80 | 2356.98 | 2343.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 2366.00 | 2357.69 | 2346.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2400.00 | 2360.83 | 2353.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 2383.20 | 2392.66 | 2378.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 2427.90 | 2431.20 | 2431.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 2427.90 | 2431.20 | 2431.26 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 2441.70 | 2431.88 | 2431.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 2468.90 | 2439.28 | 2434.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 2478.00 | 2482.04 | 2467.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 10:15:00 | 2478.00 | 2482.04 | 2467.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2478.00 | 2482.04 | 2467.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 2478.00 | 2482.04 | 2467.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 2467.70 | 2479.17 | 2467.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 2467.90 | 2479.17 | 2467.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 2465.60 | 2476.46 | 2467.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 2465.60 | 2476.46 | 2467.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 2470.00 | 2475.17 | 2467.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 2470.00 | 2475.17 | 2467.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 2469.40 | 2474.01 | 2467.87 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 2439.90 | 2464.22 | 2464.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 14:15:00 | 2435.80 | 2449.34 | 2456.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 2348.00 | 2347.43 | 2373.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 2368.70 | 2347.43 | 2373.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2353.90 | 2348.72 | 2371.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 2337.50 | 2346.52 | 2366.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:45:00 | 2336.10 | 2341.75 | 2361.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 2330.30 | 2343.40 | 2360.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 2392.90 | 2362.06 | 2363.98 | SL hit (close>static) qty=1.00 sl=2387.40 alert=retest2 |

### Cycle 236 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 2413.50 | 2372.16 | 2367.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 2426.80 | 2383.09 | 2373.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 2472.00 | 2478.07 | 2454.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 2472.00 | 2478.07 | 2454.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 15:00:00 | 1297.15 | 2023-05-18 10:15:00 | 1285.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-05-26 15:15:00 | 1311.00 | 2023-05-29 15:15:00 | 1300.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-06-02 12:45:00 | 1338.95 | 2023-06-08 14:15:00 | 1367.30 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2023-06-02 13:15:00 | 1347.90 | 2023-06-08 14:15:00 | 1367.30 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2023-06-12 11:15:00 | 1375.60 | 2023-06-12 11:15:00 | 1379.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-06-16 09:15:00 | 1415.70 | 2023-06-23 12:15:00 | 1406.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-07-03 13:30:00 | 1423.10 | 2023-07-13 13:15:00 | 1401.05 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2023-07-25 09:30:00 | 1383.10 | 2023-07-25 13:15:00 | 1400.05 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-07-25 10:30:00 | 1385.10 | 2023-07-25 13:15:00 | 1400.05 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-07-28 09:15:00 | 1404.85 | 2023-07-28 11:15:00 | 1395.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-08-01 10:45:00 | 1394.00 | 2023-08-01 13:15:00 | 1402.25 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-08-01 12:00:00 | 1395.05 | 2023-08-01 13:15:00 | 1402.25 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-08-01 12:30:00 | 1393.10 | 2023-08-01 13:15:00 | 1402.25 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-08-04 09:30:00 | 1407.40 | 2023-08-04 12:15:00 | 1398.45 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-08-17 10:45:00 | 1405.95 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-08-17 13:15:00 | 1408.00 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-08-17 15:15:00 | 1407.00 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-08-18 14:30:00 | 1408.60 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-21 10:45:00 | 1399.15 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-08-23 10:15:00 | 1393.40 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2023-08-23 11:30:00 | 1398.75 | 2023-08-23 12:15:00 | 1421.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-08-29 09:30:00 | 1421.85 | 2023-08-29 14:15:00 | 1414.80 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-08-29 14:00:00 | 1417.70 | 2023-08-29 14:15:00 | 1414.80 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-09-04 09:15:00 | 1465.90 | 2023-09-04 14:15:00 | 1425.35 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2023-09-05 10:45:00 | 1450.45 | 2023-09-07 14:15:00 | 1442.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-09-05 14:00:00 | 1458.00 | 2023-09-07 14:15:00 | 1442.75 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-09-06 09:45:00 | 1452.60 | 2023-09-07 14:15:00 | 1442.75 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-09-06 13:15:00 | 1448.15 | 2023-09-15 14:15:00 | 1595.50 | TARGET_HIT | 1.00 | 10.17% |
| BUY | retest2 | 2023-09-06 13:45:00 | 1451.30 | 2023-09-15 14:15:00 | 1603.80 | TARGET_HIT | 1.00 | 10.51% |
| BUY | retest2 | 2023-09-07 10:00:00 | 1460.00 | 2023-09-15 14:15:00 | 1597.86 | TARGET_HIT | 1.00 | 9.44% |
| BUY | retest2 | 2023-09-08 09:15:00 | 1457.60 | 2023-09-15 14:15:00 | 1603.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-08 14:00:00 | 1462.00 | 2023-09-15 14:15:00 | 1608.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-11 13:45:00 | 1460.90 | 2023-09-15 14:15:00 | 1606.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-12 10:45:00 | 1465.00 | 2023-09-15 14:15:00 | 1611.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-12 12:15:00 | 1461.50 | 2023-09-15 14:15:00 | 1607.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-12 14:30:00 | 1465.40 | 2023-09-15 14:15:00 | 1611.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 09:15:00 | 1465.20 | 2023-09-15 14:15:00 | 1611.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 09:45:00 | 1467.95 | 2023-09-15 14:15:00 | 1614.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-26 14:00:00 | 1533.10 | 2023-09-29 09:15:00 | 1543.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-09-27 11:45:00 | 1532.50 | 2023-09-29 09:15:00 | 1543.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-09-27 13:15:00 | 1531.65 | 2023-09-29 09:15:00 | 1543.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-09-28 09:15:00 | 1530.15 | 2023-09-29 13:15:00 | 1564.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2023-09-28 13:00:00 | 1518.70 | 2023-09-29 13:15:00 | 1564.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2023-09-28 13:30:00 | 1519.00 | 2023-09-29 13:15:00 | 1564.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2023-09-29 09:15:00 | 1512.60 | 2023-09-29 13:15:00 | 1564.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2023-10-05 09:15:00 | 1566.00 | 2023-10-05 14:15:00 | 1539.95 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-10-05 10:30:00 | 1560.40 | 2023-10-05 14:15:00 | 1539.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-10-13 11:30:00 | 1530.00 | 2023-10-16 09:15:00 | 1572.95 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2023-10-13 12:30:00 | 1531.80 | 2023-10-16 09:15:00 | 1572.95 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2023-10-13 13:15:00 | 1530.00 | 2023-10-16 09:15:00 | 1572.95 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-11-08 09:15:00 | 1427.00 | 2023-11-15 15:15:00 | 1569.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-22 11:15:00 | 1725.15 | 2023-12-22 14:15:00 | 1752.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-12-22 11:45:00 | 1720.95 | 2023-12-22 14:15:00 | 1752.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2023-12-22 12:30:00 | 1723.05 | 2023-12-22 14:15:00 | 1752.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2023-12-29 09:15:00 | 1860.20 | 2023-12-29 12:15:00 | 1953.21 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-12-29 09:15:00 | 1860.20 | 2023-12-29 14:15:00 | 1892.10 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest2 | 2024-01-09 09:15:00 | 2159.95 | 2024-01-09 12:15:00 | 2079.30 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-01-17 09:30:00 | 2318.20 | 2024-01-20 10:15:00 | 2276.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-01-17 15:00:00 | 2277.75 | 2024-01-20 10:15:00 | 2276.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-01-18 10:00:00 | 2285.00 | 2024-01-20 10:15:00 | 2276.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-01-20 10:15:00 | 2280.00 | 2024-01-20 10:15:00 | 2276.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-01-25 14:45:00 | 2218.45 | 2024-01-31 10:15:00 | 2260.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-02-01 14:00:00 | 2238.90 | 2024-02-05 11:15:00 | 2462.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-02 09:15:00 | 2239.95 | 2024-02-05 11:15:00 | 2463.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-16 11:15:00 | 2243.85 | 2024-02-16 11:15:00 | 2262.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-02-28 10:45:00 | 2147.25 | 2024-02-29 11:15:00 | 2199.70 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-02-29 11:30:00 | 2153.05 | 2024-02-29 12:15:00 | 2260.00 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2024-03-04 11:15:00 | 2133.70 | 2024-03-07 11:15:00 | 2027.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 10:15:00 | 2129.55 | 2024-03-07 11:15:00 | 2029.01 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2024-03-04 11:15:00 | 2133.70 | 2024-03-11 09:15:00 | 2065.20 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-03-05 10:15:00 | 2129.55 | 2024-03-11 09:15:00 | 2065.20 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2024-03-05 13:15:00 | 2135.80 | 2024-03-11 13:15:00 | 2023.07 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2024-03-05 13:15:00 | 2135.80 | 2024-03-13 12:15:00 | 1916.60 | TARGET_HIT | 0.50 | 10.26% |
| BUY | retest2 | 2024-03-20 09:30:00 | 1998.75 | 2024-03-22 10:15:00 | 1955.05 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-03-20 10:15:00 | 1992.65 | 2024-03-22 10:15:00 | 1955.05 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-03-21 09:15:00 | 2014.80 | 2024-03-22 10:15:00 | 1955.05 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-03-26 14:45:00 | 1935.00 | 2024-04-01 09:15:00 | 1972.65 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-03-27 09:45:00 | 1927.55 | 2024-04-01 09:15:00 | 1972.65 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-03-28 11:00:00 | 1935.00 | 2024-04-01 09:15:00 | 1972.65 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-03-28 13:15:00 | 1937.10 | 2024-04-01 09:15:00 | 1972.65 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-04-08 15:00:00 | 1925.45 | 2024-04-22 11:15:00 | 1926.60 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-04-09 09:30:00 | 1925.00 | 2024-04-22 11:15:00 | 1926.60 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2024-04-24 09:15:00 | 1956.05 | 2024-04-24 13:15:00 | 2053.85 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-24 09:15:00 | 1956.05 | 2024-04-29 09:15:00 | 2151.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-02 09:15:00 | 2105.00 | 2024-05-02 12:15:00 | 2092.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-05-02 10:30:00 | 2108.65 | 2024-05-02 12:15:00 | 2092.25 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-07 13:00:00 | 2042.80 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-05-07 14:00:00 | 2040.55 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-05-07 15:15:00 | 2036.60 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-05-08 11:00:00 | 2035.45 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-05-14 10:00:00 | 1993.70 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-05-14 13:15:00 | 1994.55 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-05-14 14:15:00 | 1992.00 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-05-15 15:00:00 | 1989.95 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-05-16 10:15:00 | 1980.10 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-06-13 09:15:00 | 2640.15 | 2024-06-18 15:15:00 | 2637.25 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-06-18 15:00:00 | 2650.00 | 2024-06-18 15:15:00 | 2637.25 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-06-21 13:00:00 | 2535.80 | 2024-06-24 09:15:00 | 2653.15 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2024-06-21 14:00:00 | 2557.30 | 2024-06-24 09:15:00 | 2653.15 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-06-21 14:30:00 | 2546.20 | 2024-06-24 09:15:00 | 2653.15 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2024-07-01 09:15:00 | 2662.60 | 2024-07-01 12:15:00 | 2632.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-04 14:30:00 | 2615.00 | 2024-07-11 12:15:00 | 2583.50 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-07-26 09:15:00 | 2695.05 | 2024-07-30 15:15:00 | 2695.00 | STOP_HIT | 1.00 | -0.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 2814.25 | 2024-08-13 12:15:00 | 2834.70 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2024-08-07 09:15:00 | 2844.60 | 2024-08-13 12:15:00 | 2834.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-08-23 12:15:00 | 2926.05 | 2024-08-23 15:15:00 | 2950.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-08-23 14:30:00 | 2935.15 | 2024-08-23 15:15:00 | 2950.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-08-30 12:15:00 | 2767.65 | 2024-09-02 14:15:00 | 2780.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-30 13:00:00 | 2767.45 | 2024-09-02 14:15:00 | 2780.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-30 14:15:00 | 2766.00 | 2024-09-02 14:15:00 | 2780.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-11 09:15:00 | 2860.95 | 2024-09-16 15:15:00 | 2828.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-09-18 15:15:00 | 2786.00 | 2024-09-26 09:15:00 | 2733.60 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2024-10-04 09:15:00 | 2712.30 | 2024-10-04 09:15:00 | 2739.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-10-11 09:15:00 | 2765.00 | 2024-10-11 10:15:00 | 2751.15 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-10-11 11:45:00 | 2775.10 | 2024-10-14 10:15:00 | 2743.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-10-15 11:15:00 | 2723.95 | 2024-10-21 11:15:00 | 2694.00 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2024-10-17 10:00:00 | 2709.75 | 2024-10-21 11:15:00 | 2694.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2024-10-22 11:45:00 | 2628.05 | 2024-10-24 10:15:00 | 2665.35 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-23 13:45:00 | 2641.65 | 2024-10-24 10:15:00 | 2665.35 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-11-07 09:30:00 | 2607.75 | 2024-11-13 09:15:00 | 2477.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:00:00 | 2617.30 | 2024-11-13 09:15:00 | 2486.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:30:00 | 2613.05 | 2024-11-13 09:15:00 | 2482.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 14:00:00 | 2614.10 | 2024-11-13 09:15:00 | 2483.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:30:00 | 2607.75 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2024-11-07 13:00:00 | 2617.30 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 4.98% |
| SELL | retest2 | 2024-11-07 13:30:00 | 2613.05 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2024-11-07 14:00:00 | 2614.10 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 4.86% |
| SELL | retest2 | 2024-11-08 10:30:00 | 2562.80 | 2024-11-14 09:15:00 | 2434.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 10:30:00 | 2562.80 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2024-11-08 14:00:00 | 2559.50 | 2024-11-14 09:15:00 | 2431.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 2559.50 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2024-11-11 11:15:00 | 2564.35 | 2024-11-14 09:15:00 | 2436.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:15:00 | 2564.35 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2024-11-12 12:30:00 | 2564.40 | 2024-11-14 09:15:00 | 2436.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:30:00 | 2564.40 | 2024-11-14 09:15:00 | 2487.00 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2024-11-21 11:45:00 | 2380.55 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-11-21 13:45:00 | 2378.00 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-11-22 11:30:00 | 2380.50 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-11-25 13:15:00 | 2375.00 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-11-27 10:15:00 | 2379.15 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-11-27 12:15:00 | 2375.85 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-11-27 13:30:00 | 2379.00 | 2024-11-28 09:15:00 | 2411.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-12-03 09:15:00 | 2415.15 | 2024-12-04 12:15:00 | 2399.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-06 10:45:00 | 2358.45 | 2024-12-16 09:15:00 | 2240.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 11:45:00 | 2360.10 | 2024-12-16 09:15:00 | 2242.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 10:45:00 | 2358.45 | 2024-12-16 12:15:00 | 2265.90 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-12-11 11:45:00 | 2360.10 | 2024-12-16 12:15:00 | 2265.90 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2024-12-24 10:45:00 | 2261.70 | 2024-12-27 14:15:00 | 2271.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-12-24 11:45:00 | 2254.45 | 2024-12-27 14:15:00 | 2271.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-24 15:15:00 | 2265.00 | 2024-12-27 14:15:00 | 2271.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-12-26 09:45:00 | 2260.85 | 2024-12-27 14:15:00 | 2271.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-01-08 10:15:00 | 2215.65 | 2025-01-13 11:15:00 | 2104.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 2220.35 | 2025-01-13 11:15:00 | 2109.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 2214.45 | 2025-01-13 11:15:00 | 2103.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 2221.70 | 2025-01-13 11:15:00 | 2110.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 2201.20 | 2025-01-13 13:15:00 | 2091.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 2215.65 | 2025-01-14 15:15:00 | 2087.80 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-01-09 10:15:00 | 2220.35 | 2025-01-14 15:15:00 | 2087.80 | STOP_HIT | 0.50 | 5.97% |
| SELL | retest2 | 2025-01-09 10:45:00 | 2214.45 | 2025-01-14 15:15:00 | 2087.80 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2025-01-09 13:00:00 | 2221.70 | 2025-01-14 15:15:00 | 2087.80 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-01-09 15:15:00 | 2201.20 | 2025-01-14 15:15:00 | 2087.80 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2025-01-27 09:15:00 | 2064.00 | 2025-02-01 11:15:00 | 1960.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 2064.00 | 2025-02-03 14:15:00 | 1983.10 | STOP_HIT | 0.50 | 3.92% |
| BUY | retest2 | 2025-02-24 13:00:00 | 2632.95 | 2025-02-28 10:15:00 | 2523.70 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2025-02-27 15:00:00 | 2624.85 | 2025-02-28 10:15:00 | 2523.70 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-03-27 10:00:00 | 2791.60 | 2025-03-28 12:15:00 | 2838.45 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-27 10:45:00 | 2788.90 | 2025-03-28 12:15:00 | 2838.45 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-04-17 14:15:00 | 2893.80 | 2025-04-21 13:15:00 | 2839.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-04-22 15:15:00 | 2802.40 | 2025-04-23 11:15:00 | 2865.30 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-04-28 11:15:00 | 2836.10 | 2025-04-29 10:15:00 | 2879.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-04-28 14:15:00 | 2845.80 | 2025-04-29 10:15:00 | 2879.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-04-29 09:15:00 | 2824.10 | 2025-04-29 10:15:00 | 2879.80 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-05-08 12:30:00 | 2815.70 | 2025-05-13 15:15:00 | 2809.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-05-12 11:30:00 | 2801.00 | 2025-05-13 15:15:00 | 2809.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-05-13 12:30:00 | 2790.00 | 2025-05-13 15:15:00 | 2809.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2025-05-26 09:15:00 | 3040.90 | 2025-05-26 10:15:00 | 2978.30 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-05-27 09:15:00 | 2971.00 | 2025-05-28 13:15:00 | 3268.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 3300.00 | 2025-06-24 13:15:00 | 3234.70 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-07-03 13:15:00 | 3284.60 | 2025-07-07 10:15:00 | 3352.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-07-04 13:15:00 | 3286.00 | 2025-07-07 10:15:00 | 3352.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-04 14:45:00 | 3286.00 | 2025-07-07 10:15:00 | 3352.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-04 15:15:00 | 3284.70 | 2025-07-07 10:15:00 | 3352.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-16 09:15:00 | 3188.80 | 2025-07-24 09:15:00 | 3131.30 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-07-16 10:30:00 | 3188.80 | 2025-07-24 09:15:00 | 3131.30 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-07-17 10:45:00 | 3189.30 | 2025-07-24 09:15:00 | 3131.30 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-07-31 10:45:00 | 3188.00 | 2025-07-31 15:15:00 | 3135.20 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-31 11:15:00 | 3191.30 | 2025-07-31 15:15:00 | 3135.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-31 14:30:00 | 3191.40 | 2025-07-31 15:15:00 | 3135.20 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-12 09:15:00 | 2651.30 | 2025-08-13 10:15:00 | 2696.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-08-14 13:45:00 | 2755.30 | 2025-08-26 15:15:00 | 2764.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-08-18 13:00:00 | 2741.80 | 2025-08-26 15:15:00 | 2764.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-08-18 13:30:00 | 2757.80 | 2025-08-26 15:15:00 | 2764.00 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest1 | 2025-09-16 14:45:00 | 2733.90 | 2025-09-18 14:15:00 | 2751.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2025-09-17 09:45:00 | 2751.70 | 2025-09-18 14:15:00 | 2751.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-09-30 15:15:00 | 2653.30 | 2025-10-01 10:15:00 | 2697.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-06 15:00:00 | 2738.50 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-10-07 09:15:00 | 2729.00 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-10-07 09:45:00 | 2747.50 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-08 09:30:00 | 2734.10 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-10-09 09:30:00 | 2753.70 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-09 14:30:00 | 2736.00 | 2025-10-15 10:15:00 | 2739.80 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-10-17 14:00:00 | 2717.10 | 2025-10-17 15:15:00 | 2744.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-10-20 10:15:00 | 2722.10 | 2025-10-20 13:15:00 | 2745.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-31 15:15:00 | 2620.00 | 2025-11-03 09:15:00 | 2685.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-11-10 09:15:00 | 2562.10 | 2025-11-19 12:15:00 | 2516.70 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2025-11-21 11:00:00 | 2522.60 | 2025-11-21 15:15:00 | 2490.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-21 12:00:00 | 2519.60 | 2025-11-21 15:15:00 | 2490.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2451.30 | 2025-11-25 14:15:00 | 2496.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-25 12:15:00 | 2470.00 | 2025-11-25 14:15:00 | 2496.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-25 13:00:00 | 2463.50 | 2025-11-25 14:15:00 | 2496.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-11-25 13:45:00 | 2468.50 | 2025-11-25 14:15:00 | 2496.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-11-26 09:15:00 | 2472.40 | 2025-11-26 11:15:00 | 2491.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-27 12:45:00 | 2514.90 | 2025-12-02 10:15:00 | 2498.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-11-27 13:30:00 | 2511.00 | 2025-12-02 10:15:00 | 2498.80 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-11-28 14:15:00 | 2518.00 | 2025-12-02 10:15:00 | 2498.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-10 10:00:00 | 2607.10 | 2025-12-15 13:15:00 | 2577.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-11 09:45:00 | 2594.30 | 2025-12-15 13:15:00 | 2577.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-12 10:30:00 | 2590.10 | 2025-12-15 13:15:00 | 2577.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-12 13:00:00 | 2584.00 | 2025-12-15 13:15:00 | 2577.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-30 09:15:00 | 2455.40 | 2026-01-01 09:15:00 | 2452.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-01-01 09:15:00 | 2459.10 | 2026-01-01 09:15:00 | 2452.40 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-01-05 13:00:00 | 2473.70 | 2026-01-06 09:15:00 | 2454.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-01-06 11:45:00 | 2470.70 | 2026-01-08 09:15:00 | 2450.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-13 12:00:00 | 2365.80 | 2026-01-14 12:15:00 | 2404.60 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-01-13 12:45:00 | 2365.00 | 2026-01-14 12:15:00 | 2404.60 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-13 15:15:00 | 2366.70 | 2026-01-14 12:15:00 | 2404.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-13 11:15:00 | 2634.30 | 2026-02-13 14:15:00 | 2596.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-02-13 13:30:00 | 2636.80 | 2026-02-13 14:15:00 | 2596.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-02-18 11:00:00 | 2596.70 | 2026-02-23 09:15:00 | 2623.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-02-19 09:45:00 | 2598.40 | 2026-02-23 09:15:00 | 2623.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-19 10:30:00 | 2595.80 | 2026-02-23 09:15:00 | 2623.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-02-20 14:30:00 | 2600.50 | 2026-02-23 09:15:00 | 2623.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-02-25 11:00:00 | 2649.50 | 2026-02-27 09:15:00 | 2599.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-03-27 09:15:00 | 2299.30 | 2026-04-01 09:15:00 | 2394.60 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2026-03-27 10:30:00 | 2307.70 | 2026-04-01 09:15:00 | 2394.60 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-03-27 11:00:00 | 2306.40 | 2026-04-01 09:15:00 | 2394.60 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-04-10 09:15:00 | 2400.00 | 2026-04-22 09:15:00 | 2427.90 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2026-04-13 09:30:00 | 2383.20 | 2026-04-22 09:15:00 | 2427.90 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2026-05-04 12:00:00 | 2337.50 | 2026-05-05 11:15:00 | 2392.90 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-05-04 13:45:00 | 2336.10 | 2026-05-05 11:15:00 | 2392.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-05-04 15:15:00 | 2330.30 | 2026-05-05 11:15:00 | 2392.90 | STOP_HIT | 1.00 | -2.69% |
