# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1646.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 36 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 79 |
| PARTIAL | 9 |
| TARGET_HIT | 0 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 56
- **Target hits / Stop hits / Partials:** 0 / 80 / 9
- **Avg / median % per leg:** -0.39% / -1.21%
- **Sum % (uncompounded):** -35.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 5 | 14.7% | 0 | 34 | 0 | -2.18% | -74.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.24% | -1.2% |
| BUY @ 3rd Alert (retest2) | 33 | 5 | 15.2% | 0 | 33 | 0 | -2.21% | -73.0% |
| SELL (all) | 55 | 28 | 50.9% | 0 | 46 | 9 | 0.71% | 39.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 28 | 50.9% | 0 | 46 | 9 | 0.71% | 39.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.24% | -1.2% |
| retest2 (combined) | 88 | 33 | 37.5% | 0 | 79 | 9 | -0.38% | -33.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1039.50 | 1019.95 | 1018.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1048.70 | 1036.48 | 1029.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1033.00 | 1037.66 | 1031.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1037.10 | 1037.54 | 1032.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1039.90 | 1037.44 | 1032.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 1043.90 | 1037.57 | 1033.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 1063.50 | 1071.38 | 1072.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 1063.50 | 1071.38 | 1072.21 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1180.90 | 1093.28 | 1082.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1284.40 | 1145.52 | 1108.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1357.10 | 1360.15 | 1290.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 1355.80 | 1358.53 | 1323.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1355.80 | 1358.53 | 1323.90 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1330.10 | 1338.63 | 1339.15 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1348.40 | 1340.01 | 1339.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 1349.90 | 1343.41 | 1341.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 1326.70 | 1341.92 | 1341.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1336.80 | 1340.90 | 1340.95 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1345.70 | 1333.88 | 1333.18 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1329.10 | 1332.77 | 1332.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 14:15:00 | 1325.00 | 1331.08 | 1332.06 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 1344.00 | 1333.33 | 1332.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 1358.20 | 1339.49 | 1336.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 1367.10 | 1368.85 | 1361.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 1367.10 | 1368.85 | 1361.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1366.60 | 1369.47 | 1363.37 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 1327.80 | 1356.19 | 1358.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1317.00 | 1328.99 | 1333.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1322.00 | 1318.49 | 1323.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1337.50 | 1322.29 | 1324.95 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1349.90 | 1330.44 | 1328.36 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 1334.80 | 1336.65 | 1336.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 1316.00 | 1331.79 | 1334.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1337.30 | 1319.29 | 1324.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1311.00 | 1317.64 | 1323.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 1305.00 | 1317.64 | 1323.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 1301.70 | 1305.08 | 1313.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1305.30 | 1305.26 | 1312.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:45:00 | 1300.00 | 1305.48 | 1311.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1285.00 | 1299.34 | 1306.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 1282.00 | 1299.34 | 1306.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 1282.10 | 1293.98 | 1302.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:45:00 | 1283.00 | 1287.46 | 1297.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 1283.30 | 1284.83 | 1294.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1285.50 | 1272.92 | 1281.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1285.50 | 1272.92 | 1281.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1284.40 | 1275.21 | 1281.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 1280.00 | 1278.53 | 1282.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 1280.00 | 1278.53 | 1282.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:15:00 | 1239.75 | 1252.75 | 1263.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:15:00 | 1240.03 | 1252.75 | 1263.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1236.62 | 1249.20 | 1261.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1235.00 | 1249.20 | 1261.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1250.50 | 1246.47 | 1255.00 | SL hit (close>ema200) qty=0.50 sl=1246.47 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1270.00 | 1253.69 | 1251.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 1283.40 | 1269.95 | 1261.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 1358.60 | 1364.50 | 1348.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1351.60 | 1364.73 | 1362.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1351.60 | 1364.73 | 1362.53 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 1352.00 | 1360.12 | 1360.69 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1399.00 | 1365.74 | 1362.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 1418.00 | 1376.19 | 1367.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 1408.10 | 1411.20 | 1394.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1427.00 | 1424.68 | 1416.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1427.00 | 1424.68 | 1416.00 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1399.70 | 1411.49 | 1412.99 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 1425.00 | 1414.19 | 1414.08 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1394.90 | 1410.33 | 1412.34 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 1422.10 | 1410.84 | 1410.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1431.00 | 1415.37 | 1412.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1425.00 | 1426.45 | 1420.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1405.90 | 1421.98 | 1419.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1405.90 | 1421.98 | 1419.13 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1390.40 | 1415.67 | 1416.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1374.30 | 1396.63 | 1404.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 1412.10 | 1402.54 | 1401.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1426.70 | 1410.17 | 1405.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1425.00 | 1426.53 | 1417.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 1427.00 | 1426.62 | 1418.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1427.00 | 1426.62 | 1418.65 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 1387.30 | 1419.15 | 1420.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 1374.20 | 1393.28 | 1403.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 1379.60 | 1377.13 | 1392.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1392.70 | 1382.79 | 1391.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1394.20 | 1382.79 | 1391.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1382.00 | 1382.63 | 1390.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1374.00 | 1382.63 | 1390.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1397.10 | 1385.52 | 1391.02 | SL hit (close>static) qty=1.00 sl=1395.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1404.10 | 1394.45 | 1394.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1414.30 | 1403.07 | 1398.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 1453.00 | 1453.66 | 1438.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1462.00 | 1453.66 | 1438.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1455.60 | 1454.05 | 1440.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 1450.90 | 1454.05 | 1440.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1450.60 | 1456.69 | 1447.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1450.60 | 1456.69 | 1447.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1451.10 | 1455.57 | 1447.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 1443.90 | 1453.24 | 1447.19 | SL hit (close<ema400) qty=1.00 sl=1447.19 alert=retest1 |

### Cycle 24 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1465.10 | 1488.31 | 1489.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1422.80 | 1464.19 | 1475.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1395.70 | 1392.07 | 1408.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 1395.70 | 1392.07 | 1408.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1391.00 | 1379.45 | 1385.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:15:00 | 1395.30 | 1379.45 | 1385.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1392.40 | 1382.04 | 1386.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:15:00 | 1379.90 | 1384.88 | 1386.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:30:00 | 1374.80 | 1379.91 | 1382.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 1382.20 | 1374.67 | 1375.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1384.70 | 1374.67 | 1375.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 1385.30 | 1376.79 | 1376.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1385.30 | 1376.79 | 1376.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1389.10 | 1379.25 | 1377.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1375.50 | 1379.15 | 1377.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1367.20 | 1376.76 | 1376.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1367.20 | 1376.76 | 1376.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 1361.00 | 1373.61 | 1375.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 1357.70 | 1370.43 | 1373.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1377.90 | 1368.40 | 1371.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1374.20 | 1369.56 | 1371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1375.20 | 1369.56 | 1371.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1384.60 | 1375.05 | 1373.81 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 1360.50 | 1373.82 | 1373.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 1348.40 | 1367.13 | 1370.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 1341.70 | 1351.71 | 1354.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 1341.10 | 1346.44 | 1350.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:45:00 | 1343.90 | 1344.51 | 1348.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1337.00 | 1344.81 | 1348.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1329.90 | 1328.08 | 1330.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 1328.40 | 1328.56 | 1330.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1328.50 | 1328.56 | 1330.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1334.60 | 1329.76 | 1330.56 | SL hit (close>static) qty=1.00 sl=1330.40 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1304.80 | 1293.72 | 1292.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1328.70 | 1305.20 | 1298.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1410.00 | 1413.97 | 1389.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1410.00 | 1413.97 | 1389.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1401.20 | 1410.21 | 1391.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1422.80 | 1404.59 | 1398.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 1390.00 | 1409.00 | 1403.94 | SL hit (close<static) qty=1.00 sl=1390.10 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1379.80 | 1399.24 | 1400.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 1368.90 | 1389.55 | 1395.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1361.20 | 1352.55 | 1367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1364.70 | 1354.98 | 1367.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1364.20 | 1354.98 | 1367.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1361.50 | 1356.28 | 1366.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1366.20 | 1356.28 | 1366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1363.50 | 1358.99 | 1365.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:15:00 | 1365.80 | 1358.99 | 1365.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1365.80 | 1360.35 | 1365.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 1367.00 | 1360.35 | 1365.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1361.00 | 1360.48 | 1365.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1354.60 | 1360.48 | 1365.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 1356.00 | 1358.11 | 1363.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1356.30 | 1354.61 | 1358.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 1380.90 | 1358.21 | 1357.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1380.90 | 1358.21 | 1357.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 1383.40 | 1363.25 | 1359.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1384.00 | 1386.64 | 1377.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1384.00 | 1386.64 | 1377.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1383.90 | 1385.64 | 1378.62 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1354.70 | 1374.30 | 1374.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 1348.40 | 1359.87 | 1365.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1347.40 | 1337.45 | 1346.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1343.00 | 1338.56 | 1346.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 1342.20 | 1338.56 | 1346.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 1342.60 | 1338.67 | 1344.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 1338.80 | 1336.86 | 1339.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1373.50 | 1330.36 | 1327.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 1373.50 | 1330.36 | 1327.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 1381.50 | 1340.59 | 1331.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 1372.00 | 1361.25 | 1349.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1404.80 | 1359.46 | 1352.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1371.00 | 1389.83 | 1388.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1340.90 | 1380.05 | 1384.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1340.90 | 1380.05 | 1384.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 11:15:00 | 1326.60 | 1365.11 | 1376.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1405.40 | 1359.93 | 1367.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1395.00 | 1366.94 | 1370.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 1381.90 | 1366.94 | 1370.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1388.30 | 1374.55 | 1373.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 1388.30 | 1374.55 | 1373.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1420.60 | 1384.68 | 1379.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1354.60 | 1378.36 | 1381.04 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 1369.90 | 1362.85 | 1362.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 1373.30 | 1364.94 | 1363.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1413.00 | 1400.95 | 1392.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1417.90 | 1400.95 | 1392.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1415.00 | 1402.84 | 1394.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1417.80 | 1408.08 | 1398.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1418.10 | 1428.18 | 1420.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1418.10 | 1428.18 | 1420.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1409.30 | 1424.41 | 1419.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1410.30 | 1424.41 | 1419.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1414.50 | 1422.43 | 1418.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1383.20 | 1414.58 | 1415.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1383.20 | 1414.58 | 1415.56 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1409.40 | 1403.74 | 1403.68 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1395.50 | 1402.29 | 1403.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 1394.60 | 1399.99 | 1401.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 1393.30 | 1394.47 | 1397.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1406.90 | 1392.34 | 1393.66 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1408.10 | 1395.49 | 1394.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 1418.80 | 1404.16 | 1399.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 1406.00 | 1407.49 | 1402.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:45:00 | 1405.00 | 1407.49 | 1402.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1401.30 | 1406.25 | 1402.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 1402.10 | 1406.25 | 1402.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1404.30 | 1405.86 | 1402.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 1401.30 | 1405.86 | 1402.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1403.00 | 1405.29 | 1402.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 1403.00 | 1405.29 | 1402.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1411.00 | 1406.43 | 1403.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 1415.00 | 1406.45 | 1404.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1429.10 | 1406.95 | 1405.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 1405.90 | 1416.32 | 1417.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 1405.90 | 1416.32 | 1417.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 1401.70 | 1413.40 | 1415.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1392.80 | 1385.13 | 1391.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1399.00 | 1387.90 | 1392.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1386.70 | 1387.90 | 1392.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 1387.00 | 1387.43 | 1391.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 1385.80 | 1386.31 | 1390.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 1382.90 | 1388.15 | 1389.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1387.00 | 1386.98 | 1388.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1389.70 | 1386.98 | 1388.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1385.00 | 1386.59 | 1388.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:30:00 | 1383.80 | 1386.59 | 1388.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1385.60 | 1386.14 | 1387.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 1383.20 | 1385.33 | 1387.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 1383.00 | 1386.06 | 1387.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:00:00 | 1384.60 | 1386.06 | 1387.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 1411.10 | 1390.04 | 1388.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 10:15:00 | 1411.10 | 1390.04 | 1388.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 1414.50 | 1394.93 | 1390.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1406.00 | 1410.13 | 1401.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1406.00 | 1410.13 | 1401.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1400.20 | 1408.15 | 1400.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1400.20 | 1408.15 | 1400.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1386.10 | 1403.74 | 1399.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 1409.20 | 1403.74 | 1399.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 1401.00 | 1406.56 | 1403.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 1401.50 | 1405.04 | 1403.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:00:00 | 1403.30 | 1407.39 | 1406.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1405.00 | 1406.82 | 1406.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1392.20 | 1406.82 | 1406.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1386.30 | 1402.71 | 1404.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1386.30 | 1402.71 | 1404.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1381.40 | 1398.45 | 1402.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1389.50 | 1386.67 | 1394.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1402.70 | 1389.87 | 1395.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1382.20 | 1389.87 | 1395.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1381.20 | 1388.14 | 1393.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:30:00 | 1376.90 | 1386.17 | 1392.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1375.50 | 1386.17 | 1392.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1403.60 | 1385.47 | 1384.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 1403.60 | 1385.47 | 1384.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1412.20 | 1393.30 | 1388.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 14:15:00 | 1528.00 | 1532.70 | 1506.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1528.00 | 1532.70 | 1506.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1518.80 | 1536.77 | 1522.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1518.80 | 1536.77 | 1522.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1524.40 | 1534.30 | 1522.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1509.30 | 1534.30 | 1522.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1526.40 | 1532.72 | 1522.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 1503.80 | 1532.72 | 1522.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1534.20 | 1533.01 | 1523.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 1522.20 | 1533.01 | 1523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1523.80 | 1531.17 | 1523.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 1521.60 | 1531.17 | 1523.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1537.80 | 1532.50 | 1525.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 1545.30 | 1532.50 | 1525.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1541.90 | 1534.02 | 1526.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 1545.50 | 1535.61 | 1527.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1496.00 | 1528.81 | 1526.27 | SL hit (close<static) qty=1.00 sl=1522.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 1496.10 | 1522.27 | 1523.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 1481.20 | 1514.05 | 1519.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 1537.00 | 1509.26 | 1513.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1541.20 | 1515.65 | 1516.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 1541.20 | 1515.65 | 1516.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 1536.80 | 1519.88 | 1518.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 1550.60 | 1531.79 | 1524.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1554.20 | 1557.95 | 1545.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 1554.20 | 1557.95 | 1545.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1551.70 | 1556.70 | 1545.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1552.30 | 1556.70 | 1545.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1561.00 | 1558.99 | 1552.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1549.50 | 1558.99 | 1552.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1552.60 | 1557.72 | 1552.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 1553.70 | 1557.72 | 1552.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1553.20 | 1556.81 | 1552.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:30:00 | 1565.20 | 1557.45 | 1553.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 1544.60 | 1553.93 | 1552.11 | SL hit (close<static) qty=1.00 sl=1547.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 1532.20 | 1549.62 | 1550.94 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 1559.10 | 1550.24 | 1549.82 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1549.20 | 1551.64 | 1551.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1480.00 | 1537.31 | 1545.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 1547.10 | 1539.49 | 1545.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 1520.50 | 1535.69 | 1543.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 1513.90 | 1530.17 | 1540.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 13:15:00 | 1438.20 | 1473.50 | 1500.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 1442.50 | 1440.74 | 1462.61 | SL hit (close>ema200) qty=0.50 sl=1440.74 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1497.30 | 1463.10 | 1461.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1512.80 | 1480.95 | 1475.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 1475.40 | 1496.62 | 1497.85 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1516.40 | 1496.09 | 1495.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1520.00 | 1503.75 | 1498.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 1536.70 | 1536.90 | 1523.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 12:15:00 | 1533.90 | 1534.68 | 1524.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1533.90 | 1534.68 | 1524.60 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1491.20 | 1515.88 | 1518.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1480.90 | 1500.20 | 1508.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 1475.20 | 1474.43 | 1485.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 1489.00 | 1477.11 | 1485.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1489.00 | 1477.11 | 1485.03 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1448.50 | 1437.84 | 1436.97 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 15:15:00 | 1431.90 | 1435.90 | 1436.19 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1438.50 | 1436.42 | 1436.40 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 10:15:00 | 1431.90 | 1435.51 | 1435.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 1417.90 | 1431.27 | 1433.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1464.80 | 1436.82 | 1435.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 12:15:00 | 1475.00 | 1456.10 | 1448.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1425.10 | 1444.01 | 1446.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1417.20 | 1438.65 | 1443.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 09:15:00 | 1490.00 | 1451.31 | 1448.01 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 1438.20 | 1451.82 | 1453.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 12:15:00 | 1429.60 | 1445.55 | 1449.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 1454.80 | 1428.29 | 1437.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 12:15:00 | 1514.10 | 1445.45 | 1444.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1522.00 | 1460.76 | 1451.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1549.70 | 1554.99 | 1528.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:45:00 | 1546.40 | 1554.99 | 1528.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1543.60 | 1555.05 | 1537.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1543.20 | 1555.05 | 1537.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1534.80 | 1551.00 | 1536.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1533.50 | 1551.00 | 1536.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1523.40 | 1545.48 | 1535.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 1523.40 | 1545.48 | 1535.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 1515.00 | 1528.19 | 1529.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1470.20 | 1516.59 | 1524.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 1455.00 | 1452.59 | 1473.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 1455.00 | 1452.59 | 1473.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1507.90 | 1465.58 | 1474.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1507.90 | 1465.58 | 1474.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1498.00 | 1472.07 | 1476.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1486.30 | 1472.07 | 1476.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 1489.90 | 1475.07 | 1477.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 1501.40 | 1480.34 | 1479.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1501.40 | 1480.34 | 1479.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1506.60 | 1493.13 | 1486.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1499.30 | 1520.91 | 1508.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1505.50 | 1517.83 | 1507.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 1510.10 | 1517.83 | 1507.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1501.10 | 1514.49 | 1507.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1501.10 | 1514.49 | 1507.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1475.90 | 1506.77 | 1504.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1475.90 | 1506.77 | 1504.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1474.50 | 1500.31 | 1501.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1468.20 | 1493.89 | 1498.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:30:00 | 1482.20 | 1490.11 | 1495.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1509.80 | 1481.65 | 1481.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1509.80 | 1481.65 | 1481.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 1534.30 | 1492.18 | 1486.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 1592.20 | 1623.83 | 1597.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 1596.00 | 1618.27 | 1597.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 1603.50 | 1618.27 | 1597.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 1605.00 | 1614.61 | 1597.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 14:00:00 | 1604.00 | 1612.49 | 1597.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1652.90 | 1606.46 | 1597.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1600.30 | 1615.92 | 1606.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 1599.60 | 1615.92 | 1606.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1598.60 | 1612.46 | 1605.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1593.00 | 1612.46 | 1605.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1588.80 | 1607.73 | 1604.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 1593.00 | 1607.73 | 1604.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1506.40 | 1584.62 | 1594.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1506.40 | 1584.62 | 1594.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 1478.20 | 1563.34 | 1583.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1551.50 | 1547.41 | 1568.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1551.50 | 1547.41 | 1568.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1558.70 | 1550.42 | 1565.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1566.50 | 1550.42 | 1565.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1549.20 | 1550.17 | 1564.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 1549.20 | 1550.17 | 1564.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1560.00 | 1552.14 | 1563.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:00:00 | 1560.00 | 1552.14 | 1563.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 1558.20 | 1553.35 | 1563.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:45:00 | 1575.70 | 1553.35 | 1563.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1560.20 | 1554.53 | 1560.87 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1574.40 | 1563.40 | 1562.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 1594.30 | 1571.93 | 1566.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 1602.80 | 1608.13 | 1596.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1600.00 | 1606.50 | 1597.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1591.20 | 1606.50 | 1597.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1603.00 | 1605.80 | 1597.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1624.70 | 1605.80 | 1597.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:00:00 | 1610.00 | 1628.93 | 1626.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1628.00 | 1636.84 | 1637.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1628.00 | 1636.84 | 1637.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 09:15:00 | 1617.10 | 1632.89 | 1635.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 1608.50 | 1607.39 | 1618.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:00:00 | 1608.50 | 1607.39 | 1618.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1609.20 | 1602.25 | 1609.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 1612.30 | 1602.25 | 1609.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1600.00 | 1601.80 | 1608.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 1597.50 | 1600.68 | 1607.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 1594.00 | 1600.31 | 1606.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 1633.00 | 1606.00 | 1607.81 | SL hit (close>static) qty=1.00 sl=1624.10 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 1657.10 | 1616.22 | 1612.29 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 1607.70 | 1636.69 | 1636.77 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1703.10 | 1646.50 | 1641.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1751.90 | 1681.45 | 1658.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1697.20 | 1707.75 | 1682.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 1697.20 | 1707.75 | 1682.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1668.80 | 1697.74 | 1682.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 1668.80 | 1697.74 | 1682.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1684.50 | 1695.09 | 1682.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 1687.70 | 1692.66 | 1682.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 1686.10 | 1688.04 | 1681.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1695.50 | 1688.04 | 1681.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1643.00 | 1716.32 | 1722.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 1643.00 | 1716.32 | 1722.75 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1039.90 | 2025-05-22 13:15:00 | 1063.50 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-05-14 10:15:00 | 1043.90 | 2025-05-22 13:15:00 | 1063.50 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-06-27 11:15:00 | 1305.00 | 2025-07-07 10:15:00 | 1239.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 09:45:00 | 1301.70 | 2025-07-07 10:15:00 | 1240.03 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1305.30 | 2025-07-07 11:15:00 | 1236.62 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-06-30 12:45:00 | 1300.00 | 2025-07-07 11:15:00 | 1235.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-27 11:15:00 | 1305.00 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-06-30 09:45:00 | 1301.70 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1305.30 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-06-30 12:45:00 | 1300.00 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-07-01 10:15:00 | 1282.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-07-01 12:00:00 | 1282.10 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-07-01 14:45:00 | 1283.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-07-02 09:30:00 | 1283.30 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-07-03 14:30:00 | 1280.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-07-03 15:15:00 | 1280.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1374.00 | 2025-08-12 09:15:00 | 1397.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2025-08-18 09:15:00 | 1462.00 | 2025-08-19 09:15:00 | 1443.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-19 11:15:00 | 1462.50 | 2025-08-26 09:15:00 | 1465.10 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-09-04 14:15:00 | 1379.90 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-05 14:30:00 | 1374.80 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-09 12:30:00 | 1382.20 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-09 13:00:00 | 1384.70 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-18 14:30:00 | 1341.70 | 2025-09-25 09:15:00 | 1334.60 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-09-19 11:45:00 | 1341.10 | 2025-09-25 09:15:00 | 1334.60 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-09-19 14:45:00 | 1343.90 | 2025-09-25 13:15:00 | 1331.30 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1337.00 | 2025-09-25 13:15:00 | 1331.30 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1328.40 | 2025-09-26 15:15:00 | 1276.70 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1328.50 | 2025-09-29 09:15:00 | 1274.62 | PARTIAL | 0.50 | 4.06% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1328.90 | 2025-09-29 09:15:00 | 1274.04 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2025-09-25 13:00:00 | 1328.70 | 2025-09-29 09:15:00 | 1270.15 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1328.40 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1328.50 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1328.90 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-09-25 13:00:00 | 1328.70 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1422.80 | 2025-10-13 10:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-10-16 10:15:00 | 1354.60 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1356.00 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 11:00:00 | 1356.30 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-29 12:15:00 | 1342.20 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-29 14:45:00 | 1342.60 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-10-31 09:45:00 | 1338.80 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-11-07 11:30:00 | 1372.00 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-11-10 09:15:00 | 1404.80 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1371.00 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-11-13 11:15:00 | 1381.90 | 2025-11-13 12:15:00 | 1388.30 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1413.00 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-11-27 14:15:00 | 1417.90 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1415.00 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-11-28 09:45:00 | 1417.80 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1393.30 | 2025-12-09 10:15:00 | 1406.90 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-11 14:15:00 | 1415.00 | 2025-12-15 13:15:00 | 1405.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1429.10 | 2025-12-15 13:15:00 | 1405.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1386.70 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-18 11:00:00 | 1387.00 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-18 12:30:00 | 1385.80 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-19 11:45:00 | 1382.90 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-22 10:30:00 | 1383.20 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-12-22 12:30:00 | 1383.00 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-12-22 13:00:00 | 1384.60 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-24 12:15:00 | 1409.20 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-26 11:15:00 | 1401.00 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-26 13:30:00 | 1401.50 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-29 14:00:00 | 1403.30 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-31 10:30:00 | 1376.90 | 2026-01-01 14:15:00 | 1403.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1375.50 | 2026-01-01 14:15:00 | 1403.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-01-09 13:15:00 | 1545.30 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-09 13:45:00 | 1541.90 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-01-09 14:30:00 | 1545.50 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-01-19 12:30:00 | 1565.20 | 2026-01-19 14:15:00 | 1544.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-19 14:30:00 | 1556.00 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-20 09:45:00 | 1557.90 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-01-20 10:30:00 | 1556.80 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1513.90 | 2026-01-28 13:15:00 | 1438.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1513.90 | 2026-01-29 15:15:00 | 1442.50 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1486.30 | 2026-03-17 12:15:00 | 1501.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-17 11:45:00 | 1489.90 | 2026-03-17 12:15:00 | 1501.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-20 10:30:00 | 1482.20 | 2026-03-24 10:15:00 | 1509.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-30 12:15:00 | 1603.50 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.06% |
| BUY | retest2 | 2026-03-30 13:15:00 | 1605.00 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest2 | 2026-03-30 14:00:00 | 1604.00 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.08% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1652.90 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -8.86% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1624.70 | 2026-04-20 15:15:00 | 1628.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-04-16 14:00:00 | 1610.00 | 2026-04-20 15:15:00 | 1628.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-04-23 11:30:00 | 1597.50 | 2026-04-23 15:15:00 | 1633.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-23 14:15:00 | 1594.00 | 2026-04-23 15:15:00 | 1633.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-04-30 13:45:00 | 1687.70 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-04-30 14:45:00 | 1686.10 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-30 15:15:00 | 1695.50 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -3.10% |
