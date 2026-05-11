# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1671.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 106 |
| ALERT2 | 105 |
| ALERT2_SKIP | 49 |
| ALERT3 | 295 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 137 |
| PARTIAL | 18 |
| TARGET_HIT | 2 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 158 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 98
- **Target hits / Stop hits / Partials:** 2 / 138 / 18
- **Avg / median % per leg:** 0.48% / -0.63%
- **Sum % (uncompounded):** 75.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 18 | 25.4% | 0 | 71 | 0 | -0.61% | -43.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.37% | -2.7% |
| BUY @ 3rd Alert (retest2) | 69 | 18 | 26.1% | 0 | 69 | 0 | -0.58% | -40.3% |
| SELL (all) | 87 | 42 | 48.3% | 2 | 67 | 18 | 1.37% | 119.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.97% | -2.0% |
| SELL @ 3rd Alert (retest2) | 86 | 42 | 48.8% | 2 | 66 | 18 | 1.41% | 121.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.57% | -4.7% |
| retest2 (combined) | 155 | 60 | 38.7% | 2 | 135 | 18 | 0.52% | 80.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1285.70 | 1266.98 | 1265.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 1299.35 | 1276.19 | 1270.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 10:15:00 | 1274.30 | 1279.17 | 1274.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 10:15:00 | 1274.30 | 1279.17 | 1274.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1274.30 | 1279.17 | 1274.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 1275.85 | 1279.17 | 1274.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 1284.65 | 1280.27 | 1275.33 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 1248.40 | 1272.35 | 1272.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 1241.70 | 1258.66 | 1265.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 1247.30 | 1243.76 | 1253.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:15:00 | 1249.45 | 1243.76 | 1253.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1270.00 | 1250.20 | 1254.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 1270.00 | 1250.20 | 1254.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1266.90 | 1253.54 | 1255.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 1268.60 | 1253.54 | 1255.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1260.00 | 1255.47 | 1256.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 1257.00 | 1255.47 | 1256.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1268.85 | 1258.15 | 1257.51 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 1251.35 | 1256.79 | 1256.95 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 1262.90 | 1258.01 | 1257.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 11:15:00 | 1275.05 | 1261.42 | 1259.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 12:15:00 | 1285.15 | 1286.51 | 1277.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:00:00 | 1285.15 | 1286.51 | 1277.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1279.95 | 1285.20 | 1278.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:45:00 | 1279.95 | 1285.20 | 1278.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1293.40 | 1286.84 | 1279.40 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 1268.90 | 1278.63 | 1278.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 1249.60 | 1270.71 | 1274.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 11:15:00 | 1263.20 | 1262.37 | 1266.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 11:30:00 | 1261.85 | 1262.37 | 1266.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1268.00 | 1263.49 | 1266.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:00:00 | 1268.00 | 1263.49 | 1266.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 1267.15 | 1264.22 | 1266.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 1269.30 | 1264.22 | 1266.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1269.50 | 1265.28 | 1266.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:30:00 | 1270.00 | 1265.28 | 1266.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1272.00 | 1266.62 | 1267.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1266.85 | 1266.62 | 1267.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 1266.70 | 1263.14 | 1264.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1271.45 | 1246.16 | 1243.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1271.45 | 1246.16 | 1243.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 1291.80 | 1277.36 | 1263.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1246.00 | 1271.09 | 1261.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1246.00 | 1271.09 | 1261.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1246.00 | 1271.09 | 1261.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1246.00 | 1271.09 | 1261.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1218.45 | 1260.56 | 1258.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1218.45 | 1260.56 | 1258.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 1231.55 | 1254.76 | 1255.62 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1276.85 | 1258.12 | 1255.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 1288.35 | 1264.16 | 1258.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1352.30 | 1354.51 | 1338.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 1344.00 | 1354.51 | 1338.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1341.95 | 1352.00 | 1338.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1341.30 | 1352.00 | 1338.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1342.20 | 1349.73 | 1341.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:45:00 | 1344.20 | 1349.73 | 1341.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 1336.10 | 1347.00 | 1340.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 1336.10 | 1347.00 | 1340.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1332.40 | 1344.08 | 1339.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 1332.40 | 1344.08 | 1339.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1338.00 | 1342.87 | 1339.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 1339.75 | 1342.87 | 1339.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1431.10 | 1450.03 | 1443.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 1440.05 | 1450.03 | 1443.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1433.15 | 1446.66 | 1442.60 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 13:15:00 | 1429.95 | 1439.51 | 1439.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 1417.25 | 1433.71 | 1437.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 1424.90 | 1415.45 | 1423.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 1424.90 | 1415.45 | 1423.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1424.90 | 1415.45 | 1423.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 1424.90 | 1415.45 | 1423.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1425.60 | 1417.48 | 1423.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:15:00 | 1425.60 | 1417.48 | 1423.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1420.70 | 1418.13 | 1423.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:30:00 | 1414.45 | 1422.47 | 1424.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 1414.95 | 1422.47 | 1424.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1434.30 | 1422.70 | 1424.09 | SL hit (close>static) qty=1.00 sl=1427.80 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 1437.00 | 1425.56 | 1425.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 1443.10 | 1431.34 | 1428.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 1428.65 | 1431.67 | 1429.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 1428.65 | 1431.67 | 1429.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1428.65 | 1431.67 | 1429.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 1428.65 | 1431.67 | 1429.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1434.80 | 1432.29 | 1429.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 10:15:00 | 1439.55 | 1431.54 | 1429.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 12:00:00 | 1438.65 | 1433.03 | 1430.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 1422.55 | 1430.83 | 1430.18 | SL hit (close<static) qty=1.00 sl=1427.85 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 10:15:00 | 1423.70 | 1430.08 | 1430.23 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 1434.00 | 1428.68 | 1428.46 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 1425.90 | 1428.09 | 1428.23 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 1437.30 | 1429.93 | 1429.06 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1399.40 | 1424.61 | 1426.84 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 1436.10 | 1418.38 | 1417.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 15:15:00 | 1438.95 | 1422.50 | 1419.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 1424.15 | 1427.15 | 1423.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 12:45:00 | 1425.70 | 1427.15 | 1423.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1421.95 | 1426.11 | 1423.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 1420.55 | 1426.11 | 1423.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1420.40 | 1424.97 | 1423.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 1420.40 | 1424.97 | 1423.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1421.75 | 1424.32 | 1422.92 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 1411.50 | 1421.76 | 1421.88 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 1431.40 | 1420.94 | 1420.88 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1398.45 | 1416.74 | 1419.21 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 1428.95 | 1413.66 | 1411.60 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 1408.00 | 1412.39 | 1412.42 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1423.95 | 1414.70 | 1413.47 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 1399.55 | 1412.31 | 1412.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 13:15:00 | 1390.25 | 1407.90 | 1410.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 1400.95 | 1398.77 | 1405.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 1400.95 | 1398.77 | 1405.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1400.25 | 1399.26 | 1404.29 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1413.50 | 1406.85 | 1406.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 11:15:00 | 1433.90 | 1416.29 | 1411.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 10:15:00 | 1432.00 | 1433.58 | 1424.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 11:00:00 | 1432.00 | 1433.58 | 1424.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1429.35 | 1432.74 | 1424.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 1425.70 | 1432.74 | 1424.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1430.00 | 1432.19 | 1425.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 1425.35 | 1432.19 | 1425.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 1424.65 | 1430.68 | 1425.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 1424.65 | 1430.68 | 1425.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1425.15 | 1429.58 | 1425.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:15:00 | 1426.00 | 1429.58 | 1425.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 1426.00 | 1428.86 | 1425.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 1405.05 | 1428.86 | 1425.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1423.15 | 1427.72 | 1424.99 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 12:15:00 | 1407.00 | 1420.18 | 1421.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 13:15:00 | 1404.70 | 1417.08 | 1420.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 1397.95 | 1396.53 | 1406.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 14:00:00 | 1397.95 | 1396.53 | 1406.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1400.50 | 1397.32 | 1406.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 1406.65 | 1397.32 | 1406.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1410.00 | 1399.86 | 1406.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1417.35 | 1399.86 | 1406.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1418.15 | 1403.52 | 1407.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1419.15 | 1403.52 | 1407.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1400.50 | 1402.91 | 1406.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:30:00 | 1396.00 | 1402.53 | 1406.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:30:00 | 1398.95 | 1401.30 | 1405.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 1408.75 | 1386.01 | 1385.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 1408.75 | 1386.01 | 1385.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 1421.90 | 1393.18 | 1389.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 1378.30 | 1396.93 | 1392.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 1378.30 | 1396.93 | 1392.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1378.30 | 1396.93 | 1392.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:15:00 | 1444.00 | 1402.13 | 1395.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:45:00 | 1430.65 | 1425.17 | 1414.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:30:00 | 1429.30 | 1431.61 | 1420.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 1432.90 | 1427.93 | 1421.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1419.40 | 1428.63 | 1423.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 1419.40 | 1428.63 | 1423.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1415.55 | 1426.01 | 1423.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1415.55 | 1426.01 | 1423.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1418.50 | 1424.51 | 1422.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 1423.80 | 1424.51 | 1422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1421.90 | 1424.45 | 1422.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 1420.40 | 1424.45 | 1422.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 1407.80 | 1421.12 | 1421.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 1407.80 | 1421.12 | 1421.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 1397.95 | 1416.49 | 1419.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1404.40 | 1401.67 | 1409.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 1404.40 | 1401.67 | 1409.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 1394.65 | 1400.27 | 1408.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:30:00 | 1390.15 | 1398.70 | 1406.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 13:30:00 | 1393.60 | 1397.06 | 1405.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 14:15:00 | 1379.35 | 1368.98 | 1367.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1379.35 | 1368.98 | 1367.92 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 1356.80 | 1366.90 | 1367.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 1351.90 | 1363.90 | 1366.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 1349.85 | 1348.68 | 1353.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:45:00 | 1349.35 | 1348.68 | 1353.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1352.00 | 1349.64 | 1353.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 1352.00 | 1349.64 | 1353.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1347.75 | 1349.27 | 1352.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 1341.70 | 1347.78 | 1351.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 1343.10 | 1347.45 | 1350.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:00:00 | 1340.90 | 1346.14 | 1350.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 10:15:00 | 1343.50 | 1340.19 | 1339.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1343.50 | 1340.19 | 1339.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 1360.10 | 1345.82 | 1342.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 12:15:00 | 1349.95 | 1357.77 | 1351.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 12:15:00 | 1349.95 | 1357.77 | 1351.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1349.95 | 1357.77 | 1351.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 1349.95 | 1357.77 | 1351.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 1351.75 | 1356.57 | 1351.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:45:00 | 1359.85 | 1355.39 | 1352.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 1355.00 | 1373.16 | 1375.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1355.00 | 1373.16 | 1375.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1348.00 | 1359.94 | 1367.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1380.00 | 1362.36 | 1367.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 1380.00 | 1362.36 | 1367.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1380.00 | 1362.36 | 1367.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 1380.00 | 1362.36 | 1367.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1383.00 | 1366.49 | 1368.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1383.00 | 1366.49 | 1368.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1388.00 | 1373.41 | 1371.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 1398.85 | 1386.19 | 1378.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 13:15:00 | 1450.00 | 1453.74 | 1437.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 14:00:00 | 1450.00 | 1453.74 | 1437.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1430.55 | 1449.84 | 1439.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 1431.30 | 1449.84 | 1439.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1440.00 | 1447.87 | 1439.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 1446.40 | 1447.37 | 1440.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 1445.95 | 1447.08 | 1440.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 1510.05 | 1518.82 | 1519.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 12:15:00 | 1510.05 | 1518.82 | 1519.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 13:15:00 | 1505.95 | 1516.24 | 1517.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 15:15:00 | 1515.00 | 1514.57 | 1516.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 1521.25 | 1515.91 | 1517.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1521.25 | 1515.91 | 1517.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 1521.25 | 1515.91 | 1517.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1525.85 | 1517.90 | 1517.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:45:00 | 1524.10 | 1517.90 | 1517.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 1529.00 | 1520.12 | 1518.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 1531.95 | 1522.48 | 1520.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1515.30 | 1521.05 | 1519.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 1515.30 | 1521.05 | 1519.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1515.30 | 1521.05 | 1519.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1515.30 | 1521.05 | 1519.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1519.45 | 1520.73 | 1519.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 1525.50 | 1520.73 | 1519.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1595.00 | 1611.02 | 1612.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 1595.00 | 1611.02 | 1612.01 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 1614.00 | 1612.47 | 1612.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 1627.40 | 1616.70 | 1614.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 09:15:00 | 1624.95 | 1625.87 | 1620.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1624.95 | 1625.87 | 1620.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1624.95 | 1625.87 | 1620.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 1625.55 | 1625.87 | 1620.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1612.80 | 1623.26 | 1619.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 1612.80 | 1623.26 | 1619.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1603.10 | 1619.23 | 1617.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:45:00 | 1602.05 | 1619.23 | 1617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 1602.60 | 1615.90 | 1616.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 1596.65 | 1612.05 | 1614.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 15:15:00 | 1613.45 | 1612.27 | 1614.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 15:15:00 | 1613.45 | 1612.27 | 1614.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1613.45 | 1612.27 | 1614.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 1618.60 | 1612.27 | 1614.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1614.45 | 1612.71 | 1614.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 1615.00 | 1612.71 | 1614.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1612.30 | 1612.63 | 1614.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1612.30 | 1612.63 | 1614.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 1619.90 | 1612.12 | 1613.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:45:00 | 1620.75 | 1612.12 | 1613.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 1612.50 | 1612.19 | 1613.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1610.75 | 1612.19 | 1613.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:00:00 | 1609.70 | 1611.70 | 1613.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:45:00 | 1606.25 | 1609.00 | 1611.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:15:00 | 1530.21 | 1571.30 | 1589.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:15:00 | 1529.21 | 1571.30 | 1589.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:15:00 | 1525.94 | 1571.30 | 1589.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 1483.15 | 1482.68 | 1503.61 | SL hit (close>ema200) qty=0.50 sl=1482.68 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1529.15 | 1511.67 | 1511.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1577.60 | 1528.27 | 1519.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 1529.25 | 1551.74 | 1541.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 1529.25 | 1551.74 | 1541.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1529.25 | 1551.74 | 1541.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 1529.25 | 1551.74 | 1541.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1521.55 | 1545.70 | 1539.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 1523.40 | 1545.70 | 1539.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 1518.00 | 1533.33 | 1534.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 1495.70 | 1523.05 | 1529.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 1503.65 | 1503.37 | 1515.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 15:00:00 | 1503.65 | 1503.37 | 1515.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1514.05 | 1504.69 | 1514.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:00:00 | 1503.50 | 1506.30 | 1512.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 14:15:00 | 1504.70 | 1506.49 | 1512.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1503.65 | 1509.25 | 1512.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:45:00 | 1502.80 | 1506.85 | 1511.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1506.40 | 1506.14 | 1510.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 1506.40 | 1506.14 | 1510.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1502.00 | 1505.32 | 1509.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 1497.85 | 1505.32 | 1509.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 1497.75 | 1504.77 | 1508.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 1497.30 | 1504.47 | 1507.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:45:00 | 1495.65 | 1502.31 | 1506.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1480.05 | 1486.53 | 1494.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 1470.00 | 1484.59 | 1493.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1428.33 | 1460.00 | 1475.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1429.46 | 1460.00 | 1475.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1428.47 | 1460.00 | 1475.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1427.66 | 1460.00 | 1475.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 1464.70 | 1456.70 | 1468.45 | SL hit (close>ema200) qty=0.50 sl=1456.70 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1284.80 | 1267.01 | 1266.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1300.25 | 1279.55 | 1272.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1271.20 | 1287.38 | 1279.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1271.20 | 1287.38 | 1279.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1271.20 | 1287.38 | 1279.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1271.20 | 1287.38 | 1279.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1269.85 | 1283.87 | 1278.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1270.00 | 1283.87 | 1278.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1283.00 | 1283.92 | 1280.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1287.75 | 1283.92 | 1280.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1286.00 | 1284.34 | 1281.26 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1266.10 | 1277.61 | 1278.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1259.65 | 1274.02 | 1277.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 1265.00 | 1264.74 | 1269.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 09:15:00 | 1266.60 | 1264.74 | 1269.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1260.00 | 1263.79 | 1268.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 1250.40 | 1260.42 | 1266.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 1187.88 | 1213.94 | 1231.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 15:15:00 | 1214.90 | 1210.85 | 1221.80 | SL hit (close>ema200) qty=0.50 sl=1210.85 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 1244.40 | 1227.69 | 1227.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1250.30 | 1238.04 | 1232.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 11:15:00 | 1237.15 | 1238.44 | 1233.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 12:00:00 | 1237.15 | 1238.44 | 1233.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 1231.20 | 1236.99 | 1233.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 1231.20 | 1236.99 | 1233.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1224.25 | 1234.44 | 1232.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 1224.25 | 1234.44 | 1232.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1236.00 | 1234.76 | 1233.13 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 1205.85 | 1227.55 | 1230.05 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1252.85 | 1224.41 | 1221.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 12:15:00 | 1268.75 | 1250.01 | 1238.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 1264.55 | 1275.16 | 1265.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 1264.55 | 1275.16 | 1265.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1264.55 | 1275.16 | 1265.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 1264.55 | 1275.16 | 1265.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1267.90 | 1273.71 | 1265.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1267.90 | 1273.71 | 1265.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1263.35 | 1271.64 | 1265.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 1262.30 | 1271.64 | 1265.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1265.40 | 1270.39 | 1265.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 1267.50 | 1270.39 | 1265.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1266.90 | 1269.69 | 1265.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1266.90 | 1269.69 | 1265.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1243.70 | 1264.23 | 1263.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 1243.70 | 1264.23 | 1263.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 1237.85 | 1258.95 | 1261.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 1234.45 | 1254.05 | 1258.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1248.00 | 1244.52 | 1251.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1248.00 | 1244.52 | 1251.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1248.00 | 1244.52 | 1251.49 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1269.65 | 1251.42 | 1251.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1292.00 | 1268.53 | 1261.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 1279.70 | 1286.73 | 1278.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 1279.70 | 1286.73 | 1278.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1279.70 | 1286.73 | 1278.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 1279.70 | 1286.73 | 1278.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1279.90 | 1285.36 | 1278.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:45:00 | 1288.00 | 1280.13 | 1278.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:45:00 | 1289.95 | 1281.25 | 1278.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1288.00 | 1281.25 | 1278.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:45:00 | 1288.50 | 1283.96 | 1280.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1281.60 | 1283.72 | 1280.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 1281.60 | 1283.72 | 1280.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1282.25 | 1283.42 | 1281.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1314.50 | 1283.42 | 1281.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 1294.90 | 1326.09 | 1328.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1294.90 | 1326.09 | 1328.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 1255.75 | 1284.69 | 1299.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1224.10 | 1219.81 | 1234.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 1224.10 | 1219.81 | 1234.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1189.95 | 1185.12 | 1192.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 1190.05 | 1185.12 | 1192.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1191.75 | 1186.26 | 1190.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1191.75 | 1186.26 | 1190.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1187.15 | 1186.44 | 1189.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1197.55 | 1186.44 | 1189.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1194.10 | 1187.97 | 1190.24 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 1211.05 | 1192.59 | 1192.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 1228.90 | 1206.84 | 1200.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 1206.10 | 1207.81 | 1201.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 14:00:00 | 1206.10 | 1207.81 | 1201.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1189.55 | 1206.36 | 1202.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 1189.55 | 1206.36 | 1202.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1193.30 | 1203.75 | 1201.97 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 1192.75 | 1200.62 | 1200.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 1184.80 | 1196.30 | 1198.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 15:15:00 | 1186.95 | 1185.97 | 1190.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 1214.15 | 1185.97 | 1190.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 51 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1229.75 | 1194.73 | 1194.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 1244.50 | 1204.68 | 1198.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1296.90 | 1304.86 | 1276.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1296.90 | 1304.86 | 1276.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1287.55 | 1295.82 | 1283.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 1287.55 | 1295.82 | 1283.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1290.65 | 1294.79 | 1284.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 1287.60 | 1294.79 | 1284.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1276.60 | 1290.50 | 1284.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 1276.60 | 1290.50 | 1284.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1277.65 | 1287.93 | 1283.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 1273.15 | 1287.93 | 1283.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1269.55 | 1280.11 | 1280.69 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 14:15:00 | 1287.00 | 1278.22 | 1277.13 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1247.80 | 1273.38 | 1275.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 1230.00 | 1249.59 | 1259.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1250.00 | 1237.39 | 1248.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1250.00 | 1237.39 | 1248.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1250.00 | 1237.39 | 1248.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1254.60 | 1237.39 | 1248.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1252.30 | 1240.37 | 1248.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1252.30 | 1240.37 | 1248.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1263.90 | 1245.08 | 1250.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 1263.90 | 1245.08 | 1250.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1268.80 | 1249.82 | 1251.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 1268.80 | 1249.82 | 1251.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 1268.00 | 1253.46 | 1253.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 1277.05 | 1260.73 | 1256.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 11:15:00 | 1263.20 | 1263.89 | 1259.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 11:15:00 | 1263.20 | 1263.89 | 1259.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 1263.20 | 1263.89 | 1259.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 1263.20 | 1263.89 | 1259.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 1260.30 | 1263.17 | 1259.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 1260.30 | 1263.17 | 1259.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 1255.35 | 1261.61 | 1259.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 1255.35 | 1261.61 | 1259.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1253.65 | 1260.02 | 1258.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 1257.00 | 1260.02 | 1258.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 1255.80 | 1259.17 | 1258.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 1288.30 | 1259.17 | 1258.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 1248.95 | 1272.20 | 1269.75 | SL hit (close<static) qty=1.00 sl=1251.25 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 1250.05 | 1267.77 | 1267.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 1240.00 | 1255.77 | 1261.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 1253.80 | 1250.45 | 1255.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1253.80 | 1250.45 | 1255.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1253.80 | 1250.45 | 1255.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 1261.10 | 1250.45 | 1255.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1252.30 | 1251.11 | 1254.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 1257.30 | 1251.11 | 1254.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1255.90 | 1252.07 | 1254.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 1255.90 | 1252.07 | 1254.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1251.90 | 1252.03 | 1254.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:00:00 | 1251.90 | 1252.03 | 1254.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1234.45 | 1244.98 | 1250.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 1250.70 | 1244.98 | 1250.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1258.30 | 1245.89 | 1248.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1258.30 | 1245.89 | 1248.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 15:15:00 | 1270.00 | 1250.71 | 1250.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 09:15:00 | 1244.00 | 1249.37 | 1249.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 10:15:00 | 1224.50 | 1244.39 | 1247.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 1248.00 | 1243.35 | 1246.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 1248.00 | 1243.35 | 1246.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 1248.00 | 1243.35 | 1246.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 1248.00 | 1243.35 | 1246.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1246.05 | 1243.89 | 1246.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 1246.05 | 1243.89 | 1246.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1246.00 | 1244.31 | 1246.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 1245.60 | 1244.31 | 1246.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1230.05 | 1241.46 | 1244.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 1228.45 | 1235.59 | 1240.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1254.00 | 1218.26 | 1218.73 | SL hit (close>static) qty=1.00 sl=1252.35 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 1253.60 | 1225.33 | 1221.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1275.50 | 1238.72 | 1229.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 1279.50 | 1283.07 | 1263.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:00:00 | 1279.50 | 1283.07 | 1263.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1288.95 | 1281.88 | 1270.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:15:00 | 1298.00 | 1284.55 | 1273.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 13:15:00 | 1254.65 | 1286.78 | 1277.61 | SL hit (close<static) qty=1.00 sl=1268.95 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1240.10 | 1268.86 | 1272.61 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 1304.15 | 1277.57 | 1275.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 11:15:00 | 1313.50 | 1284.75 | 1278.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 1379.05 | 1386.21 | 1368.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 1379.05 | 1386.21 | 1368.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1367.10 | 1380.59 | 1369.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 1368.55 | 1380.59 | 1369.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1379.15 | 1380.30 | 1370.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1386.75 | 1380.55 | 1371.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 1387.20 | 1381.01 | 1373.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 1385.45 | 1381.01 | 1373.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 1365.75 | 1376.65 | 1373.01 | SL hit (close<static) qty=1.00 sl=1366.50 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1338.80 | 1369.53 | 1370.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 1330.80 | 1361.79 | 1367.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1340.20 | 1321.85 | 1333.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1340.20 | 1321.85 | 1333.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1340.20 | 1321.85 | 1333.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1335.10 | 1321.85 | 1333.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1346.00 | 1326.68 | 1334.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1352.80 | 1326.68 | 1334.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 1353.00 | 1339.11 | 1339.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 1397.35 | 1352.66 | 1345.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 1381.45 | 1382.94 | 1368.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 09:45:00 | 1383.55 | 1382.94 | 1368.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1372.55 | 1380.86 | 1368.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 1372.55 | 1380.86 | 1368.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 1371.00 | 1378.89 | 1368.90 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 10:15:00 | 1353.00 | 1364.96 | 1365.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 1343.50 | 1355.42 | 1359.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 15:15:00 | 1350.00 | 1349.84 | 1354.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 15:15:00 | 1350.00 | 1349.84 | 1354.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 1350.00 | 1349.84 | 1354.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 1350.55 | 1349.84 | 1354.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1369.40 | 1353.75 | 1356.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1369.40 | 1353.75 | 1356.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1366.60 | 1356.32 | 1357.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1370.90 | 1356.32 | 1357.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1376.30 | 1360.32 | 1358.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1384.55 | 1372.84 | 1366.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 1376.65 | 1387.92 | 1381.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 1376.65 | 1387.92 | 1381.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1376.65 | 1387.92 | 1381.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 1376.65 | 1387.92 | 1381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1385.10 | 1387.35 | 1381.79 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1350.35 | 1373.36 | 1376.38 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1389.50 | 1374.99 | 1374.89 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 1365.80 | 1373.69 | 1374.44 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 1418.10 | 1380.40 | 1376.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 14:15:00 | 1440.50 | 1416.46 | 1398.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 1420.00 | 1420.66 | 1403.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 10:00:00 | 1420.00 | 1420.66 | 1403.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 1409.95 | 1418.84 | 1407.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 1409.95 | 1418.84 | 1407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1401.25 | 1414.50 | 1407.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1401.25 | 1414.50 | 1407.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1396.30 | 1410.86 | 1406.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 1420.00 | 1410.86 | 1406.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 12:00:00 | 1408.30 | 1411.43 | 1407.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 1421.80 | 1445.38 | 1448.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 1421.80 | 1445.38 | 1448.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 1412.05 | 1438.71 | 1445.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 1439.60 | 1438.89 | 1444.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 1439.60 | 1438.89 | 1444.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1449.75 | 1441.06 | 1445.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:45:00 | 1451.60 | 1441.06 | 1445.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1439.20 | 1440.69 | 1444.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 1431.65 | 1443.50 | 1444.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 13:15:00 | 1449.70 | 1444.81 | 1444.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 1449.70 | 1444.81 | 1444.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1462.60 | 1449.12 | 1446.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 11:15:00 | 1447.60 | 1449.85 | 1447.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 11:15:00 | 1447.60 | 1449.85 | 1447.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1447.60 | 1449.85 | 1447.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 1448.95 | 1449.85 | 1447.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1440.85 | 1448.05 | 1447.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 1440.85 | 1448.05 | 1447.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1443.00 | 1447.04 | 1446.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:30:00 | 1445.25 | 1447.04 | 1446.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1437.00 | 1444.68 | 1445.62 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1453.10 | 1446.37 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1475.40 | 1457.70 | 1452.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1487.00 | 1504.10 | 1491.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 1487.00 | 1504.10 | 1491.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1487.00 | 1504.10 | 1491.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 1487.00 | 1504.10 | 1491.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1486.55 | 1500.59 | 1491.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 1482.95 | 1500.59 | 1491.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1485.55 | 1497.58 | 1490.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 1485.55 | 1497.58 | 1490.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1483.55 | 1494.78 | 1490.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:45:00 | 1484.00 | 1494.78 | 1490.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1532.00 | 1539.15 | 1528.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 1535.80 | 1539.15 | 1528.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 1519.10 | 1531.77 | 1527.31 | SL hit (close<static) qty=1.00 sl=1522.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1518.00 | 1525.37 | 1525.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 1512.95 | 1522.95 | 1524.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1531.95 | 1522.24 | 1523.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1531.95 | 1522.24 | 1523.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1531.95 | 1522.24 | 1523.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1531.95 | 1522.24 | 1523.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1528.00 | 1523.39 | 1524.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1517.75 | 1523.39 | 1524.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1526.10 | 1522.95 | 1523.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 1526.10 | 1522.95 | 1523.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1532.00 | 1524.76 | 1524.45 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 1518.80 | 1523.43 | 1523.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 1496.55 | 1518.38 | 1521.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 11:15:00 | 1456.15 | 1448.75 | 1460.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:00:00 | 1456.15 | 1448.75 | 1460.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 1455.45 | 1450.09 | 1459.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 1455.45 | 1450.09 | 1459.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1437.00 | 1403.10 | 1420.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 1437.00 | 1403.10 | 1420.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 1435.20 | 1409.52 | 1421.96 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 1455.50 | 1429.42 | 1428.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 1463.55 | 1436.24 | 1432.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 1439.00 | 1442.20 | 1435.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 10:00:00 | 1439.00 | 1442.20 | 1435.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 1394.50 | 1432.66 | 1432.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 1394.50 | 1432.66 | 1432.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 11:15:00 | 1418.40 | 1429.81 | 1430.79 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1465.55 | 1437.77 | 1434.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1493.40 | 1454.43 | 1442.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 10:15:00 | 1569.40 | 1570.69 | 1541.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 11:00:00 | 1569.40 | 1570.69 | 1541.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1595.70 | 1633.86 | 1605.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:00:00 | 1595.70 | 1633.86 | 1605.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 1598.80 | 1626.84 | 1605.25 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 15:15:00 | 1561.00 | 1590.44 | 1593.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 1550.00 | 1568.57 | 1578.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 15:15:00 | 1557.00 | 1553.71 | 1565.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 09:15:00 | 1559.70 | 1553.71 | 1565.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1541.10 | 1551.19 | 1562.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 1496.40 | 1541.11 | 1552.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 1521.60 | 1537.10 | 1545.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:15:00 | 1520.50 | 1537.10 | 1545.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 1521.00 | 1499.47 | 1499.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 1521.00 | 1499.47 | 1499.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 1531.30 | 1505.84 | 1502.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 1531.80 | 1537.40 | 1526.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:45:00 | 1529.50 | 1537.40 | 1526.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 1528.40 | 1535.60 | 1527.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 1570.40 | 1535.60 | 1527.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1566.30 | 1541.74 | 1530.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 1582.60 | 1549.91 | 1535.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 1573.50 | 1561.34 | 1544.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 1575.00 | 1564.07 | 1547.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1524.70 | 1548.07 | 1550.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1524.70 | 1548.07 | 1550.41 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1576.00 | 1548.03 | 1544.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1577.00 | 1557.82 | 1549.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1573.90 | 1575.86 | 1565.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 1573.90 | 1575.86 | 1565.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1601.00 | 1581.66 | 1570.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1611.50 | 1600.30 | 1589.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 1607.80 | 1601.80 | 1591.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 1608.10 | 1616.99 | 1615.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1609.90 | 1614.23 | 1614.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1609.90 | 1614.23 | 1614.74 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1649.80 | 1621.92 | 1618.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 1651.60 | 1634.52 | 1625.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 11:15:00 | 1634.50 | 1640.80 | 1632.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 1634.50 | 1640.80 | 1632.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1632.80 | 1639.20 | 1632.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1632.80 | 1639.20 | 1632.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1620.80 | 1635.52 | 1631.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 1619.50 | 1635.52 | 1631.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1631.20 | 1634.65 | 1631.72 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1615.40 | 1629.80 | 1629.96 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1638.20 | 1628.18 | 1628.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 1643.40 | 1634.76 | 1631.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 1654.50 | 1654.57 | 1646.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:45:00 | 1650.80 | 1654.57 | 1646.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1645.90 | 1654.20 | 1649.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 1645.70 | 1654.20 | 1649.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1646.60 | 1652.68 | 1649.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 1634.20 | 1652.68 | 1649.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 1614.80 | 1645.10 | 1646.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 1607.50 | 1629.85 | 1638.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1579.00 | 1575.17 | 1593.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 1579.00 | 1575.17 | 1593.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1591.50 | 1580.26 | 1592.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 1575.60 | 1580.85 | 1591.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 13:15:00 | 1496.82 | 1532.25 | 1556.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1517.20 | 1516.16 | 1530.48 | SL hit (close>ema200) qty=0.50 sl=1516.16 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1614.00 | 1547.19 | 1542.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1644.50 | 1595.36 | 1571.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 1630.10 | 1631.75 | 1614.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 1613.80 | 1631.75 | 1614.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1630.20 | 1631.44 | 1615.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 1609.60 | 1631.44 | 1615.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1616.20 | 1627.47 | 1616.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:45:00 | 1615.70 | 1627.47 | 1616.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1611.30 | 1624.24 | 1616.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 1613.70 | 1624.24 | 1616.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1602.10 | 1619.81 | 1614.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1602.10 | 1619.81 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1609.00 | 1615.61 | 1613.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1624.70 | 1615.61 | 1613.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1600.50 | 1614.81 | 1613.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1600.50 | 1614.81 | 1613.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 1595.20 | 1610.89 | 1612.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1581.60 | 1602.75 | 1608.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1562.90 | 1560.09 | 1574.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1562.90 | 1560.09 | 1574.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1566.70 | 1564.96 | 1571.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:15:00 | 1569.30 | 1564.96 | 1571.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1560.30 | 1564.02 | 1570.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 1558.00 | 1564.02 | 1570.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 1586.60 | 1562.58 | 1565.76 | SL hit (close>static) qty=1.00 sl=1576.40 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 1582.90 | 1569.40 | 1568.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 13:15:00 | 1587.80 | 1574.90 | 1571.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1577.00 | 1578.58 | 1574.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1577.00 | 1578.58 | 1574.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1563.70 | 1575.61 | 1573.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 1566.90 | 1575.61 | 1573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1547.70 | 1570.02 | 1570.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1543.40 | 1564.70 | 1568.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 1551.50 | 1550.62 | 1557.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:30:00 | 1552.50 | 1550.62 | 1557.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1557.90 | 1552.07 | 1557.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1557.90 | 1552.07 | 1557.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1550.10 | 1551.68 | 1556.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1549.70 | 1551.68 | 1556.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1569.00 | 1556.70 | 1558.08 | SL hit (close>static) qty=1.00 sl=1565.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1572.40 | 1559.84 | 1559.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1587.00 | 1567.65 | 1563.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1600.40 | 1602.44 | 1594.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 1600.40 | 1602.44 | 1594.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1619.90 | 1605.93 | 1596.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:00:00 | 1622.40 | 1610.07 | 1600.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 1621.40 | 1613.86 | 1602.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:45:00 | 1639.20 | 1620.51 | 1608.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 1620.40 | 1631.69 | 1625.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1628.00 | 1630.39 | 1626.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1599.80 | 1630.39 | 1626.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1591.00 | 1622.51 | 1623.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1591.00 | 1622.51 | 1623.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 1585.10 | 1609.91 | 1616.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 1519.40 | 1518.39 | 1529.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:30:00 | 1510.30 | 1517.23 | 1527.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1524.00 | 1519.49 | 1524.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1524.90 | 1519.49 | 1524.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1540.00 | 1523.59 | 1526.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1540.00 | 1523.59 | 1526.18 | SL hit (close>ema400) qty=1.00 sl=1526.18 alert=retest1 |

### Cycle 95 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1545.60 | 1528.00 | 1527.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 1565.10 | 1542.25 | 1535.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1546.00 | 1549.99 | 1541.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 1546.00 | 1549.99 | 1541.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1548.40 | 1548.68 | 1543.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 1543.50 | 1548.68 | 1543.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1530.40 | 1544.60 | 1542.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 1534.80 | 1544.60 | 1542.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1531.90 | 1542.06 | 1541.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1531.90 | 1542.06 | 1541.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1532.30 | 1540.11 | 1540.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 1528.50 | 1534.91 | 1537.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1541.50 | 1536.21 | 1537.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 1533.30 | 1535.47 | 1537.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1549.60 | 1534.34 | 1535.03 | SL hit (close>static) qty=1.00 sl=1547.70 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1546.00 | 1536.67 | 1536.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1556.60 | 1544.09 | 1539.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1542.50 | 1547.71 | 1543.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 1542.50 | 1547.71 | 1543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1542.60 | 1546.69 | 1543.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 1552.70 | 1549.01 | 1544.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1541.50 | 1553.12 | 1553.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1541.50 | 1553.12 | 1553.34 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1565.70 | 1555.64 | 1554.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1572.70 | 1559.05 | 1556.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1564.40 | 1566.15 | 1561.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1559.40 | 1566.15 | 1561.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1559.70 | 1564.86 | 1561.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 1559.70 | 1564.86 | 1561.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1561.50 | 1564.19 | 1561.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 1565.50 | 1564.19 | 1561.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 1566.50 | 1565.61 | 1562.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1552.90 | 1562.86 | 1561.96 | SL hit (close<static) qty=1.00 sl=1556.10 alert=retest2 |

### Cycle 100 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1553.50 | 1560.98 | 1561.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 1549.70 | 1553.26 | 1556.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1490.50 | 1489.73 | 1502.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:30:00 | 1491.90 | 1489.73 | 1502.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1487.00 | 1487.72 | 1497.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1483.50 | 1487.72 | 1497.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 1480.10 | 1486.68 | 1496.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 1409.33 | 1446.12 | 1461.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 1406.09 | 1446.12 | 1461.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1439.80 | 1434.33 | 1448.89 | SL hit (close>ema200) qty=0.50 sl=1434.33 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1472.80 | 1453.65 | 1453.57 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1455.00 | 1458.84 | 1459.33 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 1464.80 | 1458.72 | 1458.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 11:15:00 | 1472.00 | 1461.38 | 1459.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1470.80 | 1473.34 | 1467.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 1470.80 | 1473.34 | 1467.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1471.20 | 1472.91 | 1467.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1471.20 | 1472.91 | 1467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1469.30 | 1472.19 | 1467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1469.30 | 1472.19 | 1467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1467.20 | 1471.19 | 1467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 1467.20 | 1471.19 | 1467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1464.00 | 1469.75 | 1467.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 1464.00 | 1469.75 | 1467.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1450.30 | 1465.86 | 1465.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1450.30 | 1465.86 | 1465.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1450.00 | 1462.69 | 1464.26 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1479.50 | 1465.34 | 1464.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 1483.30 | 1468.93 | 1466.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1469.90 | 1475.29 | 1471.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 1472.40 | 1475.29 | 1471.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1460.50 | 1472.33 | 1470.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 1460.50 | 1472.33 | 1470.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1463.00 | 1470.46 | 1469.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 1459.40 | 1470.46 | 1469.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 1454.80 | 1466.30 | 1467.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 1450.60 | 1463.16 | 1466.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1470.00 | 1462.26 | 1465.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 1472.50 | 1462.26 | 1465.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1473.00 | 1464.41 | 1465.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 1473.00 | 1464.41 | 1465.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1460.10 | 1464.94 | 1465.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 1459.00 | 1462.76 | 1464.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 1458.70 | 1463.49 | 1464.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1527.40 | 1475.61 | 1469.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1527.40 | 1475.61 | 1469.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1530.10 | 1522.18 | 1509.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 1524.00 | 1526.26 | 1516.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 1524.00 | 1526.26 | 1516.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1519.10 | 1524.55 | 1517.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1520.00 | 1524.55 | 1517.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1520.10 | 1523.33 | 1518.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 1520.40 | 1523.33 | 1518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1514.60 | 1521.58 | 1518.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 1517.60 | 1521.58 | 1518.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1519.00 | 1521.07 | 1518.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1520.90 | 1520.45 | 1518.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:45:00 | 1520.20 | 1520.46 | 1519.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:15:00 | 1520.30 | 1520.46 | 1519.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 1509.00 | 1519.94 | 1519.91 | SL hit (close<static) qty=1.00 sl=1514.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1511.50 | 1518.26 | 1519.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1504.90 | 1515.58 | 1517.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1448.40 | 1448.19 | 1465.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:30:00 | 1449.00 | 1448.19 | 1465.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1451.30 | 1437.70 | 1448.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1451.30 | 1437.70 | 1448.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1452.60 | 1440.68 | 1448.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1451.70 | 1440.68 | 1448.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1450.30 | 1444.64 | 1449.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1438.00 | 1444.64 | 1449.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1447.80 | 1446.08 | 1449.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:15:00 | 1444.60 | 1446.08 | 1449.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1443.90 | 1445.64 | 1448.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 1444.50 | 1445.52 | 1448.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1440.50 | 1445.52 | 1448.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1435.70 | 1443.55 | 1447.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1426.60 | 1439.67 | 1444.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 1425.50 | 1435.30 | 1440.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 1427.60 | 1434.62 | 1440.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1468.80 | 1443.39 | 1442.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1468.80 | 1443.39 | 1442.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 1487.70 | 1469.87 | 1459.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 1491.10 | 1492.38 | 1479.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1497.20 | 1492.38 | 1479.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1510.00 | 1516.93 | 1507.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 1510.00 | 1516.93 | 1507.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1520.20 | 1517.58 | 1508.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1494.30 | 1511.59 | 1508.63 | SL hit (close<ema400) qty=1.00 sl=1508.63 alert=retest1 |

### Cycle 110 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 1597.10 | 1615.89 | 1617.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1589.90 | 1608.57 | 1613.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1586.50 | 1582.17 | 1593.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1586.50 | 1582.17 | 1593.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1586.70 | 1579.95 | 1588.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1586.70 | 1579.95 | 1588.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1590.70 | 1582.10 | 1588.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1587.50 | 1582.10 | 1588.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1594.70 | 1584.62 | 1589.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1589.70 | 1584.62 | 1589.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1585.30 | 1584.75 | 1588.99 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 1604.60 | 1590.60 | 1590.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 1611.90 | 1594.86 | 1592.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1585.20 | 1595.48 | 1593.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1585.20 | 1595.48 | 1593.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1597.60 | 1595.90 | 1593.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:00:00 | 1603.50 | 1597.98 | 1595.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 1602.90 | 1598.83 | 1595.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 1603.90 | 1599.84 | 1596.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1576.00 | 1595.58 | 1595.14 | SL hit (close<static) qty=1.00 sl=1582.10 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 1563.70 | 1589.20 | 1592.29 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 1602.30 | 1589.00 | 1587.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 1611.60 | 1593.52 | 1589.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1613.60 | 1619.04 | 1609.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1613.60 | 1619.04 | 1609.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1608.00 | 1616.83 | 1609.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1608.00 | 1616.83 | 1609.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1608.20 | 1615.10 | 1609.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 1607.00 | 1615.10 | 1609.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1607.50 | 1613.58 | 1609.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:45:00 | 1615.00 | 1610.69 | 1608.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 1654.20 | 1662.01 | 1662.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1654.20 | 1662.01 | 1662.09 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 1668.90 | 1662.74 | 1662.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1687.70 | 1673.26 | 1668.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 1732.80 | 1740.58 | 1721.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1732.80 | 1740.58 | 1721.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1720.90 | 1733.06 | 1723.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1720.90 | 1733.06 | 1723.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1723.40 | 1731.13 | 1723.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:15:00 | 1718.60 | 1731.13 | 1723.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1716.20 | 1728.14 | 1722.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1713.60 | 1728.14 | 1722.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1723.90 | 1727.29 | 1722.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1729.00 | 1727.29 | 1722.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1689.00 | 1719.91 | 1720.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1689.00 | 1719.91 | 1720.26 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1716.90 | 1711.84 | 1711.65 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1694.60 | 1708.96 | 1710.41 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1739.30 | 1713.76 | 1712.26 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1675.70 | 1717.56 | 1721.17 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1739.40 | 1709.08 | 1708.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1752.50 | 1725.56 | 1716.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1718.20 | 1732.59 | 1722.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 1722.20 | 1732.59 | 1722.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1712.40 | 1728.55 | 1721.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1712.40 | 1728.55 | 1721.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1724.80 | 1727.80 | 1722.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 1731.90 | 1729.40 | 1723.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 1729.90 | 1730.57 | 1728.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:30:00 | 1730.00 | 1728.12 | 1727.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1716.00 | 1725.69 | 1726.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1716.00 | 1725.69 | 1726.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 10:15:00 | 1710.60 | 1720.95 | 1723.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1722.00 | 1715.67 | 1718.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1722.00 | 1715.67 | 1718.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1713.20 | 1715.17 | 1718.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 1710.10 | 1714.06 | 1717.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1709.50 | 1713.84 | 1716.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 1705.10 | 1712.09 | 1715.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 1705.50 | 1692.68 | 1692.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1705.50 | 1692.68 | 1692.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1708.00 | 1695.75 | 1693.74 | Break + close above crossover candle high |

### Cycle 124 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 1673.80 | 1691.36 | 1691.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 1665.30 | 1686.15 | 1689.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1689.50 | 1672.59 | 1679.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 1689.50 | 1672.59 | 1679.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1679.50 | 1673.97 | 1679.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:30:00 | 1676.90 | 1673.28 | 1678.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 1676.80 | 1666.89 | 1672.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 1684.90 | 1676.18 | 1675.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1684.90 | 1676.18 | 1675.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1705.10 | 1683.82 | 1679.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1730.00 | 1733.65 | 1724.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 1725.50 | 1733.65 | 1724.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1721.20 | 1730.99 | 1724.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1721.20 | 1730.99 | 1724.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1722.50 | 1729.29 | 1724.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 1720.50 | 1729.29 | 1724.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1723.50 | 1728.74 | 1725.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1727.00 | 1728.74 | 1725.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1718.40 | 1726.67 | 1724.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:45:00 | 1739.00 | 1728.70 | 1725.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1709.50 | 1723.58 | 1724.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1709.50 | 1723.58 | 1724.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1684.90 | 1711.64 | 1717.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1677.30 | 1673.35 | 1686.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 1677.30 | 1673.35 | 1686.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1700.00 | 1679.58 | 1686.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1700.00 | 1679.58 | 1686.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1707.40 | 1685.15 | 1688.24 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1725.50 | 1697.02 | 1693.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 1731.30 | 1708.35 | 1699.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1722.20 | 1722.69 | 1711.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1722.20 | 1722.69 | 1711.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1716.10 | 1721.37 | 1711.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 15:00:00 | 1725.00 | 1722.10 | 1712.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1706.70 | 1719.71 | 1713.41 | SL hit (close<static) qty=1.00 sl=1708.90 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 1720.00 | 1728.60 | 1729.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 1708.00 | 1723.21 | 1726.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1726.70 | 1723.67 | 1726.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1727.50 | 1723.67 | 1726.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1728.70 | 1724.68 | 1726.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1728.70 | 1724.68 | 1726.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1736.40 | 1727.02 | 1727.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1736.40 | 1727.02 | 1727.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1734.00 | 1728.42 | 1728.05 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1712.80 | 1725.29 | 1726.67 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1731.90 | 1727.09 | 1726.82 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1718.30 | 1726.60 | 1726.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1707.00 | 1718.69 | 1722.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1685.30 | 1683.66 | 1697.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1685.30 | 1683.66 | 1697.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1677.80 | 1680.89 | 1690.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 1675.00 | 1678.99 | 1688.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 14:15:00 | 1591.25 | 1622.77 | 1646.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 1683.40 | 1628.85 | 1644.81 | SL hit (close>ema200) qty=0.50 sl=1628.85 alert=retest2 |

### Cycle 133 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1694.30 | 1661.65 | 1657.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 1701.70 | 1681.03 | 1669.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1697.20 | 1698.86 | 1689.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:30:00 | 1697.10 | 1698.86 | 1689.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1690.30 | 1697.31 | 1690.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1691.10 | 1697.31 | 1690.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1695.30 | 1696.91 | 1690.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1690.50 | 1696.91 | 1690.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1693.60 | 1696.25 | 1691.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 1693.90 | 1696.25 | 1691.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1693.50 | 1695.70 | 1691.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 1694.60 | 1695.70 | 1691.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1693.80 | 1695.32 | 1691.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1693.80 | 1695.32 | 1691.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1698.00 | 1695.86 | 1692.20 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1681.40 | 1689.81 | 1690.53 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 1700.50 | 1692.04 | 1691.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 15:15:00 | 1704.30 | 1694.49 | 1692.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1696.40 | 1696.51 | 1694.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 1694.10 | 1696.51 | 1694.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1697.50 | 1696.71 | 1694.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 1705.00 | 1698.01 | 1695.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:30:00 | 1705.60 | 1698.81 | 1695.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1703.00 | 1699.44 | 1696.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1703.60 | 1700.28 | 1697.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1720.60 | 1704.34 | 1699.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 1726.00 | 1713.37 | 1705.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1744.60 | 1776.43 | 1779.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1744.60 | 1776.43 | 1779.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1742.20 | 1769.58 | 1775.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 1707.50 | 1706.85 | 1720.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 1707.50 | 1706.85 | 1720.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1697.10 | 1702.75 | 1714.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 1689.10 | 1699.51 | 1710.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1684.50 | 1697.49 | 1707.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1688.20 | 1695.63 | 1706.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1679.50 | 1695.43 | 1705.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1681.70 | 1692.68 | 1702.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 1719.40 | 1699.89 | 1700.35 | SL hit (close>static) qty=1.00 sl=1714.70 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 1705.10 | 1700.93 | 1700.78 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1678.40 | 1697.63 | 1699.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1660.80 | 1682.30 | 1691.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1657.50 | 1641.18 | 1656.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1655.50 | 1641.18 | 1656.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1643.60 | 1641.66 | 1655.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1640.50 | 1641.66 | 1655.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 1664.40 | 1650.06 | 1655.03 | SL hit (close>static) qty=1.00 sl=1659.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1651.60 | 1640.95 | 1640.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1661.10 | 1644.98 | 1642.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1623.70 | 1643.08 | 1642.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 1623.70 | 1643.08 | 1642.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 1627.80 | 1640.03 | 1641.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 1611.30 | 1630.89 | 1636.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1596.90 | 1584.12 | 1601.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1596.90 | 1584.12 | 1601.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1687.10 | 1606.74 | 1608.65 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1679.40 | 1621.27 | 1615.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 1698.10 | 1648.21 | 1629.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 1722.00 | 1723.70 | 1702.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:00:00 | 1722.00 | 1723.70 | 1702.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1711.00 | 1718.40 | 1706.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1708.80 | 1718.40 | 1706.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1733.10 | 1755.34 | 1740.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1733.10 | 1755.34 | 1740.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1747.00 | 1753.67 | 1741.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:45:00 | 1751.90 | 1753.70 | 1742.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 1751.00 | 1752.24 | 1742.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 1725.00 | 1744.99 | 1741.03 | SL hit (close<static) qty=1.00 sl=1728.30 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 1719.40 | 1736.58 | 1737.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1704.70 | 1722.70 | 1729.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 10:15:00 | 1726.50 | 1723.46 | 1729.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 1726.50 | 1723.46 | 1729.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1735.50 | 1725.87 | 1729.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1735.50 | 1725.87 | 1729.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1731.70 | 1727.03 | 1730.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 1738.20 | 1727.03 | 1730.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1729.40 | 1728.09 | 1730.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1729.40 | 1728.09 | 1730.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1738.00 | 1730.08 | 1730.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1718.90 | 1730.08 | 1730.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1723.80 | 1713.90 | 1713.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1723.80 | 1713.90 | 1713.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 1733.80 | 1717.88 | 1715.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1719.90 | 1720.30 | 1716.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1719.30 | 1720.30 | 1716.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1717.90 | 1719.82 | 1716.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 1717.90 | 1719.82 | 1716.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1718.60 | 1719.58 | 1717.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 1717.60 | 1719.58 | 1717.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1712.50 | 1718.16 | 1716.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:45:00 | 1713.20 | 1718.16 | 1716.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1719.00 | 1718.33 | 1716.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:30:00 | 1713.10 | 1718.33 | 1716.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1717.60 | 1718.18 | 1716.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 1717.20 | 1718.18 | 1716.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1716.00 | 1717.75 | 1716.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 1709.20 | 1717.75 | 1716.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1706.00 | 1715.40 | 1715.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 1699.00 | 1712.12 | 1714.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1701.80 | 1683.23 | 1691.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1701.80 | 1683.23 | 1691.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1702.60 | 1687.10 | 1692.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1702.60 | 1687.10 | 1692.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1687.00 | 1687.08 | 1691.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1676.70 | 1685.63 | 1689.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1711.00 | 1690.70 | 1691.31 | SL hit (close>static) qty=1.00 sl=1704.30 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 11:15:00 | 1714.10 | 1695.38 | 1693.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 1720.80 | 1700.47 | 1695.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1697.80 | 1733.94 | 1723.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 1697.80 | 1733.94 | 1723.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1704.90 | 1728.13 | 1722.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1730.80 | 1724.16 | 1721.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1709.40 | 1726.99 | 1728.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1709.40 | 1726.99 | 1728.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1691.70 | 1716.62 | 1722.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1666.40 | 1662.17 | 1681.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 1647.60 | 1660.60 | 1676.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1651.80 | 1657.60 | 1669.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 1482.84 | 1620.90 | 1644.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1474.20 | 1431.54 | 1428.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1476.30 | 1452.66 | 1439.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1428.00 | 1454.01 | 1444.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 1433.60 | 1454.01 | 1444.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1429.30 | 1449.07 | 1442.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 1421.40 | 1449.07 | 1442.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1418.20 | 1437.00 | 1438.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1376.00 | 1421.09 | 1430.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1393.90 | 1377.69 | 1398.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1382.40 | 1377.69 | 1398.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1313.28 | 1362.48 | 1381.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1349.10 | 1344.25 | 1365.38 | SL hit (close>ema200) qty=0.50 sl=1344.25 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1391.50 | 1369.76 | 1368.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1408.90 | 1381.14 | 1374.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1529.00 | 1529.38 | 1498.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1569.80 | 1529.38 | 1498.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1529.80 | 1560.76 | 1536.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1529.80 | 1560.76 | 1536.28 | SL hit (close<ema400) qty=1.00 sl=1536.28 alert=retest1 |

### Cycle 150 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1505.50 | 1524.59 | 1526.09 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1537.40 | 1527.15 | 1527.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1597.30 | 1555.23 | 1543.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1570.50 | 1570.81 | 1556.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 1570.50 | 1570.81 | 1556.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1567.30 | 1569.82 | 1558.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 1570.20 | 1569.82 | 1558.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:45:00 | 1572.90 | 1569.99 | 1559.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1569.00 | 1572.42 | 1565.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1584.90 | 1571.63 | 1568.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1570.50 | 1581.32 | 1576.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:15:00 | 1568.90 | 1581.32 | 1576.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1573.90 | 1579.84 | 1576.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 1564.80 | 1574.02 | 1574.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 1564.80 | 1574.02 | 1574.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1535.50 | 1564.87 | 1570.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1553.50 | 1544.03 | 1551.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1553.50 | 1544.03 | 1551.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1568.00 | 1548.82 | 1553.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 1569.30 | 1548.82 | 1553.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1570.00 | 1553.06 | 1554.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1545.00 | 1553.06 | 1554.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1560.00 | 1551.70 | 1553.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1560.00 | 1551.70 | 1553.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1554.00 | 1552.16 | 1553.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 1549.10 | 1552.31 | 1553.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 1558.80 | 1554.97 | 1554.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1558.80 | 1554.97 | 1554.65 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 1537.40 | 1553.00 | 1554.20 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1565.00 | 1552.19 | 1551.61 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1499.30 | 1541.54 | 1547.12 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 1570.50 | 1544.76 | 1544.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1634.70 | 1562.75 | 1552.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 1687.50 | 1691.82 | 1668.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:45:00 | 1685.00 | 1691.82 | 1668.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1676.30 | 1687.58 | 1675.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:00:00 | 1689.60 | 1685.01 | 1677.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 1671.00 | 1680.34 | 1676.48 | SL hit (close<static) qty=1.00 sl=1672.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 1261.50 | 2024-05-13 10:15:00 | 1285.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-05-28 09:15:00 | 1266.85 | 2024-06-03 09:15:00 | 1271.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-05-29 10:15:00 | 1266.70 | 2024-06-03 09:15:00 | 1271.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-06-25 10:30:00 | 1414.45 | 2024-06-25 12:15:00 | 1434.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-06-25 11:00:00 | 1414.95 | 2024-06-25 12:15:00 | 1434.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-06-27 10:15:00 | 1439.55 | 2024-06-27 13:15:00 | 1422.55 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-06-27 12:00:00 | 1438.65 | 2024-06-27 13:15:00 | 1422.55 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-06-27 15:00:00 | 1438.40 | 2024-06-28 10:15:00 | 1423.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-07-24 11:30:00 | 1396.00 | 2024-07-26 12:15:00 | 1408.75 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-24 12:30:00 | 1398.95 | 2024-07-26 12:15:00 | 1408.75 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-29 11:15:00 | 1444.00 | 2024-08-01 11:15:00 | 1407.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-07-30 10:45:00 | 1430.65 | 2024-08-01 11:15:00 | 1407.80 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-30 13:30:00 | 1429.30 | 2024-08-01 11:15:00 | 1407.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-31 09:45:00 | 1432.90 | 2024-08-01 11:15:00 | 1407.80 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-08-02 12:30:00 | 1390.15 | 2024-08-07 14:15:00 | 1379.35 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2024-08-02 13:30:00 | 1393.60 | 2024-08-07 14:15:00 | 1379.35 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2024-08-13 09:30:00 | 1341.70 | 2024-08-16 10:15:00 | 1343.50 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-08-13 11:15:00 | 1343.10 | 2024-08-16 10:15:00 | 1343.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-08-13 12:00:00 | 1340.90 | 2024-08-16 10:15:00 | 1343.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-08-20 09:45:00 | 1359.85 | 2024-08-23 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-08-30 11:45:00 | 1446.40 | 2024-09-10 12:15:00 | 1510.05 | STOP_HIT | 1.00 | 4.40% |
| BUY | retest2 | 2024-08-30 13:00:00 | 1445.95 | 2024-09-10 12:15:00 | 1510.05 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2024-09-11 15:15:00 | 1525.50 | 2024-09-25 11:15:00 | 1595.00 | STOP_HIT | 1.00 | 4.56% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1610.75 | 2024-10-03 10:15:00 | 1530.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 15:00:00 | 1609.70 | 2024-10-03 10:15:00 | 1529.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:45:00 | 1606.25 | 2024-10-03 10:15:00 | 1525.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1610.75 | 2024-10-08 09:15:00 | 1483.15 | STOP_HIT | 0.50 | 7.92% |
| SELL | retest2 | 2024-09-30 15:00:00 | 1609.70 | 2024-10-08 09:15:00 | 1483.15 | STOP_HIT | 0.50 | 7.86% |
| SELL | retest2 | 2024-10-01 09:45:00 | 1606.25 | 2024-10-08 09:15:00 | 1483.15 | STOP_HIT | 0.50 | 7.66% |
| SELL | retest2 | 2024-10-14 13:00:00 | 1503.50 | 2024-10-18 09:15:00 | 1428.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 14:15:00 | 1504.70 | 2024-10-18 09:15:00 | 1429.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:15:00 | 1503.65 | 2024-10-18 09:15:00 | 1428.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1502.80 | 2024-10-18 09:15:00 | 1427.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 13:00:00 | 1503.50 | 2024-10-18 13:15:00 | 1464.70 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2024-10-14 14:15:00 | 1504.70 | 2024-10-18 13:15:00 | 1464.70 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-10-15 09:15:00 | 1503.65 | 2024-10-18 13:15:00 | 1464.70 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1502.80 | 2024-10-18 13:15:00 | 1464.70 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2024-10-15 13:15:00 | 1497.85 | 2024-10-22 12:15:00 | 1422.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 14:30:00 | 1497.75 | 2024-10-22 12:15:00 | 1422.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 1497.30 | 2024-10-22 12:15:00 | 1422.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:45:00 | 1495.65 | 2024-10-22 12:15:00 | 1420.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 1470.00 | 2024-10-23 09:15:00 | 1396.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:15:00 | 1497.85 | 2024-10-23 13:15:00 | 1420.20 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2024-10-15 14:30:00 | 1497.75 | 2024-10-23 13:15:00 | 1420.20 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2024-10-16 09:15:00 | 1497.30 | 2024-10-23 13:15:00 | 1420.20 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2024-10-16 09:45:00 | 1495.65 | 2024-10-23 13:15:00 | 1420.20 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2024-10-17 11:15:00 | 1470.00 | 2024-10-23 13:15:00 | 1420.20 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2024-11-12 12:00:00 | 1250.40 | 2024-11-14 09:15:00 | 1187.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:00:00 | 1250.40 | 2024-11-14 15:15:00 | 1214.90 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2024-12-09 09:45:00 | 1288.00 | 2024-12-13 10:15:00 | 1294.90 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-12-09 10:45:00 | 1289.95 | 2024-12-13 10:15:00 | 1294.90 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-12-09 11:15:00 | 1288.00 | 2024-12-13 10:15:00 | 1294.90 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-12-09 12:45:00 | 1288.50 | 2024-12-13 10:15:00 | 1294.90 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1314.50 | 2024-12-13 10:15:00 | 1294.90 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1288.30 | 2025-01-17 10:15:00 | 1248.95 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-01-24 13:30:00 | 1228.45 | 2025-01-28 12:15:00 | 1254.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-01-31 11:15:00 | 1298.00 | 2025-01-31 13:15:00 | 1254.65 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-01-31 14:30:00 | 1301.05 | 2025-02-01 09:15:00 | 1268.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-02-07 09:15:00 | 1386.75 | 2025-02-07 13:15:00 | 1365.75 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-02-07 11:30:00 | 1387.20 | 2025-02-07 13:15:00 | 1365.75 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-02-07 12:15:00 | 1385.45 | 2025-02-07 13:15:00 | 1365.75 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-03-03 09:15:00 | 1420.00 | 2025-03-10 14:15:00 | 1421.80 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-03-03 12:00:00 | 1408.30 | 2025-03-10 14:15:00 | 1421.80 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-03-12 11:30:00 | 1431.65 | 2025-03-12 13:15:00 | 1449.70 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-03-25 09:15:00 | 1535.80 | 2025-03-25 11:15:00 | 1519.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1536.40 | 2025-03-26 14:15:00 | 1518.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-03-26 10:30:00 | 1533.40 | 2025-03-26 14:15:00 | 1518.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-03-26 12:00:00 | 1533.50 | 2025-03-26 14:15:00 | 1518.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1496.40 | 2025-05-05 10:15:00 | 1521.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-04-28 14:45:00 | 1521.60 | 2025-05-05 10:15:00 | 1521.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-04-28 15:15:00 | 1520.50 | 2025-05-05 10:15:00 | 1521.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-05-07 11:00:00 | 1582.60 | 2025-05-09 09:15:00 | 1524.70 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-05-07 14:00:00 | 1573.50 | 2025-05-09 09:15:00 | 1524.70 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-05-07 15:00:00 | 1575.00 | 2025-05-09 09:15:00 | 1524.70 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1611.50 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-05-15 14:00:00 | 1607.80 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-05-20 13:00:00 | 1608.10 | 2025-05-20 14:15:00 | 1609.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-06-03 11:15:00 | 1575.60 | 2025-06-04 13:15:00 | 1496.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 11:15:00 | 1575.60 | 2025-06-06 09:15:00 | 1517.20 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-06-17 11:15:00 | 1558.00 | 2025-06-18 09:15:00 | 1586.60 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1549.70 | 2025-06-23 10:15:00 | 1569.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-26 14:00:00 | 1622.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-06-26 14:30:00 | 1621.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-27 10:45:00 | 1639.20 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-06-30 13:45:00 | 1620.40 | 2025-07-01 09:15:00 | 1591.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest1 | 2025-07-08 10:30:00 | 1510.30 | 2025-07-09 09:15:00 | 1540.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-14 11:45:00 | 1533.30 | 2025-07-15 10:15:00 | 1549.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-16 13:30:00 | 1552.70 | 2025-07-21 09:15:00 | 1541.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-22 12:15:00 | 1565.50 | 2025-07-23 09:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-22 15:00:00 | 1566.50 | 2025-07-23 09:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1483.50 | 2025-08-01 11:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:45:00 | 1480.10 | 2025-08-01 11:15:00 | 1406.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1483.50 | 2025-08-04 09:15:00 | 1439.80 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2025-07-30 10:45:00 | 1480.10 | 2025-08-04 09:15:00 | 1439.80 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-08-14 12:30:00 | 1459.00 | 2025-08-18 09:15:00 | 1527.40 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2025-08-14 13:45:00 | 1458.70 | 2025-08-18 09:15:00 | 1527.40 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1520.90 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-08-22 13:45:00 | 1520.20 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-22 14:15:00 | 1520.30 | 2025-08-25 12:15:00 | 1509.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-02 11:15:00 | 1444.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1443.90 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1444.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1440.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1426.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-09-03 13:15:00 | 1425.50 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-09-03 13:45:00 | 1427.60 | 2025-09-04 09:15:00 | 1468.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2025-09-09 09:15:00 | 1497.20 | 2025-09-11 14:15:00 | 1494.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1532.00 | 2025-09-25 12:15:00 | 1597.10 | STOP_HIT | 1.00 | 4.25% |
| BUY | retest2 | 2025-10-01 13:00:00 | 1603.50 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-01 13:30:00 | 1602.90 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-01 15:00:00 | 1603.90 | 2025-10-03 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-09 10:45:00 | 1615.00 | 2025-10-17 14:15:00 | 1654.20 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1729.00 | 2025-10-29 09:15:00 | 1689.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-11-11 12:30:00 | 1731.90 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-12 14:00:00 | 1729.90 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-12 14:30:00 | 1730.00 | 2025-11-12 15:15:00 | 1716.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-14 12:45:00 | 1710.10 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1709.50 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-11-17 10:00:00 | 1705.10 | 2025-11-20 14:15:00 | 1705.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-11-24 11:30:00 | 1676.90 | 2025-11-25 14:15:00 | 1684.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1676.80 | 2025-11-25 14:15:00 | 1684.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-02 10:45:00 | 1739.00 | 2025-12-02 12:15:00 | 1709.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-08 15:00:00 | 1725.00 | 2025-12-09 09:15:00 | 1706.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-09 13:00:00 | 1726.50 | 2025-12-11 15:15:00 | 1720.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1675.00 | 2025-12-22 14:15:00 | 1591.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1675.00 | 2025-12-23 09:15:00 | 1683.40 | STOP_HIT | 0.50 | -0.50% |
| BUY | retest2 | 2025-12-31 13:30:00 | 1705.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-12-31 14:30:00 | 1705.60 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2026-01-01 09:15:00 | 1703.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2026-01-01 10:00:00 | 1703.60 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2026-01-01 15:00:00 | 1726.00 | 2026-01-08 10:15:00 | 1744.60 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2026-01-14 12:00:00 | 1689.10 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1684.50 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1688.20 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1679.50 | 2026-01-19 10:15:00 | 1719.40 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1640.50 | 2026-01-22 14:15:00 | 1664.40 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-27 10:45:00 | 1640.10 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1635.00 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-27 13:00:00 | 1633.80 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-28 14:30:00 | 1632.30 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1614.70 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-01-29 11:30:00 | 1625.50 | 2026-01-29 15:15:00 | 1651.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-10 11:45:00 | 1751.90 | 2026-02-10 14:15:00 | 1725.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-02-10 12:45:00 | 1751.00 | 2026-02-10 14:15:00 | 1725.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1718.90 | 2026-02-17 13:15:00 | 1723.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-02-24 09:30:00 | 1676.70 | 2026-02-24 10:15:00 | 1711.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1730.80 | 2026-03-02 10:15:00 | 1709.40 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-05 12:45:00 | 1647.60 | 2026-03-09 09:15:00 | 1482.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1651.80 | 2026-03-09 09:15:00 | 1486.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1382.40 | 2026-04-02 09:15:00 | 1313.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1382.40 | 2026-04-02 13:15:00 | 1349.10 | STOP_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1569.80 | 2026-04-13 09:15:00 | 1529.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-04-17 10:15:00 | 1570.20 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-17 10:45:00 | 1572.90 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-04-20 10:00:00 | 1569.00 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1584.90 | 2026-04-22 14:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1549.10 | 2026-04-27 14:15:00 | 1558.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-05-08 14:00:00 | 1689.60 | 2026-05-08 15:15:00 | 1671.00 | STOP_HIT | 1.00 | -1.10% |
