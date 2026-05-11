# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 760.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 126 |
| ALERT1 | 94 |
| ALERT2 | 94 |
| ALERT2_SKIP | 54 |
| ALERT3 | 266 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 129 |
| PARTIAL | 27 |
| TARGET_HIT | 16 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 161 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 82 / 79
- **Target hits / Stop hits / Partials:** 16 / 118 / 27
- **Avg / median % per leg:** 1.67% / 0.39%
- **Sum % (uncompounded):** 269.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 19 | 37.3% | 6 | 45 | 0 | 0.80% | 40.9% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.69% | -1.4% |
| BUY @ 3rd Alert (retest2) | 49 | 18 | 36.7% | 6 | 43 | 0 | 0.86% | 42.2% |
| SELL (all) | 110 | 63 | 57.3% | 10 | 73 | 27 | 2.08% | 228.5% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.04% | -3.1% |
| SELL @ 3rd Alert (retest2) | 107 | 62 | 57.9% | 10 | 70 | 27 | 2.16% | 231.6% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.90% | -4.5% |
| retest2 (combined) | 156 | 80 | 51.3% | 16 | 113 | 27 | 1.76% | 273.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1087.00 | 1038.58 | 1034.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 1111.50 | 1076.56 | 1057.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1200.75 | 1208.07 | 1174.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 1190.65 | 1202.75 | 1178.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1190.65 | 1202.75 | 1178.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 1235.40 | 1195.36 | 1184.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 1232.75 | 1252.36 | 1252.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1232.75 | 1252.36 | 1252.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 1216.60 | 1237.25 | 1242.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 1237.40 | 1232.43 | 1237.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 1237.40 | 1232.43 | 1237.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1237.40 | 1232.43 | 1237.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 1237.40 | 1232.43 | 1237.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1231.55 | 1232.25 | 1236.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1269.40 | 1232.25 | 1236.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1260.25 | 1237.85 | 1238.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1268.30 | 1237.85 | 1238.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 1286.55 | 1247.59 | 1243.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 1299.75 | 1268.82 | 1255.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1307.25 | 1335.98 | 1310.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1307.25 | 1335.98 | 1310.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1307.25 | 1335.98 | 1310.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 1312.20 | 1335.98 | 1310.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1300.00 | 1328.79 | 1309.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1300.00 | 1328.79 | 1309.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1249.45 | 1312.92 | 1304.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 1249.45 | 1312.92 | 1304.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 1235.00 | 1297.33 | 1297.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1186.05 | 1256.99 | 1277.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1260.35 | 1229.53 | 1248.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1260.35 | 1229.53 | 1248.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1260.35 | 1229.53 | 1248.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 1277.70 | 1229.53 | 1248.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1255.15 | 1234.66 | 1249.51 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 1338.40 | 1260.74 | 1259.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1423.30 | 1327.25 | 1295.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 1377.85 | 1382.94 | 1354.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:30:00 | 1377.45 | 1382.94 | 1354.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1380.85 | 1391.47 | 1378.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 1380.85 | 1391.47 | 1378.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1366.05 | 1386.38 | 1377.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 1366.05 | 1386.38 | 1377.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1345.00 | 1378.11 | 1374.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1345.00 | 1378.11 | 1374.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 1345.05 | 1371.50 | 1371.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 14:15:00 | 1340.55 | 1365.31 | 1368.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 1362.25 | 1360.97 | 1366.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-13 10:15:00 | 1376.00 | 1360.97 | 1366.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1383.90 | 1365.55 | 1367.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 1383.90 | 1365.55 | 1367.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1369.35 | 1366.31 | 1367.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 1365.00 | 1366.05 | 1367.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 09:15:00 | 1412.70 | 1374.05 | 1370.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 1412.70 | 1374.05 | 1370.62 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 1354.05 | 1370.18 | 1370.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 10:15:00 | 1350.00 | 1361.73 | 1366.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 1381.50 | 1365.69 | 1367.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 1381.50 | 1365.69 | 1367.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1381.50 | 1365.69 | 1367.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:45:00 | 1378.10 | 1365.69 | 1367.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1368.90 | 1366.33 | 1367.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 13:15:00 | 1368.85 | 1366.33 | 1367.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 14:30:00 | 1364.90 | 1365.92 | 1367.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 1359.90 | 1365.92 | 1367.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 09:30:00 | 1366.95 | 1337.95 | 1347.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1351.20 | 1340.60 | 1347.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:30:00 | 1362.20 | 1340.60 | 1347.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1338.40 | 1341.19 | 1346.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 1331.10 | 1339.75 | 1345.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1356.75 | 1342.88 | 1345.61 | SL hit (close>static) qty=1.00 sl=1349.90 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 12:15:00 | 1348.70 | 1347.27 | 1347.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 1368.80 | 1351.58 | 1349.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 15:15:00 | 1353.40 | 1353.51 | 1350.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:15:00 | 1342.90 | 1353.51 | 1350.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1352.25 | 1353.26 | 1350.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1347.10 | 1353.26 | 1350.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1340.65 | 1350.74 | 1349.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 1340.10 | 1350.74 | 1349.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1364.60 | 1353.51 | 1351.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 13:30:00 | 1368.90 | 1356.89 | 1353.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 13:15:00 | 1371.15 | 1379.33 | 1369.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 1358.20 | 1370.33 | 1370.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 1358.20 | 1370.33 | 1370.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 15:15:00 | 1355.00 | 1367.27 | 1369.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 1354.00 | 1352.53 | 1359.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 1354.00 | 1352.53 | 1359.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1354.00 | 1352.53 | 1359.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 1350.40 | 1353.81 | 1358.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 1342.55 | 1354.23 | 1356.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 15:15:00 | 1348.75 | 1350.39 | 1353.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 1399.35 | 1359.92 | 1357.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 1399.35 | 1359.92 | 1357.17 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 1353.25 | 1373.30 | 1373.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 11:15:00 | 1350.00 | 1368.64 | 1371.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 09:15:00 | 1327.30 | 1307.12 | 1317.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1327.30 | 1307.12 | 1317.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1327.30 | 1307.12 | 1317.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:30:00 | 1336.95 | 1307.12 | 1317.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1321.00 | 1309.89 | 1318.00 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 13:15:00 | 1337.90 | 1322.33 | 1322.26 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 1297.20 | 1319.69 | 1322.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 1275.00 | 1299.46 | 1306.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 1303.35 | 1278.80 | 1289.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1303.35 | 1278.80 | 1289.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1303.35 | 1278.80 | 1289.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 1303.35 | 1278.80 | 1289.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1319.95 | 1287.03 | 1292.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:45:00 | 1317.70 | 1287.03 | 1292.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1289.25 | 1293.77 | 1294.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:15:00 | 1283.50 | 1293.77 | 1294.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 11:00:00 | 1286.60 | 1292.33 | 1293.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 13:15:00 | 1219.33 | 1244.38 | 1256.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 13:15:00 | 1222.27 | 1244.38 | 1256.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 1209.45 | 1201.15 | 1218.89 | SL hit (close>ema200) qty=0.50 sl=1201.15 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 1241.00 | 1224.58 | 1223.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 1252.90 | 1236.04 | 1229.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 1272.95 | 1278.36 | 1265.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 1272.95 | 1278.36 | 1265.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1258.95 | 1273.94 | 1265.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 1258.95 | 1273.94 | 1265.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1254.95 | 1270.14 | 1264.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 1254.20 | 1270.14 | 1264.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1261.30 | 1267.92 | 1264.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 1261.30 | 1267.92 | 1264.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1264.30 | 1267.19 | 1264.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1276.80 | 1263.82 | 1263.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 1259.00 | 1263.95 | 1263.75 | SL hit (close<static) qty=1.00 sl=1260.50 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 1257.45 | 1262.65 | 1263.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 1251.00 | 1260.32 | 1262.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1199.00 | 1159.29 | 1187.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1199.00 | 1159.29 | 1187.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1199.00 | 1159.29 | 1187.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1162.00 | 1167.45 | 1183.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:45:00 | 1162.90 | 1164.63 | 1173.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:45:00 | 1162.05 | 1167.50 | 1173.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 1189.15 | 1177.51 | 1176.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 1189.15 | 1177.51 | 1176.72 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 1163.80 | 1174.75 | 1175.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 1159.60 | 1171.72 | 1174.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 1135.00 | 1134.68 | 1145.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 14:00:00 | 1135.00 | 1134.68 | 1145.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1122.60 | 1123.81 | 1132.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:30:00 | 1117.70 | 1124.48 | 1131.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:00:00 | 1117.10 | 1123.00 | 1130.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 13:30:00 | 1118.95 | 1122.42 | 1128.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 1114.00 | 1122.63 | 1128.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1134.15 | 1123.55 | 1127.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 1150.00 | 1131.67 | 1130.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 1150.00 | 1131.67 | 1130.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1154.00 | 1138.98 | 1134.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 1145.50 | 1145.96 | 1141.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 09:15:00 | 1145.00 | 1145.96 | 1141.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1150.80 | 1146.93 | 1142.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:15:00 | 1173.05 | 1146.93 | 1142.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:15:00 | 1159.00 | 1162.81 | 1158.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 1156.15 | 1158.00 | 1156.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 12:15:00 | 1155.30 | 1156.75 | 1156.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 1152.90 | 1155.75 | 1156.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 1152.90 | 1155.75 | 1156.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 10:15:00 | 1137.80 | 1151.77 | 1154.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1146.75 | 1138.10 | 1144.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 1146.75 | 1138.10 | 1144.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1146.75 | 1138.10 | 1144.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 1146.75 | 1138.10 | 1144.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1146.50 | 1139.78 | 1144.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 1156.70 | 1139.78 | 1144.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1142.45 | 1141.58 | 1144.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1138.45 | 1141.58 | 1144.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 1139.50 | 1142.08 | 1144.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 1168.55 | 1150.24 | 1148.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 1168.55 | 1150.24 | 1148.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 1187.80 | 1166.63 | 1157.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 13:15:00 | 1179.05 | 1179.61 | 1169.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 15:15:00 | 1168.25 | 1176.58 | 1169.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1168.25 | 1176.58 | 1169.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 1181.60 | 1176.58 | 1169.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:45:00 | 1182.95 | 1179.71 | 1171.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-03 09:15:00 | 1299.76 | 1242.44 | 1225.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 1339.95 | 1349.31 | 1349.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 14:15:00 | 1326.35 | 1344.72 | 1347.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 1369.90 | 1345.54 | 1347.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1369.90 | 1345.54 | 1347.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1369.90 | 1345.54 | 1347.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 1369.90 | 1345.54 | 1347.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1354.00 | 1347.23 | 1348.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 1361.35 | 1347.23 | 1348.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1346.10 | 1340.12 | 1343.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 1326.00 | 1340.13 | 1342.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 1327.65 | 1330.71 | 1335.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 1327.05 | 1330.57 | 1334.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 11:45:00 | 1328.50 | 1331.27 | 1334.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1330.85 | 1331.19 | 1333.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1330.85 | 1331.19 | 1333.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1351.00 | 1335.15 | 1335.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 1350.10 | 1335.15 | 1335.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 1343.00 | 1336.72 | 1336.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 1343.00 | 1336.72 | 1336.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 1389.40 | 1350.74 | 1343.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 1414.00 | 1415.71 | 1396.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:00:00 | 1414.00 | 1415.71 | 1396.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1380.95 | 1413.96 | 1403.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:30:00 | 1380.95 | 1413.96 | 1403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 1369.95 | 1405.16 | 1400.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 1369.95 | 1405.16 | 1400.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 1369.65 | 1392.40 | 1395.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 1367.05 | 1371.19 | 1377.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 11:15:00 | 1374.00 | 1371.75 | 1377.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 11:15:00 | 1374.00 | 1371.75 | 1377.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 1374.00 | 1371.75 | 1377.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:30:00 | 1372.10 | 1371.75 | 1377.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 1365.25 | 1370.45 | 1376.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:30:00 | 1368.90 | 1370.45 | 1376.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 1364.80 | 1369.32 | 1375.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:30:00 | 1364.30 | 1369.32 | 1375.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1379.50 | 1371.36 | 1375.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 1379.50 | 1371.36 | 1375.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 1399.00 | 1376.89 | 1377.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:15:00 | 1394.10 | 1376.89 | 1377.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1352.55 | 1372.02 | 1375.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:15:00 | 1348.50 | 1372.02 | 1375.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:45:00 | 1346.30 | 1367.82 | 1373.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 11:30:00 | 1348.85 | 1362.53 | 1370.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:00:00 | 1343.00 | 1356.93 | 1362.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1346.65 | 1349.15 | 1355.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:45:00 | 1338.65 | 1346.73 | 1354.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 1281.08 | 1297.80 | 1314.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 1278.98 | 1297.80 | 1314.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 1281.41 | 1297.80 | 1314.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 1275.85 | 1297.80 | 1314.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:15:00 | 1271.72 | 1297.80 | 1314.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 1284.35 | 1277.94 | 1291.62 | SL hit (close>ema200) qty=0.50 sl=1277.94 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 1303.50 | 1296.68 | 1296.41 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 1281.45 | 1298.16 | 1299.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1275.45 | 1288.79 | 1294.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1295.40 | 1282.62 | 1287.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 1295.40 | 1282.62 | 1287.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1295.40 | 1282.62 | 1287.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 1291.05 | 1282.62 | 1287.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1294.95 | 1285.09 | 1288.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 1282.20 | 1285.09 | 1288.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 1245.70 | 1261.50 | 1274.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 1218.09 | 1254.73 | 1269.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1183.41 | 1222.56 | 1244.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 15:15:00 | 1199.75 | 1194.83 | 1209.41 | SL hit (close>ema200) qty=0.50 sl=1194.83 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1205.95 | 1178.01 | 1176.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 1210.15 | 1188.55 | 1181.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 1181.45 | 1193.65 | 1187.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 1181.45 | 1193.65 | 1187.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1181.45 | 1193.65 | 1187.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 1181.45 | 1193.65 | 1187.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1184.50 | 1191.82 | 1187.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 1184.00 | 1191.82 | 1187.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1173.20 | 1188.10 | 1186.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1173.20 | 1188.10 | 1186.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 1170.00 | 1184.48 | 1184.74 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 1258.65 | 1197.12 | 1190.01 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 1179.25 | 1188.84 | 1189.27 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 1207.65 | 1190.53 | 1189.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 1219.60 | 1204.49 | 1197.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 13:15:00 | 1216.20 | 1229.05 | 1215.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 1216.20 | 1229.05 | 1215.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 1216.20 | 1229.05 | 1215.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 1216.20 | 1229.05 | 1215.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1204.90 | 1224.22 | 1214.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:45:00 | 1206.00 | 1224.22 | 1214.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 1202.00 | 1219.78 | 1213.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 1199.05 | 1219.78 | 1213.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1194.90 | 1214.80 | 1211.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1194.90 | 1214.80 | 1211.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1185.00 | 1208.84 | 1209.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 1179.55 | 1199.17 | 1204.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1133.80 | 1117.72 | 1135.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1133.80 | 1117.72 | 1135.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1133.80 | 1117.72 | 1135.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1138.00 | 1117.72 | 1135.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1144.45 | 1123.06 | 1136.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1144.45 | 1123.06 | 1136.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1132.05 | 1124.86 | 1135.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 1127.25 | 1124.86 | 1135.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:15:00 | 1124.75 | 1125.50 | 1135.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 1121.10 | 1126.08 | 1134.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1070.89 | 1113.65 | 1126.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1068.51 | 1113.65 | 1126.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1065.04 | 1113.65 | 1126.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 1089.85 | 1084.70 | 1099.67 | SL hit (close>ema200) qty=0.50 sl=1084.70 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 1135.70 | 1107.24 | 1104.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 1149.10 | 1134.11 | 1122.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 15:15:00 | 1188.75 | 1191.80 | 1176.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1214.50 | 1191.80 | 1176.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1208.40 | 1210.54 | 1202.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:45:00 | 1204.55 | 1210.54 | 1202.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1227.70 | 1213.53 | 1205.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:15:00 | 1229.35 | 1213.53 | 1205.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:30:00 | 1230.15 | 1223.97 | 1212.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 1245.15 | 1222.42 | 1214.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 1219.35 | 1232.23 | 1226.13 | SL hit (close<ema400) qty=1.00 sl=1226.13 alert=retest1 |

### Cycle 34 — SELL (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 12:15:00 | 1280.80 | 1295.13 | 1295.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 1266.05 | 1284.18 | 1288.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 13:15:00 | 1246.10 | 1245.10 | 1258.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:45:00 | 1249.25 | 1245.10 | 1258.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1255.05 | 1248.04 | 1256.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 1263.85 | 1248.04 | 1256.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1253.35 | 1249.10 | 1256.10 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 1268.75 | 1256.84 | 1255.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 1273.90 | 1260.26 | 1257.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 1284.90 | 1292.76 | 1282.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1284.90 | 1292.76 | 1282.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1284.90 | 1292.76 | 1282.47 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 1267.90 | 1278.47 | 1278.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1260.95 | 1274.96 | 1277.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 1283.40 | 1276.65 | 1277.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 1283.40 | 1276.65 | 1277.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1283.40 | 1276.65 | 1277.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 1283.40 | 1276.65 | 1277.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1280.05 | 1277.33 | 1277.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 12:15:00 | 1276.70 | 1277.33 | 1277.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 1304.45 | 1281.83 | 1279.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 1304.45 | 1281.83 | 1279.79 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1256.70 | 1275.61 | 1278.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1236.20 | 1264.70 | 1272.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 1326.40 | 1273.89 | 1275.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 1326.40 | 1273.89 | 1275.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1326.40 | 1273.89 | 1275.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 1326.40 | 1273.89 | 1275.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 1286.75 | 1276.46 | 1276.16 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 11:15:00 | 1260.55 | 1273.28 | 1274.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 15:15:00 | 1255.00 | 1264.80 | 1269.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 1264.90 | 1260.96 | 1265.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 1264.90 | 1260.96 | 1265.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1264.90 | 1260.96 | 1265.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:45:00 | 1266.15 | 1260.96 | 1265.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1266.05 | 1261.98 | 1265.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:30:00 | 1262.50 | 1261.98 | 1265.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1267.85 | 1263.15 | 1266.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 1267.85 | 1263.15 | 1266.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1260.05 | 1262.53 | 1265.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 1259.20 | 1262.53 | 1265.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1247.00 | 1259.43 | 1263.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:45:00 | 1243.00 | 1256.51 | 1262.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 1243.65 | 1256.51 | 1262.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:30:00 | 1241.95 | 1249.93 | 1257.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 1238.25 | 1249.32 | 1256.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1250.00 | 1246.75 | 1253.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 1252.85 | 1246.75 | 1253.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1252.10 | 1248.18 | 1252.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:45:00 | 1252.55 | 1248.18 | 1252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1238.00 | 1246.14 | 1251.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:45:00 | 1234.90 | 1241.64 | 1247.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:30:00 | 1233.15 | 1237.22 | 1244.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 10:15:00 | 1233.35 | 1239.75 | 1240.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 1251.95 | 1242.19 | 1241.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 1251.95 | 1242.19 | 1241.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 1256.85 | 1248.86 | 1245.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1281.20 | 1286.98 | 1274.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 10:00:00 | 1281.20 | 1286.98 | 1274.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1258.20 | 1281.22 | 1273.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1258.20 | 1281.22 | 1273.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1269.85 | 1278.95 | 1272.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1261.00 | 1278.95 | 1272.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1279.85 | 1279.62 | 1274.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 1279.85 | 1279.62 | 1274.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1248.55 | 1273.40 | 1271.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 1248.55 | 1273.40 | 1271.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1257.50 | 1270.22 | 1270.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 1242.80 | 1262.13 | 1266.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 15:15:00 | 1252.50 | 1252.03 | 1258.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 09:15:00 | 1243.50 | 1252.03 | 1258.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1237.70 | 1249.17 | 1256.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:45:00 | 1228.30 | 1242.64 | 1252.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1166.88 | 1197.43 | 1217.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 1105.47 | 1138.61 | 1171.11 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1140.00 | 1113.17 | 1111.14 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1123.40 | 1129.16 | 1129.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1107.90 | 1122.35 | 1125.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 15:15:00 | 1065.50 | 1064.33 | 1080.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 09:15:00 | 1075.75 | 1064.33 | 1080.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1061.95 | 1063.85 | 1078.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:45:00 | 1046.05 | 1059.67 | 1072.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 993.75 | 1017.57 | 1036.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 1020.00 | 1014.55 | 1028.63 | SL hit (close>ema200) qty=0.50 sl=1014.55 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1071.00 | 1037.47 | 1034.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1091.50 | 1048.28 | 1039.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1140.20 | 1155.46 | 1132.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 1140.20 | 1155.46 | 1132.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1171.50 | 1157.94 | 1137.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 1146.10 | 1157.94 | 1137.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1213.50 | 1169.24 | 1145.93 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 1155.80 | 1159.00 | 1159.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 1134.90 | 1154.18 | 1156.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 12:15:00 | 1149.00 | 1147.89 | 1152.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:30:00 | 1148.95 | 1147.89 | 1152.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1154.30 | 1149.16 | 1152.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1154.30 | 1149.16 | 1152.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1149.20 | 1149.17 | 1152.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1136.00 | 1149.17 | 1152.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:00:00 | 1147.20 | 1148.95 | 1151.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 1148.90 | 1148.59 | 1151.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1089.84 | 1110.23 | 1125.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1091.45 | 1110.23 | 1125.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 1079.20 | 1096.05 | 1114.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 1032.48 | 1069.69 | 1095.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1028.50 | 1007.37 | 1005.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 1030.00 | 1023.03 | 1016.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 1015.10 | 1021.44 | 1016.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 14:15:00 | 1015.10 | 1021.44 | 1016.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 1015.10 | 1021.44 | 1016.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 1015.10 | 1021.44 | 1016.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 1030.00 | 1023.15 | 1017.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1033.90 | 1023.15 | 1017.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:45:00 | 1036.80 | 1025.02 | 1018.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1033.70 | 1024.97 | 1019.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 13:15:00 | 1008.05 | 1020.27 | 1018.45 | SL hit (close<static) qty=1.00 sl=1012.05 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 1004.50 | 1017.12 | 1017.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 997.60 | 1011.59 | 1014.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 986.00 | 981.75 | 991.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 986.00 | 981.75 | 991.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 986.00 | 981.75 | 991.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 986.00 | 981.75 | 991.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 989.95 | 983.39 | 991.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:45:00 | 988.80 | 983.39 | 991.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 972.65 | 978.30 | 984.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:00:00 | 959.50 | 972.97 | 981.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:00:00 | 942.55 | 959.63 | 972.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 09:30:00 | 953.20 | 950.41 | 952.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 966.20 | 956.07 | 954.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 966.20 | 956.07 | 954.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 984.15 | 962.31 | 958.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 961.15 | 963.33 | 959.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 961.15 | 963.33 | 959.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 961.15 | 963.33 | 959.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:30:00 | 961.50 | 963.33 | 959.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 966.00 | 963.97 | 960.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:30:00 | 964.20 | 963.97 | 960.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 959.55 | 963.09 | 960.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 959.55 | 963.09 | 960.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 960.90 | 962.65 | 960.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 976.15 | 962.65 | 960.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 15:15:00 | 952.50 | 963.39 | 962.84 | SL hit (close<static) qty=1.00 sl=956.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 946.00 | 961.18 | 962.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 940.35 | 957.01 | 960.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 954.00 | 952.82 | 957.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 954.00 | 952.82 | 957.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 954.00 | 952.82 | 957.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:45:00 | 956.95 | 952.82 | 957.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 954.70 | 953.20 | 956.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:15:00 | 960.00 | 953.20 | 956.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 960.00 | 954.56 | 957.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 963.85 | 956.92 | 957.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 974.30 | 960.39 | 959.46 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 946.35 | 961.90 | 961.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 938.35 | 949.97 | 955.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 951.75 | 943.91 | 948.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 951.75 | 943.91 | 948.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 951.75 | 943.91 | 948.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 948.00 | 943.91 | 948.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 933.50 | 941.82 | 947.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 924.30 | 938.95 | 944.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 948.80 | 945.37 | 945.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 14:15:00 | 948.80 | 945.37 | 945.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 953.00 | 946.89 | 945.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 957.00 | 957.63 | 953.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:00:00 | 957.00 | 957.63 | 953.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 968.15 | 959.74 | 955.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 957.40 | 959.74 | 955.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 984.85 | 990.31 | 981.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 982.05 | 990.31 | 981.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 979.15 | 988.08 | 981.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 979.15 | 988.08 | 981.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 987.50 | 987.96 | 981.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 997.35 | 984.95 | 981.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 990.50 | 986.06 | 982.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 975.00 | 979.94 | 980.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 975.00 | 979.94 | 980.38 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 982.25 | 980.84 | 980.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 992.80 | 983.56 | 981.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 10:15:00 | 983.25 | 986.79 | 984.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 983.25 | 986.79 | 984.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 983.25 | 986.79 | 984.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 983.25 | 986.79 | 984.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 979.85 | 985.40 | 983.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 979.85 | 985.40 | 983.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 979.45 | 984.21 | 983.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 979.45 | 984.21 | 983.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 974.85 | 982.34 | 982.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 966.50 | 978.44 | 980.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 983.30 | 966.91 | 971.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 983.30 | 966.91 | 971.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 983.30 | 966.91 | 971.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 976.80 | 966.91 | 971.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 989.00 | 971.32 | 972.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 988.60 | 971.32 | 972.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 986.70 | 974.40 | 974.10 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 963.75 | 976.80 | 977.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 951.95 | 966.76 | 972.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 920.35 | 911.22 | 931.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 920.35 | 911.22 | 931.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 920.35 | 911.22 | 931.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 905.00 | 917.23 | 925.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:30:00 | 909.20 | 914.73 | 923.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 909.00 | 911.04 | 919.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 13:15:00 | 934.70 | 921.67 | 921.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 934.70 | 921.67 | 921.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 955.95 | 932.36 | 926.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 1038.00 | 1040.44 | 1026.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 09:45:00 | 1040.60 | 1040.44 | 1026.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1028.00 | 1037.34 | 1028.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 1028.00 | 1037.34 | 1028.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1027.00 | 1035.27 | 1028.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 1028.95 | 1035.27 | 1028.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1036.45 | 1035.51 | 1029.22 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1016.00 | 1024.82 | 1025.67 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1039.20 | 1028.40 | 1027.03 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1012.50 | 1025.61 | 1027.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 1008.95 | 1022.28 | 1025.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 1018.20 | 1017.58 | 1022.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 1018.20 | 1017.58 | 1022.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1031.30 | 1019.56 | 1022.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:45:00 | 1028.25 | 1019.56 | 1022.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1029.15 | 1021.48 | 1023.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 1032.80 | 1021.48 | 1023.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 1025.05 | 1022.62 | 1023.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 1028.00 | 1022.62 | 1023.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 1028.65 | 1023.83 | 1023.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 1028.65 | 1023.83 | 1023.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 14:15:00 | 1025.90 | 1024.24 | 1024.07 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1022.00 | 1023.79 | 1023.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1002.80 | 1017.11 | 1020.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1002.50 | 1002.43 | 1010.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:45:00 | 1001.10 | 1002.43 | 1010.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1015.60 | 1004.22 | 1009.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:45:00 | 1015.50 | 1004.22 | 1009.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1017.20 | 1006.81 | 1010.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 1017.20 | 1006.81 | 1010.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1011.60 | 1011.91 | 1012.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 1005.20 | 1010.12 | 1011.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:15:00 | 1008.70 | 1010.20 | 1011.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 1007.60 | 1005.51 | 1007.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:15:00 | 1008.90 | 1006.40 | 1007.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 1018.70 | 1008.86 | 1008.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1018.70 | 1008.86 | 1008.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 1023.00 | 1011.69 | 1009.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1011.70 | 1013.90 | 1011.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 12:15:00 | 1011.70 | 1013.90 | 1011.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1011.70 | 1013.90 | 1011.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 1011.70 | 1013.90 | 1011.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1000.70 | 1011.26 | 1010.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1000.70 | 1011.26 | 1010.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1005.50 | 1010.11 | 1010.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 995.90 | 1007.26 | 1008.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 1004.30 | 1001.50 | 1005.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 1004.30 | 1001.50 | 1005.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1010.10 | 1003.22 | 1005.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:45:00 | 1007.70 | 1003.22 | 1005.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1011.90 | 1004.96 | 1006.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 1011.90 | 1004.96 | 1006.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 1022.90 | 1010.15 | 1008.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1033.20 | 1014.76 | 1010.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1067.30 | 1079.37 | 1064.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1067.30 | 1079.37 | 1064.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1067.30 | 1079.37 | 1064.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 1065.60 | 1079.37 | 1064.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1066.60 | 1074.71 | 1065.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 1068.80 | 1074.71 | 1065.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1065.70 | 1072.91 | 1065.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 1065.70 | 1072.91 | 1065.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1066.90 | 1071.70 | 1065.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:30:00 | 1067.40 | 1071.70 | 1065.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1065.20 | 1070.40 | 1065.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 1065.20 | 1070.40 | 1065.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1067.00 | 1069.72 | 1065.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1091.90 | 1069.72 | 1065.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1104.00 | 1076.58 | 1068.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 1124.10 | 1108.26 | 1094.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 1100.00 | 1104.23 | 1104.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1100.00 | 1104.23 | 1104.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 1095.00 | 1101.21 | 1102.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 1089.90 | 1084.29 | 1089.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 1089.90 | 1084.29 | 1089.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1089.90 | 1084.29 | 1089.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1089.90 | 1084.29 | 1089.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1083.10 | 1084.05 | 1089.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 1077.50 | 1081.43 | 1087.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1095.30 | 1085.27 | 1085.76 | SL hit (close>static) qty=1.00 sl=1090.90 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 1094.10 | 1087.04 | 1086.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1100.90 | 1090.72 | 1088.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1090.00 | 1097.45 | 1092.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1090.00 | 1097.45 | 1092.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1090.00 | 1097.45 | 1092.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 1090.00 | 1097.45 | 1092.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1098.10 | 1097.58 | 1093.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 1100.00 | 1095.55 | 1092.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1107.80 | 1096.68 | 1094.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-03 10:15:00 | 1210.00 | 1166.03 | 1136.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 1237.10 | 1249.82 | 1251.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 13:15:00 | 1226.70 | 1242.86 | 1247.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 1238.30 | 1237.90 | 1243.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 1238.30 | 1237.90 | 1243.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1227.30 | 1235.78 | 1241.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1213.30 | 1232.62 | 1238.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 1152.63 | 1168.36 | 1178.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1156.20 | 1147.47 | 1160.94 | SL hit (close>ema200) qty=0.50 sl=1147.47 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 1165.30 | 1155.46 | 1154.52 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 1143.10 | 1152.26 | 1153.18 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 1159.20 | 1153.44 | 1152.75 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 1141.70 | 1151.09 | 1151.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 11:15:00 | 1137.90 | 1147.13 | 1149.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 1114.40 | 1114.40 | 1123.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:30:00 | 1117.50 | 1114.40 | 1123.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1090.30 | 1097.37 | 1106.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 1098.30 | 1097.37 | 1106.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1063.50 | 1058.21 | 1069.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 1062.70 | 1058.21 | 1069.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1068.70 | 1062.43 | 1069.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 1069.80 | 1062.43 | 1069.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1073.10 | 1064.57 | 1069.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1073.10 | 1064.57 | 1069.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1076.00 | 1066.85 | 1070.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1080.70 | 1066.85 | 1070.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1079.40 | 1070.87 | 1071.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1079.40 | 1070.87 | 1071.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 1082.50 | 1073.20 | 1072.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 1086.20 | 1075.80 | 1073.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 09:15:00 | 1101.00 | 1107.38 | 1096.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:00:00 | 1101.00 | 1107.38 | 1096.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1098.40 | 1105.58 | 1096.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1098.40 | 1105.58 | 1096.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1089.40 | 1102.34 | 1095.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 1089.40 | 1102.34 | 1095.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1084.00 | 1098.68 | 1094.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 1084.00 | 1098.68 | 1094.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 1077.30 | 1091.33 | 1091.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1070.30 | 1082.23 | 1087.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 15:15:00 | 1080.40 | 1079.97 | 1084.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1070.20 | 1079.97 | 1084.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1085.00 | 1071.04 | 1075.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 1085.00 | 1071.04 | 1075.40 | SL hit (close>ema400) qty=1.00 sl=1075.40 alert=retest1 |

### Cycle 77 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1085.00 | 1077.87 | 1077.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1097.00 | 1088.85 | 1083.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1121.30 | 1126.39 | 1116.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 1121.30 | 1126.39 | 1116.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1118.80 | 1124.87 | 1117.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1114.00 | 1124.87 | 1117.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1117.80 | 1123.46 | 1117.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1117.80 | 1123.46 | 1117.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1119.80 | 1122.73 | 1117.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 1116.90 | 1122.73 | 1117.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1123.20 | 1122.82 | 1117.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1124.70 | 1122.82 | 1117.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 1109.60 | 1120.18 | 1117.13 | SL hit (close<static) qty=1.00 sl=1115.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1097.30 | 1113.81 | 1114.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1095.10 | 1102.85 | 1107.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 1065.10 | 1059.88 | 1074.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:45:00 | 1065.00 | 1059.88 | 1074.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1060.60 | 1059.69 | 1067.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 1060.60 | 1059.69 | 1067.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1006.90 | 989.83 | 999.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:45:00 | 1010.20 | 989.83 | 999.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 980.80 | 988.02 | 997.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 979.90 | 990.53 | 995.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 980.20 | 987.13 | 992.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 975.20 | 984.88 | 989.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 980.50 | 984.13 | 987.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 992.20 | 985.34 | 987.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 991.00 | 985.34 | 987.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 991.40 | 986.55 | 987.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 989.00 | 986.55 | 987.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 956.50 | 951.67 | 957.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 955.10 | 951.67 | 957.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 963.50 | 954.04 | 958.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 963.50 | 954.04 | 958.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 956.90 | 954.61 | 958.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 965.40 | 960.20 | 959.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 965.40 | 960.20 | 959.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 970.50 | 962.26 | 960.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 959.20 | 965.17 | 962.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 959.20 | 965.17 | 962.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 959.20 | 965.17 | 962.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 959.20 | 965.17 | 962.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 960.70 | 964.28 | 962.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 957.70 | 964.28 | 962.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 957.90 | 963.00 | 962.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 957.50 | 963.00 | 962.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 970.30 | 964.22 | 962.81 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 913.20 | 954.71 | 958.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 906.60 | 945.09 | 954.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 14:15:00 | 942.00 | 931.70 | 943.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 942.00 | 931.70 | 943.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 942.00 | 931.70 | 943.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 938.50 | 931.70 | 943.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 958.30 | 936.96 | 943.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 958.30 | 936.96 | 943.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 957.10 | 940.99 | 944.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 960.00 | 940.99 | 944.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 965.10 | 949.04 | 948.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 976.80 | 964.55 | 961.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 968.00 | 968.51 | 965.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 09:15:00 | 959.50 | 968.51 | 965.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 966.30 | 968.07 | 965.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 960.00 | 968.07 | 965.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 957.70 | 966.00 | 964.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 957.10 | 966.00 | 964.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 958.90 | 964.58 | 964.16 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 957.50 | 963.16 | 963.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 942.90 | 958.24 | 961.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 12:15:00 | 949.40 | 948.85 | 954.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 12:30:00 | 948.00 | 948.85 | 954.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 952.80 | 949.71 | 953.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 951.40 | 949.71 | 953.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 955.10 | 950.79 | 954.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 938.10 | 950.79 | 954.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 940.00 | 934.68 | 933.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 940.00 | 934.68 | 933.98 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 931.05 | 934.03 | 934.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 927.60 | 932.10 | 933.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 920.75 | 920.54 | 925.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 920.75 | 920.54 | 925.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 920.75 | 920.54 | 925.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 912.95 | 918.12 | 923.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 912.25 | 917.01 | 922.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 907.60 | 913.77 | 919.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:15:00 | 912.85 | 914.02 | 918.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 915.95 | 909.66 | 913.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 931.60 | 917.22 | 916.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 931.60 | 917.22 | 916.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 943.40 | 929.36 | 923.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 932.55 | 933.80 | 928.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 928.55 | 933.80 | 928.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 930.30 | 933.10 | 928.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 928.00 | 933.10 | 928.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 945.50 | 955.13 | 949.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 945.50 | 955.13 | 949.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 947.45 | 953.60 | 949.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 948.80 | 953.60 | 949.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 950.85 | 953.05 | 949.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 963.65 | 953.54 | 949.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 957.35 | 955.55 | 951.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:45:00 | 956.80 | 953.59 | 952.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 948.05 | 952.40 | 952.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 948.05 | 952.40 | 952.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 942.45 | 950.41 | 951.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 926.15 | 917.89 | 922.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 926.15 | 917.89 | 922.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 926.15 | 917.89 | 922.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 908.40 | 921.28 | 922.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 910.75 | 895.65 | 895.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 910.75 | 895.65 | 895.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 923.85 | 903.45 | 899.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 908.85 | 910.93 | 904.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 908.60 | 910.93 | 904.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 909.60 | 910.67 | 904.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:15:00 | 905.95 | 910.67 | 904.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 908.00 | 910.13 | 905.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 907.75 | 910.13 | 905.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 943.25 | 922.74 | 915.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 946.45 | 922.74 | 915.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 913.50 | 926.06 | 926.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 913.50 | 926.06 | 926.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 911.85 | 923.21 | 925.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 931.10 | 920.67 | 923.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 931.10 | 920.67 | 923.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 931.10 | 920.67 | 923.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 931.10 | 920.67 | 923.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 941.25 | 924.78 | 924.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 941.25 | 924.78 | 924.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 945.65 | 928.96 | 926.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 951.60 | 933.48 | 928.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 974.60 | 977.12 | 966.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 974.95 | 977.12 | 966.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 978.50 | 980.19 | 974.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 978.50 | 980.19 | 974.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 976.50 | 979.45 | 974.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 970.80 | 979.45 | 974.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 977.05 | 978.97 | 974.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 977.05 | 978.97 | 974.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 998.60 | 982.90 | 977.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 1004.30 | 982.90 | 977.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:30:00 | 1004.50 | 991.24 | 982.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1004.45 | 991.02 | 983.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 1003.55 | 995.15 | 986.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1009.85 | 1016.65 | 1011.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1009.85 | 1016.65 | 1011.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1013.35 | 1015.99 | 1012.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1020.50 | 1015.99 | 1012.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1028.00 | 1035.72 | 1035.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1028.00 | 1035.72 | 1035.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 1015.00 | 1029.83 | 1032.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 989.90 | 986.67 | 998.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 973.30 | 981.19 | 989.61 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 966.10 | 969.77 | 977.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:00:00 | 959.50 | 966.66 | 974.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 955.90 | 965.65 | 973.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 957.20 | 965.65 | 973.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 958.60 | 964.24 | 972.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 959.40 | 949.98 | 954.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 959.40 | 949.98 | 954.78 | SL hit (close>ema400) qty=1.00 sl=954.78 alert=retest1 |

### Cycle 91 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 908.70 | 898.70 | 897.53 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 891.50 | 900.50 | 901.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 886.75 | 892.76 | 895.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 886.80 | 886.54 | 889.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 886.50 | 886.18 | 889.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 891.20 | 887.19 | 889.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 891.20 | 887.19 | 889.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 892.40 | 888.23 | 889.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 891.65 | 888.23 | 889.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 890.80 | 889.19 | 889.83 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 894.85 | 890.45 | 890.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 905.00 | 893.36 | 891.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 892.45 | 893.18 | 891.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 892.45 | 893.18 | 891.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 892.45 | 893.18 | 891.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 891.60 | 893.18 | 891.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 885.00 | 891.54 | 891.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 885.00 | 891.54 | 891.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 12:15:00 | 885.25 | 890.28 | 890.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 13:15:00 | 882.80 | 888.79 | 889.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 876.00 | 859.83 | 864.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 876.00 | 859.83 | 864.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 876.00 | 859.83 | 864.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 876.00 | 859.83 | 864.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 875.45 | 862.95 | 865.87 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 882.80 | 868.98 | 868.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 887.05 | 878.65 | 875.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 874.05 | 878.75 | 875.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 874.05 | 878.75 | 875.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 874.05 | 878.75 | 875.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 874.80 | 878.75 | 875.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 875.30 | 878.06 | 875.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 875.10 | 878.06 | 875.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 876.00 | 877.65 | 875.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 874.55 | 877.65 | 875.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 876.30 | 877.38 | 875.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 875.00 | 877.38 | 875.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 877.20 | 877.34 | 875.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 875.90 | 877.34 | 875.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 882.30 | 878.33 | 876.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:15:00 | 877.95 | 878.33 | 876.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 877.95 | 878.26 | 876.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 870.10 | 878.26 | 876.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 870.60 | 876.73 | 876.13 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 869.85 | 875.35 | 875.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 863.00 | 872.02 | 873.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 865.00 | 858.56 | 862.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 865.00 | 858.56 | 862.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 865.00 | 858.56 | 862.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 866.05 | 858.56 | 862.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 870.10 | 860.87 | 863.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 870.00 | 860.87 | 863.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 878.55 | 865.96 | 865.12 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 13:15:00 | 864.65 | 865.03 | 865.04 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 867.50 | 865.52 | 865.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 874.20 | 867.66 | 866.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 868.80 | 871.01 | 869.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 868.80 | 871.01 | 869.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 868.80 | 871.01 | 869.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 868.70 | 871.01 | 869.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 875.05 | 871.82 | 869.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 879.90 | 873.92 | 871.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 877.45 | 877.36 | 874.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 883.70 | 876.29 | 874.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 871.20 | 876.25 | 876.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 871.20 | 876.25 | 876.47 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 881.30 | 877.18 | 876.85 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 870.30 | 876.13 | 876.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 864.80 | 872.72 | 874.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 873.95 | 872.64 | 874.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 873.95 | 872.64 | 874.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 873.95 | 872.64 | 874.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 873.95 | 872.64 | 874.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 870.00 | 872.11 | 873.96 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 883.60 | 875.80 | 875.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 886.40 | 877.92 | 876.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 877.45 | 880.90 | 878.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 877.45 | 880.90 | 878.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 877.45 | 880.90 | 878.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 877.45 | 880.90 | 878.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 884.25 | 881.57 | 878.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 889.00 | 883.50 | 880.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:00:00 | 886.55 | 890.01 | 885.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 886.25 | 890.77 | 888.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 887.65 | 890.77 | 888.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 890.90 | 890.80 | 888.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 887.25 | 890.80 | 888.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 895.85 | 896.37 | 893.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 893.65 | 896.37 | 893.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 895.45 | 896.18 | 893.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 899.70 | 893.66 | 892.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:45:00 | 898.65 | 894.43 | 893.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 890.05 | 893.56 | 893.02 | SL hit (close<static) qty=1.00 sl=891.90 alert=retest2 |

### Cycle 104 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 885.85 | 891.83 | 892.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 883.70 | 888.79 | 890.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 890.75 | 889.18 | 890.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 890.75 | 889.18 | 890.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 890.75 | 889.18 | 890.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 879.50 | 886.92 | 889.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:45:00 | 881.20 | 884.10 | 887.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 837.14 | 844.94 | 852.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 13:15:00 | 835.52 | 840.56 | 847.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 791.55 | 812.99 | 826.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 105 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 751.00 | 741.60 | 740.86 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 728.25 | 741.53 | 742.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 720.65 | 737.36 | 740.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 736.85 | 734.82 | 738.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 736.85 | 734.82 | 738.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 736.85 | 734.82 | 738.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 736.85 | 734.82 | 738.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 749.00 | 737.65 | 739.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 759.60 | 737.65 | 739.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 771.00 | 744.32 | 742.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 777.50 | 750.96 | 745.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 772.90 | 780.77 | 771.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 772.90 | 780.77 | 771.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 772.90 | 780.77 | 771.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 772.90 | 780.77 | 771.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 767.70 | 778.16 | 771.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 764.55 | 778.16 | 771.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 766.85 | 775.90 | 770.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 775.95 | 772.86 | 770.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 764.00 | 769.60 | 769.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 764.00 | 769.60 | 769.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 757.55 | 767.19 | 768.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 778.70 | 768.37 | 768.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 778.70 | 768.37 | 768.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 778.70 | 768.37 | 768.68 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 786.25 | 771.94 | 770.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 791.10 | 775.77 | 772.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 796.00 | 796.79 | 789.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 795.10 | 796.52 | 790.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 784.80 | 796.21 | 793.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 784.50 | 796.21 | 793.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 789.00 | 794.77 | 793.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 794.85 | 793.75 | 792.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 769.00 | 789.13 | 791.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 769.00 | 789.13 | 791.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 766.65 | 777.22 | 784.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 739.75 | 738.79 | 746.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:30:00 | 743.15 | 738.79 | 746.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 738.55 | 739.87 | 743.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:30:00 | 742.65 | 739.87 | 743.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 740.00 | 737.91 | 740.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 740.85 | 737.91 | 740.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 740.80 | 738.49 | 740.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 753.50 | 738.49 | 740.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 749.55 | 740.70 | 741.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 751.70 | 740.70 | 741.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 754.40 | 743.44 | 742.36 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 737.05 | 743.28 | 743.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 734.50 | 741.53 | 742.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 724.45 | 720.33 | 727.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 724.45 | 720.33 | 727.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 724.45 | 720.33 | 727.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 727.05 | 720.33 | 727.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 707.75 | 716.11 | 721.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 698.50 | 712.31 | 719.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 663.57 | 679.83 | 692.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 662.55 | 661.87 | 675.14 | SL hit (close>ema200) qty=0.50 sl=661.87 alert=retest2 |

### Cycle 113 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 672.25 | 666.48 | 666.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 673.20 | 668.41 | 667.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 672.90 | 673.92 | 670.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:45:00 | 673.20 | 673.92 | 670.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 670.90 | 673.32 | 670.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 670.90 | 673.32 | 670.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 670.50 | 672.76 | 670.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:15:00 | 670.90 | 672.76 | 670.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 670.90 | 672.38 | 670.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 653.45 | 672.38 | 670.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 659.50 | 669.81 | 669.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 655.55 | 669.81 | 669.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 679.80 | 672.17 | 670.77 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 664.45 | 669.28 | 669.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 655.55 | 665.69 | 667.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 659.00 | 657.41 | 661.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:15:00 | 638.35 | 657.41 | 661.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 658.60 | 651.45 | 656.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 658.60 | 651.45 | 656.11 | SL hit (close>ema400) qty=1.00 sl=656.11 alert=retest1 |

### Cycle 115 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 670.15 | 655.14 | 654.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 681.25 | 660.36 | 657.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 662.40 | 673.54 | 666.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 662.40 | 673.54 | 666.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 662.40 | 673.54 | 666.25 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 659.00 | 663.94 | 663.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 652.00 | 659.32 | 661.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 636.00 | 635.92 | 644.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 636.00 | 635.92 | 644.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 636.00 | 635.92 | 644.06 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 665.00 | 646.55 | 645.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 718.35 | 671.30 | 659.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 688.25 | 688.47 | 672.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 688.25 | 688.47 | 672.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 683.00 | 688.43 | 675.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 661.90 | 688.43 | 675.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 651.90 | 681.12 | 673.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 651.90 | 681.12 | 673.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 647.00 | 674.30 | 670.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 647.00 | 674.30 | 670.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 651.50 | 665.42 | 667.18 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 671.10 | 664.45 | 664.34 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 649.30 | 662.65 | 663.61 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 681.40 | 666.27 | 664.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 689.65 | 670.95 | 667.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 692.80 | 693.38 | 685.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 692.80 | 693.38 | 685.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 699.25 | 712.17 | 703.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 699.25 | 712.17 | 703.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 708.85 | 711.50 | 703.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 721.80 | 708.39 | 705.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 715.55 | 720.71 | 714.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 787.11 | 740.27 | 733.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 771.85 | 782.74 | 782.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 765.80 | 779.35 | 781.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 776.40 | 776.26 | 778.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 776.40 | 776.26 | 778.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 775.00 | 776.00 | 778.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 783.25 | 776.00 | 778.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 785.25 | 777.85 | 779.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 789.30 | 777.85 | 779.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 785.30 | 779.34 | 779.75 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 786.00 | 780.67 | 780.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 789.50 | 782.44 | 781.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 15:15:00 | 798.10 | 799.12 | 792.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:15:00 | 802.90 | 799.12 | 792.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 796.00 | 798.14 | 795.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 800.75 | 798.14 | 795.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 788.55 | 796.22 | 794.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 788.55 | 796.22 | 794.82 | SL hit (close<ema400) qty=1.00 sl=794.82 alert=retest1 |

### Cycle 124 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 788.90 | 793.18 | 793.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 781.40 | 789.49 | 791.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 11:15:00 | 790.20 | 789.31 | 790.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 790.20 | 789.31 | 790.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 790.20 | 789.31 | 790.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 790.20 | 789.31 | 790.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 790.25 | 789.50 | 790.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 790.55 | 789.50 | 790.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 790.10 | 789.62 | 790.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 790.00 | 789.62 | 790.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 789.90 | 789.67 | 790.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:15:00 | 790.15 | 789.67 | 790.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 790.15 | 789.77 | 790.57 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 801.45 | 792.11 | 791.56 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 776.70 | 793.24 | 793.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 14:15:00 | 773.25 | 782.39 | 787.85 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 09:15:00 | 1235.40 | 2024-05-28 12:15:00 | 1232.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-13 13:00:00 | 1365.00 | 2024-06-14 09:15:00 | 1412.70 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-06-18 13:15:00 | 1368.85 | 2024-06-21 09:15:00 | 1356.75 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-06-18 14:30:00 | 1364.90 | 2024-06-21 12:15:00 | 1348.70 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-06-18 15:15:00 | 1359.90 | 2024-06-21 12:15:00 | 1348.70 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-06-20 09:30:00 | 1366.95 | 2024-06-21 12:15:00 | 1348.70 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2024-06-20 14:15:00 | 1331.10 | 2024-06-21 12:15:00 | 1348.70 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-24 13:30:00 | 1368.90 | 2024-06-26 14:15:00 | 1358.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-25 13:15:00 | 1371.15 | 2024-06-26 14:15:00 | 1358.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-28 12:15:00 | 1350.40 | 2024-07-02 09:15:00 | 1399.35 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-07-01 11:15:00 | 1342.55 | 2024-07-02 09:15:00 | 1399.35 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-07-01 15:15:00 | 1348.75 | 2024-07-02 09:15:00 | 1399.35 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-07-18 10:15:00 | 1283.50 | 2024-07-23 13:15:00 | 1219.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 11:00:00 | 1286.60 | 2024-07-23 13:15:00 | 1222.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 10:15:00 | 1283.50 | 2024-07-25 09:15:00 | 1209.45 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-07-18 11:00:00 | 1286.60 | 2024-07-25 09:15:00 | 1209.45 | STOP_HIT | 0.50 | 6.00% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1276.80 | 2024-08-01 11:15:00 | 1259.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1162.00 | 2024-08-08 11:15:00 | 1189.15 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-08-07 13:45:00 | 1162.90 | 2024-08-08 11:15:00 | 1189.15 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-07 14:45:00 | 1162.05 | 2024-08-08 11:15:00 | 1189.15 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-08-14 09:30:00 | 1117.70 | 2024-08-16 12:15:00 | 1150.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-08-14 11:00:00 | 1117.10 | 2024-08-16 12:15:00 | 1150.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-08-14 13:30:00 | 1118.95 | 2024-08-16 12:15:00 | 1150.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-08-14 15:15:00 | 1114.00 | 2024-08-16 12:15:00 | 1150.00 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-08-20 10:15:00 | 1173.05 | 2024-08-22 13:15:00 | 1152.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-21 15:15:00 | 1159.00 | 2024-08-22 13:15:00 | 1152.90 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-08-22 10:30:00 | 1156.15 | 2024-08-22 13:15:00 | 1152.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-08-22 12:15:00 | 1155.30 | 2024-08-22 13:15:00 | 1152.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-26 13:15:00 | 1138.45 | 2024-08-27 09:15:00 | 1168.55 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-08-26 14:15:00 | 1139.50 | 2024-08-27 09:15:00 | 1168.55 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1181.60 | 2024-09-03 09:15:00 | 1299.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-29 09:45:00 | 1182.95 | 2024-09-03 09:15:00 | 1301.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-23 15:15:00 | 1326.00 | 2024-09-25 14:15:00 | 1343.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-09-24 15:00:00 | 1327.65 | 2024-09-25 14:15:00 | 1343.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-09-25 09:15:00 | 1327.05 | 2024-09-25 14:15:00 | 1343.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-25 11:45:00 | 1328.50 | 2024-09-25 14:15:00 | 1343.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-08 10:15:00 | 1348.50 | 2024-10-14 09:15:00 | 1281.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-08 10:45:00 | 1346.30 | 2024-10-14 09:15:00 | 1278.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-08 11:30:00 | 1348.85 | 2024-10-14 09:15:00 | 1281.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 12:00:00 | 1343.00 | 2024-10-14 09:15:00 | 1275.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 10:45:00 | 1338.65 | 2024-10-14 09:15:00 | 1271.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-08 10:15:00 | 1348.50 | 2024-10-15 11:15:00 | 1284.35 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2024-10-08 10:45:00 | 1346.30 | 2024-10-15 11:15:00 | 1284.35 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2024-10-08 11:30:00 | 1348.85 | 2024-10-15 11:15:00 | 1284.35 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2024-10-09 12:00:00 | 1343.00 | 2024-10-15 11:15:00 | 1284.35 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2024-10-10 10:45:00 | 1338.65 | 2024-10-15 11:15:00 | 1284.35 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1282.20 | 2024-10-22 10:15:00 | 1218.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:30:00 | 1245.70 | 2024-10-23 09:15:00 | 1183.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1282.20 | 2024-10-24 15:15:00 | 1199.75 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2024-10-22 09:30:00 | 1245.70 | 2024-10-24 15:15:00 | 1199.75 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2024-11-12 12:15:00 | 1127.25 | 2024-11-13 09:15:00 | 1070.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 13:15:00 | 1124.75 | 2024-11-13 09:15:00 | 1068.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 14:15:00 | 1121.10 | 2024-11-13 09:15:00 | 1065.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 1127.25 | 2024-11-14 11:15:00 | 1089.85 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2024-11-12 13:15:00 | 1124.75 | 2024-11-14 11:15:00 | 1089.85 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2024-11-12 14:15:00 | 1121.10 | 2024-11-14 11:15:00 | 1089.85 | STOP_HIT | 0.50 | 2.79% |
| BUY | retest1 | 2024-11-25 09:15:00 | 1214.50 | 2024-11-29 09:15:00 | 1219.35 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2024-11-27 10:15:00 | 1229.35 | 2024-12-09 12:15:00 | 1280.80 | STOP_HIT | 1.00 | 4.19% |
| BUY | retest2 | 2024-11-27 12:30:00 | 1230.15 | 2024-12-09 12:15:00 | 1280.80 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2024-11-28 09:15:00 | 1245.15 | 2024-12-09 12:15:00 | 1280.80 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2024-11-29 10:45:00 | 1232.60 | 2024-12-09 12:15:00 | 1280.80 | STOP_HIT | 1.00 | 3.91% |
| BUY | retest2 | 2024-12-03 09:15:00 | 1256.80 | 2024-12-09 12:15:00 | 1280.80 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2024-12-19 12:15:00 | 1276.70 | 2024-12-19 14:15:00 | 1304.45 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-12-26 10:45:00 | 1243.00 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-12-26 11:15:00 | 1243.65 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-26 13:30:00 | 1241.95 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-26 15:15:00 | 1238.25 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-30 11:45:00 | 1234.90 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-12-30 13:30:00 | 1233.15 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-01-01 10:15:00 | 1233.35 | 2025-01-01 10:15:00 | 1251.95 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-08 11:45:00 | 1228.30 | 2025-01-10 09:15:00 | 1166.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:45:00 | 1228.30 | 2025-01-13 11:15:00 | 1105.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 1046.05 | 2025-01-28 09:15:00 | 993.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 1046.05 | 2025-01-28 13:15:00 | 1020.00 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1136.00 | 2025-02-11 09:15:00 | 1089.84 | PARTIAL | 0.50 | 4.06% |
| SELL | retest2 | 2025-02-07 12:00:00 | 1147.20 | 2025-02-11 09:15:00 | 1091.45 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1148.90 | 2025-02-11 12:15:00 | 1079.20 | PARTIAL | 0.50 | 6.07% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1136.00 | 2025-02-12 09:15:00 | 1032.48 | TARGET_HIT | 0.50 | 9.11% |
| SELL | retest2 | 2025-02-07 12:00:00 | 1147.20 | 2025-02-12 09:15:00 | 1034.01 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1148.90 | 2025-02-12 14:15:00 | 1022.40 | TARGET_HIT | 0.50 | 11.01% |
| BUY | retest2 | 2025-02-21 09:15:00 | 1033.90 | 2025-02-21 13:15:00 | 1008.05 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-02-21 09:45:00 | 1036.80 | 2025-02-21 13:15:00 | 1008.05 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-02-21 11:30:00 | 1033.70 | 2025-02-21 13:15:00 | 1008.05 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-02-28 12:00:00 | 959.50 | 2025-03-05 11:15:00 | 966.20 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-02-28 15:00:00 | 942.55 | 2025-03-05 11:15:00 | 966.20 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-03-05 09:30:00 | 953.20 | 2025-03-05 11:15:00 | 966.20 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-07 09:15:00 | 976.15 | 2025-03-07 15:15:00 | 952.50 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-03-10 11:00:00 | 967.35 | 2025-03-10 15:15:00 | 946.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-03-10 14:45:00 | 961.60 | 2025-03-10 15:15:00 | 946.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-03-18 12:30:00 | 924.30 | 2025-03-19 14:15:00 | 948.80 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-03-26 09:15:00 | 997.35 | 2025-03-26 15:15:00 | 975.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-03-26 10:00:00 | 990.50 | 2025-03-26 15:15:00 | 975.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-04-09 10:00:00 | 905.00 | 2025-04-11 13:15:00 | 934.70 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-09 10:30:00 | 909.20 | 2025-04-11 13:15:00 | 934.70 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-09 15:00:00 | 909.00 | 2025-04-11 13:15:00 | 934.70 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1005.20 | 2025-05-07 14:15:00 | 1018.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-05-06 13:15:00 | 1008.70 | 2025-05-07 14:15:00 | 1018.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-05-07 13:15:00 | 1007.60 | 2025-05-07 14:15:00 | 1018.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-05-07 14:15:00 | 1008.90 | 2025-05-07 14:15:00 | 1018.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-19 13:00:00 | 1124.10 | 2025-05-22 15:15:00 | 1100.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-05-27 13:30:00 | 1077.50 | 2025-05-29 09:15:00 | 1095.30 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-30 11:45:00 | 1100.00 | 2025-06-03 10:15:00 | 1210.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1107.80 | 2025-06-03 11:15:00 | 1218.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1213.30 | 2025-06-19 10:15:00 | 1152.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1213.30 | 2025-06-20 09:15:00 | 1156.20 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest1 | 2025-07-11 09:15:00 | 1070.20 | 2025-07-14 09:15:00 | 1085.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-18 14:15:00 | 1124.70 | 2025-07-18 14:15:00 | 1109.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-04 09:15:00 | 979.90 | 2025-08-12 12:15:00 | 965.40 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-08-04 10:45:00 | 980.20 | 2025-08-12 12:15:00 | 965.40 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2025-08-05 10:15:00 | 975.20 | 2025-08-12 12:15:00 | 965.40 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-08-05 12:45:00 | 980.50 | 2025-08-12 12:15:00 | 965.40 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-08-29 09:15:00 | 938.10 | 2025-09-03 12:15:00 | 940.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-09-08 13:15:00 | 912.95 | 2025-09-10 12:15:00 | 931.60 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-08 13:45:00 | 912.25 | 2025-09-10 12:15:00 | 931.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-09-09 09:45:00 | 907.60 | 2025-09-10 12:15:00 | 931.60 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-09-09 11:15:00 | 912.85 | 2025-09-10 12:15:00 | 931.60 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-09-17 09:15:00 | 963.65 | 2025-09-19 09:15:00 | 948.05 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-09-17 11:45:00 | 957.35 | 2025-09-19 09:15:00 | 948.05 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-18 09:45:00 | 956.80 | 2025-09-19 09:15:00 | 948.05 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-26 09:15:00 | 908.40 | 2025-10-07 09:15:00 | 910.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-10-10 10:15:00 | 946.45 | 2025-10-14 11:15:00 | 913.50 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-10-23 12:15:00 | 1004.30 | 2025-11-04 10:15:00 | 1028.00 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2025-10-23 14:30:00 | 1004.50 | 2025-11-04 10:15:00 | 1028.00 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1004.45 | 2025-11-04 10:15:00 | 1028.00 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-10-24 10:30:00 | 1003.55 | 2025-11-04 10:15:00 | 1028.00 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1020.50 | 2025-11-04 10:15:00 | 1028.00 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest1 | 2025-11-11 09:30:00 | 973.30 | 2025-11-17 09:15:00 | 959.40 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2025-11-12 13:00:00 | 959.50 | 2025-11-21 09:15:00 | 911.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 13:30:00 | 955.90 | 2025-11-21 09:15:00 | 910.67 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-11-12 14:15:00 | 957.20 | 2025-11-21 10:15:00 | 908.10 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-11-12 15:00:00 | 958.60 | 2025-11-21 10:15:00 | 909.34 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-11-17 14:00:00 | 944.30 | 2025-11-24 09:15:00 | 897.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 937.80 | 2025-11-24 10:15:00 | 890.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 13:00:00 | 959.50 | 2025-11-24 15:15:00 | 863.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 13:30:00 | 955.90 | 2025-11-24 15:15:00 | 860.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 957.20 | 2025-11-24 15:15:00 | 861.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 958.60 | 2025-11-24 15:15:00 | 862.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 14:00:00 | 944.30 | 2025-11-25 12:15:00 | 885.90 | STOP_HIT | 0.50 | 6.18% |
| SELL | retest2 | 2025-11-18 09:15:00 | 937.80 | 2025-11-25 12:15:00 | 885.90 | STOP_HIT | 0.50 | 5.53% |
| BUY | retest2 | 2025-12-23 13:30:00 | 879.90 | 2025-12-29 12:15:00 | 871.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-24 14:30:00 | 877.45 | 2025-12-29 12:15:00 | 871.20 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-26 09:15:00 | 883.70 | 2025-12-29 12:15:00 | 871.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-01 11:45:00 | 889.00 | 2026-01-07 10:15:00 | 890.05 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-01-02 10:00:00 | 886.55 | 2026-01-07 10:15:00 | 890.05 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-01-05 09:30:00 | 886.25 | 2026-01-07 12:15:00 | 885.85 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-01-05 10:00:00 | 887.65 | 2026-01-07 12:15:00 | 885.85 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-01-07 09:15:00 | 899.70 | 2026-01-07 12:15:00 | 885.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-07 09:45:00 | 898.65 | 2026-01-07 12:15:00 | 885.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-08 10:45:00 | 879.50 | 2026-01-16 09:15:00 | 837.14 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-01-08 12:45:00 | 881.20 | 2026-01-16 13:15:00 | 835.52 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-08 10:45:00 | 879.50 | 2026-01-20 09:15:00 | 791.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 12:45:00 | 881.20 | 2026-01-20 09:15:00 | 793.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-06 10:00:00 | 775.95 | 2026-02-06 12:15:00 | 764.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-02-12 13:30:00 | 794.85 | 2026-02-13 09:15:00 | 769.00 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-02-27 10:45:00 | 698.50 | 2026-03-04 09:15:00 | 663.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 698.50 | 2026-03-05 09:15:00 | 662.55 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest1 | 2026-03-16 09:15:00 | 638.35 | 2026-03-16 14:15:00 | 658.60 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-03-17 09:15:00 | 642.50 | 2026-03-18 10:15:00 | 670.15 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-04-10 09:30:00 | 721.80 | 2026-04-16 09:15:00 | 787.11 | TARGET_HIT | 1.00 | 9.05% |
| BUY | retest2 | 2026-04-13 09:45:00 | 715.55 | 2026-04-21 09:15:00 | 793.98 | TARGET_HIT | 1.00 | 10.96% |
| BUY | retest1 | 2026-04-29 09:15:00 | 802.90 | 2026-04-30 09:15:00 | 788.55 | STOP_HIT | 1.00 | -1.79% |
