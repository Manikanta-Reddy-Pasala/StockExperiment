# Aditya Infotech Ltd. (CPPLUS)

## Backtest Summary

- **Window:** 2025-08-05 09:15:00 → 2026-05-11 15:15:00 (1304 bars)
- **Last close:** 2515.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 50 |
| ALERT1 | 33 |
| ALERT2 | 33 |
| ALERT2_SKIP | 21 |
| ALERT3 | 99 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 40 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 20
- **Target hits / Stop hits / Partials:** 6 / 35 / 13
- **Avg / median % per leg:** 2.51% / 3.57%
- **Sum % (uncompounded):** 135.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 8 | 30.8% | 5 | 21 | 0 | 0.91% | 23.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| BUY @ 3rd Alert (retest2) | 25 | 8 | 32.0% | 5 | 20 | 0 | 1.02% | 25.5% |
| SELL (all) | 28 | 26 | 92.9% | 1 | 14 | 13 | 3.99% | 111.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 26 | 92.9% | 1 | 14 | 13 | 3.99% | 111.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| retest2 (combined) | 53 | 34 | 64.2% | 6 | 34 | 13 | 2.59% | 137.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1050.00 | 1085.53 | 1090.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1047.00 | 1060.73 | 1066.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1074.00 | 1063.38 | 1066.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1074.00 | 1063.38 | 1066.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1074.00 | 1063.38 | 1066.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 1071.25 | 1063.38 | 1066.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1093.00 | 1069.30 | 1069.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 1093.00 | 1069.30 | 1069.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 1091.00 | 1073.64 | 1071.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1114.45 | 1090.20 | 1081.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 1246.10 | 1247.57 | 1195.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:45:00 | 1254.70 | 1247.57 | 1195.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1303.80 | 1313.88 | 1290.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1300.00 | 1313.88 | 1290.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1289.65 | 1305.42 | 1293.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 1289.65 | 1305.42 | 1293.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1282.80 | 1300.90 | 1292.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1282.80 | 1300.90 | 1292.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1272.00 | 1295.12 | 1290.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1249.10 | 1295.12 | 1290.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1249.00 | 1285.89 | 1286.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 09:15:00 | 1229.70 | 1257.39 | 1265.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 1253.00 | 1245.01 | 1255.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 14:15:00 | 1253.00 | 1245.01 | 1255.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1253.00 | 1245.01 | 1255.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1253.00 | 1245.01 | 1255.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1255.20 | 1247.05 | 1255.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1282.90 | 1247.05 | 1255.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1310.60 | 1259.76 | 1260.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1310.60 | 1259.76 | 1260.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1310.60 | 1269.93 | 1264.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1322.50 | 1302.32 | 1285.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1299.80 | 1301.81 | 1287.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1299.80 | 1301.81 | 1287.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1290.90 | 1298.77 | 1288.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1290.90 | 1298.77 | 1288.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1290.90 | 1297.19 | 1288.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 1285.60 | 1297.19 | 1288.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1376.10 | 1312.97 | 1296.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1377.10 | 1325.60 | 1303.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 1388.60 | 1374.13 | 1344.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 1382.10 | 1378.56 | 1354.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1386.00 | 1373.46 | 1359.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1386.20 | 1376.01 | 1362.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1344.80 | 1360.48 | 1361.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 1344.80 | 1360.48 | 1361.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 1333.00 | 1344.84 | 1351.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 1350.00 | 1343.58 | 1349.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 1350.00 | 1343.58 | 1349.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1350.00 | 1343.58 | 1349.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1350.00 | 1343.58 | 1349.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1348.00 | 1344.46 | 1349.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1392.00 | 1344.46 | 1349.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1407.90 | 1357.15 | 1354.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 1440.00 | 1396.04 | 1378.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 15:15:00 | 1440.00 | 1440.76 | 1423.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 1438.50 | 1440.76 | 1423.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1417.50 | 1432.52 | 1423.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 1417.50 | 1432.52 | 1423.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1418.00 | 1429.62 | 1423.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 1417.00 | 1429.62 | 1423.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1422.00 | 1427.18 | 1423.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1422.00 | 1427.18 | 1423.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1431.10 | 1427.96 | 1423.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1448.10 | 1427.96 | 1423.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 1421.60 | 1440.60 | 1441.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1421.60 | 1440.60 | 1441.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 1410.20 | 1434.52 | 1439.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 10:15:00 | 1418.70 | 1415.23 | 1426.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 10:30:00 | 1418.70 | 1415.23 | 1426.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1420.30 | 1416.92 | 1423.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:30:00 | 1421.70 | 1416.92 | 1423.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1421.50 | 1417.83 | 1423.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 1408.00 | 1417.83 | 1423.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1407.00 | 1415.67 | 1421.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:00:00 | 1396.10 | 1408.47 | 1415.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 1326.29 | 1345.02 | 1369.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-26 13:15:00 | 1256.49 | 1292.81 | 1329.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 1317.70 | 1299.20 | 1297.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 1321.00 | 1303.56 | 1299.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 1409.00 | 1413.19 | 1397.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 1409.00 | 1413.19 | 1397.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1393.00 | 1408.00 | 1398.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1391.30 | 1408.00 | 1398.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1386.00 | 1403.60 | 1397.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1379.00 | 1403.60 | 1397.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1377.00 | 1398.28 | 1395.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1377.00 | 1398.28 | 1395.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1342.80 | 1387.18 | 1390.42 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 1392.90 | 1381.09 | 1380.93 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 1367.60 | 1379.29 | 1380.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 1361.00 | 1375.63 | 1378.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 1358.50 | 1353.99 | 1364.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 1358.50 | 1353.99 | 1364.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1358.50 | 1353.99 | 1364.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 1358.50 | 1353.99 | 1364.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1369.60 | 1356.31 | 1363.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1369.60 | 1356.31 | 1363.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1376.80 | 1360.41 | 1364.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1374.10 | 1360.41 | 1364.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1405.70 | 1374.62 | 1370.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1428.80 | 1390.20 | 1378.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 1380.40 | 1392.44 | 1383.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1380.40 | 1392.44 | 1383.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1380.40 | 1392.44 | 1383.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1380.40 | 1392.44 | 1383.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1382.20 | 1390.39 | 1382.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 1373.60 | 1390.39 | 1382.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1381.80 | 1388.67 | 1382.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 1378.70 | 1388.67 | 1382.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1374.00 | 1385.74 | 1382.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1374.00 | 1385.74 | 1382.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1383.70 | 1385.33 | 1382.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 1397.80 | 1387.71 | 1383.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 1382.40 | 1406.91 | 1408.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 1382.40 | 1406.91 | 1408.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 1377.60 | 1401.05 | 1405.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 1386.60 | 1385.66 | 1395.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 11:00:00 | 1386.60 | 1385.66 | 1395.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1391.00 | 1386.73 | 1394.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 1391.00 | 1386.73 | 1394.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1381.30 | 1385.64 | 1393.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:45:00 | 1376.10 | 1381.56 | 1388.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 1375.80 | 1376.44 | 1383.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 1372.00 | 1375.55 | 1382.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 1378.80 | 1376.28 | 1379.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1383.70 | 1377.76 | 1379.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 1383.70 | 1377.76 | 1379.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1380.00 | 1378.21 | 1379.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1374.90 | 1378.21 | 1379.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 14:15:00 | 1307.29 | 1336.10 | 1351.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 14:15:00 | 1307.01 | 1336.10 | 1351.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 14:15:00 | 1309.86 | 1336.10 | 1351.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 14:15:00 | 1306.15 | 1336.10 | 1351.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1338.00 | 1323.72 | 1336.84 | SL hit (close>ema200) qty=0.50 sl=1323.72 alert=retest2 |

### Cycle 14 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 1403.80 | 1345.74 | 1338.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1483.50 | 1403.59 | 1373.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 11:15:00 | 1500.00 | 1501.32 | 1456.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:30:00 | 1495.90 | 1501.32 | 1456.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1495.00 | 1494.06 | 1463.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 1503.80 | 1498.30 | 1470.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 1505.00 | 1502.57 | 1477.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 1500.70 | 1502.84 | 1482.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1511.00 | 1502.27 | 1483.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1485.80 | 1499.09 | 1488.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 1485.80 | 1499.09 | 1488.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1505.90 | 1500.45 | 1489.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1592.00 | 1498.90 | 1491.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-14 09:15:00 | 1654.18 | 1624.69 | 1572.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1668.00 | 1684.20 | 1684.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1658.90 | 1672.11 | 1677.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1646.30 | 1645.47 | 1659.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1646.30 | 1645.47 | 1659.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1646.30 | 1645.47 | 1659.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1666.80 | 1645.47 | 1659.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1651.00 | 1646.57 | 1658.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:45:00 | 1653.00 | 1646.57 | 1658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1600.00 | 1628.45 | 1644.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 1598.70 | 1628.45 | 1644.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:45:00 | 1597.20 | 1622.44 | 1640.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 1596.90 | 1599.57 | 1618.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:30:00 | 1592.20 | 1594.16 | 1612.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:15:00 | 1518.76 | 1543.14 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:15:00 | 1517.34 | 1543.14 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:15:00 | 1517.06 | 1543.14 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:15:00 | 1512.59 | 1543.14 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 1534.90 | 1530.93 | 1554.42 | SL hit (close>ema200) qty=0.50 sl=1530.93 alert=retest2 |

### Cycle 16 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 1523.60 | 1494.07 | 1490.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1536.80 | 1502.62 | 1494.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1545.00 | 1562.18 | 1547.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1545.00 | 1562.18 | 1547.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1545.00 | 1562.18 | 1547.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1545.00 | 1562.18 | 1547.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1539.90 | 1557.73 | 1547.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 1541.50 | 1557.73 | 1547.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1538.80 | 1545.03 | 1543.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1547.00 | 1545.03 | 1543.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1533.30 | 1542.68 | 1542.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1533.30 | 1542.68 | 1542.89 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 1548.80 | 1542.98 | 1542.93 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1537.40 | 1541.86 | 1542.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1526.40 | 1538.77 | 1540.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1532.00 | 1529.23 | 1534.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1532.00 | 1529.23 | 1534.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1532.00 | 1529.23 | 1534.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1532.00 | 1529.23 | 1534.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1539.10 | 1528.44 | 1531.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 1535.20 | 1528.44 | 1531.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1533.60 | 1529.48 | 1531.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:15:00 | 1541.80 | 1529.48 | 1531.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1542.30 | 1532.04 | 1532.90 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 1542.40 | 1534.11 | 1533.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1549.00 | 1537.09 | 1535.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1577.90 | 1583.83 | 1568.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 1580.70 | 1583.83 | 1568.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1571.80 | 1579.52 | 1568.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 1576.10 | 1579.52 | 1568.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1572.70 | 1578.15 | 1569.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 1569.50 | 1578.15 | 1569.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1554.70 | 1573.46 | 1567.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 1554.70 | 1573.46 | 1567.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1552.00 | 1569.17 | 1566.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1557.00 | 1569.17 | 1566.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 1540.80 | 1560.83 | 1562.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 1516.30 | 1548.01 | 1556.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1491.30 | 1477.56 | 1490.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1491.30 | 1477.56 | 1490.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1491.30 | 1477.56 | 1490.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1491.30 | 1477.56 | 1490.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1468.00 | 1475.65 | 1488.03 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1525.90 | 1496.74 | 1493.03 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 1476.30 | 1499.05 | 1500.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 11:15:00 | 1472.80 | 1493.80 | 1498.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 1439.60 | 1389.09 | 1403.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1439.60 | 1389.09 | 1403.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1439.60 | 1389.09 | 1403.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 1439.60 | 1389.09 | 1403.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1408.80 | 1393.03 | 1403.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1380.50 | 1402.01 | 1404.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 1413.90 | 1404.00 | 1403.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1413.90 | 1404.00 | 1403.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1437.00 | 1416.63 | 1410.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 1422.70 | 1423.58 | 1415.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 1422.70 | 1423.58 | 1415.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1407.80 | 1420.42 | 1414.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 1407.80 | 1420.42 | 1414.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1421.90 | 1420.72 | 1415.44 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 1395.60 | 1410.79 | 1411.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1384.30 | 1395.28 | 1400.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 10:15:00 | 1397.50 | 1393.34 | 1398.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 10:15:00 | 1397.50 | 1393.34 | 1398.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1397.50 | 1393.34 | 1398.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 1397.50 | 1393.34 | 1398.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1382.90 | 1391.25 | 1397.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 1383.20 | 1391.25 | 1397.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1389.80 | 1386.93 | 1392.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 1389.80 | 1386.93 | 1392.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1409.20 | 1391.39 | 1393.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1409.20 | 1391.39 | 1393.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 11:15:00 | 1416.00 | 1396.31 | 1395.87 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 1375.00 | 1394.35 | 1395.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 1361.90 | 1387.86 | 1392.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1394.10 | 1382.40 | 1386.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1394.10 | 1382.40 | 1386.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1394.10 | 1382.40 | 1386.97 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1412.00 | 1391.02 | 1389.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1428.50 | 1398.51 | 1393.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 1396.20 | 1413.01 | 1404.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 1396.20 | 1413.01 | 1404.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1396.20 | 1413.01 | 1404.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1396.20 | 1413.01 | 1404.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1396.00 | 1409.61 | 1403.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:15:00 | 1395.00 | 1409.61 | 1403.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1395.00 | 1406.69 | 1402.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:15:00 | 1390.00 | 1406.69 | 1402.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1392.60 | 1401.20 | 1400.77 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1383.10 | 1397.58 | 1399.17 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 1407.80 | 1397.51 | 1396.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1410.10 | 1400.02 | 1397.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1391.10 | 1400.97 | 1398.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1391.10 | 1400.97 | 1398.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1391.10 | 1400.97 | 1398.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 1409.00 | 1403.61 | 1401.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 1384.40 | 1400.55 | 1400.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 1384.40 | 1400.55 | 1400.73 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 1425.90 | 1403.05 | 1400.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 1453.50 | 1418.25 | 1408.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1494.10 | 1515.81 | 1489.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1494.10 | 1515.81 | 1489.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1460.10 | 1504.67 | 1486.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 1462.20 | 1504.67 | 1486.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1472.40 | 1498.21 | 1485.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:00:00 | 1479.50 | 1489.78 | 1483.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 1485.90 | 1486.57 | 1482.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 15:15:00 | 1539.00 | 1549.89 | 1551.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 1539.00 | 1549.89 | 1551.01 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 09:15:00 | 1643.50 | 1568.61 | 1559.42 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 1570.00 | 1580.63 | 1581.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 1564.60 | 1575.95 | 1579.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 1579.80 | 1576.09 | 1578.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1579.80 | 1576.09 | 1578.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1579.80 | 1576.09 | 1578.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 1562.50 | 1572.70 | 1576.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1484.38 | 1503.93 | 1523.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1485.10 | 1474.98 | 1498.55 | SL hit (close>ema200) qty=0.50 sl=1474.98 alert=retest2 |

### Cycle 36 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1592.50 | 1515.40 | 1512.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 1603.00 | 1532.92 | 1520.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1710.80 | 1718.83 | 1681.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1710.80 | 1718.83 | 1681.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1704.30 | 1715.93 | 1683.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 1689.00 | 1715.93 | 1683.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1687.10 | 1710.81 | 1686.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 1688.50 | 1710.81 | 1686.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1687.20 | 1706.09 | 1686.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 1687.20 | 1706.09 | 1686.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1687.00 | 1702.27 | 1686.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 1687.00 | 1702.27 | 1686.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 1683.00 | 1698.42 | 1686.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 1718.70 | 1698.42 | 1686.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 1665.90 | 1692.76 | 1692.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 11:15:00 | 1665.90 | 1692.76 | 1692.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1640.20 | 1670.35 | 1680.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1670.00 | 1652.30 | 1663.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1670.00 | 1652.30 | 1663.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1670.00 | 1652.30 | 1663.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1670.00 | 1652.30 | 1663.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1660.70 | 1653.98 | 1663.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 1637.80 | 1651.35 | 1660.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 1555.91 | 1593.03 | 1619.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 1583.00 | 1575.33 | 1601.03 | SL hit (close>ema200) qty=0.50 sl=1575.33 alert=retest2 |

### Cycle 38 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1634.50 | 1603.97 | 1600.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1681.00 | 1623.25 | 1611.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 14:15:00 | 1672.70 | 1681.24 | 1660.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1672.70 | 1681.24 | 1660.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1672.70 | 1681.24 | 1660.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1672.70 | 1681.24 | 1660.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1665.00 | 1677.99 | 1661.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1734.00 | 1677.99 | 1661.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 12:15:00 | 1652.00 | 1684.62 | 1681.19 | SL hit (close<static) qty=1.00 sl=1660.00 alert=retest2 |

### Cycle 39 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1639.70 | 1671.84 | 1675.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 1635.00 | 1664.47 | 1672.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1663.50 | 1655.28 | 1664.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1663.50 | 1655.28 | 1664.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1663.50 | 1655.28 | 1664.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1667.50 | 1655.28 | 1664.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1677.20 | 1659.66 | 1665.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1677.20 | 1659.66 | 1665.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1675.90 | 1662.91 | 1666.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 1679.00 | 1662.91 | 1666.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1679.00 | 1666.13 | 1667.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1731.80 | 1666.13 | 1667.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1832.00 | 1699.30 | 1682.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 1867.60 | 1796.93 | 1748.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 1826.10 | 1830.48 | 1786.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1788.80 | 1822.55 | 1790.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1788.80 | 1822.55 | 1790.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 1759.30 | 1822.55 | 1790.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1793.50 | 1816.74 | 1790.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 11:45:00 | 1795.20 | 1812.55 | 1791.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:00:00 | 1800.30 | 1810.10 | 1792.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 1796.90 | 1811.07 | 1801.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 1800.00 | 1803.01 | 1799.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1791.90 | 1800.30 | 1798.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 1773.80 | 1800.30 | 1798.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-02 10:15:00 | 1765.80 | 1793.40 | 1795.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1765.80 | 1793.40 | 1795.74 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 1820.50 | 1800.88 | 1798.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1850.60 | 1827.49 | 1817.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 1893.60 | 1901.42 | 1881.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1893.60 | 1901.42 | 1881.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1893.60 | 1901.42 | 1881.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 1916.60 | 1898.38 | 1888.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 13:15:00 | 2108.26 | 2040.83 | 2007.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 2182.00 | 2213.67 | 2215.03 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2259.00 | 2222.74 | 2219.03 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 2201.50 | 2232.37 | 2233.31 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 2273.60 | 2237.66 | 2235.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 2294.40 | 2259.92 | 2247.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 2325.50 | 2326.72 | 2297.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 14:15:00 | 2341.50 | 2326.72 | 2297.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2316.10 | 2322.44 | 2300.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:30:00 | 2323.00 | 2317.53 | 2304.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 2324.70 | 2317.53 | 2304.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 2300.80 | 2314.18 | 2304.24 | SL hit (close<ema400) qty=1.00 sl=2304.24 alert=retest1 |

### Cycle 47 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 2288.20 | 2302.42 | 2303.20 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 2330.00 | 2307.94 | 2305.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 2364.10 | 2321.90 | 2312.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 2392.00 | 2427.56 | 2399.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 2392.00 | 2427.56 | 2399.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2392.00 | 2427.56 | 2399.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 2392.00 | 2427.56 | 2399.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 2390.30 | 2420.11 | 2398.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 2335.00 | 2420.11 | 2398.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 2262.20 | 2388.53 | 2386.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 2262.20 | 2388.53 | 2386.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 2265.00 | 2363.82 | 2375.01 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 2491.20 | 2388.32 | 2382.36 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-04 09:15:00 | 1377.10 | 2025-09-09 10:15:00 | 1344.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-09-05 09:15:00 | 1388.60 | 2025-09-09 10:15:00 | 1344.80 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-09-05 12:00:00 | 1382.10 | 2025-09-09 10:15:00 | 1344.80 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-09-08 09:15:00 | 1386.00 | 2025-09-09 10:15:00 | 1344.80 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1448.10 | 2025-09-19 11:15:00 | 1421.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-23 15:00:00 | 1396.10 | 2025-09-25 14:15:00 | 1326.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:00:00 | 1396.10 | 2025-09-26 13:15:00 | 1256.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-17 10:00:00 | 1397.80 | 2025-10-24 11:15:00 | 1382.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-28 09:45:00 | 1376.10 | 2025-10-31 14:15:00 | 1307.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1375.80 | 2025-10-31 14:15:00 | 1307.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 15:00:00 | 1372.00 | 2025-10-31 14:15:00 | 1309.86 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2025-10-29 14:00:00 | 1378.80 | 2025-10-31 14:15:00 | 1306.15 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-10-28 09:45:00 | 1376.10 | 2025-11-03 13:15:00 | 1338.00 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1375.80 | 2025-11-03 13:15:00 | 1338.00 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-10-28 15:00:00 | 1372.00 | 2025-11-03 13:15:00 | 1338.00 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2025-10-29 14:00:00 | 1378.80 | 2025-11-03 13:15:00 | 1338.00 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1374.90 | 2025-11-06 10:15:00 | 1403.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-11-11 09:45:00 | 1503.80 | 2025-11-14 09:15:00 | 1654.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 12:00:00 | 1505.00 | 2025-11-14 09:15:00 | 1655.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 13:45:00 | 1500.70 | 2025-11-14 09:15:00 | 1650.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 15:15:00 | 1511.00 | 2025-11-14 09:15:00 | 1662.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1592.00 | 2025-11-21 10:15:00 | 1668.00 | STOP_HIT | 1.00 | 4.77% |
| SELL | retest2 | 2025-11-26 10:15:00 | 1598.70 | 2025-12-01 10:15:00 | 1518.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 10:45:00 | 1597.20 | 2025-12-01 10:15:00 | 1517.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:00:00 | 1596.90 | 2025-12-01 10:15:00 | 1517.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:30:00 | 1592.20 | 2025-12-01 10:15:00 | 1512.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 10:15:00 | 1598.70 | 2025-12-01 13:15:00 | 1534.90 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-11-26 10:45:00 | 1597.20 | 2025-12-01 13:15:00 | 1534.90 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-11-27 10:00:00 | 1596.90 | 2025-12-01 13:15:00 | 1534.90 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-11-27 11:30:00 | 1592.20 | 2025-12-01 13:15:00 | 1534.90 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1534.80 | 2025-12-09 09:15:00 | 1458.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:45:00 | 1533.90 | 2025-12-09 09:15:00 | 1457.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1534.80 | 2025-12-09 14:15:00 | 1482.90 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2025-12-08 10:45:00 | 1533.90 | 2025-12-09 14:15:00 | 1482.90 | STOP_HIT | 0.50 | 3.32% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1547.00 | 2025-12-17 09:15:00 | 1533.30 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1380.50 | 2026-01-12 13:15:00 | 1413.90 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-01-30 10:30:00 | 1409.00 | 2026-01-30 14:15:00 | 1384.40 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-05 14:00:00 | 1479.50 | 2026-02-12 15:15:00 | 1539.00 | STOP_HIT | 1.00 | 4.02% |
| BUY | retest2 | 2026-02-05 15:15:00 | 1485.90 | 2026-02-12 15:15:00 | 1539.00 | STOP_HIT | 1.00 | 3.57% |
| SELL | retest2 | 2026-02-19 14:45:00 | 1562.50 | 2026-02-24 09:15:00 | 1484.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 14:45:00 | 1562.50 | 2026-02-24 14:15:00 | 1485.10 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2026-03-05 09:15:00 | 1718.70 | 2026-03-06 11:15:00 | 1665.90 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-10 12:30:00 | 1637.80 | 2026-03-12 09:15:00 | 1555.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:30:00 | 1637.80 | 2026-03-12 13:15:00 | 1583.00 | STOP_HIT | 0.50 | 3.35% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1734.00 | 2026-03-23 12:15:00 | 1652.00 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-03-30 11:45:00 | 1795.20 | 2026-04-02 10:15:00 | 1765.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-03-30 13:00:00 | 1800.30 | 2026-04-02 10:15:00 | 1765.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-04-01 13:15:00 | 1796.90 | 2026-04-02 10:15:00 | 1765.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-01 15:15:00 | 1800.00 | 2026-04-02 10:15:00 | 1765.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-13 09:45:00 | 1916.60 | 2026-04-16 13:15:00 | 2108.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2026-04-28 14:15:00 | 2341.50 | 2026-04-29 13:15:00 | 2300.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-29 12:30:00 | 2323.00 | 2026-04-30 09:15:00 | 2291.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-29 13:00:00 | 2324.70 | 2026-04-30 09:15:00 | 2291.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-04-29 14:30:00 | 2345.00 | 2026-04-30 09:15:00 | 2291.00 | STOP_HIT | 1.00 | -2.30% |
