# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1105.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 89 |
| ALERT2 | 85 |
| ALERT2_SKIP | 37 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 124 |
| PARTIAL | 29 |
| TARGET_HIT | 21 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 155 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 93 / 62
- **Target hits / Stop hits / Partials:** 21 / 105 / 29
- **Avg / median % per leg:** 2.41% / 1.63%
- **Sum % (uncompounded):** 372.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 26 | 44.8% | 6 | 52 | 0 | 1.01% | 58.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.08% | -4.2% |
| BUY @ 3rd Alert (retest2) | 56 | 26 | 46.4% | 6 | 50 | 0 | 1.12% | 62.9% |
| SELL (all) | 97 | 67 | 69.1% | 15 | 53 | 29 | 3.24% | 314.2% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.26% | 6.3% |
| SELL @ 3rd Alert (retest2) | 92 | 63 | 68.5% | 15 | 49 | 28 | 3.35% | 307.9% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 6 | 1 | 0.30% | 2.1% |
| retest2 (combined) | 148 | 89 | 60.1% | 21 | 99 | 28 | 2.51% | 370.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1154.85 | 1175.34 | 1176.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 13:15:00 | 1147.40 | 1161.78 | 1169.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 15:15:00 | 1164.95 | 1162.21 | 1168.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 09:15:00 | 1163.20 | 1162.21 | 1168.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1165.75 | 1162.92 | 1167.80 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 1184.85 | 1172.83 | 1171.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 1191.70 | 1176.61 | 1173.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 11:15:00 | 1297.05 | 1298.51 | 1279.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 12:00:00 | 1297.05 | 1298.51 | 1279.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1285.35 | 1297.81 | 1287.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 1285.35 | 1297.81 | 1287.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1277.70 | 1293.79 | 1286.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 1277.05 | 1293.79 | 1286.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1273.80 | 1289.79 | 1285.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 1273.80 | 1289.79 | 1285.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 1272.15 | 1281.68 | 1282.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 1263.00 | 1276.09 | 1279.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 11:15:00 | 1280.35 | 1276.85 | 1279.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 1280.35 | 1276.85 | 1279.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1280.35 | 1276.85 | 1279.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 1280.35 | 1276.85 | 1279.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 1281.75 | 1277.83 | 1279.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:30:00 | 1275.40 | 1277.88 | 1279.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1295.05 | 1280.53 | 1280.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 09:15:00 | 1295.05 | 1280.53 | 1280.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 10:15:00 | 1306.05 | 1285.64 | 1282.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 10:15:00 | 1298.50 | 1299.78 | 1292.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 11:00:00 | 1298.50 | 1299.78 | 1292.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1289.15 | 1297.65 | 1292.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 1289.15 | 1297.65 | 1292.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1282.45 | 1294.61 | 1291.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 1282.45 | 1294.61 | 1291.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 1283.00 | 1289.08 | 1289.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 1274.95 | 1286.25 | 1288.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1174.40 | 1170.46 | 1193.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1174.40 | 1170.46 | 1193.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1158.20 | 1168.01 | 1189.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:45:00 | 1154.00 | 1168.01 | 1189.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1184.25 | 1173.27 | 1188.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1188.25 | 1173.27 | 1188.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1182.95 | 1175.21 | 1188.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:30:00 | 1181.85 | 1175.21 | 1188.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1203.15 | 1181.76 | 1188.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 1216.50 | 1181.76 | 1188.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 1204.30 | 1194.38 | 1193.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 1215.00 | 1203.29 | 1198.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1227.40 | 1231.88 | 1220.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 1227.40 | 1231.88 | 1220.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1293.50 | 1303.93 | 1294.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 1293.50 | 1303.93 | 1294.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1294.85 | 1302.11 | 1294.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 1288.10 | 1302.11 | 1294.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1285.05 | 1298.70 | 1293.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 1285.05 | 1298.70 | 1293.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1284.65 | 1295.89 | 1292.68 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 1276.15 | 1290.20 | 1290.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 1266.00 | 1282.07 | 1286.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 10:15:00 | 1283.40 | 1282.33 | 1286.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 10:15:00 | 1283.40 | 1282.33 | 1286.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1283.40 | 1282.33 | 1286.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1283.40 | 1282.33 | 1286.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1296.35 | 1285.14 | 1287.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:45:00 | 1295.00 | 1285.14 | 1287.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 1317.45 | 1291.60 | 1289.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1369.50 | 1310.88 | 1299.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 1329.35 | 1342.36 | 1325.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 14:15:00 | 1329.35 | 1342.36 | 1325.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 1329.35 | 1342.36 | 1325.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 1329.35 | 1342.36 | 1325.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1344.80 | 1351.68 | 1339.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 1344.80 | 1351.68 | 1339.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1341.00 | 1349.54 | 1339.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1362.00 | 1349.54 | 1339.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 1362.80 | 1351.82 | 1341.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 1371.95 | 1354.65 | 1348.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-01 15:15:00 | 1498.20 | 1466.29 | 1446.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 1468.00 | 1484.09 | 1484.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 1433.40 | 1473.95 | 1479.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 1457.35 | 1450.96 | 1461.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 1457.35 | 1450.96 | 1461.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1457.35 | 1450.96 | 1461.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 1462.00 | 1450.96 | 1461.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1480.65 | 1456.90 | 1463.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 1480.65 | 1456.90 | 1463.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1487.70 | 1463.06 | 1465.87 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 1487.95 | 1468.04 | 1467.88 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 1435.05 | 1461.68 | 1465.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 1420.00 | 1447.98 | 1457.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 1435.00 | 1434.59 | 1446.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:45:00 | 1421.75 | 1430.73 | 1443.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 1350.66 | 1382.11 | 1400.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1380.95 | 1374.68 | 1387.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1380.95 | 1374.68 | 1387.80 | SL hit (close>ema200) qty=0.50 sl=1374.68 alert=retest1 |

### Cycle 12 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 1386.95 | 1377.25 | 1376.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 1394.55 | 1386.08 | 1381.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 1450.00 | 1453.09 | 1434.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:00:00 | 1450.00 | 1453.09 | 1434.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 1433.65 | 1448.60 | 1435.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 1433.65 | 1448.60 | 1435.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1443.95 | 1447.67 | 1436.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1454.60 | 1448.13 | 1437.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 1453.20 | 1449.29 | 1439.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1465.45 | 1444.54 | 1441.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 1452.20 | 1454.38 | 1449.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1447.15 | 1452.93 | 1448.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1447.15 | 1452.93 | 1448.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1449.65 | 1452.27 | 1448.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:30:00 | 1455.65 | 1460.63 | 1453.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 15:15:00 | 1460.00 | 1476.24 | 1477.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 1460.00 | 1476.24 | 1477.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1411.85 | 1454.71 | 1465.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 1423.10 | 1412.40 | 1429.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 1423.10 | 1412.40 | 1429.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 1418.00 | 1402.77 | 1413.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 1418.00 | 1402.77 | 1413.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1423.95 | 1407.01 | 1414.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 1423.95 | 1407.01 | 1414.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1456.40 | 1416.88 | 1418.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 1456.40 | 1416.88 | 1418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1455.95 | 1424.70 | 1421.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 1470.00 | 1439.33 | 1429.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 1460.40 | 1461.02 | 1450.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 1460.40 | 1461.02 | 1450.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1443.00 | 1455.78 | 1450.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 1454.85 | 1455.78 | 1450.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1415.95 | 1447.81 | 1447.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1415.95 | 1447.81 | 1447.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 1420.65 | 1442.38 | 1444.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 15:15:00 | 1408.05 | 1422.84 | 1432.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 14:15:00 | 1413.90 | 1404.79 | 1416.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 1413.90 | 1404.79 | 1416.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1410.05 | 1405.84 | 1416.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1387.25 | 1405.84 | 1416.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 12:15:00 | 1353.35 | 1350.60 | 1350.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 1353.35 | 1350.60 | 1350.58 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 1349.20 | 1350.32 | 1350.45 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 1352.65 | 1350.78 | 1350.65 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 1347.85 | 1350.18 | 1350.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 1343.65 | 1348.22 | 1349.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 10:15:00 | 1348.40 | 1346.82 | 1348.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 1348.40 | 1346.82 | 1348.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1348.40 | 1346.82 | 1348.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1348.40 | 1346.82 | 1348.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1350.80 | 1347.62 | 1348.56 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 13:15:00 | 1355.05 | 1349.96 | 1349.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 15:15:00 | 1359.60 | 1352.85 | 1350.96 | Break + close above crossover candle high |

### Cycle 21 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 1337.05 | 1349.69 | 1349.70 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1358.90 | 1349.58 | 1349.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 1376.80 | 1360.81 | 1355.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1447.80 | 1455.46 | 1436.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 1447.80 | 1455.46 | 1436.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1435.85 | 1451.75 | 1441.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 1435.85 | 1451.75 | 1441.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 1450.00 | 1451.40 | 1442.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 1421.00 | 1451.40 | 1442.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1432.20 | 1447.56 | 1441.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1424.60 | 1447.56 | 1441.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 1418.50 | 1438.21 | 1438.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 1414.15 | 1430.57 | 1434.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1436.00 | 1427.17 | 1431.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1436.00 | 1427.17 | 1431.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1436.00 | 1427.17 | 1431.62 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 1440.05 | 1432.16 | 1432.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 1446.60 | 1436.20 | 1434.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 11:15:00 | 1432.30 | 1436.81 | 1434.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 11:15:00 | 1432.30 | 1436.81 | 1434.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1432.30 | 1436.81 | 1434.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 1432.30 | 1436.81 | 1434.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1434.85 | 1436.42 | 1434.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 1437.30 | 1436.42 | 1434.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 1427.60 | 1434.66 | 1434.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 1428.50 | 1434.66 | 1434.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1441.50 | 1436.03 | 1434.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 1450.15 | 1436.03 | 1434.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1448.80 | 1440.93 | 1437.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:15:00 | 1452.15 | 1442.35 | 1438.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1449.75 | 1446.94 | 1442.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1460.45 | 1449.64 | 1444.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 15:00:00 | 1469.95 | 1460.46 | 1452.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1469.95 | 1464.26 | 1455.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 1470.05 | 1472.80 | 1464.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 1493.95 | 1519.60 | 1520.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 1493.95 | 1519.60 | 1520.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 11:15:00 | 1483.30 | 1496.32 | 1505.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 1466.95 | 1465.97 | 1479.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 1466.95 | 1465.97 | 1479.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 1463.00 | 1465.38 | 1477.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 1469.85 | 1465.38 | 1477.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1466.60 | 1465.62 | 1476.55 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 11:15:00 | 1495.00 | 1477.33 | 1477.17 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 1468.00 | 1477.10 | 1477.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 1445.00 | 1470.68 | 1474.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 12:15:00 | 1464.00 | 1462.91 | 1469.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 1464.00 | 1462.91 | 1469.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1422.50 | 1452.25 | 1462.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1402.85 | 1443.69 | 1457.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 1407.75 | 1437.90 | 1453.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1406.90 | 1423.10 | 1439.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:30:00 | 1405.00 | 1409.41 | 1422.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 1406.80 | 1407.63 | 1417.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:30:00 | 1400.00 | 1406.07 | 1416.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 09:45:00 | 1405.00 | 1405.08 | 1413.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1403.10 | 1405.08 | 1413.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 1404.20 | 1404.91 | 1412.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1406.85 | 1402.16 | 1407.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1400.80 | 1405.07 | 1406.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 1401.55 | 1403.07 | 1405.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 1401.05 | 1402.89 | 1404.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1400.00 | 1404.16 | 1405.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1386.20 | 1400.57 | 1403.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-15 15:15:00 | 1409.40 | 1402.02 | 1402.53 | SL hit (close>static) qty=1.00 sl=1408.20 alert=retest2 |

### Cycle 28 — BUY (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 14:15:00 | 1219.65 | 1211.41 | 1211.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 1229.30 | 1216.36 | 1213.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 15:15:00 | 1227.05 | 1227.15 | 1221.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:15:00 | 1225.90 | 1227.15 | 1221.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1228.05 | 1227.33 | 1222.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 1222.20 | 1227.33 | 1222.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1230.10 | 1233.04 | 1228.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:15:00 | 1222.80 | 1233.04 | 1228.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1225.70 | 1231.57 | 1228.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1219.05 | 1231.57 | 1228.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1226.90 | 1230.64 | 1227.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:30:00 | 1232.05 | 1229.97 | 1227.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 1223.50 | 1226.63 | 1226.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 1223.50 | 1226.63 | 1226.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 1215.05 | 1224.31 | 1225.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1150.50 | 1150.45 | 1167.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1150.50 | 1150.45 | 1167.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1151.00 | 1151.45 | 1159.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 1134.90 | 1147.44 | 1156.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 1131.40 | 1147.44 | 1156.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 13:15:00 | 1163.10 | 1154.08 | 1152.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 1163.10 | 1154.08 | 1152.86 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 1144.60 | 1152.89 | 1153.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 1141.15 | 1148.16 | 1150.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 1148.25 | 1143.99 | 1147.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 1148.25 | 1143.99 | 1147.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1148.25 | 1143.99 | 1147.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 1148.25 | 1143.99 | 1147.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1138.85 | 1142.96 | 1146.66 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 1170.90 | 1149.24 | 1148.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1237.00 | 1169.61 | 1158.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 1221.35 | 1223.28 | 1213.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:30:00 | 1222.15 | 1223.28 | 1213.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1216.30 | 1221.88 | 1213.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1212.90 | 1221.88 | 1213.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1210.00 | 1219.51 | 1213.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1210.00 | 1219.51 | 1213.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1207.15 | 1217.04 | 1212.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 1207.15 | 1217.04 | 1212.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 1201.75 | 1210.42 | 1210.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 10:15:00 | 1200.00 | 1208.33 | 1209.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 12:15:00 | 1207.90 | 1206.90 | 1208.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 12:15:00 | 1207.90 | 1206.90 | 1208.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1207.90 | 1206.90 | 1208.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 1206.50 | 1206.90 | 1208.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1207.90 | 1207.10 | 1208.61 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 1219.85 | 1210.81 | 1209.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1227.00 | 1217.33 | 1213.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 1218.90 | 1220.55 | 1216.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 12:00:00 | 1218.90 | 1220.55 | 1216.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1214.05 | 1218.95 | 1216.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1221.50 | 1217.31 | 1216.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 14:15:00 | 1212.30 | 1215.12 | 1215.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 1212.30 | 1215.12 | 1215.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 1202.40 | 1210.84 | 1213.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1181.90 | 1170.87 | 1182.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 1181.90 | 1170.87 | 1182.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1181.90 | 1170.87 | 1182.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 1181.90 | 1170.87 | 1182.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1184.45 | 1173.59 | 1182.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:15:00 | 1200.35 | 1173.59 | 1182.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1198.00 | 1178.47 | 1184.11 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 1217.10 | 1189.63 | 1188.42 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 1176.50 | 1195.32 | 1195.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1163.00 | 1180.61 | 1187.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 13:15:00 | 1144.00 | 1143.79 | 1150.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 14:00:00 | 1144.00 | 1143.79 | 1150.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 1144.20 | 1143.87 | 1150.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 1144.20 | 1143.87 | 1150.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1144.95 | 1144.27 | 1148.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1148.40 | 1144.27 | 1148.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1148.70 | 1145.06 | 1147.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1148.70 | 1145.06 | 1147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1149.00 | 1145.84 | 1148.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 1153.70 | 1145.84 | 1148.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1148.00 | 1146.28 | 1148.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 1153.05 | 1146.28 | 1148.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1152.85 | 1147.59 | 1148.48 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 1154.30 | 1149.93 | 1149.45 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 1147.00 | 1149.14 | 1149.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 1134.05 | 1146.12 | 1147.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 1115.80 | 1115.74 | 1123.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 15:15:00 | 1122.00 | 1117.80 | 1121.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1122.00 | 1117.80 | 1121.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1110.60 | 1117.80 | 1121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1128.00 | 1119.84 | 1121.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 1128.00 | 1119.84 | 1121.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1132.05 | 1122.28 | 1122.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:30:00 | 1127.55 | 1122.28 | 1122.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1146.45 | 1127.11 | 1124.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 12:15:00 | 1157.60 | 1133.21 | 1127.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 1138.80 | 1139.81 | 1132.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 15:00:00 | 1138.80 | 1139.81 | 1132.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1142.00 | 1140.25 | 1133.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 1126.85 | 1138.04 | 1132.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1135.00 | 1137.43 | 1132.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 1132.10 | 1137.43 | 1132.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 1133.40 | 1136.63 | 1132.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 1136.45 | 1136.63 | 1132.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1144.30 | 1138.16 | 1134.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 1146.40 | 1138.16 | 1134.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:00:00 | 1148.75 | 1140.28 | 1135.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:45:00 | 1147.20 | 1153.30 | 1150.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 1139.95 | 1148.04 | 1148.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 1139.95 | 1148.04 | 1148.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 12:15:00 | 1136.90 | 1144.20 | 1146.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 1145.00 | 1144.36 | 1146.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 1145.00 | 1144.36 | 1146.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1122.35 | 1138.72 | 1143.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:30:00 | 1117.90 | 1133.80 | 1140.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:30:00 | 1118.80 | 1130.78 | 1138.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 12:15:00 | 1062.01 | 1078.12 | 1097.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 12:15:00 | 1062.86 | 1078.12 | 1097.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 1080.50 | 1072.27 | 1084.77 | SL hit (close>ema200) qty=0.50 sl=1072.27 alert=retest2 |

### Cycle 42 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1069.80 | 1050.12 | 1047.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 1076.40 | 1060.98 | 1053.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 1060.00 | 1060.79 | 1054.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 1060.00 | 1060.79 | 1054.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1054.50 | 1059.02 | 1055.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 1054.50 | 1059.02 | 1055.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 1055.20 | 1058.26 | 1055.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:15:00 | 1052.85 | 1058.26 | 1055.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1051.45 | 1056.89 | 1055.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 1052.25 | 1056.89 | 1055.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1057.30 | 1056.98 | 1055.23 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 1048.25 | 1053.43 | 1053.97 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 1060.80 | 1055.25 | 1054.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 1063.80 | 1056.96 | 1055.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 1054.55 | 1056.48 | 1055.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 1054.55 | 1056.48 | 1055.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1054.55 | 1056.48 | 1055.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1054.55 | 1056.48 | 1055.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1057.50 | 1056.68 | 1055.54 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 1045.00 | 1053.20 | 1054.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1027.75 | 1045.46 | 1050.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1038.70 | 1034.09 | 1040.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1038.70 | 1034.09 | 1040.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1043.80 | 1036.03 | 1040.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1042.15 | 1036.03 | 1040.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1044.35 | 1037.69 | 1041.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1035.40 | 1040.66 | 1041.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:45:00 | 1040.60 | 1041.10 | 1041.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 1038.10 | 1040.59 | 1041.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 983.63 | 1008.56 | 1021.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 988.57 | 1008.56 | 1021.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 986.19 | 1008.56 | 1021.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1003.55 | 996.99 | 1007.68 | SL hit (close>ema200) qty=0.50 sl=996.99 alert=retest2 |

### Cycle 46 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 881.85 | 877.47 | 877.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 15:15:00 | 889.90 | 879.96 | 878.53 | Break + close above crossover candle high |

### Cycle 47 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 862.35 | 876.43 | 877.06 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 15:15:00 | 875.80 | 872.66 | 872.28 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 867.55 | 871.70 | 871.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 865.60 | 870.48 | 871.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 12:15:00 | 872.40 | 870.86 | 871.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 12:15:00 | 872.40 | 870.86 | 871.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 872.40 | 870.86 | 871.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:30:00 | 878.10 | 870.86 | 871.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 873.15 | 871.32 | 871.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 873.15 | 871.32 | 871.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 875.40 | 872.14 | 871.93 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 848.00 | 867.73 | 870.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 834.25 | 852.43 | 860.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 847.90 | 846.26 | 854.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 847.90 | 846.26 | 854.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 857.20 | 848.45 | 855.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 857.20 | 848.45 | 855.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 859.70 | 850.70 | 855.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 860.75 | 850.70 | 855.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 857.00 | 851.96 | 855.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 852.85 | 851.96 | 855.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 861.75 | 853.92 | 856.22 | SL hit (close>static) qty=1.00 sl=860.45 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 867.00 | 858.76 | 857.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 872.60 | 862.78 | 859.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 880.35 | 881.37 | 876.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 880.35 | 881.37 | 876.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 877.80 | 880.63 | 877.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 877.80 | 880.63 | 877.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 878.00 | 880.10 | 877.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 865.00 | 880.10 | 877.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 875.35 | 879.15 | 876.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 871.60 | 879.15 | 876.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 879.15 | 879.15 | 877.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:00:00 | 884.85 | 880.08 | 877.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 864.30 | 875.80 | 876.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 864.30 | 875.80 | 876.54 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 882.30 | 877.82 | 877.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 14:15:00 | 884.45 | 879.14 | 877.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 879.95 | 880.40 | 878.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 879.95 | 880.40 | 878.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 879.95 | 880.40 | 878.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 875.00 | 880.40 | 878.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 879.40 | 880.20 | 878.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 879.40 | 880.20 | 878.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 873.75 | 878.91 | 878.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 873.75 | 878.91 | 878.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 880.65 | 879.26 | 878.60 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 872.65 | 877.82 | 878.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 865.85 | 874.20 | 876.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 10:15:00 | 865.05 | 864.66 | 869.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:45:00 | 867.45 | 864.66 | 869.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 868.10 | 865.35 | 869.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 868.10 | 865.35 | 869.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 865.65 | 865.05 | 868.39 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 880.50 | 870.34 | 870.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 894.25 | 875.12 | 872.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 15:15:00 | 880.25 | 884.75 | 879.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 15:15:00 | 880.25 | 884.75 | 879.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 880.25 | 884.75 | 879.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 899.15 | 884.75 | 879.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:45:00 | 891.05 | 895.12 | 889.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:15:00 | 890.80 | 894.09 | 889.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:15:00 | 893.75 | 892.69 | 889.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 891.50 | 892.45 | 890.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 884.00 | 889.45 | 889.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 884.00 | 889.45 | 889.60 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 890.20 | 889.31 | 889.30 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 15:15:00 | 889.00 | 889.25 | 889.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 888.60 | 889.12 | 889.21 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 889.90 | 889.27 | 889.27 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 883.80 | 888.71 | 889.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 879.50 | 886.64 | 887.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 11:15:00 | 829.70 | 829.41 | 842.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 12:00:00 | 829.70 | 829.41 | 842.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 789.95 | 782.15 | 787.07 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 801.05 | 790.63 | 789.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 806.00 | 793.70 | 791.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 803.05 | 806.07 | 801.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:00:00 | 803.05 | 806.07 | 801.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 800.10 | 804.87 | 801.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:15:00 | 799.15 | 804.87 | 801.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 798.70 | 803.64 | 800.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:45:00 | 799.40 | 803.64 | 800.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 799.55 | 802.82 | 800.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 12:45:00 | 797.75 | 802.82 | 800.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 15:15:00 | 795.75 | 799.53 | 799.60 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 800.20 | 799.66 | 799.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 10:15:00 | 814.05 | 802.54 | 800.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 806.55 | 808.66 | 805.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 806.55 | 808.66 | 805.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 806.55 | 808.66 | 805.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 806.25 | 808.66 | 805.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 818.35 | 810.60 | 806.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 828.00 | 810.60 | 806.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:00:00 | 829.70 | 818.71 | 813.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:30:00 | 824.95 | 818.69 | 814.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 825.25 | 817.31 | 814.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 815.30 | 816.91 | 814.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 815.30 | 816.91 | 814.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 816.45 | 816.82 | 815.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 813.85 | 816.82 | 815.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 814.00 | 816.26 | 814.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 814.00 | 816.26 | 814.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 814.25 | 815.85 | 814.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 814.25 | 815.85 | 814.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 814.30 | 815.54 | 814.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:45:00 | 814.10 | 815.54 | 814.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 813.00 | 815.03 | 814.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 811.75 | 815.03 | 814.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 813.00 | 814.63 | 814.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 816.55 | 814.63 | 814.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 803.00 | 815.46 | 815.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 803.00 | 815.46 | 815.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 796.35 | 811.64 | 814.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 803.80 | 803.32 | 807.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 10:15:00 | 803.80 | 803.32 | 807.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 803.80 | 803.32 | 807.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 803.80 | 803.32 | 807.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 808.25 | 803.99 | 805.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 807.20 | 803.99 | 805.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 813.30 | 805.86 | 806.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 813.30 | 805.86 | 806.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 813.05 | 807.29 | 807.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 12:15:00 | 815.00 | 808.84 | 807.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 811.60 | 812.74 | 810.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 12:15:00 | 811.60 | 812.74 | 810.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 811.60 | 812.74 | 810.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 811.60 | 812.74 | 810.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 805.05 | 811.20 | 810.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 805.05 | 811.20 | 810.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 800.70 | 809.10 | 809.52 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 814.70 | 808.63 | 808.01 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 800.80 | 810.69 | 810.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 794.85 | 806.29 | 808.68 | Break + close below crossover candle low |

### Cycle 70 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 834.20 | 808.53 | 808.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 840.25 | 825.44 | 817.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 829.30 | 830.65 | 823.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:45:00 | 826.40 | 830.65 | 823.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 831.85 | 829.99 | 824.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 826.10 | 829.99 | 824.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 851.40 | 847.12 | 841.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:15:00 | 845.50 | 847.12 | 841.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 847.40 | 847.17 | 842.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 861.05 | 849.76 | 845.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-16 09:15:00 | 947.15 | 918.52 | 895.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1031.50 | 1043.35 | 1044.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1027.10 | 1038.53 | 1041.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1018.50 | 1017.22 | 1024.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1025.50 | 1017.00 | 1020.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1025.50 | 1017.00 | 1020.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1025.50 | 1017.00 | 1020.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1026.50 | 1018.90 | 1021.21 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 1027.80 | 1023.15 | 1022.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1039.50 | 1027.66 | 1024.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 1055.30 | 1055.39 | 1044.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:45:00 | 1055.00 | 1055.39 | 1044.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1045.70 | 1050.83 | 1048.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1045.70 | 1050.83 | 1048.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1050.50 | 1050.77 | 1048.54 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1040.00 | 1046.52 | 1047.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 15:15:00 | 1035.00 | 1041.72 | 1044.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 1035.40 | 1034.84 | 1038.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1028.80 | 1034.65 | 1038.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:00:00 | 1029.90 | 1033.70 | 1037.72 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1023.10 | 1021.86 | 1025.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1026.80 | 1021.86 | 1025.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1018.00 | 1020.34 | 1023.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 1018.00 | 1020.34 | 1023.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1021.00 | 1019.94 | 1022.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 1025.10 | 1021.20 | 1022.16 | SL hit (close>ema400) qty=1.00 sl=1022.16 alert=retest1 |

### Cycle 74 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1035.40 | 1023.58 | 1022.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1050.00 | 1030.79 | 1026.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 1090.10 | 1102.95 | 1083.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 13:45:00 | 1116.60 | 1101.49 | 1089.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 14:30:00 | 1121.00 | 1104.73 | 1092.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1098.40 | 1104.58 | 1095.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 1097.90 | 1104.58 | 1095.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1095.50 | 1102.77 | 1095.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 1095.50 | 1102.77 | 1095.63 | SL hit (close<ema400) qty=1.00 sl=1095.63 alert=retest1 |

### Cycle 75 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1073.10 | 1090.47 | 1091.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 1069.90 | 1079.89 | 1085.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 1075.50 | 1074.47 | 1080.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 1075.50 | 1074.47 | 1080.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1075.00 | 1074.88 | 1079.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1086.00 | 1074.88 | 1079.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1107.80 | 1081.47 | 1082.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1107.80 | 1081.47 | 1082.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 1110.90 | 1087.35 | 1084.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 1146.60 | 1119.66 | 1104.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1151.10 | 1156.85 | 1146.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 1151.10 | 1156.85 | 1146.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1154.50 | 1161.74 | 1154.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1152.30 | 1161.74 | 1154.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1169.70 | 1163.33 | 1155.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1171.80 | 1165.53 | 1158.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 1172.90 | 1165.53 | 1158.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:00:00 | 1186.10 | 1175.76 | 1166.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 1208.70 | 1179.96 | 1177.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1193.00 | 1182.57 | 1179.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 1218.70 | 1193.50 | 1189.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 1224.40 | 1198.20 | 1191.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1222.80 | 1198.20 | 1191.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1219.90 | 1202.54 | 1194.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1214.30 | 1214.04 | 1204.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:15:00 | 1218.90 | 1214.73 | 1206.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 1223.50 | 1215.84 | 1208.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1200.00 | 1216.94 | 1223.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:00:00 | 1164.20 | 1172.05 | 1184.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:00:00 | 1161.80 | 1165.85 | 1177.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 1170.10 | 1167.43 | 1176.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 1169.50 | 1167.43 | 1176.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1170.10 | 1167.97 | 1175.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 1163.60 | 1167.09 | 1174.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1189.70 | 1170.62 | 1173.02 | SL hit (close>static) qty=1.00 sl=1179.70 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 1160.00 | 1174.82 | 1176.70 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1191.50 | 1178.22 | 1177.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 12:15:00 | 1195.10 | 1187.24 | 1184.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1285.00 | 1291.61 | 1262.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 1279.90 | 1286.59 | 1265.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1279.90 | 1286.59 | 1265.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 1268.00 | 1286.59 | 1265.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1268.00 | 1288.49 | 1279.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1268.00 | 1288.49 | 1279.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1271.40 | 1285.07 | 1278.65 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1245.00 | 1272.97 | 1274.01 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1296.00 | 1274.52 | 1273.08 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1275.00 | 1281.62 | 1282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 1272.80 | 1279.86 | 1281.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 1274.20 | 1279.23 | 1280.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 1274.30 | 1277.81 | 1279.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 1274.60 | 1276.34 | 1278.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1271.10 | 1275.95 | 1277.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1283.70 | 1276.73 | 1277.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 1283.90 | 1276.73 | 1277.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1276.70 | 1276.72 | 1277.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1275.30 | 1276.72 | 1277.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.49 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.58 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.87 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1211.53 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 1207.54 | 1215.84 | 1228.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |

### Cycle 84 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 1238.60 | 1225.92 | 1225.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 1242.80 | 1232.35 | 1228.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1236.50 | 1238.27 | 1233.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1235.50 | 1237.71 | 1233.80 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 1218.70 | 1230.11 | 1231.13 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 1241.80 | 1231.80 | 1231.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1245.00 | 1235.13 | 1232.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 1234.70 | 1236.85 | 1234.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 1234.70 | 1236.85 | 1234.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1243.00 | 1238.08 | 1235.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 1243.80 | 1238.62 | 1235.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1249.40 | 1238.90 | 1236.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1244.00 | 1250.74 | 1244.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1243.60 | 1249.29 | 1244.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1244.20 | 1248.27 | 1244.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1256.10 | 1248.27 | 1244.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 1250.90 | 1247.65 | 1244.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1242.10 | 1245.76 | 1244.29 | SL hit (close<static) qty=1.00 sl=1242.20 alert=retest2 |

### Cycle 87 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 1227.60 | 1238.65 | 1241.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1231.20 | 1225.27 | 1230.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1225.10 | 1224.04 | 1228.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1224.90 | 1224.04 | 1228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1229.10 | 1225.05 | 1228.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1229.10 | 1225.05 | 1228.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1233.70 | 1226.78 | 1228.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 1233.70 | 1226.78 | 1228.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1229.00 | 1227.22 | 1228.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 1232.70 | 1227.22 | 1228.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1229.80 | 1227.74 | 1228.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:45:00 | 1229.30 | 1227.74 | 1228.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1233.30 | 1228.85 | 1229.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 1232.60 | 1228.85 | 1229.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 1235.00 | 1230.08 | 1229.88 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 1225.10 | 1230.05 | 1230.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 1221.20 | 1226.23 | 1228.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 1215.70 | 1213.58 | 1218.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:45:00 | 1214.70 | 1213.58 | 1218.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 1199.60 | 1210.78 | 1216.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1186.90 | 1206.81 | 1213.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 1197.00 | 1203.96 | 1210.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 1193.80 | 1199.91 | 1207.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 1219.10 | 1206.01 | 1206.25 | SL hit (close>static) qty=1.00 sl=1217.90 alert=retest2 |

### Cycle 90 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1223.00 | 1209.41 | 1207.77 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 1198.40 | 1207.90 | 1208.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1192.50 | 1203.43 | 1205.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1194.50 | 1189.22 | 1194.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1187.90 | 1188.96 | 1194.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1184.00 | 1186.57 | 1192.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1194.10 | 1180.17 | 1179.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1194.10 | 1180.17 | 1179.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 1206.00 | 1191.70 | 1185.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1210.70 | 1203.34 | 1197.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 1210.30 | 1204.73 | 1198.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 1209.40 | 1207.10 | 1201.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 1209.20 | 1207.58 | 1202.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1196.00 | 1205.26 | 1201.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1196.00 | 1205.26 | 1201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1208.00 | 1205.81 | 1202.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1197.40 | 1205.81 | 1202.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1194.10 | 1203.47 | 1201.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 1190.20 | 1203.47 | 1201.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1199.00 | 1202.57 | 1201.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 1191.30 | 1202.57 | 1201.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1199.40 | 1201.94 | 1201.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 1199.10 | 1201.94 | 1201.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1199.60 | 1201.47 | 1200.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 1197.60 | 1201.47 | 1200.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1201.00 | 1202.42 | 1201.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1201.00 | 1202.42 | 1201.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1205.20 | 1202.98 | 1201.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1199.20 | 1202.98 | 1201.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1207.60 | 1203.90 | 1202.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1204.10 | 1203.90 | 1202.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1231.50 | 1212.66 | 1207.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1236.70 | 1223.32 | 1214.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1251.40 | 1237.28 | 1236.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 1268.00 | 1252.07 | 1245.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 1255.30 | 1256.37 | 1249.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1246.20 | 1254.33 | 1249.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1248.60 | 1254.33 | 1249.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1243.20 | 1252.11 | 1248.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1241.10 | 1252.11 | 1248.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1248.00 | 1251.29 | 1248.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 1233.40 | 1251.29 | 1248.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1228.00 | 1246.63 | 1246.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1214.40 | 1240.18 | 1243.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:45:00 | 1233.00 | 1226.14 | 1232.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1233.70 | 1227.65 | 1232.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 1233.70 | 1227.65 | 1232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1234.00 | 1228.92 | 1232.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1233.90 | 1228.92 | 1232.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1236.70 | 1230.48 | 1232.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 1235.00 | 1230.48 | 1232.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1229.00 | 1230.18 | 1232.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 1239.20 | 1230.18 | 1232.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1218.40 | 1212.45 | 1217.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 1216.70 | 1212.45 | 1217.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1217.10 | 1213.38 | 1217.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1217.10 | 1213.38 | 1217.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1215.30 | 1213.76 | 1217.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 1212.60 | 1213.41 | 1216.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 1211.70 | 1213.54 | 1215.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 1219.30 | 1214.93 | 1215.84 | SL hit (close>static) qty=1.00 sl=1217.70 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1229.70 | 1217.88 | 1217.10 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1210.40 | 1217.92 | 1218.21 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1227.50 | 1218.26 | 1217.92 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1214.90 | 1217.30 | 1217.55 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 1219.40 | 1217.72 | 1217.72 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1210.10 | 1216.20 | 1217.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1175.50 | 1205.79 | 1212.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1189.80 | 1186.87 | 1197.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1189.80 | 1186.87 | 1197.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1186.70 | 1186.84 | 1196.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 1185.80 | 1186.84 | 1196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1195.40 | 1188.55 | 1196.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1195.80 | 1188.55 | 1196.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1120.70 | 1112.45 | 1117.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1120.70 | 1112.45 | 1117.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1133.80 | 1116.72 | 1119.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1133.80 | 1116.72 | 1119.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1137.60 | 1120.90 | 1120.74 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 1119.00 | 1122.76 | 1122.89 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 1124.20 | 1123.05 | 1123.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 15:15:00 | 1125.00 | 1123.44 | 1123.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1119.00 | 1124.40 | 1123.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1122.50 | 1124.02 | 1123.72 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 1118.00 | 1122.82 | 1123.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1113.70 | 1119.95 | 1121.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1084.40 | 1083.64 | 1089.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:45:00 | 1083.00 | 1083.64 | 1089.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1072.10 | 1077.06 | 1084.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 1072.10 | 1077.06 | 1084.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1087.80 | 1079.81 | 1084.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1086.80 | 1079.81 | 1084.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1092.60 | 1082.36 | 1085.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:30:00 | 1102.30 | 1082.36 | 1085.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 1100.00 | 1088.42 | 1087.44 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 1088.90 | 1090.77 | 1090.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 1080.30 | 1087.67 | 1089.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 1083.00 | 1074.86 | 1079.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1081.40 | 1076.17 | 1079.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1085.10 | 1076.17 | 1079.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1074.40 | 1075.82 | 1079.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1069.50 | 1075.82 | 1079.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 1071.60 | 1075.73 | 1079.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 1071.90 | 1074.97 | 1078.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1072.00 | 1074.95 | 1078.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1072.30 | 1069.61 | 1073.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1072.30 | 1069.61 | 1073.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1075.70 | 1070.83 | 1073.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1069.80 | 1070.83 | 1073.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1069.90 | 1069.98 | 1072.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1055.00 | 1044.39 | 1041.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1075.20 | 1083.27 | 1068.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 1075.20 | 1083.27 | 1068.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1085.00 | 1087.11 | 1082.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1079.30 | 1087.11 | 1082.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1079.00 | 1085.49 | 1082.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1077.40 | 1085.49 | 1082.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1073.10 | 1083.01 | 1081.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 1073.10 | 1083.01 | 1081.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1074.90 | 1081.39 | 1080.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 1074.40 | 1081.39 | 1080.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 1072.90 | 1079.69 | 1080.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 1068.30 | 1076.47 | 1078.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1030.00 | 1052.80 | 1060.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 1029.10 | 1049.34 | 1057.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:15:00 | 978.50 | 1008.82 | 1028.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 12:15:00 | 977.64 | 1003.52 | 1024.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 988.70 | 979.27 | 992.99 | SL hit (close>ema200) qty=0.50 sl=979.27 alert=retest2 |

### Cycle 110 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 979.85 | 969.82 | 968.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 983.70 | 976.26 | 972.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 1006.80 | 1008.73 | 1003.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 09:15:00 | 1000.40 | 1008.73 | 1003.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1008.15 | 1008.61 | 1003.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 999.40 | 1008.61 | 1003.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1005.00 | 1007.89 | 1003.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 1005.00 | 1007.89 | 1003.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 995.60 | 1005.43 | 1003.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 995.15 | 1005.43 | 1003.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 995.00 | 1003.34 | 1002.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 995.00 | 1003.34 | 1002.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 994.85 | 1001.65 | 1001.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 975.80 | 995.31 | 998.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 990.85 | 990.50 | 995.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 990.85 | 990.50 | 995.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 992.85 | 990.97 | 995.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 994.80 | 990.97 | 995.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 995.00 | 991.78 | 995.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 995.00 | 991.78 | 995.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 995.00 | 992.42 | 995.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 995.00 | 992.42 | 995.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 995.00 | 992.94 | 995.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 994.25 | 992.94 | 995.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 987.55 | 991.86 | 994.59 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 998.00 | 993.32 | 993.22 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 991.20 | 992.90 | 993.03 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 996.00 | 993.52 | 993.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 1001.60 | 995.13 | 994.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 985.90 | 996.95 | 995.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 989.00 | 995.36 | 994.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 986.70 | 995.36 | 994.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 987.15 | 993.72 | 994.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 980.60 | 990.47 | 992.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 985.30 | 970.76 | 975.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 988.75 | 974.36 | 976.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 988.75 | 974.36 | 976.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 991.70 | 980.53 | 979.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 994.00 | 983.22 | 980.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 968.30 | 983.46 | 981.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 956.85 | 978.14 | 979.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 948.80 | 972.27 | 976.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 889.00 | 886.37 | 901.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 15:15:00 | 882.40 | 887.85 | 899.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 894.60 | 888.33 | 897.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 891.90 | 888.33 | 897.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 903.65 | 892.51 | 896.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 903.65 | 892.51 | 896.39 | SL hit (close>ema400) qty=1.00 sl=896.39 alert=retest1 |

### Cycle 118 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 913.35 | 900.67 | 899.41 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 891.45 | 897.56 | 898.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 878.00 | 893.65 | 896.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 895.80 | 888.58 | 891.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 894.65 | 889.80 | 892.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 905.45 | 889.80 | 892.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 906.10 | 893.06 | 893.38 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 909.80 | 896.40 | 894.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 912.40 | 899.60 | 896.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 912.65 | 913.37 | 906.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 911.75 | 913.37 | 906.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 908.95 | 911.71 | 907.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 908.65 | 911.71 | 907.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 907.95 | 910.96 | 907.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 907.00 | 910.96 | 907.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 912.50 | 911.27 | 907.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 907.60 | 911.27 | 907.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 900.85 | 909.18 | 906.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 900.85 | 909.18 | 906.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 904.35 | 908.22 | 906.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:15:00 | 905.00 | 908.22 | 906.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 939.95 | 947.70 | 948.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 939.95 | 947.70 | 948.48 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 953.00 | 949.44 | 949.17 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 946.55 | 948.86 | 948.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 941.20 | 947.33 | 948.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 944.85 | 936.64 | 940.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 941.40 | 937.59 | 940.20 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 963.10 | 944.24 | 942.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 968.10 | 949.01 | 945.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 990.00 | 990.59 | 976.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 990.65 | 990.59 | 976.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 983.15 | 989.00 | 981.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 995.00 | 987.76 | 983.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 972.30 | 986.70 | 987.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 972.30 | 986.70 | 987.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 963.80 | 982.12 | 985.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 962.50 | 957.88 | 965.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:00:00 | 962.50 | 957.88 | 965.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 965.15 | 959.34 | 965.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 965.15 | 959.34 | 965.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 966.70 | 960.81 | 965.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 960.40 | 962.40 | 965.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 961.95 | 962.68 | 964.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 969.25 | 963.99 | 965.32 | SL hit (close>static) qty=1.00 sl=968.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 951.55 | 935.00 | 934.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 974.40 | 950.05 | 942.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 960.20 | 960.48 | 950.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 960.20 | 960.48 | 950.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 959.30 | 959.94 | 952.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 934.55 | 959.94 | 952.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 934.10 | 954.78 | 950.62 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 935.15 | 947.45 | 947.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 916.75 | 938.38 | 943.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 897.20 | 890.58 | 904.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 897.20 | 890.58 | 904.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 900.00 | 892.46 | 904.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 900.00 | 892.46 | 904.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 910.15 | 896.39 | 904.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 910.15 | 896.39 | 904.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 918.30 | 900.77 | 905.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:45:00 | 917.00 | 900.77 | 905.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 949.35 | 914.82 | 911.06 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 891.65 | 926.27 | 927.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 891.00 | 919.22 | 924.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 902.00 | 900.12 | 909.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 902.00 | 900.12 | 909.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 906.45 | 901.38 | 909.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 907.60 | 901.38 | 909.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 907.10 | 902.53 | 909.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 907.50 | 902.53 | 909.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 905.70 | 903.62 | 908.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 905.05 | 903.62 | 908.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 950.05 | 913.03 | 912.01 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 913.25 | 932.07 | 932.99 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 969.20 | 939.50 | 936.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 991.00 | 966.23 | 951.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 955.80 | 964.79 | 953.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 956.10 | 963.05 | 953.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 960.05 | 963.05 | 953.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 962.85 | 963.01 | 954.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 1056.06 | 1007.00 | 990.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 1181.95 | 1218.04 | 1220.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1138.70 | 1196.08 | 1210.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1107.60 | 1086.14 | 1117.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:30:00 | 1112.90 | 1086.14 | 1117.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1109.00 | 1104.73 | 1113.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:15:00 | 1104.50 | 1107.22 | 1112.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:45:00 | 1104.40 | 1106.66 | 1112.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 1098.10 | 1104.94 | 1110.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 1103.00 | 1103.84 | 1109.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1107.60 | 1103.96 | 1107.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1107.60 | 1103.96 | 1107.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1102.70 | 1103.70 | 1107.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:30:00 | 1105.00 | 1103.70 | 1107.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1103.30 | 1103.62 | 1107.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 1108.30 | 1103.62 | 1107.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1105.00 | 1103.90 | 1106.88 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 14:30:00 | 1275.40 | 2024-05-27 09:15:00 | 1295.05 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-06-21 09:15:00 | 1362.00 | 2024-07-01 15:15:00 | 1498.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-21 09:45:00 | 1362.80 | 2024-07-01 15:15:00 | 1499.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 09:15:00 | 1371.95 | 2024-07-01 15:15:00 | 1509.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-07-11 09:45:00 | 1421.75 | 2024-07-15 10:15:00 | 1350.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-11 09:45:00 | 1421.75 | 2024-07-16 09:15:00 | 1380.95 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-07-16 13:15:00 | 1377.60 | 2024-07-18 09:15:00 | 1389.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-07-18 10:45:00 | 1374.90 | 2024-07-22 14:15:00 | 1386.95 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-07-22 10:30:00 | 1379.30 | 2024-07-22 14:15:00 | 1386.95 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1454.60 | 2024-08-01 15:15:00 | 1460.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-07-26 11:15:00 | 1453.20 | 2024-08-01 15:15:00 | 1460.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1465.45 | 2024-08-01 15:15:00 | 1460.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-07-30 09:15:00 | 1452.20 | 2024-08-01 15:15:00 | 1460.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-07-30 11:30:00 | 1455.65 | 2024-08-01 15:15:00 | 1460.00 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-08-14 09:15:00 | 1387.25 | 2024-08-27 12:15:00 | 1353.35 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2024-09-12 15:15:00 | 1450.15 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2024-09-13 11:15:00 | 1448.80 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2024-09-13 12:15:00 | 1452.15 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1449.75 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 3.05% |
| BUY | retest2 | 2024-09-16 15:00:00 | 1469.95 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2024-09-17 10:15:00 | 1469.95 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2024-09-18 10:00:00 | 1470.05 | 2024-09-26 09:15:00 | 1493.95 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2024-10-07 10:45:00 | 1402.85 | 2024-10-15 15:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-07 11:45:00 | 1407.75 | 2024-10-15 15:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-10-08 09:30:00 | 1406.90 | 2024-10-15 15:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-10-09 09:30:00 | 1405.00 | 2024-10-15 15:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-10-09 14:30:00 | 1400.00 | 2024-10-22 09:15:00 | 1337.36 | PARTIAL | 0.50 | 4.47% |
| SELL | retest2 | 2024-10-10 09:45:00 | 1405.00 | 2024-10-22 09:15:00 | 1336.56 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-10-10 10:15:00 | 1403.10 | 2024-10-22 09:15:00 | 1334.75 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-10-10 11:00:00 | 1404.20 | 2024-10-22 09:15:00 | 1334.75 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-10-14 09:15:00 | 1400.80 | 2024-10-22 11:15:00 | 1332.71 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2024-10-14 12:30:00 | 1401.55 | 2024-10-22 11:15:00 | 1330.00 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2024-10-14 13:45:00 | 1401.05 | 2024-10-22 11:15:00 | 1332.94 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2024-10-15 09:15:00 | 1400.00 | 2024-10-22 11:15:00 | 1333.99 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2024-10-16 12:45:00 | 1380.80 | 2024-10-22 13:15:00 | 1311.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:45:00 | 1381.10 | 2024-10-22 13:15:00 | 1312.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 15:15:00 | 1377.50 | 2024-10-22 13:15:00 | 1308.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1381.00 | 2024-10-22 13:15:00 | 1311.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 1358.00 | 2024-10-22 13:15:00 | 1290.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1359.40 | 2024-10-22 13:15:00 | 1291.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1359.60 | 2024-10-22 13:15:00 | 1291.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 14:30:00 | 1400.00 | 2024-10-23 09:15:00 | 1262.57 | TARGET_HIT | 0.50 | 9.82% |
| SELL | retest2 | 2024-10-10 09:45:00 | 1405.00 | 2024-10-23 09:15:00 | 1266.98 | TARGET_HIT | 0.50 | 9.82% |
| SELL | retest2 | 2024-10-10 10:15:00 | 1403.10 | 2024-10-23 09:15:00 | 1266.21 | TARGET_HIT | 0.50 | 9.76% |
| SELL | retest2 | 2024-10-10 11:00:00 | 1404.20 | 2024-10-23 09:15:00 | 1264.50 | TARGET_HIT | 0.50 | 9.95% |
| SELL | retest2 | 2024-10-14 09:15:00 | 1400.80 | 2024-10-23 09:15:00 | 1260.00 | TARGET_HIT | 0.50 | 10.05% |
| SELL | retest2 | 2024-10-14 12:30:00 | 1401.55 | 2024-10-23 09:15:00 | 1264.50 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2024-10-14 13:45:00 | 1401.05 | 2024-10-23 09:15:00 | 1262.79 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2024-10-15 09:15:00 | 1400.00 | 2024-10-23 09:15:00 | 1263.78 | TARGET_HIT | 0.50 | 9.73% |
| SELL | retest2 | 2024-10-16 12:45:00 | 1380.80 | 2024-10-24 09:15:00 | 1242.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 11:45:00 | 1381.10 | 2024-10-24 09:15:00 | 1242.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 15:15:00 | 1377.50 | 2024-10-24 09:15:00 | 1239.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1381.00 | 2024-10-24 09:15:00 | 1242.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 1358.00 | 2024-10-24 09:15:00 | 1222.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1359.40 | 2024-10-24 09:15:00 | 1223.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1359.60 | 2024-10-24 09:15:00 | 1223.64 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-07 12:30:00 | 1232.05 | 2024-11-07 15:15:00 | 1223.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-11-18 10:30:00 | 1134.90 | 2024-11-19 13:15:00 | 1163.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-11-18 11:00:00 | 1131.40 | 2024-11-19 13:15:00 | 1163.10 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1221.50 | 2024-12-04 14:15:00 | 1212.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-31 13:15:00 | 1146.40 | 2025-01-03 09:15:00 | 1139.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-12-31 14:00:00 | 1148.75 | 2025-01-03 09:15:00 | 1139.95 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-01-02 13:45:00 | 1147.20 | 2025-01-03 09:15:00 | 1139.95 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-01-06 10:30:00 | 1117.90 | 2025-01-08 12:15:00 | 1062.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:30:00 | 1118.80 | 2025-01-08 12:15:00 | 1062.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:30:00 | 1117.90 | 2025-01-09 11:15:00 | 1080.50 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-01-06 11:30:00 | 1118.80 | 2025-01-09 11:15:00 | 1080.50 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1035.40 | 2025-01-28 09:15:00 | 983.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 10:45:00 | 1040.60 | 2025-01-28 09:15:00 | 988.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 1038.10 | 2025-01-28 09:15:00 | 986.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1035.40 | 2025-01-29 09:15:00 | 1003.55 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-01-24 10:45:00 | 1040.60 | 2025-01-29 09:15:00 | 1003.55 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2025-01-24 13:30:00 | 1038.10 | 2025-01-29 09:15:00 | 1003.55 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-03-04 09:15:00 | 852.85 | 2025-03-04 09:15:00 | 861.75 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-03-04 10:30:00 | 854.50 | 2025-03-04 11:15:00 | 862.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-03-04 11:15:00 | 855.20 | 2025-03-04 11:15:00 | 862.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-03-04 13:30:00 | 855.50 | 2025-03-04 14:15:00 | 867.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-10 13:00:00 | 884.85 | 2025-03-11 09:15:00 | 864.30 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-03-19 09:15:00 | 899.15 | 2025-03-21 14:15:00 | 884.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-03-20 10:45:00 | 891.05 | 2025-03-21 14:15:00 | 884.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-03-20 12:15:00 | 890.80 | 2025-03-21 14:15:00 | 884.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-03-20 14:15:00 | 893.75 | 2025-03-21 14:15:00 | 884.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-04-21 11:15:00 | 828.00 | 2025-04-25 09:15:00 | 803.00 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2025-04-22 11:00:00 | 829.70 | 2025-04-25 09:15:00 | 803.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-04-22 13:30:00 | 824.95 | 2025-04-25 09:15:00 | 803.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-04-23 09:15:00 | 825.25 | 2025-04-25 09:15:00 | 803.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-04-24 09:15:00 | 816.55 | 2025-04-25 09:15:00 | 803.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-14 09:15:00 | 861.05 | 2025-05-16 09:15:00 | 947.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 1028.80 | 2025-06-20 15:15:00 | 1025.10 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest1 | 2025-06-17 10:00:00 | 1029.90 | 2025-06-20 15:15:00 | 1025.10 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest1 | 2025-06-26 13:45:00 | 1116.60 | 2025-06-27 11:15:00 | 1095.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-06-26 14:30:00 | 1121.00 | 2025-06-27 11:15:00 | 1095.50 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-09 12:45:00 | 1171.80 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2025-07-09 13:15:00 | 1172.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2025-07-10 10:00:00 | 1186.10 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 2.58% |
| BUY | retest2 | 2025-07-14 09:15:00 | 1208.70 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-07-17 10:00:00 | 1218.70 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-07-17 10:30:00 | 1224.40 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-17 11:15:00 | 1222.80 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-17 12:00:00 | 1219.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-18 12:15:00 | 1218.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-18 13:30:00 | 1223.50 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-28 11:00:00 | 1164.20 | 2025-07-30 09:15:00 | 1189.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1161.80 | 2025-07-31 11:15:00 | 1180.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-29 09:30:00 | 1170.10 | 2025-07-31 11:15:00 | 1180.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1169.50 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-29 12:00:00 | 1163.60 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-30 15:15:00 | 1169.00 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-31 09:45:00 | 1164.10 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-21 11:30:00 | 1274.20 | 2025-08-29 09:15:00 | 1210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:00:00 | 1274.30 | 2025-08-29 09:15:00 | 1210.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:45:00 | 1274.60 | 2025-08-29 09:15:00 | 1210.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1271.10 | 2025-08-29 09:15:00 | 1211.53 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1275.30 | 2025-08-29 15:15:00 | 1207.54 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2025-08-21 11:30:00 | 1274.20 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-08-21 15:00:00 | 1274.30 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-08-22 12:45:00 | 1274.60 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1271.10 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1275.30 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.62% |
| BUY | retest2 | 2025-09-09 14:30:00 | 1243.80 | 2025-09-11 12:15:00 | 1242.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1249.40 | 2025-09-11 12:15:00 | 1242.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-10 14:15:00 | 1244.00 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-10 14:45:00 | 1243.60 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1256.10 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-11 10:45:00 | 1250.90 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-23 09:15:00 | 1186.90 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-23 10:30:00 | 1197.00 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1193.80 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1184.00 | 2025-10-01 13:15:00 | 1194.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1210.70 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-07 10:00:00 | 1210.30 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-10-07 12:45:00 | 1209.40 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2025-10-07 13:45:00 | 1209.20 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1236.70 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-28 10:00:00 | 1212.60 | 2025-10-28 15:15:00 | 1219.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-28 14:00:00 | 1211.70 | 2025-10-28 15:15:00 | 1219.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1069.50 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2025-12-01 13:15:00 | 1071.60 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-01 14:00:00 | 1071.90 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1072.00 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1069.80 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1069.90 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1030.00 | 2025-12-24 11:15:00 | 978.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1029.10 | 2025-12-24 12:15:00 | 977.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1030.00 | 2025-12-29 09:15:00 | 988.70 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1029.10 | 2025-12-29 09:15:00 | 988.70 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest1 | 2026-01-29 15:15:00 | 882.40 | 2026-01-30 14:15:00 | 903.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-05 11:15:00 | 905.00 | 2026-02-13 10:15:00 | 939.95 | STOP_HIT | 1.00 | 3.86% |
| BUY | retest2 | 2026-02-20 15:00:00 | 995.00 | 2026-02-24 11:15:00 | 972.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-02-26 15:00:00 | 960.40 | 2026-02-27 10:15:00 | 969.25 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-27 10:15:00 | 961.95 | 2026-02-27 10:15:00 | 969.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-27 13:30:00 | 960.70 | 2026-03-02 09:15:00 | 912.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 13:30:00 | 960.70 | 2026-03-04 12:15:00 | 941.70 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2026-04-02 11:15:00 | 960.05 | 2026-04-08 09:15:00 | 1056.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:00:00 | 962.85 | 2026-04-08 09:15:00 | 1059.14 | TARGET_HIT | 1.00 | 10.00% |
