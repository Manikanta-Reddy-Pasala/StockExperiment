# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-11 15:15:00 (3716 bars)
- **Last close:** 885.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 96 |
| ALERT2 | 95 |
| ALERT2_SKIP | 54 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 149 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 142 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 167 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 65 / 102
- **Target hits / Stop hits / Partials:** 7 / 142 / 18
- **Avg / median % per leg:** 0.73% / -0.63%
- **Sum % (uncompounded):** 122.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 22 | 29.3% | 4 | 71 | 0 | -0.34% | -25.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.44% | 0.9% |
| BUY @ 3rd Alert (retest2) | 73 | 20 | 27.4% | 4 | 69 | 0 | -0.37% | -26.6% |
| SELL (all) | 92 | 43 | 46.7% | 3 | 71 | 18 | 1.61% | 148.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 92 | 43 | 46.7% | 3 | 71 | 18 | 1.61% | 148.3% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.44% | 0.9% |
| retest2 (combined) | 165 | 63 | 38.2% | 7 | 140 | 18 | 0.74% | 121.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1261.45 | 1248.57 | 1247.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 1265.95 | 1252.05 | 1248.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 11:15:00 | 1256.85 | 1259.20 | 1255.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 11:15:00 | 1256.85 | 1259.20 | 1255.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 1256.85 | 1259.20 | 1255.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:00:00 | 1256.85 | 1259.20 | 1255.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 1251.80 | 1257.72 | 1254.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:15:00 | 1247.35 | 1257.72 | 1254.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1237.55 | 1253.68 | 1253.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 1237.55 | 1253.68 | 1253.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 15:15:00 | 1250.00 | 1252.89 | 1252.92 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1266.00 | 1255.51 | 1254.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 1285.25 | 1265.67 | 1261.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 1294.75 | 1296.66 | 1287.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 1294.75 | 1296.66 | 1287.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1294.50 | 1295.00 | 1288.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 1307.40 | 1295.00 | 1288.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 15:00:00 | 1301.85 | 1298.12 | 1296.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 12:15:00 | 1290.00 | 1295.52 | 1295.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 12:15:00 | 1290.00 | 1295.52 | 1295.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1282.10 | 1292.05 | 1294.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 14:15:00 | 1284.20 | 1283.80 | 1288.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 1284.20 | 1283.80 | 1288.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1282.55 | 1283.55 | 1287.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 1279.50 | 1283.81 | 1287.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:45:00 | 1279.80 | 1283.10 | 1286.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 1278.95 | 1283.03 | 1286.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:15:00 | 1272.00 | 1282.18 | 1285.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1285.25 | 1281.17 | 1284.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:45:00 | 1265.45 | 1277.45 | 1281.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 1265.80 | 1267.54 | 1273.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1292.05 | 1276.21 | 1275.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1292.05 | 1276.21 | 1275.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 1323.00 | 1292.21 | 1283.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1287.15 | 1305.87 | 1294.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1287.15 | 1305.87 | 1294.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1287.15 | 1305.87 | 1294.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1274.00 | 1305.87 | 1294.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1195.25 | 1283.74 | 1285.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1131.35 | 1253.27 | 1271.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 1160.40 | 1144.99 | 1187.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 1160.40 | 1144.99 | 1187.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1218.00 | 1158.44 | 1183.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1218.00 | 1158.44 | 1183.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1197.45 | 1166.25 | 1184.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1219.00 | 1166.25 | 1184.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1215.90 | 1184.44 | 1190.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:45:00 | 1213.05 | 1184.44 | 1190.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 1233.20 | 1199.56 | 1196.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1235.00 | 1210.72 | 1202.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1274.00 | 1282.52 | 1260.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 15:15:00 | 1274.00 | 1282.52 | 1260.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1274.00 | 1282.52 | 1260.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:45:00 | 1295.95 | 1283.60 | 1269.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 1295.95 | 1283.60 | 1269.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1296.25 | 1285.57 | 1272.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:45:00 | 1302.10 | 1287.44 | 1274.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1317.80 | 1296.89 | 1286.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:00:00 | 1323.25 | 1302.16 | 1289.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:00:00 | 1323.95 | 1307.31 | 1296.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 1323.55 | 1309.85 | 1298.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 13:15:00 | 1326.55 | 1314.28 | 1304.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1334.95 | 1318.41 | 1307.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 1358.00 | 1326.33 | 1312.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 1425.55 | 1351.41 | 1326.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 1335.10 | 1372.50 | 1376.20 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 1415.70 | 1381.22 | 1378.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 1434.80 | 1402.58 | 1391.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 1383.65 | 1406.24 | 1400.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 1383.65 | 1406.24 | 1400.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1383.65 | 1406.24 | 1400.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 1383.65 | 1406.24 | 1400.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1386.75 | 1402.34 | 1398.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 1393.60 | 1402.34 | 1398.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 1376.00 | 1393.91 | 1395.39 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 1400.00 | 1396.25 | 1396.16 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 1388.95 | 1395.40 | 1395.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 09:15:00 | 1375.05 | 1391.20 | 1393.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 1384.45 | 1381.24 | 1387.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 13:45:00 | 1384.05 | 1381.24 | 1387.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1382.45 | 1381.48 | 1386.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1382.45 | 1381.48 | 1386.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1382.95 | 1381.77 | 1386.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1433.25 | 1381.77 | 1386.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1425.75 | 1390.57 | 1390.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 15:15:00 | 1484.00 | 1456.54 | 1437.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1475.65 | 1477.41 | 1461.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 09:30:00 | 1475.10 | 1477.41 | 1461.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1461.15 | 1470.49 | 1461.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 1461.15 | 1470.49 | 1461.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1460.15 | 1468.42 | 1461.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:30:00 | 1462.50 | 1468.42 | 1461.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1500.00 | 1474.74 | 1465.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1529.45 | 1480.79 | 1468.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 1514.85 | 1534.73 | 1523.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 1507.00 | 1517.30 | 1518.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 1507.00 | 1517.30 | 1518.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 13:15:00 | 1504.05 | 1508.39 | 1511.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 10:15:00 | 1482.00 | 1479.22 | 1490.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 1482.00 | 1479.22 | 1490.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1495.00 | 1482.38 | 1491.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:00:00 | 1495.00 | 1482.38 | 1491.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1483.65 | 1482.63 | 1490.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:15:00 | 1489.50 | 1482.63 | 1490.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1477.50 | 1481.61 | 1489.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:45:00 | 1474.90 | 1481.45 | 1488.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 1476.00 | 1481.45 | 1488.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:30:00 | 1476.20 | 1478.13 | 1485.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 1475.25 | 1478.70 | 1484.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1487.50 | 1473.46 | 1478.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 1487.50 | 1473.46 | 1478.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1486.40 | 1476.05 | 1479.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 1484.70 | 1476.05 | 1479.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1487.90 | 1478.42 | 1479.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 1488.95 | 1478.42 | 1479.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 1496.20 | 1481.98 | 1481.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 1496.20 | 1481.98 | 1481.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1511.40 | 1489.30 | 1484.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 1468.75 | 1491.99 | 1488.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 1468.75 | 1491.99 | 1488.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1468.75 | 1491.99 | 1488.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1462.25 | 1491.99 | 1488.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1469.10 | 1487.41 | 1486.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 1465.75 | 1487.41 | 1486.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 1463.95 | 1482.72 | 1484.64 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 1489.85 | 1486.13 | 1485.85 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 1484.20 | 1485.61 | 1485.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 11:15:00 | 1472.65 | 1482.29 | 1484.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 12:15:00 | 1487.85 | 1483.41 | 1484.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 1487.85 | 1483.41 | 1484.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 1487.85 | 1483.41 | 1484.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 1488.00 | 1483.41 | 1484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 1484.05 | 1483.53 | 1484.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 1476.30 | 1482.31 | 1483.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 1501.30 | 1486.13 | 1485.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1501.30 | 1486.13 | 1485.20 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 1486.50 | 1502.67 | 1503.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 1473.15 | 1492.41 | 1498.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 13:15:00 | 1449.25 | 1444.46 | 1462.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 14:00:00 | 1449.25 | 1444.46 | 1462.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 1413.05 | 1406.97 | 1419.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 1407.00 | 1405.01 | 1416.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 1429.80 | 1409.22 | 1415.48 | SL hit (close>static) qty=1.00 sl=1422.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 1426.45 | 1419.13 | 1418.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1438.00 | 1426.18 | 1422.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 10:15:00 | 1423.65 | 1425.67 | 1422.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 10:15:00 | 1423.65 | 1425.67 | 1422.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 1423.65 | 1425.67 | 1422.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 1423.65 | 1425.67 | 1422.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 1397.95 | 1420.13 | 1420.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 1390.40 | 1410.79 | 1415.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 1413.10 | 1406.75 | 1412.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 1413.10 | 1406.75 | 1412.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1413.10 | 1406.75 | 1412.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 1412.40 | 1406.75 | 1412.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 1414.45 | 1408.29 | 1412.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 1414.45 | 1408.29 | 1412.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1415.80 | 1409.79 | 1412.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:15:00 | 1421.45 | 1409.79 | 1412.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 1421.50 | 1412.13 | 1413.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 1421.50 | 1412.13 | 1413.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 1417.85 | 1415.27 | 1414.96 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 1408.60 | 1413.94 | 1414.38 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 1431.85 | 1417.67 | 1416.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 1444.00 | 1424.90 | 1419.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 15:15:00 | 1485.00 | 1488.56 | 1473.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:15:00 | 1507.85 | 1488.56 | 1473.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 13:15:00 | 1497.00 | 1496.39 | 1483.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1509.00 | 1530.03 | 1524.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 1509.00 | 1530.03 | 1524.70 | SL hit (close<ema400) qty=1.00 sl=1524.70 alert=retest1 |

### Cycle 26 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 1505.75 | 1520.60 | 1521.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 10:15:00 | 1488.00 | 1505.12 | 1512.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 14:15:00 | 1512.45 | 1495.47 | 1504.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 14:15:00 | 1512.45 | 1495.47 | 1504.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1512.45 | 1495.47 | 1504.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 1512.45 | 1495.47 | 1504.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1510.00 | 1498.38 | 1504.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1514.30 | 1498.38 | 1504.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1496.65 | 1498.03 | 1504.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 1495.05 | 1498.03 | 1504.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 13:15:00 | 1494.30 | 1498.35 | 1502.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 14:30:00 | 1490.15 | 1495.67 | 1500.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 1500.00 | 1476.40 | 1473.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1500.00 | 1476.40 | 1473.90 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 1438.45 | 1485.01 | 1487.36 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 1465.60 | 1453.83 | 1452.79 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1433.95 | 1451.83 | 1452.19 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 1457.90 | 1449.83 | 1449.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1463.00 | 1456.39 | 1453.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 15:15:00 | 1461.00 | 1461.98 | 1457.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 1463.40 | 1461.98 | 1457.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1465.90 | 1462.76 | 1458.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:45:00 | 1469.75 | 1463.94 | 1459.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:00:00 | 1472.65 | 1465.85 | 1462.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:30:00 | 1471.00 | 1467.58 | 1463.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 1476.05 | 1483.44 | 1484.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 14:15:00 | 1476.05 | 1483.44 | 1484.16 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1494.75 | 1484.27 | 1484.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 1498.40 | 1487.10 | 1485.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 1481.00 | 1486.39 | 1485.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 1481.00 | 1486.39 | 1485.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1481.00 | 1486.39 | 1485.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 1483.95 | 1486.39 | 1485.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1525.40 | 1494.19 | 1489.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 1525.40 | 1494.19 | 1489.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1586.05 | 1597.35 | 1588.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:45:00 | 1580.90 | 1597.35 | 1588.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1591.00 | 1596.08 | 1588.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:45:00 | 1598.60 | 1596.40 | 1590.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:00:00 | 1596.15 | 1596.35 | 1590.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 14:45:00 | 1601.55 | 1603.21 | 1598.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 1600.00 | 1603.04 | 1599.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 1602.45 | 1602.92 | 1599.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 1599.00 | 1602.92 | 1599.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 1585.30 | 1599.40 | 1598.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 1585.30 | 1599.40 | 1598.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 1567.05 | 1592.93 | 1595.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 1567.05 | 1592.93 | 1595.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 1534.00 | 1575.66 | 1586.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 1532.50 | 1528.83 | 1545.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:45:00 | 1534.95 | 1528.83 | 1545.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 1523.55 | 1528.91 | 1540.42 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1548.65 | 1541.99 | 1541.78 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 1540.00 | 1541.71 | 1541.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 1525.45 | 1538.46 | 1540.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 1503.30 | 1500.60 | 1512.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 1503.30 | 1500.60 | 1512.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1503.30 | 1500.60 | 1512.26 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 1532.90 | 1512.66 | 1510.29 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 1505.00 | 1508.82 | 1509.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 1493.40 | 1505.45 | 1507.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 1493.90 | 1463.76 | 1477.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 1493.90 | 1463.76 | 1477.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1493.90 | 1463.76 | 1477.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 1493.90 | 1463.76 | 1477.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1497.90 | 1470.59 | 1479.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1481.20 | 1470.59 | 1479.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1484.80 | 1475.59 | 1479.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 1484.80 | 1475.59 | 1479.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1477.50 | 1475.97 | 1479.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1474.60 | 1480.17 | 1480.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 1400.87 | 1408.03 | 1426.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1380.00 | 1379.86 | 1398.35 | SL hit (close>ema200) qty=0.50 sl=1379.86 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1402.65 | 1386.93 | 1385.35 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 1372.60 | 1383.38 | 1384.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 1364.20 | 1375.43 | 1380.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1377.95 | 1372.06 | 1376.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1377.95 | 1372.06 | 1376.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1377.95 | 1372.06 | 1376.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1377.95 | 1372.06 | 1376.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1380.00 | 1373.64 | 1377.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1380.00 | 1373.64 | 1377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1362.30 | 1371.38 | 1375.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:30:00 | 1356.00 | 1367.26 | 1373.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:00:00 | 1350.80 | 1367.26 | 1373.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1358.00 | 1349.44 | 1356.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 1356.10 | 1353.40 | 1357.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1362.00 | 1355.12 | 1357.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 1359.65 | 1355.12 | 1357.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 1383.85 | 1360.87 | 1360.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1383.85 | 1360.87 | 1360.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1388.80 | 1366.45 | 1362.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 1387.45 | 1389.42 | 1379.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:45:00 | 1386.70 | 1389.42 | 1379.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1383.20 | 1387.75 | 1380.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1383.20 | 1387.75 | 1380.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1391.80 | 1388.56 | 1381.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1384.70 | 1388.56 | 1381.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1372.55 | 1385.36 | 1380.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 1367.95 | 1385.36 | 1380.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1346.05 | 1377.50 | 1377.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 1346.05 | 1377.50 | 1377.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1317.85 | 1365.57 | 1371.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 1290.85 | 1350.63 | 1364.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 1311.50 | 1304.25 | 1329.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:45:00 | 1311.60 | 1304.25 | 1329.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1324.75 | 1299.90 | 1317.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1331.90 | 1299.90 | 1317.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1325.00 | 1304.92 | 1318.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:45:00 | 1332.30 | 1304.92 | 1318.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1311.60 | 1283.96 | 1294.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1311.60 | 1283.96 | 1294.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1295.05 | 1286.18 | 1294.83 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 1305.00 | 1299.32 | 1299.15 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1296.25 | 1298.71 | 1298.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 1289.15 | 1294.13 | 1296.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1310.05 | 1296.65 | 1297.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1310.05 | 1296.65 | 1297.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1310.05 | 1296.65 | 1297.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 1317.65 | 1296.65 | 1297.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1306.05 | 1298.53 | 1297.91 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 12:15:00 | 1295.25 | 1297.20 | 1297.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 13:15:00 | 1293.85 | 1296.53 | 1297.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1281.00 | 1273.12 | 1280.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1281.00 | 1273.12 | 1280.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1281.00 | 1273.12 | 1280.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 1281.00 | 1273.12 | 1280.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1274.65 | 1273.42 | 1280.19 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 1304.50 | 1286.16 | 1284.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1316.70 | 1296.25 | 1290.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 1317.25 | 1319.50 | 1310.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:30:00 | 1317.65 | 1319.50 | 1310.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1322.00 | 1319.01 | 1311.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:00:00 | 1322.75 | 1319.76 | 1312.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 1323.05 | 1319.63 | 1313.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 1322.55 | 1320.22 | 1314.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 1323.15 | 1320.28 | 1314.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1303.80 | 1323.41 | 1321.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 1303.80 | 1323.41 | 1321.97 | SL hit (close<static) qty=1.00 sl=1311.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 1310.90 | 1320.91 | 1320.96 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 1348.25 | 1322.41 | 1321.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 1352.80 | 1331.99 | 1325.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 12:15:00 | 1328.25 | 1333.47 | 1328.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 12:15:00 | 1328.25 | 1333.47 | 1328.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1328.25 | 1333.47 | 1328.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:30:00 | 1328.00 | 1333.47 | 1328.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1343.10 | 1335.39 | 1329.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:30:00 | 1337.95 | 1335.39 | 1329.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1361.35 | 1361.19 | 1352.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1356.20 | 1361.19 | 1352.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1351.10 | 1359.17 | 1352.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:00:00 | 1351.10 | 1359.17 | 1352.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1367.75 | 1360.89 | 1353.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 15:15:00 | 1370.00 | 1362.49 | 1354.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 1316.05 | 1351.08 | 1354.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1316.05 | 1351.08 | 1354.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1314.10 | 1338.85 | 1347.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 1243.25 | 1235.50 | 1252.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 1243.25 | 1235.50 | 1252.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1243.25 | 1235.50 | 1252.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1243.25 | 1235.50 | 1252.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1256.80 | 1239.94 | 1251.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:30:00 | 1256.35 | 1239.94 | 1251.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1256.25 | 1243.20 | 1252.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:45:00 | 1261.75 | 1243.20 | 1252.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1255.00 | 1249.62 | 1253.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 1249.35 | 1249.62 | 1253.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1260.20 | 1241.52 | 1244.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1266.70 | 1241.52 | 1244.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1259.25 | 1245.06 | 1245.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1263.25 | 1245.06 | 1245.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1271.00 | 1250.88 | 1248.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1282.55 | 1263.28 | 1255.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 1282.05 | 1286.04 | 1273.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:30:00 | 1292.30 | 1286.04 | 1273.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 1275.90 | 1284.02 | 1274.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:45:00 | 1272.25 | 1284.02 | 1274.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 1289.00 | 1285.02 | 1276.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:15:00 | 1296.00 | 1285.02 | 1276.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 14:30:00 | 1295.00 | 1288.66 | 1279.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 1295.00 | 1288.66 | 1279.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 1294.35 | 1288.34 | 1280.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1284.30 | 1287.53 | 1281.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:15:00 | 1292.25 | 1287.05 | 1281.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 1263.50 | 1281.04 | 1280.87 | SL hit (close<static) qty=1.00 sl=1272.55 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 10:15:00 | 1268.45 | 1278.52 | 1279.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1255.75 | 1267.97 | 1273.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1272.60 | 1258.24 | 1263.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1272.60 | 1258.24 | 1263.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1272.60 | 1258.24 | 1263.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1272.60 | 1258.24 | 1263.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1269.55 | 1260.50 | 1264.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 1275.50 | 1260.50 | 1264.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1275.25 | 1263.45 | 1265.14 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 1278.75 | 1268.32 | 1267.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 1285.00 | 1271.66 | 1268.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 11:15:00 | 1335.50 | 1340.33 | 1324.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 12:00:00 | 1335.50 | 1340.33 | 1324.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1325.05 | 1336.42 | 1325.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:45:00 | 1327.35 | 1336.42 | 1325.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1332.85 | 1335.71 | 1325.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:45:00 | 1330.10 | 1335.71 | 1325.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1322.00 | 1333.33 | 1326.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 1315.35 | 1333.33 | 1326.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1324.70 | 1331.61 | 1326.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 1322.35 | 1331.61 | 1326.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1364.15 | 1335.49 | 1329.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 1373.45 | 1354.57 | 1344.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:00:00 | 1376.05 | 1354.57 | 1344.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:30:00 | 1375.75 | 1380.03 | 1377.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 1347.45 | 1373.52 | 1374.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 1347.45 | 1373.52 | 1374.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 1342.40 | 1354.11 | 1359.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 1183.35 | 1140.84 | 1165.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 1183.35 | 1140.84 | 1165.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1183.35 | 1140.84 | 1165.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 1183.35 | 1140.84 | 1165.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1141.20 | 1140.91 | 1163.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:00:00 | 1138.65 | 1140.46 | 1160.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:00:00 | 1140.15 | 1140.40 | 1158.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 1140.10 | 1140.12 | 1157.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 11:45:00 | 1134.75 | 1140.97 | 1151.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1081.72 | 1098.63 | 1111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1083.14 | 1098.63 | 1111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1083.09 | 1098.63 | 1111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1078.01 | 1098.63 | 1111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 15:15:00 | 1091.90 | 1082.93 | 1096.06 | SL hit (close>ema200) qty=0.50 sl=1082.93 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 09:15:00 | 1113.10 | 1102.09 | 1101.25 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 1094.65 | 1103.92 | 1104.35 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 1139.10 | 1106.36 | 1102.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1151.05 | 1118.97 | 1109.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1195.15 | 1200.80 | 1186.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 1195.15 | 1200.80 | 1186.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1225.00 | 1205.64 | 1189.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:30:00 | 1237.15 | 1214.11 | 1194.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 1260.85 | 1271.53 | 1272.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 1260.85 | 1271.53 | 1272.17 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 1282.05 | 1273.63 | 1273.07 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1264.40 | 1271.93 | 1272.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 14:15:00 | 1261.40 | 1269.82 | 1271.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 10:15:00 | 1277.25 | 1268.38 | 1270.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 10:15:00 | 1277.25 | 1268.38 | 1270.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1277.25 | 1268.38 | 1270.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1277.25 | 1268.38 | 1270.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1260.15 | 1266.74 | 1269.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:45:00 | 1268.85 | 1266.74 | 1269.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1277.50 | 1268.89 | 1269.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:30:00 | 1274.15 | 1268.89 | 1269.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 13:15:00 | 1294.05 | 1273.92 | 1272.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 14:15:00 | 1315.25 | 1282.19 | 1276.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 11:15:00 | 1280.50 | 1286.98 | 1281.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 11:15:00 | 1280.50 | 1286.98 | 1281.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 1280.50 | 1286.98 | 1281.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:30:00 | 1284.15 | 1286.98 | 1281.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 1275.00 | 1284.58 | 1280.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 1275.00 | 1284.58 | 1280.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 1275.45 | 1282.75 | 1280.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:45:00 | 1273.50 | 1282.75 | 1280.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1267.50 | 1283.53 | 1281.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:00:00 | 1267.50 | 1283.53 | 1281.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1293.95 | 1285.62 | 1282.51 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 13:15:00 | 1271.65 | 1280.72 | 1280.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 14:15:00 | 1256.85 | 1275.95 | 1278.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 1126.90 | 1115.79 | 1137.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 1126.90 | 1115.79 | 1137.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1126.90 | 1115.79 | 1137.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 1127.60 | 1115.79 | 1137.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1106.55 | 1105.39 | 1116.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:30:00 | 1107.60 | 1105.39 | 1116.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1094.30 | 1102.99 | 1112.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 1106.85 | 1102.99 | 1112.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1092.05 | 1098.33 | 1108.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:45:00 | 1099.80 | 1098.33 | 1108.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1101.95 | 1090.64 | 1097.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1101.95 | 1090.64 | 1097.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1105.45 | 1093.60 | 1098.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 1105.45 | 1093.60 | 1098.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 1100.15 | 1094.91 | 1098.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1087.15 | 1096.93 | 1099.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 1093.90 | 1092.23 | 1095.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 1039.20 | 1053.88 | 1068.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 1032.79 | 1052.16 | 1066.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 1049.55 | 1048.17 | 1061.81 | SL hit (close>ema200) qty=0.50 sl=1048.17 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 1055.25 | 1038.32 | 1036.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 1078.15 | 1058.87 | 1049.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 14:15:00 | 1076.60 | 1077.90 | 1069.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 1076.60 | 1077.90 | 1069.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1078.60 | 1078.04 | 1069.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 1089.85 | 1079.21 | 1071.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:00:00 | 1081.50 | 1079.24 | 1075.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:45:00 | 1089.15 | 1080.72 | 1076.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 12:00:00 | 1082.25 | 1081.03 | 1076.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 1069.70 | 1078.76 | 1076.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 1069.70 | 1078.76 | 1076.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 1066.10 | 1076.23 | 1075.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 1067.20 | 1076.23 | 1075.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 1061.25 | 1073.23 | 1074.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 14:15:00 | 1061.25 | 1073.23 | 1074.05 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1084.50 | 1074.89 | 1074.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 1093.10 | 1083.92 | 1080.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 1134.25 | 1135.06 | 1118.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:00:00 | 1134.25 | 1135.06 | 1118.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1144.05 | 1152.29 | 1144.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1144.85 | 1152.29 | 1144.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1146.45 | 1151.12 | 1144.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1143.65 | 1151.12 | 1144.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1149.85 | 1150.87 | 1144.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:15:00 | 1142.95 | 1150.87 | 1144.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1143.70 | 1149.43 | 1144.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 1144.85 | 1149.43 | 1144.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1141.10 | 1147.77 | 1144.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 1141.55 | 1147.59 | 1144.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1144.60 | 1147.50 | 1145.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 1144.60 | 1147.50 | 1145.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1143.50 | 1146.70 | 1145.27 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1135.75 | 1143.62 | 1144.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 1130.90 | 1141.08 | 1142.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1140.90 | 1140.04 | 1141.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1140.90 | 1140.04 | 1141.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1140.90 | 1140.04 | 1141.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 1140.20 | 1140.04 | 1141.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1137.40 | 1139.51 | 1141.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 1142.55 | 1139.51 | 1141.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1132.30 | 1137.51 | 1140.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:45:00 | 1124.05 | 1134.13 | 1138.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 1132.00 | 1107.94 | 1106.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1132.00 | 1107.94 | 1106.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1136.25 | 1113.60 | 1109.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 15:15:00 | 1110.00 | 1117.45 | 1113.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 15:15:00 | 1110.00 | 1117.45 | 1113.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 1110.00 | 1117.45 | 1113.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 1100.75 | 1117.45 | 1113.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1093.20 | 1112.60 | 1111.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1093.20 | 1112.60 | 1111.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1107.50 | 1111.58 | 1111.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1096.85 | 1111.58 | 1111.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 1123.80 | 1118.66 | 1114.81 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1050.75 | 1105.68 | 1109.93 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1082.15 | 1075.57 | 1075.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1087.95 | 1078.49 | 1076.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 1127.20 | 1128.49 | 1116.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 1124.40 | 1128.49 | 1116.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1135.00 | 1131.40 | 1124.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 1136.20 | 1132.44 | 1125.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1147.00 | 1162.03 | 1162.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1147.00 | 1162.03 | 1162.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 12:15:00 | 1143.90 | 1158.40 | 1160.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 1158.00 | 1145.88 | 1150.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 15:15:00 | 1158.00 | 1145.88 | 1150.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1158.00 | 1145.88 | 1150.20 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1161.30 | 1152.85 | 1151.75 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 1143.60 | 1151.55 | 1152.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 1137.90 | 1148.82 | 1151.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 1150.00 | 1145.11 | 1148.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 15:15:00 | 1150.00 | 1145.11 | 1148.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1150.00 | 1145.11 | 1148.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 1127.50 | 1137.22 | 1141.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:15:00 | 1127.20 | 1135.14 | 1140.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:30:00 | 1119.10 | 1131.43 | 1137.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1170.90 | 1143.38 | 1140.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1170.90 | 1143.38 | 1140.48 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1133.00 | 1143.39 | 1143.66 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 1158.00 | 1142.67 | 1142.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1178.70 | 1149.87 | 1145.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1175.40 | 1180.29 | 1167.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:45:00 | 1175.40 | 1180.29 | 1167.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1166.20 | 1176.53 | 1168.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1166.20 | 1176.53 | 1168.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1164.00 | 1174.02 | 1167.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1164.00 | 1174.02 | 1167.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1161.00 | 1171.42 | 1167.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 1162.40 | 1171.42 | 1167.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1173.60 | 1171.95 | 1168.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 1172.20 | 1171.95 | 1168.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 1167.00 | 1171.41 | 1168.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:00:00 | 1167.00 | 1171.41 | 1168.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1169.20 | 1170.97 | 1168.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 1173.00 | 1170.57 | 1168.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1176.00 | 1171.50 | 1169.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 1173.50 | 1172.14 | 1170.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 1203.00 | 1228.35 | 1228.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 1203.00 | 1228.35 | 1228.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 1192.00 | 1205.64 | 1214.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1205.00 | 1203.89 | 1210.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 1205.00 | 1203.89 | 1210.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1192.00 | 1198.70 | 1205.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:30:00 | 1190.20 | 1197.40 | 1202.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1226.70 | 1202.46 | 1201.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1226.70 | 1202.46 | 1201.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 1227.90 | 1221.93 | 1217.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 1245.40 | 1246.98 | 1237.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 1245.40 | 1246.98 | 1237.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1239.30 | 1245.45 | 1237.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 1239.70 | 1245.45 | 1237.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1239.10 | 1244.18 | 1237.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 1238.60 | 1244.18 | 1237.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1235.40 | 1242.42 | 1237.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:15:00 | 1233.70 | 1242.42 | 1237.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1233.10 | 1240.56 | 1237.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1233.10 | 1240.56 | 1237.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1235.30 | 1237.90 | 1236.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:15:00 | 1239.40 | 1237.90 | 1236.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:45:00 | 1239.20 | 1239.66 | 1237.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 1275.00 | 1289.76 | 1290.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1275.00 | 1289.76 | 1290.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1261.70 | 1284.15 | 1288.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1289.90 | 1277.76 | 1283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1289.00 | 1280.00 | 1283.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 1282.00 | 1280.00 | 1283.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 1291.20 | 1285.43 | 1285.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 1291.20 | 1285.43 | 1285.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1294.00 | 1287.23 | 1286.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 1285.30 | 1287.68 | 1286.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1284.20 | 1286.98 | 1286.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1281.30 | 1286.98 | 1286.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 1276.10 | 1284.80 | 1285.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1273.80 | 1282.60 | 1284.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 1272.80 | 1280.35 | 1282.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1272.30 | 1277.96 | 1281.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 1272.80 | 1274.08 | 1278.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 1266.70 | 1268.90 | 1275.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1277.20 | 1266.22 | 1272.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:30:00 | 1271.40 | 1266.22 | 1272.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1271.00 | 1267.18 | 1271.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1270.00 | 1267.18 | 1271.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 1268.40 | 1267.54 | 1271.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 1269.60 | 1268.31 | 1271.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 1252.00 | 1246.85 | 1246.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 1252.00 | 1246.85 | 1246.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 1254.30 | 1250.65 | 1248.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1244.10 | 1250.83 | 1249.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1244.20 | 1249.51 | 1248.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 1247.20 | 1249.51 | 1248.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1250.50 | 1249.70 | 1249.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:30:00 | 1260.00 | 1250.44 | 1249.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 1240.20 | 1247.39 | 1248.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1240.20 | 1247.39 | 1248.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1237.80 | 1245.47 | 1247.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 1236.80 | 1236.50 | 1240.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 1236.80 | 1236.50 | 1240.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1243.90 | 1237.93 | 1240.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 1243.90 | 1237.93 | 1240.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1252.20 | 1240.79 | 1241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 1253.50 | 1240.79 | 1241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1248.30 | 1242.29 | 1242.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 1255.30 | 1248.21 | 1245.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 1242.00 | 1247.16 | 1245.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1239.70 | 1245.67 | 1244.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 1245.00 | 1246.12 | 1244.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 1246.50 | 1244.98 | 1244.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 1246.50 | 1244.98 | 1244.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 1237.20 | 1243.49 | 1243.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 1237.20 | 1243.49 | 1243.98 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1250.50 | 1244.13 | 1243.74 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1241.40 | 1244.53 | 1244.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 1233.50 | 1241.79 | 1243.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1244.00 | 1240.05 | 1241.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1243.00 | 1240.64 | 1241.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1241.40 | 1240.64 | 1241.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 1242.40 | 1240.99 | 1241.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1255.90 | 1243.97 | 1243.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1255.90 | 1243.97 | 1243.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 1267.00 | 1253.65 | 1248.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1257.70 | 1260.95 | 1255.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1257.70 | 1260.95 | 1255.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1253.90 | 1259.54 | 1255.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1253.90 | 1259.54 | 1255.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1255.50 | 1258.73 | 1255.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 1256.10 | 1259.58 | 1255.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1256.50 | 1258.45 | 1255.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 1260.00 | 1258.45 | 1255.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1251.80 | 1256.56 | 1255.70 | SL hit (close<static) qty=1.00 sl=1252.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1247.70 | 1254.79 | 1254.97 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 1258.00 | 1254.68 | 1254.37 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 1250.90 | 1254.26 | 1254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 1248.00 | 1253.01 | 1254.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1254.40 | 1253.29 | 1254.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1245.00 | 1251.63 | 1253.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 1242.50 | 1250.60 | 1252.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1242.10 | 1249.88 | 1252.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1241.30 | 1248.17 | 1251.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1239.80 | 1244.99 | 1249.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1243.90 | 1244.26 | 1248.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 1245.60 | 1244.26 | 1248.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1246.20 | 1244.65 | 1247.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 1248.00 | 1244.65 | 1247.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1245.00 | 1244.72 | 1247.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1248.50 | 1244.72 | 1247.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1246.10 | 1245.00 | 1247.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1258.40 | 1245.00 | 1247.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1248.30 | 1245.66 | 1247.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1244.90 | 1245.66 | 1247.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1241.10 | 1244.75 | 1246.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1245.80 | 1244.75 | 1246.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1246.60 | 1245.12 | 1246.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1248.20 | 1245.12 | 1246.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1241.50 | 1244.39 | 1246.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1239.30 | 1244.39 | 1246.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1234.90 | 1244.15 | 1246.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1180.38 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1179.99 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1179.23 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1177.81 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1177.33 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1173.15 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 1135.30 | 1135.10 | 1152.63 | SL hit (close>ema200) qty=0.50 sl=1135.10 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1158.60 | 1140.01 | 1137.61 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1128.90 | 1139.33 | 1139.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 1124.70 | 1133.60 | 1136.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 1129.60 | 1127.75 | 1132.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1117.40 | 1115.04 | 1122.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 1106.20 | 1111.58 | 1118.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 1106.00 | 1112.04 | 1113.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1107.90 | 1109.67 | 1112.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1107.10 | 1108.32 | 1109.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1114.30 | 1104.86 | 1106.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1114.30 | 1104.86 | 1106.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1108.60 | 1105.61 | 1106.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:15:00 | 1104.20 | 1105.61 | 1106.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 1102.80 | 1106.01 | 1106.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 1101.90 | 1105.93 | 1106.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:00:00 | 1103.00 | 1104.23 | 1105.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1104.00 | 1104.18 | 1105.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 1102.80 | 1104.18 | 1105.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1101.80 | 1103.70 | 1105.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 1108.90 | 1106.15 | 1105.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 1108.90 | 1106.15 | 1105.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 1113.10 | 1107.54 | 1106.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1121.50 | 1125.58 | 1119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1116.00 | 1123.66 | 1119.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1113.50 | 1123.66 | 1119.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1118.50 | 1122.63 | 1118.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 1126.40 | 1122.42 | 1119.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1132.40 | 1124.23 | 1120.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1119.10 | 1122.76 | 1123.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1119.10 | 1122.76 | 1123.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1109.50 | 1116.86 | 1119.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 1088.00 | 1081.33 | 1089.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 1066.80 | 1081.33 | 1089.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1073.50 | 1079.77 | 1088.36 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1101.00 | 1087.49 | 1086.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 11:15:00 | 1115.30 | 1101.20 | 1095.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1102.00 | 1104.24 | 1098.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1102.00 | 1104.24 | 1098.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1097.30 | 1102.85 | 1098.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1095.00 | 1101.28 | 1098.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1094.00 | 1099.82 | 1097.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1094.00 | 1099.82 | 1097.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1091.20 | 1098.10 | 1097.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 1090.60 | 1098.10 | 1097.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1104.70 | 1107.85 | 1104.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1104.70 | 1107.85 | 1104.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1105.00 | 1107.28 | 1104.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1100.90 | 1107.28 | 1104.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1105.70 | 1106.96 | 1104.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 1115.10 | 1108.29 | 1105.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 1121.80 | 1111.51 | 1108.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 1117.40 | 1123.61 | 1122.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 1112.00 | 1120.30 | 1120.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 1112.00 | 1120.30 | 1120.71 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 1129.30 | 1122.40 | 1121.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1133.80 | 1125.46 | 1123.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 1134.00 | 1136.23 | 1130.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 1134.00 | 1136.23 | 1130.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1143.00 | 1136.89 | 1131.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1149.00 | 1138.09 | 1132.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 1144.20 | 1138.09 | 1133.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 1143.30 | 1139.13 | 1134.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 1143.20 | 1139.95 | 1135.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1134.50 | 1139.17 | 1135.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1134.50 | 1139.17 | 1135.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1131.00 | 1137.53 | 1135.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-17 15:15:00 | 1131.00 | 1137.53 | 1135.53 | SL hit (close<static) qty=1.00 sl=1131.20 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 1112.10 | 1133.47 | 1136.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1106.70 | 1128.12 | 1133.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 1125.10 | 1124.61 | 1130.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:30:00 | 1125.30 | 1124.61 | 1130.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1125.90 | 1124.87 | 1130.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1122.10 | 1124.87 | 1130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1109.50 | 1110.50 | 1116.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 1101.40 | 1108.50 | 1115.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:15:00 | 1046.33 | 1057.30 | 1065.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1052.60 | 1051.06 | 1057.95 | SL hit (close>ema200) qty=0.50 sl=1051.06 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1025.20 | 1017.35 | 1016.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1032.00 | 1020.28 | 1018.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1020.90 | 1022.37 | 1019.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 1020.90 | 1022.37 | 1019.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1014.10 | 1026.30 | 1024.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1014.10 | 1026.30 | 1024.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1006.80 | 1022.40 | 1022.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1000.00 | 1017.92 | 1020.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1019.10 | 1012.18 | 1016.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1016.90 | 1013.12 | 1016.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1016.90 | 1013.12 | 1016.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1018.70 | 1014.24 | 1016.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 1018.70 | 1014.24 | 1016.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1030.00 | 1017.39 | 1017.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 1030.00 | 1017.39 | 1017.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1025.20 | 1018.95 | 1018.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1041.00 | 1023.36 | 1020.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1058.30 | 1059.10 | 1049.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 1058.30 | 1059.10 | 1049.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1060.10 | 1067.02 | 1064.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1060.10 | 1067.02 | 1064.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1061.00 | 1065.81 | 1064.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 1078.00 | 1068.67 | 1065.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1082.70 | 1092.66 | 1092.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1082.70 | 1092.66 | 1092.76 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1100.80 | 1093.67 | 1092.97 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 1086.50 | 1093.03 | 1093.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 1075.10 | 1087.60 | 1091.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1083.00 | 1081.23 | 1086.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 1083.00 | 1081.23 | 1086.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1040.70 | 1030.32 | 1041.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 1040.70 | 1030.32 | 1041.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1041.30 | 1032.52 | 1041.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:15:00 | 1040.10 | 1032.52 | 1041.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1040.10 | 1034.03 | 1041.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1047.30 | 1034.03 | 1041.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1035.80 | 1034.39 | 1040.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 1033.70 | 1034.65 | 1040.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 1060.00 | 1043.47 | 1043.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 1060.00 | 1043.47 | 1043.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 1065.70 | 1047.91 | 1045.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 1102.00 | 1106.41 | 1094.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 11:45:00 | 1101.00 | 1106.41 | 1094.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1113.20 | 1110.79 | 1103.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 1106.30 | 1110.79 | 1103.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1102.70 | 1109.17 | 1105.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1102.70 | 1109.17 | 1105.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1102.40 | 1107.82 | 1104.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 1111.10 | 1108.97 | 1105.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1110.30 | 1114.76 | 1114.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1110.30 | 1114.76 | 1114.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 1100.60 | 1111.93 | 1113.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1111.80 | 1107.14 | 1110.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1112.50 | 1108.21 | 1110.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:15:00 | 1110.70 | 1108.21 | 1110.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1126.00 | 1112.17 | 1111.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 1131.00 | 1115.93 | 1113.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1125.80 | 1127.48 | 1122.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 1125.80 | 1127.48 | 1122.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1123.90 | 1126.34 | 1122.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1122.10 | 1126.34 | 1122.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1121.10 | 1125.29 | 1122.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1121.10 | 1125.29 | 1122.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1122.90 | 1124.81 | 1122.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:30:00 | 1121.10 | 1124.81 | 1122.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1121.10 | 1124.07 | 1122.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1122.20 | 1124.07 | 1122.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1111.80 | 1121.62 | 1121.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1111.80 | 1121.62 | 1121.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 1110.00 | 1119.29 | 1120.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 11:15:00 | 1107.20 | 1116.87 | 1119.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 1118.60 | 1115.06 | 1117.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1110.90 | 1114.23 | 1117.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 1120.00 | 1114.23 | 1117.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1120.00 | 1115.38 | 1117.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1126.90 | 1115.38 | 1117.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1127.90 | 1117.89 | 1118.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1127.90 | 1117.89 | 1118.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1123.90 | 1119.09 | 1118.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 1133.30 | 1121.93 | 1120.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 10:15:00 | 1128.50 | 1128.66 | 1124.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 11:00:00 | 1128.50 | 1128.66 | 1124.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1126.70 | 1128.13 | 1125.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1126.70 | 1128.13 | 1125.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1129.20 | 1128.35 | 1126.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1129.90 | 1128.35 | 1126.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1126.40 | 1127.96 | 1126.13 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 1118.10 | 1124.18 | 1124.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 1117.20 | 1121.27 | 1122.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 1117.70 | 1122.70 | 1123.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1130.60 | 1124.28 | 1123.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 1130.60 | 1124.28 | 1123.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1134.00 | 1126.37 | 1125.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:45:00 | 1112.00 | 1125.30 | 1124.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1120.00 | 1124.24 | 1124.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1114.50 | 1121.74 | 1123.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1114.00 | 1113.43 | 1117.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:45:00 | 1114.60 | 1113.43 | 1117.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1115.70 | 1113.88 | 1117.33 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1130.30 | 1118.68 | 1118.61 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1111.00 | 1117.10 | 1117.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 1106.50 | 1114.98 | 1116.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1114.50 | 1111.53 | 1114.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1116.00 | 1112.42 | 1114.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1115.60 | 1112.42 | 1114.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1109.70 | 1111.88 | 1113.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1112.40 | 1111.88 | 1113.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1111.50 | 1111.77 | 1113.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:15:00 | 1110.00 | 1111.77 | 1113.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1110.00 | 1111.42 | 1113.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1114.50 | 1111.42 | 1113.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1121.00 | 1113.34 | 1113.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1122.70 | 1113.34 | 1113.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1119.30 | 1114.53 | 1114.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 1123.80 | 1119.63 | 1118.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1118.90 | 1119.63 | 1118.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1119.60 | 1119.62 | 1118.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 1123.40 | 1120.38 | 1118.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1117.00 | 1120.16 | 1119.50 | SL hit (close<static) qty=1.00 sl=1118.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1132.50 | 1138.44 | 1138.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1130.60 | 1136.50 | 1137.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1136.00 | 1135.56 | 1136.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1131.40 | 1134.73 | 1136.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 1130.80 | 1133.51 | 1135.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1126.10 | 1133.51 | 1135.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 13:15:00 | 1130.80 | 1126.49 | 1126.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 1130.80 | 1126.49 | 1126.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1133.60 | 1129.00 | 1127.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 1122.50 | 1128.34 | 1127.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1118.10 | 1126.29 | 1126.64 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1130.20 | 1127.07 | 1126.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 1138.00 | 1129.20 | 1127.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 1128.40 | 1129.23 | 1128.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1132.00 | 1129.78 | 1128.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1126.10 | 1129.78 | 1128.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1128.40 | 1129.51 | 1128.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 1128.40 | 1129.51 | 1128.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1121.70 | 1127.94 | 1127.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1121.70 | 1127.94 | 1127.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1131.30 | 1128.62 | 1128.23 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1126.70 | 1128.01 | 1128.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 1118.50 | 1124.66 | 1126.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1011.00 | 1009.83 | 1036.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:45:00 | 1011.80 | 1009.83 | 1036.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 908.00 | 890.35 | 895.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 895.50 | 890.35 | 895.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 887.00 | 889.68 | 894.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 886.10 | 889.68 | 894.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 885.40 | 888.32 | 893.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 884.30 | 887.29 | 891.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 841.79 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 841.13 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 840.08 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 11:15:00 | 797.49 | 837.16 | 860.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 872.90 | 838.72 | 835.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 941.60 | 859.30 | 845.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 885.00 | 893.73 | 881.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 887.05 | 891.59 | 884.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 887.05 | 891.59 | 884.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 875.65 | 891.59 | 884.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 887.45 | 890.76 | 884.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 890.95 | 890.80 | 885.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 880.45 | 891.93 | 888.42 | SL hit (close<static) qty=1.00 sl=881.65 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 857.75 | 881.06 | 884.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 849.75 | 874.79 | 880.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 894.45 | 874.02 | 879.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 877.10 | 874.64 | 879.02 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 889.65 | 881.32 | 881.01 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 878.60 | 880.77 | 880.79 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 882.00 | 881.02 | 880.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 886.75 | 882.26 | 881.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 866.70 | 881.67 | 881.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 857.40 | 876.81 | 879.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 852.60 | 871.97 | 876.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 876.30 | 872.30 | 876.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 889.45 | 875.73 | 877.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 889.45 | 875.73 | 877.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 881.40 | 876.86 | 877.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 897.55 | 876.86 | 877.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 903.20 | 882.13 | 880.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 910.50 | 887.80 | 882.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 1075.55 | 1083.50 | 1054.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 1075.55 | 1083.50 | 1054.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1062.50 | 1070.30 | 1062.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 1062.50 | 1070.30 | 1062.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1060.55 | 1068.35 | 1062.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 1060.55 | 1068.35 | 1062.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1061.45 | 1066.97 | 1062.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 1060.20 | 1066.97 | 1062.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1057.00 | 1064.97 | 1061.88 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1047.80 | 1057.95 | 1059.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1042.75 | 1054.91 | 1057.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 948.40 | 943.22 | 960.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 948.40 | 943.22 | 960.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 954.00 | 946.67 | 953.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 937.45 | 946.67 | 953.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 939.25 | 945.19 | 952.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 932.25 | 942.45 | 950.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 973.40 | 947.20 | 950.43 | SL hit (close>static) qty=1.00 sl=957.30 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 988.25 | 955.41 | 953.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 15:15:00 | 995.45 | 963.42 | 957.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:00:00 | 954.40 | 961.61 | 957.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 952.25 | 959.74 | 956.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 13:00:00 | 961.00 | 959.29 | 957.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:00:00 | 963.10 | 960.05 | 957.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:15:00 | 964.00 | 959.91 | 957.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 925.35 | 953.65 | 955.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 925.35 | 953.65 | 955.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 918.40 | 946.60 | 952.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 919.10 | 919.10 | 928.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 923.20 | 919.10 | 928.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 934.00 | 922.08 | 929.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:00:00 | 913.00 | 920.26 | 927.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 867.35 | 896.90 | 910.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 886.30 | 879.18 | 892.34 | SL hit (close>ema200) qty=0.50 sl=879.18 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 771.85 | 751.56 | 750.67 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 743.05 | 755.04 | 755.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 724.65 | 747.67 | 751.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 721.85 | 726.49 | 736.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 720.00 | 726.62 | 733.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 10:15:00 | 739.50 | 729.24 | 732.74 | SL hit (close>static) qty=1.00 sl=738.20 alert=retest2 |

### Cycle 133 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 748.30 | 736.48 | 735.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 750.35 | 739.26 | 736.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 778.65 | 780.46 | 767.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:45:00 | 783.00 | 780.46 | 767.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 824.00 | 826.79 | 819.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 818.60 | 826.79 | 819.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 818.00 | 825.03 | 819.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 794.75 | 825.03 | 819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 789.10 | 817.84 | 816.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 786.95 | 817.84 | 816.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 789.95 | 812.26 | 814.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 785.85 | 803.04 | 809.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 800.05 | 802.77 | 806.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:15:00 | 800.30 | 802.77 | 806.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 799.75 | 802.71 | 805.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 801.80 | 802.88 | 805.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 803.55 | 803.01 | 804.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:30:00 | 804.40 | 803.01 | 804.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 804.30 | 803.27 | 804.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 805.35 | 803.27 | 804.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 800.55 | 802.73 | 804.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 808.30 | 805.38 | 805.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 808.30 | 805.38 | 805.26 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 804.00 | 804.97 | 805.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 14:15:00 | 801.70 | 803.97 | 804.57 | Break + close below crossover candle low |

### Cycle 137 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 830.50 | 809.10 | 806.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 864.60 | 830.26 | 819.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 15:15:00 | 859.55 | 860.12 | 849.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 856.80 | 860.12 | 849.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 855.85 | 860.28 | 854.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 853.40 | 860.28 | 854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 852.00 | 858.62 | 853.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 855.35 | 858.62 | 853.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 846.85 | 856.27 | 853.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 841.85 | 856.27 | 853.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 831.00 | 851.21 | 851.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 829.30 | 844.16 | 847.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 867.00 | 849.58 | 848.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 884.20 | 868.62 | 864.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 867.25 | 875.17 | 870.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 868.60 | 873.85 | 870.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 865.65 | 873.85 | 870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 868.85 | 872.85 | 870.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:00:00 | 869.65 | 870.80 | 870.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 874.00 | 870.04 | 869.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 1244.95 | 2024-05-13 10:15:00 | 1261.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-22 09:15:00 | 1307.40 | 2024-05-24 12:15:00 | 1290.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-23 15:00:00 | 1301.85 | 2024-05-24 12:15:00 | 1290.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-28 11:15:00 | 1279.50 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-05-28 11:45:00 | 1279.80 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-05-28 12:30:00 | 1278.95 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-05-28 15:15:00 | 1272.00 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-05-30 10:45:00 | 1265.45 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-05-31 09:30:00 | 1265.80 | 2024-06-03 09:15:00 | 1292.05 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-06-11 13:45:00 | 1295.95 | 2024-06-18 09:15:00 | 1425.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 14:15:00 | 1295.95 | 2024-06-18 09:15:00 | 1425.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1296.25 | 2024-06-18 09:15:00 | 1425.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 09:45:00 | 1302.10 | 2024-06-18 09:15:00 | 1432.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 11:00:00 | 1323.25 | 2024-06-21 12:15:00 | 1335.10 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-06-13 15:00:00 | 1323.95 | 2024-06-21 12:15:00 | 1335.10 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2024-06-14 09:15:00 | 1323.55 | 2024-06-21 12:15:00 | 1335.10 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2024-06-14 13:15:00 | 1326.55 | 2024-06-21 12:15:00 | 1335.10 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-06-14 15:00:00 | 1358.00 | 2024-06-21 12:15:00 | 1335.10 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-07-08 09:15:00 | 1529.45 | 2024-07-11 09:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-07-10 11:15:00 | 1514.85 | 2024-07-11 09:15:00 | 1507.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-07-18 14:45:00 | 1474.90 | 2024-07-22 14:15:00 | 1496.20 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-18 15:15:00 | 1476.00 | 2024-07-22 14:15:00 | 1496.20 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-07-19 09:30:00 | 1476.20 | 2024-07-22 14:15:00 | 1496.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-07-19 11:30:00 | 1475.25 | 2024-07-22 14:15:00 | 1496.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-07-25 14:45:00 | 1476.30 | 2024-07-26 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-08-06 14:30:00 | 1407.00 | 2024-08-07 10:15:00 | 1429.80 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest1 | 2024-08-16 09:15:00 | 1507.85 | 2024-08-21 09:15:00 | 1509.00 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest1 | 2024-08-16 13:15:00 | 1497.00 | 2024-08-21 09:15:00 | 1509.00 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-08-23 10:15:00 | 1495.05 | 2024-08-30 09:15:00 | 1500.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-23 13:15:00 | 1494.30 | 2024-08-30 09:15:00 | 1500.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-08-23 14:30:00 | 1490.15 | 2024-08-30 09:15:00 | 1500.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-09-11 10:45:00 | 1469.75 | 2024-09-19 14:15:00 | 1476.05 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-09-12 10:00:00 | 1472.65 | 2024-09-19 14:15:00 | 1476.05 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-09-12 10:30:00 | 1471.00 | 2024-09-19 14:15:00 | 1476.05 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-09-30 10:45:00 | 1598.60 | 2024-10-03 13:15:00 | 1567.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-09-30 12:00:00 | 1596.15 | 2024-10-03 13:15:00 | 1567.05 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-10-01 14:45:00 | 1601.55 | 2024-10-03 13:15:00 | 1567.05 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-10-03 10:45:00 | 1600.00 | 2024-10-03 13:15:00 | 1567.05 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1474.60 | 2024-10-25 09:15:00 | 1400.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1474.60 | 2024-10-28 10:15:00 | 1380.00 | STOP_HIT | 0.50 | 6.42% |
| SELL | retest2 | 2024-11-04 10:30:00 | 1356.00 | 2024-11-06 11:15:00 | 1383.85 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-11-04 11:00:00 | 1350.80 | 2024-11-06 11:15:00 | 1383.85 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-11-05 15:15:00 | 1358.00 | 2024-11-06 11:15:00 | 1383.85 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-06 09:30:00 | 1356.10 | 2024-11-06 11:15:00 | 1383.85 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-11-27 10:00:00 | 1322.75 | 2024-11-29 09:15:00 | 1303.80 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-11-27 10:30:00 | 1323.05 | 2024-11-29 09:15:00 | 1303.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-27 12:00:00 | 1322.55 | 2024-11-29 09:15:00 | 1303.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-11-27 12:45:00 | 1323.15 | 2024-11-29 09:15:00 | 1303.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-04 15:15:00 | 1370.00 | 2024-12-06 09:15:00 | 1316.05 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-12-18 13:15:00 | 1296.00 | 2024-12-20 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-12-18 14:30:00 | 1295.00 | 2024-12-20 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-12-18 15:15:00 | 1295.00 | 2024-12-20 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-12-19 09:30:00 | 1294.35 | 2024-12-20 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-12-19 12:15:00 | 1292.25 | 2024-12-20 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-01-02 10:30:00 | 1373.45 | 2025-01-06 13:15:00 | 1347.45 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-01-02 11:00:00 | 1376.05 | 2025-01-06 13:15:00 | 1347.45 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-01-06 12:30:00 | 1375.75 | 2025-01-06 13:15:00 | 1347.45 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-01-16 12:00:00 | 1138.65 | 2025-01-22 09:15:00 | 1081.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:00:00 | 1140.15 | 2025-01-22 09:15:00 | 1083.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:45:00 | 1140.10 | 2025-01-22 09:15:00 | 1083.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 11:45:00 | 1134.75 | 2025-01-22 09:15:00 | 1078.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 12:00:00 | 1138.65 | 2025-01-22 15:15:00 | 1091.90 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-01-16 13:00:00 | 1140.15 | 2025-01-22 15:15:00 | 1091.90 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-01-16 13:45:00 | 1140.10 | 2025-01-22 15:15:00 | 1091.90 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-01-17 11:45:00 | 1134.75 | 2025-01-22 15:15:00 | 1091.90 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-01-23 11:45:00 | 1103.45 | 2025-01-24 09:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-23 15:00:00 | 1103.45 | 2025-01-24 09:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-01 14:30:00 | 1237.15 | 2025-02-07 09:15:00 | 1260.85 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2025-02-25 09:15:00 | 1087.15 | 2025-03-03 09:15:00 | 1039.20 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2025-02-25 15:15:00 | 1093.90 | 2025-03-03 10:15:00 | 1032.79 | PARTIAL | 0.50 | 5.59% |
| SELL | retest2 | 2025-02-25 09:15:00 | 1087.15 | 2025-03-03 12:15:00 | 1049.55 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-02-25 15:15:00 | 1093.90 | 2025-03-03 12:15:00 | 1049.55 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest2 | 2025-03-13 09:45:00 | 1089.85 | 2025-03-17 14:15:00 | 1061.25 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-03-17 10:00:00 | 1081.50 | 2025-03-17 14:15:00 | 1061.25 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-03-17 10:45:00 | 1089.15 | 2025-03-17 14:15:00 | 1061.25 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-03-17 12:00:00 | 1082.25 | 2025-03-17 14:15:00 | 1061.25 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-28 10:45:00 | 1124.05 | 2025-04-03 09:15:00 | 1132.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-04-21 09:45:00 | 1136.20 | 2025-04-25 11:15:00 | 1147.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1127.50 | 2025-05-08 09:15:00 | 1170.90 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-05-06 14:15:00 | 1127.20 | 2025-05-08 09:15:00 | 1170.90 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-05-07 09:30:00 | 1119.10 | 2025-05-08 09:15:00 | 1170.90 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2025-05-14 15:15:00 | 1173.00 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2025-05-15 09:30:00 | 1176.00 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2025-05-15 14:15:00 | 1173.50 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-05-26 13:30:00 | 1190.20 | 2025-05-28 09:15:00 | 1226.70 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-06-05 11:15:00 | 1239.40 | 2025-06-12 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-06-05 11:45:00 | 1239.20 | 2025-06-12 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 2.89% |
| SELL | retest2 | 2025-06-13 15:15:00 | 1282.00 | 2025-06-16 12:15:00 | 1291.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-06-18 12:45:00 | 1272.80 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1272.30 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-06-19 09:45:00 | 1272.80 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-19 11:30:00 | 1266.70 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-06-20 09:15:00 | 1270.00 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2025-06-20 09:45:00 | 1268.40 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-06-20 11:30:00 | 1269.60 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-06-27 12:30:00 | 1260.00 | 2025-06-27 15:15:00 | 1240.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1245.00 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-03 09:30:00 | 1246.50 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-03 10:15:00 | 1246.50 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-08 14:15:00 | 1241.40 | 2025-07-09 09:15:00 | 1255.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-08 15:00:00 | 1242.40 | 2025-07-09 09:15:00 | 1255.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-11 12:30:00 | 1256.10 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-11 14:45:00 | 1256.50 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-11 15:15:00 | 1260.00 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-17 14:30:00 | 1242.50 | 2025-07-25 11:15:00 | 1180.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1242.10 | 2025-07-25 11:15:00 | 1179.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1241.30 | 2025-07-25 11:15:00 | 1179.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1239.80 | 2025-07-25 14:15:00 | 1177.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1239.30 | 2025-07-25 14:15:00 | 1177.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 15:15:00 | 1234.90 | 2025-07-25 14:15:00 | 1173.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 14:30:00 | 1242.50 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.63% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1242.10 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.60% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1241.30 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.54% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1239.80 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.43% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1239.30 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.39% |
| SELL | retest2 | 2025-07-21 15:15:00 | 1234.90 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.07% |
| SELL | retest2 | 2025-08-08 14:00:00 | 1106.20 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-08-12 15:15:00 | 1106.00 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-08-13 09:30:00 | 1107.90 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-08-14 09:45:00 | 1107.10 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-18 11:15:00 | 1104.20 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-18 15:00:00 | 1102.80 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-19 10:15:00 | 1101.90 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-19 13:00:00 | 1103.00 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-08-22 15:00:00 | 1126.40 | 2025-08-26 11:15:00 | 1119.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1132.40 | 2025-08-26 11:15:00 | 1119.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-09 10:30:00 | 1115.10 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-10 11:00:00 | 1121.80 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-12 12:15:00 | 1117.40 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1149.00 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-09-17 10:15:00 | 1144.20 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-17 11:00:00 | 1143.30 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-17 12:00:00 | 1143.20 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-18 13:15:00 | 1144.50 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-09-18 14:45:00 | 1154.30 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-18 15:15:00 | 1150.00 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-19 11:00:00 | 1145.90 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-24 10:30:00 | 1101.40 | 2025-10-01 10:15:00 | 1046.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:30:00 | 1101.40 | 2025-10-03 09:15:00 | 1052.60 | STOP_HIT | 0.50 | 4.43% |
| BUY | retest2 | 2025-10-24 09:45:00 | 1078.00 | 2025-10-31 14:15:00 | 1082.70 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-11-12 10:30:00 | 1033.70 | 2025-11-12 13:15:00 | 1060.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-19 11:45:00 | 1111.10 | 2025-11-24 14:15:00 | 1110.30 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-05 09:30:00 | 1117.70 | 2025-12-05 10:15:00 | 1130.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-16 12:00:00 | 1123.40 | 2025-12-17 09:15:00 | 1117.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 11:00:00 | 1123.90 | 2025-12-26 11:15:00 | 1132.50 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-12-30 10:30:00 | 1130.80 | 2026-01-01 13:15:00 | 1130.80 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-30 11:15:00 | 1126.10 | 2026-01-01 13:15:00 | 1130.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-22 10:15:00 | 886.10 | 2026-01-27 09:15:00 | 841.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 885.40 | 2026-01-27 09:15:00 | 841.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 884.30 | 2026-01-27 09:15:00 | 840.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 886.10 | 2026-01-27 11:15:00 | 797.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 885.40 | 2026-01-27 11:15:00 | 796.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 884.30 | 2026-01-27 11:15:00 | 795.87 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-02 15:00:00 | 890.95 | 2026-02-03 12:15:00 | 880.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-02-03 14:15:00 | 891.00 | 2026-02-03 14:15:00 | 876.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-27 10:45:00 | 932.25 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2026-03-02 13:00:00 | 961.00 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-03-02 14:00:00 | 963.10 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-03-02 15:15:00 | 964.00 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-03-06 10:00:00 | 913.00 | 2026-03-09 09:15:00 | 867.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:00:00 | 913.00 | 2026-03-10 09:15:00 | 886.30 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-04-01 11:00:00 | 721.85 | 2026-04-02 10:15:00 | 739.50 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-04-01 14:45:00 | 720.00 | 2026-04-02 10:15:00 | 739.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-15 11:45:00 | 800.05 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-15 12:15:00 | 800.30 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-04-15 15:00:00 | 799.75 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-16 09:30:00 | 801.80 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -0.81% |
