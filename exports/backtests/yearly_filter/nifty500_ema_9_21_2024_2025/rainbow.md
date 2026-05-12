# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1311.00
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
| ALERT2 | 88 |
| ALERT2_SKIP | 47 |
| ALERT3 | 220 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 151 |
| PARTIAL | 23 |
| TARGET_HIT | 2 |
| STOP_HIT | 156 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 181 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 83 / 98
- **Target hits / Stop hits / Partials:** 2 / 156 / 23
- **Avg / median % per leg:** 0.51% / -0.40%
- **Sum % (uncompounded):** 92.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 26 | 38.2% | 1 | 67 | 0 | -0.14% | -9.6% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 6 | 0 | 0.83% | 5.0% |
| BUY @ 3rd Alert (retest2) | 62 | 23 | 37.1% | 1 | 61 | 0 | -0.24% | -14.6% |
| SELL (all) | 113 | 57 | 50.4% | 1 | 89 | 23 | 0.91% | 102.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.14% | -4.3% |
| SELL @ 3rd Alert (retest2) | 111 | 57 | 51.4% | 1 | 87 | 23 | 0.96% | 106.6% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 8 | 0 | 0.09% | 0.7% |
| retest2 (combined) | 173 | 80 | 46.2% | 2 | 148 | 23 | 0.53% | 92.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 1391.30 | 1358.00 | 1354.06 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 1335.00 | 1350.77 | 1352.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 1328.70 | 1342.96 | 1348.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 14:15:00 | 1256.45 | 1231.82 | 1253.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 1256.45 | 1231.82 | 1253.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1256.45 | 1231.82 | 1253.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1256.45 | 1231.82 | 1253.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1255.50 | 1236.56 | 1253.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1238.05 | 1236.56 | 1253.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1220.10 | 1233.27 | 1250.24 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 1289.05 | 1258.43 | 1255.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 1250.00 | 1255.99 | 1256.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 1247.00 | 1254.19 | 1255.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1237.55 | 1232.53 | 1239.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 1237.55 | 1232.53 | 1239.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1237.55 | 1232.53 | 1239.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 1237.10 | 1232.53 | 1239.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1243.45 | 1234.71 | 1239.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 1243.45 | 1234.71 | 1239.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1277.30 | 1243.23 | 1243.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 1277.30 | 1243.23 | 1243.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 1308.40 | 1256.26 | 1249.25 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1225.20 | 1260.49 | 1262.92 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1291.35 | 1266.71 | 1263.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1312.50 | 1282.24 | 1273.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 13:15:00 | 1306.60 | 1306.61 | 1295.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:45:00 | 1308.60 | 1306.61 | 1295.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1299.95 | 1309.27 | 1303.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:45:00 | 1301.90 | 1309.27 | 1303.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1299.05 | 1307.23 | 1303.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1303.30 | 1307.23 | 1303.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1304.20 | 1306.86 | 1303.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 1310.00 | 1307.49 | 1304.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 1296.00 | 1304.65 | 1303.70 | SL hit (close<static) qty=1.00 sl=1300.05 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 1309.95 | 1313.50 | 1313.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 13:15:00 | 1305.00 | 1311.55 | 1312.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 1316.35 | 1312.34 | 1312.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1316.35 | 1312.34 | 1312.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1316.35 | 1312.34 | 1312.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 10:30:00 | 1311.00 | 1311.95 | 1312.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 12:15:00 | 1318.90 | 1313.83 | 1313.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 1318.90 | 1313.83 | 1313.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1324.65 | 1316.10 | 1314.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1302.00 | 1314.38 | 1314.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1302.00 | 1314.38 | 1314.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1302.00 | 1314.38 | 1314.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 1302.00 | 1314.38 | 1314.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 1304.45 | 1312.40 | 1313.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 14:15:00 | 1291.75 | 1300.13 | 1304.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 1296.60 | 1292.42 | 1298.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 1296.60 | 1292.42 | 1298.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1296.60 | 1292.42 | 1298.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 1296.60 | 1292.42 | 1298.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1300.90 | 1294.11 | 1298.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1300.90 | 1294.11 | 1298.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1294.20 | 1294.13 | 1298.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1302.70 | 1294.13 | 1298.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1295.20 | 1294.35 | 1297.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:00:00 | 1292.00 | 1294.06 | 1297.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:15:00 | 1288.95 | 1293.76 | 1296.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1291.85 | 1278.27 | 1282.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:00:00 | 1290.60 | 1282.40 | 1283.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1277.15 | 1280.24 | 1282.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 1285.45 | 1280.24 | 1282.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1260.70 | 1255.69 | 1262.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 1260.70 | 1255.69 | 1262.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1262.90 | 1257.13 | 1262.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 1263.10 | 1257.13 | 1262.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 1257.60 | 1257.22 | 1262.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 1254.75 | 1257.22 | 1262.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:30:00 | 1250.00 | 1255.83 | 1260.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 15:00:00 | 1252.35 | 1255.13 | 1260.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:15:00 | 1253.25 | 1255.59 | 1259.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1250.10 | 1254.49 | 1259.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 1275.45 | 1259.65 | 1259.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1275.45 | 1259.65 | 1259.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 11:15:00 | 1285.75 | 1267.33 | 1262.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1291.95 | 1307.00 | 1294.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1291.95 | 1307.00 | 1294.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1291.95 | 1307.00 | 1294.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:00:00 | 1291.95 | 1307.00 | 1294.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1290.10 | 1303.62 | 1294.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 1294.90 | 1303.62 | 1294.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1284.40 | 1299.78 | 1293.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 1284.40 | 1299.78 | 1293.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 1292.00 | 1294.98 | 1292.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 1283.05 | 1294.98 | 1292.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1278.50 | 1291.68 | 1291.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 1272.25 | 1291.68 | 1291.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 1269.85 | 1287.32 | 1289.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 1265.15 | 1272.53 | 1278.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 1273.00 | 1260.72 | 1266.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 15:15:00 | 1273.00 | 1260.72 | 1266.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1273.00 | 1260.72 | 1266.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 1250.55 | 1259.29 | 1265.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 13:00:00 | 1250.00 | 1256.07 | 1262.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:45:00 | 1250.00 | 1252.35 | 1258.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 14:15:00 | 1188.02 | 1203.91 | 1217.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 14:15:00 | 1187.50 | 1203.91 | 1217.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 14:15:00 | 1187.50 | 1203.91 | 1217.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 1176.45 | 1171.66 | 1186.85 | SL hit (close>ema200) qty=0.50 sl=1171.66 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1163.00 | 1136.31 | 1133.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 1170.10 | 1150.71 | 1141.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 1180.80 | 1185.98 | 1178.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 1180.80 | 1185.98 | 1178.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1180.80 | 1185.98 | 1178.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1180.80 | 1185.98 | 1178.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1183.00 | 1185.38 | 1179.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:00:00 | 1195.80 | 1186.07 | 1181.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 1191.75 | 1188.00 | 1183.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1170.95 | 1184.38 | 1184.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1170.95 | 1184.38 | 1184.94 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 1190.90 | 1183.72 | 1182.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 1197.60 | 1186.50 | 1184.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 09:15:00 | 1190.45 | 1190.47 | 1187.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1190.45 | 1190.47 | 1187.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1190.45 | 1190.47 | 1187.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 1183.00 | 1190.47 | 1187.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1182.50 | 1188.88 | 1186.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 1182.50 | 1188.88 | 1186.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1183.50 | 1187.80 | 1186.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 1178.25 | 1187.80 | 1186.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 1184.25 | 1187.09 | 1186.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:15:00 | 1183.40 | 1187.09 | 1186.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1182.85 | 1186.24 | 1185.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1180.30 | 1186.24 | 1185.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 14:15:00 | 1178.95 | 1184.78 | 1185.29 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 1206.00 | 1188.57 | 1186.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1227.15 | 1209.22 | 1199.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1202.00 | 1228.48 | 1221.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1202.00 | 1228.48 | 1221.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1202.00 | 1228.48 | 1221.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:45:00 | 1204.40 | 1228.48 | 1221.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1202.00 | 1223.18 | 1219.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 1203.00 | 1223.18 | 1219.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 1202.95 | 1215.39 | 1216.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1194.85 | 1206.78 | 1211.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 1220.30 | 1207.74 | 1211.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 11:15:00 | 1220.30 | 1207.74 | 1211.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1220.30 | 1207.74 | 1211.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 1220.30 | 1207.74 | 1211.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1220.50 | 1210.29 | 1211.95 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 1219.30 | 1213.66 | 1213.28 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 09:15:00 | 1186.40 | 1208.42 | 1210.98 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 1212.00 | 1202.88 | 1201.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 12:15:00 | 1215.35 | 1210.11 | 1207.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 13:15:00 | 1235.00 | 1236.96 | 1229.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 13:30:00 | 1235.10 | 1236.96 | 1229.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 1235.00 | 1235.91 | 1230.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:30:00 | 1245.50 | 1236.50 | 1231.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:15:00 | 1250.00 | 1237.17 | 1233.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:45:00 | 1246.90 | 1242.57 | 1236.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 14:00:00 | 1243.00 | 1244.48 | 1239.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1238.10 | 1243.20 | 1239.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1238.10 | 1243.20 | 1239.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1237.00 | 1241.96 | 1239.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 1243.05 | 1241.96 | 1239.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 1247.35 | 1241.52 | 1239.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 1242.00 | 1242.62 | 1241.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 1242.10 | 1241.27 | 1240.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1246.90 | 1242.39 | 1241.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 13:45:00 | 1251.60 | 1244.95 | 1242.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 1261.90 | 1263.99 | 1264.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 1261.90 | 1263.99 | 1264.24 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 15:15:00 | 1266.15 | 1264.42 | 1264.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1290.00 | 1269.54 | 1266.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 1285.00 | 1285.89 | 1279.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:30:00 | 1287.25 | 1285.89 | 1279.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1279.25 | 1284.65 | 1279.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 1280.15 | 1284.65 | 1279.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 1282.95 | 1284.31 | 1280.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 1265.00 | 1284.31 | 1280.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1266.00 | 1280.65 | 1278.83 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 1269.70 | 1277.18 | 1277.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 1266.80 | 1275.10 | 1276.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 1276.40 | 1275.14 | 1276.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 1276.40 | 1275.14 | 1276.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1276.40 | 1275.14 | 1276.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1276.40 | 1275.14 | 1276.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1278.00 | 1275.71 | 1276.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1311.95 | 1275.71 | 1276.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1297.90 | 1280.15 | 1278.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 1316.25 | 1302.58 | 1292.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1330.65 | 1332.28 | 1314.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:45:00 | 1330.25 | 1332.28 | 1314.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1314.35 | 1328.49 | 1316.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 1314.00 | 1328.49 | 1316.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1310.95 | 1324.98 | 1315.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 1313.40 | 1324.98 | 1315.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1314.50 | 1322.88 | 1315.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1329.95 | 1316.43 | 1314.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 1319.00 | 1324.07 | 1320.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1329.50 | 1322.66 | 1320.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 14:15:00 | 1302.80 | 1321.85 | 1321.33 | SL hit (close<static) qty=1.00 sl=1308.10 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 1317.55 | 1320.99 | 1320.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 1299.60 | 1309.68 | 1314.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1310.05 | 1301.14 | 1307.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1310.05 | 1301.14 | 1307.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1310.05 | 1301.14 | 1307.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 1312.00 | 1301.14 | 1307.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 1370.90 | 1315.09 | 1313.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 13:15:00 | 1398.40 | 1343.84 | 1327.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 09:15:00 | 1344.45 | 1354.80 | 1337.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1344.45 | 1354.80 | 1337.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1344.45 | 1354.80 | 1337.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 1344.85 | 1354.80 | 1337.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1373.40 | 1370.08 | 1361.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 1389.45 | 1373.60 | 1365.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 10:45:00 | 1393.60 | 1377.61 | 1369.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1388.80 | 1378.10 | 1372.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 15:15:00 | 1390.00 | 1404.97 | 1405.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 1390.00 | 1404.97 | 1405.52 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 1432.30 | 1409.15 | 1406.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 1456.50 | 1418.62 | 1411.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1403.45 | 1419.86 | 1413.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1403.45 | 1419.86 | 1413.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1403.45 | 1419.86 | 1413.39 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 1390.25 | 1406.66 | 1408.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 1380.90 | 1394.96 | 1401.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 13:15:00 | 1371.10 | 1368.57 | 1381.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:00:00 | 1371.10 | 1368.57 | 1381.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1384.85 | 1373.92 | 1381.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 1381.70 | 1373.92 | 1381.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1407.90 | 1380.72 | 1383.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 1402.00 | 1380.72 | 1383.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 1397.95 | 1386.28 | 1385.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1410.00 | 1394.68 | 1390.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 1398.50 | 1405.51 | 1399.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 1398.50 | 1405.51 | 1399.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1398.50 | 1405.51 | 1399.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 12:00:00 | 1414.35 | 1406.79 | 1401.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:45:00 | 1413.85 | 1405.39 | 1402.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:15:00 | 1410.80 | 1404.57 | 1402.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:45:00 | 1410.05 | 1405.90 | 1403.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1404.00 | 1406.89 | 1404.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1436.60 | 1404.82 | 1404.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1435.60 | 1444.05 | 1444.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1435.60 | 1444.05 | 1444.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 15:15:00 | 1428.50 | 1437.56 | 1440.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1373.45 | 1372.24 | 1392.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:30:00 | 1362.85 | 1372.24 | 1392.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1375.80 | 1373.65 | 1389.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1380.65 | 1373.65 | 1389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1375.85 | 1374.95 | 1387.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1375.85 | 1374.95 | 1387.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1385.10 | 1377.33 | 1384.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 1392.75 | 1377.33 | 1384.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1375.85 | 1377.03 | 1383.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:45:00 | 1361.50 | 1372.33 | 1381.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:30:00 | 1363.75 | 1370.95 | 1379.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 1410.05 | 1381.32 | 1380.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 14:15:00 | 1410.05 | 1381.32 | 1380.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 09:15:00 | 1450.00 | 1395.00 | 1387.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 13:15:00 | 1471.00 | 1477.55 | 1451.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 14:00:00 | 1471.00 | 1477.55 | 1451.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1480.20 | 1474.11 | 1455.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 1475.35 | 1474.11 | 1455.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1547.00 | 1571.68 | 1537.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1547.00 | 1571.68 | 1537.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1564.15 | 1601.44 | 1577.37 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 15:15:00 | 1546.00 | 1565.99 | 1567.29 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1576.75 | 1568.30 | 1568.12 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 1567.10 | 1568.11 | 1568.21 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 10:15:00 | 1583.40 | 1571.17 | 1569.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 13:15:00 | 1599.15 | 1580.17 | 1574.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 1577.00 | 1585.48 | 1579.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 11:15:00 | 1577.00 | 1585.48 | 1579.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1577.00 | 1585.48 | 1579.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 1577.00 | 1585.48 | 1579.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 1580.00 | 1584.39 | 1579.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:45:00 | 1585.25 | 1584.09 | 1580.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 1569.30 | 1580.64 | 1579.57 | SL hit (close<static) qty=1.00 sl=1574.75 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 1574.40 | 1578.63 | 1578.85 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 1582.15 | 1579.34 | 1579.15 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 1572.85 | 1578.04 | 1578.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 1568.00 | 1576.03 | 1577.61 | Break + close below crossover candle low |

### Cycle 41 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 1600.50 | 1580.92 | 1579.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 1614.40 | 1592.11 | 1585.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1591.70 | 1607.40 | 1597.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 1591.70 | 1607.40 | 1597.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1591.70 | 1607.40 | 1597.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 1635.15 | 1602.22 | 1598.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 1583.90 | 1610.58 | 1610.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 1583.90 | 1610.58 | 1610.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 09:15:00 | 1574.70 | 1595.35 | 1602.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 10:15:00 | 1582.80 | 1579.55 | 1588.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 10:15:00 | 1582.80 | 1579.55 | 1588.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1582.80 | 1579.55 | 1588.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 1584.00 | 1579.55 | 1588.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 1570.10 | 1577.66 | 1587.09 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1655.40 | 1597.92 | 1591.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 1666.00 | 1620.77 | 1603.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 1630.85 | 1630.92 | 1613.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 1630.85 | 1630.92 | 1613.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1629.85 | 1638.12 | 1629.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:45:00 | 1629.10 | 1638.12 | 1629.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1630.80 | 1636.66 | 1629.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:15:00 | 1609.95 | 1636.66 | 1629.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1607.00 | 1630.73 | 1627.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 1607.00 | 1630.73 | 1627.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 1606.90 | 1625.96 | 1625.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:30:00 | 1604.60 | 1625.96 | 1625.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 1612.75 | 1623.32 | 1624.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 1598.20 | 1618.30 | 1621.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1559.05 | 1546.40 | 1570.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 1559.05 | 1546.40 | 1570.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1587.40 | 1558.14 | 1568.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 1587.40 | 1558.14 | 1568.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 1586.00 | 1563.71 | 1570.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 1583.90 | 1563.71 | 1570.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 1623.90 | 1584.53 | 1579.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1648.25 | 1616.06 | 1599.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1664.20 | 1669.59 | 1650.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 1656.80 | 1669.59 | 1650.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 1651.00 | 1663.15 | 1653.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:15:00 | 1633.40 | 1663.15 | 1653.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1648.30 | 1660.18 | 1652.94 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 10:15:00 | 1631.30 | 1648.39 | 1648.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1627.00 | 1644.11 | 1646.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 1642.05 | 1640.00 | 1643.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 14:15:00 | 1642.05 | 1640.00 | 1643.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1642.05 | 1640.00 | 1643.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 1642.05 | 1640.00 | 1643.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1646.90 | 1641.38 | 1644.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 1632.85 | 1641.38 | 1644.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1665.05 | 1646.11 | 1646.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 1665.05 | 1646.11 | 1646.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 1668.35 | 1650.56 | 1648.14 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 1633.75 | 1650.67 | 1652.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 1626.00 | 1634.23 | 1640.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 15:15:00 | 1629.00 | 1627.71 | 1634.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 1615.55 | 1627.71 | 1634.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1608.30 | 1623.82 | 1631.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 1604.25 | 1623.82 | 1631.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 1606.00 | 1617.21 | 1627.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:00:00 | 1602.05 | 1614.17 | 1624.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 1603.10 | 1618.79 | 1625.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1603.10 | 1615.65 | 1623.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 1595.00 | 1615.65 | 1623.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 1601.00 | 1597.21 | 1603.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 1591.80 | 1597.49 | 1601.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 1595.30 | 1597.35 | 1599.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1595.30 | 1596.94 | 1599.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1577.90 | 1596.94 | 1599.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:45:00 | 1579.95 | 1593.22 | 1597.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1524.04 | 1549.92 | 1566.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1525.70 | 1549.92 | 1566.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1521.95 | 1549.92 | 1566.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1522.94 | 1549.92 | 1566.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1520.95 | 1549.92 | 1566.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:15:00 | 1515.25 | 1537.98 | 1558.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:15:00 | 1512.21 | 1537.98 | 1558.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:15:00 | 1515.53 | 1537.98 | 1558.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 1552.30 | 1538.99 | 1554.92 | SL hit (close>ema200) qty=0.50 sl=1538.99 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 1545.00 | 1505.89 | 1503.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 1556.35 | 1547.60 | 1534.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 1556.20 | 1560.73 | 1547.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 1556.20 | 1560.73 | 1547.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 1558.90 | 1560.36 | 1548.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 1564.50 | 1557.93 | 1549.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 1569.20 | 1593.51 | 1596.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 1569.20 | 1593.51 | 1596.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1548.25 | 1577.18 | 1586.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 14:15:00 | 1558.70 | 1554.04 | 1570.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 14:15:00 | 1558.70 | 1554.04 | 1570.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1558.70 | 1554.04 | 1570.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 1558.70 | 1554.04 | 1570.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 1468.05 | 1461.57 | 1474.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 1468.05 | 1461.57 | 1474.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1451.70 | 1455.32 | 1464.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:30:00 | 1459.15 | 1455.32 | 1464.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1462.00 | 1456.66 | 1464.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 1448.05 | 1456.66 | 1464.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 10:00:00 | 1449.20 | 1455.17 | 1463.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1446.40 | 1457.24 | 1460.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 1445.30 | 1431.50 | 1433.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 13:15:00 | 1474.60 | 1440.12 | 1437.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1474.60 | 1440.12 | 1437.24 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1369.05 | 1436.81 | 1442.25 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1467.30 | 1406.87 | 1402.31 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 1405.05 | 1414.84 | 1416.01 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 1425.85 | 1417.04 | 1416.91 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 1407.70 | 1415.17 | 1416.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 1394.30 | 1409.84 | 1413.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 1383.50 | 1382.56 | 1394.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 14:45:00 | 1378.15 | 1382.56 | 1394.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1375.35 | 1379.99 | 1391.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:15:00 | 1367.45 | 1380.57 | 1381.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1333.45 | 1362.34 | 1369.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 1299.08 | 1332.85 | 1351.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1266.78 | 1311.65 | 1334.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 1301.35 | 1289.75 | 1306.15 | SL hit (close>ema200) qty=0.50 sl=1289.75 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1308.05 | 1289.31 | 1288.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 1326.15 | 1305.03 | 1299.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 14:15:00 | 1322.80 | 1324.62 | 1315.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 14:45:00 | 1323.10 | 1324.62 | 1315.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1316.05 | 1322.90 | 1315.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 1322.35 | 1322.90 | 1315.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1327.45 | 1323.81 | 1316.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 1335.05 | 1323.81 | 1316.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 13:00:00 | 1330.55 | 1326.19 | 1319.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 15:00:00 | 1329.05 | 1327.15 | 1321.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 1311.90 | 1324.44 | 1321.62 | SL hit (close<static) qty=1.00 sl=1314.20 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 1309.80 | 1318.11 | 1319.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 14:15:00 | 1299.10 | 1312.51 | 1316.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1245.75 | 1242.76 | 1262.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 1245.75 | 1242.76 | 1262.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1272.60 | 1251.48 | 1263.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1272.60 | 1251.48 | 1263.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1276.90 | 1256.57 | 1264.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 1276.90 | 1256.57 | 1264.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 1283.20 | 1269.05 | 1268.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 1287.55 | 1272.75 | 1270.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 1316.60 | 1317.11 | 1305.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 13:15:00 | 1316.60 | 1317.11 | 1305.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1316.60 | 1317.11 | 1305.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 1316.60 | 1317.11 | 1305.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1305.00 | 1314.20 | 1306.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 1309.00 | 1314.20 | 1306.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1297.90 | 1310.94 | 1305.46 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 1299.05 | 1302.22 | 1302.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 14:15:00 | 1283.00 | 1298.37 | 1300.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 1250.00 | 1244.46 | 1255.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1231.00 | 1244.46 | 1255.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:45:00 | 1235.70 | 1242.81 | 1252.57 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1259.80 | 1246.21 | 1253.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 1259.80 | 1246.21 | 1253.23 | SL hit (close>ema400) qty=1.00 sl=1253.23 alert=retest1 |

### Cycle 61 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1298.60 | 1260.26 | 1257.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 1308.85 | 1269.97 | 1261.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 12:15:00 | 1290.00 | 1295.53 | 1284.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:15:00 | 1287.45 | 1295.53 | 1284.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 1285.00 | 1291.51 | 1285.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 1314.50 | 1291.51 | 1285.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-28 12:15:00 | 1445.95 | 1392.32 | 1367.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 1362.00 | 1372.58 | 1372.79 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1373.10 | 1371.66 | 1371.64 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 12:15:00 | 1368.55 | 1371.65 | 1371.67 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1374.95 | 1371.69 | 1371.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 1390.00 | 1375.36 | 1373.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1386.60 | 1400.28 | 1391.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1386.60 | 1400.28 | 1391.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1386.60 | 1400.28 | 1391.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 1450.50 | 1399.35 | 1395.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 1500.40 | 1518.62 | 1519.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 1500.40 | 1518.62 | 1519.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 11:15:00 | 1496.00 | 1502.48 | 1508.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 11:15:00 | 1457.20 | 1455.11 | 1471.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 11:45:00 | 1457.70 | 1455.11 | 1471.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1406.40 | 1371.75 | 1383.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 1403.90 | 1371.75 | 1383.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1401.00 | 1377.60 | 1385.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:45:00 | 1385.00 | 1387.66 | 1388.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 09:15:00 | 1315.75 | 1354.94 | 1364.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 1361.10 | 1354.94 | 1364.42 | SL hit (close>static) qty=0.50 sl=1354.94 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1366.20 | 1357.66 | 1357.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 1374.60 | 1361.05 | 1358.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1355.90 | 1361.69 | 1359.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1355.90 | 1361.69 | 1359.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1355.90 | 1361.69 | 1359.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1355.90 | 1361.69 | 1359.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1343.00 | 1357.95 | 1358.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1330.10 | 1352.38 | 1355.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 1338.90 | 1337.53 | 1346.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:45:00 | 1339.60 | 1337.53 | 1346.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1347.00 | 1340.40 | 1345.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1358.90 | 1340.40 | 1345.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1360.50 | 1344.42 | 1346.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 13:00:00 | 1349.50 | 1348.34 | 1348.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 1348.60 | 1348.39 | 1348.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1348.60 | 1348.39 | 1348.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1354.50 | 1350.05 | 1349.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1347.80 | 1353.69 | 1351.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 1346.70 | 1353.69 | 1351.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1352.50 | 1353.45 | 1351.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 1358.20 | 1353.45 | 1351.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 1359.50 | 1372.13 | 1372.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1359.50 | 1372.13 | 1372.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1346.20 | 1354.84 | 1361.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1352.90 | 1349.62 | 1355.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 1339.50 | 1347.59 | 1354.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1362.90 | 1346.19 | 1345.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1362.90 | 1346.19 | 1345.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 1375.60 | 1360.45 | 1355.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 1403.00 | 1406.83 | 1393.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 1403.00 | 1406.83 | 1393.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1392.70 | 1404.29 | 1395.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 1392.70 | 1404.29 | 1395.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1399.40 | 1403.31 | 1396.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1412.50 | 1401.25 | 1395.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1403.80 | 1407.47 | 1403.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 1383.20 | 1404.52 | 1403.87 | SL hit (close<static) qty=1.00 sl=1392.40 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 1372.00 | 1398.01 | 1400.97 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1408.70 | 1402.70 | 1402.60 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 1400.10 | 1402.73 | 1402.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1390.90 | 1400.36 | 1401.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1394.00 | 1393.71 | 1397.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1394.00 | 1393.71 | 1397.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1410.00 | 1396.97 | 1398.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1410.00 | 1396.97 | 1398.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1421.30 | 1401.83 | 1400.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1429.90 | 1410.69 | 1405.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1414.00 | 1416.85 | 1410.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:00:00 | 1414.00 | 1416.85 | 1410.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1413.10 | 1415.39 | 1411.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 1417.00 | 1412.47 | 1411.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 1418.30 | 1415.69 | 1413.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1394.30 | 1411.21 | 1411.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1394.30 | 1411.21 | 1411.92 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 1434.30 | 1408.48 | 1406.73 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 1417.00 | 1428.40 | 1428.94 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 1438.60 | 1429.79 | 1429.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 1449.20 | 1433.68 | 1431.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1442.50 | 1447.46 | 1442.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 1454.40 | 1446.43 | 1442.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1471.80 | 1451.50 | 1447.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 1441.10 | 1452.68 | 1454.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 1441.10 | 1452.68 | 1454.15 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1476.70 | 1458.35 | 1456.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 1489.50 | 1464.58 | 1459.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 1536.00 | 1540.95 | 1526.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:45:00 | 1533.00 | 1540.95 | 1526.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1570.90 | 1544.04 | 1531.00 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1514.70 | 1524.72 | 1525.89 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1545.90 | 1526.29 | 1526.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1565.20 | 1539.92 | 1532.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 14:15:00 | 1589.20 | 1591.42 | 1569.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 1589.20 | 1591.42 | 1569.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1574.00 | 1589.77 | 1573.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1575.10 | 1589.77 | 1573.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1568.30 | 1585.48 | 1572.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1568.30 | 1585.48 | 1572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1558.70 | 1580.12 | 1571.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1558.70 | 1580.12 | 1571.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1566.70 | 1572.09 | 1569.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1561.20 | 1572.09 | 1569.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1554.80 | 1568.63 | 1568.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1557.00 | 1568.63 | 1568.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1544.60 | 1563.82 | 1566.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 1541.20 | 1559.30 | 1563.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1538.40 | 1528.67 | 1536.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1538.40 | 1528.67 | 1536.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1537.00 | 1530.33 | 1536.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 1536.60 | 1530.33 | 1536.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1531.40 | 1530.55 | 1536.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1553.90 | 1530.55 | 1536.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1571.20 | 1538.68 | 1539.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1571.20 | 1538.68 | 1539.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1544.00 | 1539.74 | 1539.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 1550.60 | 1539.74 | 1539.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1535.70 | 1538.93 | 1539.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 1532.10 | 1538.11 | 1539.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 1531.50 | 1537.29 | 1538.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1528.20 | 1537.29 | 1538.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1544.00 | 1534.76 | 1534.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1544.00 | 1534.76 | 1534.17 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1532.50 | 1533.78 | 1533.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 15:15:00 | 1521.00 | 1530.36 | 1532.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1542.40 | 1532.77 | 1533.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1542.40 | 1532.77 | 1533.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 1539.00 | 1534.02 | 1533.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 1550.10 | 1537.23 | 1535.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1539.60 | 1540.95 | 1538.22 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 1532.00 | 1537.47 | 1537.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1526.00 | 1535.18 | 1536.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1533.40 | 1530.91 | 1533.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 1533.90 | 1530.91 | 1533.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1529.50 | 1530.63 | 1533.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 1526.90 | 1529.88 | 1532.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1524.00 | 1515.76 | 1519.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1523.20 | 1517.25 | 1519.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1534.30 | 1521.44 | 1521.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1534.30 | 1521.44 | 1521.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1559.60 | 1534.52 | 1528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 1595.80 | 1597.09 | 1572.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:30:00 | 1595.90 | 1597.09 | 1572.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1565.90 | 1588.73 | 1572.45 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1551.80 | 1564.36 | 1564.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1537.30 | 1558.95 | 1562.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1550.50 | 1545.17 | 1551.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 1550.50 | 1545.17 | 1551.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1547.70 | 1545.68 | 1551.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:45:00 | 1556.40 | 1545.68 | 1551.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1525.00 | 1535.18 | 1544.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 1523.30 | 1531.83 | 1540.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:15:00 | 1521.00 | 1531.83 | 1540.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 1523.20 | 1529.17 | 1538.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 1523.30 | 1529.17 | 1538.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1523.30 | 1521.11 | 1527.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1523.30 | 1521.11 | 1527.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1510.20 | 1518.93 | 1526.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1526.70 | 1518.93 | 1526.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1528.50 | 1520.84 | 1526.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 1522.00 | 1524.14 | 1527.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 1518.00 | 1525.52 | 1527.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 1505.00 | 1483.55 | 1482.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1505.00 | 1483.55 | 1482.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1530.10 | 1500.89 | 1491.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1532.70 | 1542.45 | 1530.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1532.70 | 1542.45 | 1530.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1541.20 | 1542.20 | 1531.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 1550.00 | 1543.02 | 1532.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 1551.10 | 1542.61 | 1533.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:15:00 | 1547.60 | 1542.61 | 1533.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1555.60 | 1544.78 | 1536.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1586.20 | 1584.89 | 1577.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1566.50 | 1584.89 | 1577.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1580.10 | 1583.93 | 1578.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1579.60 | 1583.93 | 1578.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1580.60 | 1583.26 | 1578.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1558.80 | 1575.35 | 1576.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1558.80 | 1575.35 | 1576.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1540.50 | 1564.87 | 1571.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1501.90 | 1500.09 | 1512.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1501.90 | 1500.09 | 1512.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1506.70 | 1502.12 | 1511.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 1504.00 | 1502.12 | 1511.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1512.10 | 1504.12 | 1511.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 1508.60 | 1504.12 | 1511.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1503.00 | 1503.89 | 1510.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 1491.80 | 1503.89 | 1510.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 1502.00 | 1499.78 | 1505.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:00:00 | 1501.00 | 1500.03 | 1505.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1493.60 | 1501.20 | 1504.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1505.20 | 1502.00 | 1504.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 1505.20 | 1502.00 | 1504.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1505.40 | 1502.68 | 1504.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1505.10 | 1502.68 | 1504.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1507.00 | 1503.54 | 1505.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 1507.00 | 1503.54 | 1505.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 1505.10 | 1503.85 | 1505.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1505.00 | 1503.85 | 1505.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1510.00 | 1504.99 | 1505.14 | SL hit (close>static) qty=1.00 sl=1508.10 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1519.40 | 1507.87 | 1506.44 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1500.60 | 1511.09 | 1511.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1498.90 | 1508.65 | 1510.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1512.00 | 1507.51 | 1509.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 1511.70 | 1507.51 | 1509.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1506.20 | 1507.25 | 1509.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1481.80 | 1507.25 | 1509.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 09:15:00 | 1407.71 | 1424.34 | 1435.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 1416.60 | 1407.09 | 1415.37 | SL hit (close>ema200) qty=0.50 sl=1407.09 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1340.00 | 1327.66 | 1326.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1349.80 | 1340.52 | 1334.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1334.30 | 1339.56 | 1334.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1334.30 | 1339.56 | 1334.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1336.70 | 1338.99 | 1335.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 1337.70 | 1338.99 | 1335.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1332.00 | 1337.04 | 1335.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:15:00 | 1332.00 | 1337.04 | 1335.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1332.00 | 1336.03 | 1334.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:30:00 | 1335.30 | 1334.65 | 1334.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 1335.70 | 1334.65 | 1334.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1332.30 | 1333.67 | 1333.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1332.30 | 1333.67 | 1333.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 1327.90 | 1332.17 | 1333.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1335.00 | 1332.39 | 1333.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1336.00 | 1332.39 | 1333.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1328.50 | 1331.61 | 1332.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1338.40 | 1331.61 | 1332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1324.30 | 1326.35 | 1329.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 1319.10 | 1325.46 | 1328.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1355.30 | 1327.22 | 1327.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1355.30 | 1327.22 | 1327.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1360.00 | 1343.71 | 1339.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 1360.00 | 1360.12 | 1353.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 1360.00 | 1360.12 | 1353.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1354.50 | 1359.00 | 1353.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1352.90 | 1359.00 | 1353.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1365.00 | 1360.20 | 1354.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1384.50 | 1359.23 | 1357.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 1370.60 | 1366.28 | 1361.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1364.60 | 1373.90 | 1374.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1364.60 | 1373.90 | 1374.81 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1376.60 | 1373.43 | 1373.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1390.40 | 1377.07 | 1375.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1380.00 | 1386.26 | 1382.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1380.00 | 1386.26 | 1382.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1380.30 | 1385.06 | 1381.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1380.00 | 1385.06 | 1381.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1380.50 | 1384.15 | 1381.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1380.50 | 1384.15 | 1381.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1380.00 | 1383.32 | 1381.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1380.00 | 1383.32 | 1381.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1369.10 | 1380.48 | 1380.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1353.90 | 1370.52 | 1374.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 15:15:00 | 1353.90 | 1353.60 | 1362.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 1346.90 | 1353.60 | 1362.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1357.40 | 1354.36 | 1361.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1331.00 | 1355.16 | 1356.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 1335.20 | 1352.27 | 1354.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1337.60 | 1349.34 | 1353.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1305.00 | 1342.34 | 1347.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1300.40 | 1333.95 | 1343.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1292.50 | 1314.60 | 1328.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 1343.00 | 1323.24 | 1320.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1343.00 | 1323.24 | 1320.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1352.50 | 1336.40 | 1332.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1336.00 | 1336.32 | 1332.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1330.60 | 1336.32 | 1332.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1344.60 | 1337.98 | 1334.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 1349.50 | 1343.80 | 1339.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 1349.10 | 1345.62 | 1341.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 1349.40 | 1348.07 | 1344.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:30:00 | 1350.30 | 1348.16 | 1344.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1345.70 | 1350.91 | 1347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 1340.00 | 1350.91 | 1347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1351.30 | 1350.99 | 1348.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1334.80 | 1346.28 | 1347.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1334.80 | 1346.28 | 1347.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1324.40 | 1341.90 | 1345.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 12:15:00 | 1332.60 | 1331.62 | 1336.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:45:00 | 1333.20 | 1331.62 | 1336.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1336.80 | 1332.65 | 1336.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:45:00 | 1336.80 | 1332.65 | 1336.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1336.10 | 1333.34 | 1336.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 1335.10 | 1333.34 | 1336.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1335.10 | 1333.69 | 1336.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1360.00 | 1333.69 | 1336.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1351.40 | 1337.24 | 1337.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 1351.70 | 1337.24 | 1337.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 1348.10 | 1339.41 | 1338.77 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1338.40 | 1345.09 | 1345.15 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1349.50 | 1344.87 | 1344.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 1352.30 | 1347.71 | 1345.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 15:15:00 | 1347.50 | 1347.67 | 1346.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:15:00 | 1350.20 | 1347.67 | 1346.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1350.70 | 1348.27 | 1346.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:45:00 | 1366.00 | 1352.33 | 1349.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 1361.10 | 1356.11 | 1351.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1348.40 | 1365.74 | 1366.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 1348.40 | 1365.74 | 1366.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 1342.30 | 1356.88 | 1361.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1315.90 | 1315.44 | 1326.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 1315.90 | 1315.44 | 1326.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1321.50 | 1316.51 | 1325.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1321.50 | 1316.51 | 1325.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1320.80 | 1317.37 | 1324.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1318.00 | 1317.37 | 1324.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1316.50 | 1317.20 | 1324.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 1311.00 | 1315.35 | 1322.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1336.60 | 1320.21 | 1321.66 | SL hit (close>static) qty=1.00 sl=1335.10 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1352.50 | 1326.67 | 1324.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1358.50 | 1343.66 | 1334.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1356.90 | 1357.99 | 1351.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 14:45:00 | 1360.50 | 1358.39 | 1352.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 15:15:00 | 1361.90 | 1358.39 | 1352.19 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1346.00 | 1356.48 | 1352.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1346.00 | 1356.48 | 1352.43 | SL hit (close<ema400) qty=1.00 sl=1352.43 alert=retest1 |

### Cycle 108 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1337.00 | 1348.86 | 1349.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1331.70 | 1342.73 | 1346.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1320.30 | 1320.24 | 1326.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1320.30 | 1320.24 | 1326.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1313.50 | 1318.46 | 1323.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1309.80 | 1314.80 | 1319.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 1310.10 | 1314.44 | 1318.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1304.10 | 1318.23 | 1319.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:00:00 | 1307.00 | 1313.40 | 1315.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1285.20 | 1293.09 | 1300.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1270.90 | 1290.67 | 1298.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1270.00 | 1285.06 | 1292.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1244.31 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1244.59 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1241.65 | 1256.53 | 1259.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 1256.90 | 1256.21 | 1258.95 | SL hit (close>ema200) qty=0.50 sl=1256.21 alert=retest2 |

### Cycle 109 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1138.90 | 1124.46 | 1122.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1142.80 | 1128.13 | 1124.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1214.50 | 1214.73 | 1196.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1207.00 | 1214.34 | 1205.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1207.00 | 1214.34 | 1205.36 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1192.90 | 1200.88 | 1201.69 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1222.30 | 1201.37 | 1199.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1232.60 | 1207.62 | 1202.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1208.50 | 1213.14 | 1207.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 1208.50 | 1213.14 | 1207.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1216.30 | 1213.77 | 1208.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1212.80 | 1213.77 | 1208.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1209.00 | 1212.25 | 1209.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1207.20 | 1211.36 | 1209.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1202.40 | 1209.57 | 1208.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1202.40 | 1209.57 | 1208.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1212.80 | 1210.22 | 1209.07 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1192.40 | 1206.06 | 1207.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 1192.10 | 1200.14 | 1204.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 15:15:00 | 1200.00 | 1199.11 | 1202.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 1199.00 | 1199.11 | 1202.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1222.00 | 1203.83 | 1204.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1222.00 | 1203.83 | 1204.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1215.90 | 1206.24 | 1205.28 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1200.10 | 1206.51 | 1207.01 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1210.60 | 1207.12 | 1207.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1214.70 | 1209.12 | 1208.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1206.40 | 1209.08 | 1208.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 1206.40 | 1209.08 | 1208.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1207.70 | 1208.81 | 1208.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1196.70 | 1208.81 | 1208.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1204.90 | 1208.02 | 1208.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1187.80 | 1197.10 | 1202.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1190.80 | 1186.39 | 1193.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:00:00 | 1190.80 | 1186.39 | 1193.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1195.00 | 1188.11 | 1193.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 1195.00 | 1188.11 | 1193.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1201.00 | 1190.69 | 1194.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1190.00 | 1190.69 | 1194.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1199.40 | 1193.83 | 1195.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 1191.60 | 1193.83 | 1195.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 11:15:00 | 1212.30 | 1197.52 | 1196.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 1223.50 | 1205.08 | 1200.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 15:15:00 | 1204.40 | 1206.15 | 1201.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 1195.40 | 1206.15 | 1201.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1185.90 | 1202.10 | 1200.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 1185.90 | 1202.10 | 1200.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 1176.60 | 1197.00 | 1198.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 11:15:00 | 1170.10 | 1191.62 | 1195.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1195.10 | 1192.32 | 1195.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 1199.30 | 1192.32 | 1195.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1199.70 | 1193.79 | 1195.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:15:00 | 1206.60 | 1193.79 | 1195.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1210.00 | 1197.03 | 1197.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1210.00 | 1197.03 | 1197.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 1213.00 | 1200.23 | 1198.68 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 1179.30 | 1194.87 | 1196.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 11:15:00 | 1176.50 | 1191.20 | 1194.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1200.20 | 1188.38 | 1192.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1200.20 | 1188.38 | 1192.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1182.10 | 1187.12 | 1191.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1164.10 | 1187.12 | 1191.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 15:15:00 | 1176.30 | 1175.46 | 1181.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 1180.60 | 1175.79 | 1178.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:15:00 | 1180.90 | 1175.79 | 1178.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1183.40 | 1177.31 | 1179.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 1185.20 | 1177.31 | 1179.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1182.50 | 1178.35 | 1179.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1194.20 | 1178.35 | 1179.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1197.10 | 1182.10 | 1181.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1197.10 | 1182.10 | 1181.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 15:15:00 | 1206.10 | 1196.38 | 1189.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1169.30 | 1190.97 | 1187.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 1169.30 | 1190.97 | 1187.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1170.30 | 1186.83 | 1186.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1170.00 | 1186.83 | 1186.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 1172.00 | 1183.87 | 1184.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1142.00 | 1170.28 | 1177.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1126.70 | 1125.01 | 1139.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1126.70 | 1125.01 | 1139.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1120.10 | 1125.47 | 1132.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:00:00 | 1113.60 | 1123.09 | 1130.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1109.50 | 1120.02 | 1126.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 1114.00 | 1116.50 | 1118.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1142.00 | 1116.88 | 1114.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1142.00 | 1116.88 | 1114.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 1146.50 | 1128.65 | 1120.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1162.00 | 1164.12 | 1149.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 1169.80 | 1165.26 | 1151.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1154.50 | 1163.37 | 1156.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1154.50 | 1163.37 | 1156.00 | SL hit (close<ema400) qty=1.00 sl=1156.00 alert=retest1 |

### Cycle 124 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1130.50 | 1155.40 | 1158.18 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1166.80 | 1156.05 | 1155.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1193.10 | 1169.69 | 1163.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1211.10 | 1215.11 | 1192.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 1230.10 | 1217.71 | 1195.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:30:00 | 1229.20 | 1221.10 | 1199.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:00:00 | 1232.80 | 1245.44 | 1223.44 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1258.30 | 1250.36 | 1236.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1272.90 | 1253.33 | 1245.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 1265.70 | 1273.53 | 1266.92 | SL hit (close<ema400) qty=1.00 sl=1266.92 alert=retest1 |

### Cycle 126 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 1239.00 | 1258.84 | 1261.50 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1273.90 | 1256.18 | 1254.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 1278.70 | 1260.69 | 1256.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 13:15:00 | 1286.10 | 1288.41 | 1276.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 1286.10 | 1288.41 | 1276.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1300.00 | 1292.16 | 1281.17 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 1266.10 | 1277.69 | 1278.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1250.90 | 1272.33 | 1275.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1256.70 | 1246.47 | 1257.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1256.20 | 1246.47 | 1257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1261.10 | 1249.39 | 1257.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1263.40 | 1249.39 | 1257.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1256.80 | 1250.87 | 1257.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:30:00 | 1256.50 | 1250.87 | 1257.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1257.00 | 1252.10 | 1257.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 1250.20 | 1253.86 | 1257.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 1261.00 | 1249.73 | 1251.32 | SL hit (close>static) qty=1.00 sl=1259.30 alert=retest2 |

### Cycle 129 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1262.60 | 1254.09 | 1253.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 1265.80 | 1258.10 | 1255.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1256.00 | 1257.68 | 1255.30 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1251.30 | 1254.30 | 1254.37 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1268.00 | 1256.09 | 1255.08 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 1248.50 | 1254.70 | 1255.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 1241.20 | 1251.67 | 1253.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1257.90 | 1250.75 | 1252.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 1257.90 | 1250.75 | 1252.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1260.00 | 1252.60 | 1253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1270.00 | 1252.60 | 1253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1275.10 | 1257.10 | 1255.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1282.30 | 1266.22 | 1260.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 1290.50 | 1290.98 | 1280.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 1289.70 | 1290.98 | 1280.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1297.90 | 1292.37 | 1282.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1301.00 | 1292.37 | 1282.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 1351.80 | 2024-05-18 09:15:00 | 1370.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-05-13 13:45:00 | 1351.80 | 2024-05-18 09:15:00 | 1370.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-05-13 15:15:00 | 1347.00 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-05-14 10:15:00 | 1354.65 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-05-15 10:15:00 | 1344.30 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-05-16 11:15:00 | 1349.85 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-05-16 12:30:00 | 1350.75 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-05-16 13:15:00 | 1342.55 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-05-17 13:30:00 | 1338.05 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-05-17 14:30:00 | 1336.55 | 2024-05-18 11:15:00 | 1391.30 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2024-06-11 12:00:00 | 1310.00 | 2024-06-11 14:15:00 | 1296.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1320.00 | 2024-06-14 10:15:00 | 1309.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-13 10:00:00 | 1317.30 | 2024-06-14 10:15:00 | 1309.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-06-13 10:30:00 | 1311.10 | 2024-06-14 10:15:00 | 1309.95 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-06-13 12:30:00 | 1313.95 | 2024-06-14 10:15:00 | 1309.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-13 15:00:00 | 1314.50 | 2024-06-14 10:15:00 | 1309.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-06-18 10:30:00 | 1311.00 | 2024-06-18 12:15:00 | 1318.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-06-24 12:00:00 | 1292.00 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-06-24 14:15:00 | 1288.95 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2024-06-26 11:15:00 | 1291.85 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2024-06-26 13:00:00 | 1290.60 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2024-07-01 12:15:00 | 1254.75 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-07-01 13:30:00 | 1250.00 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-07-01 15:00:00 | 1252.35 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-07-02 09:15:00 | 1253.25 | 2024-07-03 09:15:00 | 1275.45 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-07-11 10:15:00 | 1250.55 | 2024-07-16 14:15:00 | 1188.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 13:00:00 | 1250.00 | 2024-07-16 14:15:00 | 1187.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 09:45:00 | 1250.00 | 2024-07-16 14:15:00 | 1187.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:15:00 | 1250.55 | 2024-07-19 10:15:00 | 1176.45 | STOP_HIT | 0.50 | 5.93% |
| SELL | retest2 | 2024-07-11 13:00:00 | 1250.00 | 2024-07-19 10:15:00 | 1176.45 | STOP_HIT | 0.50 | 5.88% |
| SELL | retest2 | 2024-07-12 09:45:00 | 1250.00 | 2024-07-19 10:15:00 | 1176.45 | STOP_HIT | 0.50 | 5.88% |
| BUY | retest2 | 2024-08-01 15:00:00 | 1195.80 | 2024-08-05 10:15:00 | 1170.95 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-08-02 10:15:00 | 1191.75 | 2024-08-05 10:15:00 | 1170.95 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-27 09:30:00 | 1245.50 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2024-08-27 15:15:00 | 1250.00 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-08-28 09:45:00 | 1246.90 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-08-28 14:00:00 | 1243.00 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1243.05 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-08-29 10:30:00 | 1247.35 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-08-29 14:45:00 | 1242.00 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2024-08-30 10:15:00 | 1242.10 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2024-08-30 13:45:00 | 1251.60 | 2024-09-04 14:15:00 | 1261.90 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-09-13 09:15:00 | 1329.95 | 2024-09-16 14:15:00 | 1302.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-09-13 15:00:00 | 1319.00 | 2024-09-16 14:15:00 | 1302.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1329.50 | 2024-09-16 14:15:00 | 1302.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-09-24 15:00:00 | 1389.45 | 2024-09-30 15:15:00 | 1390.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-09-25 10:45:00 | 1393.60 | 2024-09-30 15:15:00 | 1390.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-09-25 15:15:00 | 1388.80 | 2024-09-30 15:15:00 | 1390.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-10-10 12:00:00 | 1414.35 | 2024-10-18 11:15:00 | 1435.60 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-10-11 10:45:00 | 1413.85 | 2024-10-18 11:15:00 | 1435.60 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2024-10-11 13:15:00 | 1410.80 | 2024-10-18 11:15:00 | 1435.60 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2024-10-11 13:45:00 | 1410.05 | 2024-10-18 11:15:00 | 1435.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1436.60 | 2024-10-18 11:15:00 | 1435.60 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-10-24 12:45:00 | 1361.50 | 2024-10-25 14:15:00 | 1410.05 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-10-24 13:30:00 | 1363.75 | 2024-10-25 14:15:00 | 1410.05 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-11-08 13:45:00 | 1585.25 | 2024-11-11 09:15:00 | 1569.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-11-11 10:45:00 | 1585.05 | 2024-11-11 12:15:00 | 1574.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-11 11:15:00 | 1582.80 | 2024-11-11 12:15:00 | 1574.40 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-11-14 09:15:00 | 1635.15 | 2024-11-18 11:15:00 | 1583.90 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-12-13 10:15:00 | 1604.25 | 2024-12-23 09:15:00 | 1524.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 12:15:00 | 1606.00 | 2024-12-23 09:15:00 | 1525.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 13:00:00 | 1602.05 | 2024-12-23 09:15:00 | 1521.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 15:15:00 | 1603.10 | 2024-12-23 09:15:00 | 1522.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 1595.00 | 2024-12-23 09:15:00 | 1520.95 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2024-12-17 13:15:00 | 1601.00 | 2024-12-23 11:15:00 | 1515.25 | PARTIAL | 0.50 | 5.36% |
| SELL | retest2 | 2024-12-18 09:45:00 | 1591.80 | 2024-12-23 11:15:00 | 1512.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 15:15:00 | 1595.30 | 2024-12-23 11:15:00 | 1515.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 10:15:00 | 1604.25 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2024-12-13 12:15:00 | 1606.00 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2024-12-13 13:00:00 | 1602.05 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-12-13 15:15:00 | 1603.10 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2024-12-16 09:15:00 | 1595.00 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2024-12-17 13:15:00 | 1601.00 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2024-12-18 09:45:00 | 1591.80 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2024-12-18 15:15:00 | 1595.30 | 2024-12-23 13:15:00 | 1552.30 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1577.90 | 2024-12-26 12:15:00 | 1499.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:45:00 | 1579.95 | 2024-12-26 12:15:00 | 1500.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1577.90 | 2024-12-27 12:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2024-12-19 09:45:00 | 1579.95 | 2024-12-27 12:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.73% |
| BUY | retest2 | 2025-01-06 09:15:00 | 1564.50 | 2025-01-09 12:15:00 | 1569.20 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-01-20 09:15:00 | 1448.05 | 2025-01-23 13:15:00 | 1474.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-01-20 10:00:00 | 1449.20 | 2025-01-23 13:15:00 | 1474.60 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1446.40 | 2025-01-23 13:15:00 | 1474.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-01-23 12:30:00 | 1445.30 | 2025-01-23 13:15:00 | 1474.60 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-10 10:15:00 | 1367.45 | 2025-02-11 12:15:00 | 1299.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1333.45 | 2025-02-12 09:15:00 | 1266.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:15:00 | 1367.45 | 2025-02-13 11:15:00 | 1301.35 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1333.45 | 2025-02-13 11:15:00 | 1301.35 | STOP_HIT | 0.50 | 2.41% |
| BUY | retest2 | 2025-02-25 10:15:00 | 1335.05 | 2025-02-27 10:15:00 | 1311.90 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-25 13:00:00 | 1330.55 | 2025-02-27 10:15:00 | 1311.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-25 15:00:00 | 1329.05 | 2025-02-27 10:15:00 | 1311.90 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2025-03-13 09:15:00 | 1231.00 | 2025-03-13 11:15:00 | 1259.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest1 | 2025-03-13 10:45:00 | 1235.70 | 2025-03-13 11:15:00 | 1259.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-03-13 13:15:00 | 1252.10 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-03-17 09:30:00 | 1253.50 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-03-17 10:30:00 | 1253.35 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-03-17 11:15:00 | 1254.25 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-03-17 12:45:00 | 1237.05 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-03-17 14:15:00 | 1249.40 | 2025-03-18 09:15:00 | 1298.60 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1314.50 | 2025-03-28 12:15:00 | 1445.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 1450.50 | 2025-04-17 09:15:00 | 1500.40 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2025-04-30 14:45:00 | 1385.00 | 2025-05-06 09:15:00 | 1315.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 14:45:00 | 1385.00 | 2025-05-06 09:15:00 | 1361.10 | STOP_HIT | 0.50 | 1.73% |
| SELL | retest2 | 2025-05-12 13:00:00 | 1349.50 | 2025-05-12 13:15:00 | 1348.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-13 14:15:00 | 1358.20 | 2025-05-19 09:15:00 | 1359.50 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-05-21 12:00:00 | 1339.50 | 2025-05-23 09:15:00 | 1362.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1412.50 | 2025-06-02 14:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1403.80 | 2025-06-02 14:15:00 | 1383.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-10 09:45:00 | 1417.00 | 2025-06-11 09:15:00 | 1394.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-10 13:45:00 | 1418.30 | 2025-06-11 09:15:00 | 1394.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-20 09:30:00 | 1454.40 | 2025-06-24 13:15:00 | 1441.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1471.80 | 2025-06-24 13:15:00 | 1441.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-07-11 12:30:00 | 1532.10 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-11 13:30:00 | 1531.50 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-11 14:15:00 | 1528.20 | 2025-07-15 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-18 15:00:00 | 1526.90 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1524.00 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1523.20 | 2025-07-23 09:15:00 | 1534.30 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-30 11:30:00 | 1523.30 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-07-30 12:15:00 | 1521.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-07-30 13:45:00 | 1523.20 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-07-30 14:15:00 | 1523.30 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-08-01 11:30:00 | 1522.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-08-04 09:30:00 | 1518.00 | 2025-08-11 12:15:00 | 1505.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-08-14 11:45:00 | 1550.00 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-08-14 12:30:00 | 1551.10 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-08-14 13:15:00 | 1547.60 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1555.60 | 2025-08-25 09:15:00 | 1558.80 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-08-29 15:15:00 | 1491.80 | 2025-09-03 09:15:00 | 1510.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-01 13:00:00 | 1502.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-01 14:00:00 | 1501.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1493.60 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1505.00 | 2025-09-03 10:15:00 | 1519.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1481.80 | 2025-09-22 09:15:00 | 1407.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1481.80 | 2025-09-23 14:15:00 | 1416.60 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2025-10-14 09:30:00 | 1335.30 | 2025-10-14 11:15:00 | 1332.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-10-14 10:15:00 | 1335.70 | 2025-10-14 11:15:00 | 1332.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-16 11:15:00 | 1319.10 | 2025-10-17 09:15:00 | 1355.30 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1384.50 | 2025-10-31 11:15:00 | 1364.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-28 12:45:00 | 1370.60 | 2025-10-31 11:15:00 | 1364.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-14 09:15:00 | 1331.00 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-14 10:15:00 | 1335.20 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-14 11:00:00 | 1337.60 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1305.00 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1292.50 | 2025-11-19 13:15:00 | 1343.00 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-11-26 11:15:00 | 1349.50 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-26 12:45:00 | 1349.10 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-27 11:00:00 | 1349.40 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-27 11:30:00 | 1350.30 | 2025-12-01 10:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-10 14:45:00 | 1366.00 | 2025-12-15 12:15:00 | 1348.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-11 10:00:00 | 1361.10 | 2025-12-15 12:15:00 | 1348.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-19 11:30:00 | 1311.00 | 2025-12-22 09:15:00 | 1336.60 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-12-24 14:45:00 | 1360.50 | 2025-12-26 09:15:00 | 1346.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2025-12-24 15:15:00 | 1361.90 | 2025-12-26 09:15:00 | 1346.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1309.80 | 2026-01-16 09:15:00 | 1244.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1310.10 | 2026-01-16 09:15:00 | 1244.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1304.10 | 2026-01-16 09:15:00 | 1241.65 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1309.80 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2026-01-02 09:45:00 | 1310.10 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1304.10 | 2026-01-16 12:15:00 | 1256.90 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2026-01-06 11:00:00 | 1307.00 | 2026-01-20 09:15:00 | 1238.89 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1270.90 | 2026-01-20 13:15:00 | 1207.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1270.00 | 2026-01-20 13:15:00 | 1206.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:00:00 | 1307.00 | 2026-01-22 14:15:00 | 1173.69 | TARGET_HIT | 0.50 | 10.20% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1270.90 | 2026-01-22 15:15:00 | 1194.00 | STOP_HIT | 0.50 | 6.05% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1270.00 | 2026-01-22 15:15:00 | 1194.00 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1164.10 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-03-10 13:45:00 | 1180.60 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-10 14:15:00 | 1180.90 | 2026-03-11 09:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1113.60 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1109.50 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2026-03-20 15:00:00 | 1114.00 | 2026-03-24 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2026-03-27 11:00:00 | 1169.80 | 2026-03-27 15:15:00 | 1154.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-30 11:45:00 | 1148.90 | 2026-04-02 09:15:00 | 1123.60 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-30 14:45:00 | 1157.00 | 2026-04-02 09:15:00 | 1123.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2026-04-08 09:45:00 | 1230.10 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest1 | 2026-04-08 10:30:00 | 1229.20 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest1 | 2026-04-09 10:00:00 | 1232.80 | 2026-04-15 14:15:00 | 1265.70 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1272.90 | 2026-04-16 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-04-27 15:15:00 | 1250.20 | 2026-04-29 10:15:00 | 1261.00 | STOP_HIT | 1.00 | -0.86% |
