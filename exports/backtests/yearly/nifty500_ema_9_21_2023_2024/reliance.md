# Reliance Industries Ltd. (RELIANCE)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1436.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 157 |
| ALERT2 | 155 |
| ALERT2_SKIP | 82 |
| ALERT3 | 428 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 188 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 190 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 199 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 134
- **Target hits / Stop hits / Partials:** 1 / 190 / 8
- **Avg / median % per leg:** 0.19% / -0.56%
- **Sum % (uncompounded):** 37.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 119 | 37 | 31.1% | 1 | 118 | 0 | -0.08% | -9.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.33% | -0.3% |
| BUY @ 3rd Alert (retest2) | 118 | 37 | 31.4% | 1 | 117 | 0 | -0.08% | -9.1% |
| SELL (all) | 80 | 28 | 35.0% | 0 | 72 | 8 | 0.59% | 47.1% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 2.70% | 8.1% |
| SELL @ 3rd Alert (retest2) | 77 | 25 | 32.5% | 0 | 70 | 7 | 0.51% | 39.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.94% | 7.8% |
| retest2 (combined) | 195 | 62 | 31.8% | 1 | 187 | 7 | 0.15% | 29.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 1242.35 | 1240.39 | 1240.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 10:15:00 | 1245.25 | 1241.36 | 1240.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-15 14:15:00 | 1243.08 | 1244.68 | 1242.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-15 15:00:00 | 1243.08 | 1244.68 | 1242.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 15:15:00 | 1242.38 | 1244.22 | 1242.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-16 09:15:00 | 1246.90 | 1244.22 | 1242.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-16 10:00:00 | 1246.50 | 1244.68 | 1243.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-16 10:15:00 | 1241.80 | 1244.10 | 1243.00 | SL hit (close<static) qty=1.00 sl=1242.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 12:15:00 | 1234.53 | 1240.84 | 1241.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 1226.50 | 1236.56 | 1239.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 13:15:00 | 1218.38 | 1217.31 | 1221.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 14:00:00 | 1218.38 | 1217.31 | 1221.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1221.30 | 1218.11 | 1221.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 1221.30 | 1218.11 | 1221.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 1222.00 | 1218.89 | 1221.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 1227.10 | 1218.89 | 1221.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 1229.35 | 1220.98 | 1222.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:30:00 | 1229.75 | 1220.98 | 1222.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 1230.53 | 1222.89 | 1223.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 1230.53 | 1222.89 | 1223.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 1232.50 | 1224.81 | 1223.87 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 1222.40 | 1226.52 | 1226.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 14:15:00 | 1220.88 | 1224.75 | 1225.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 1222.40 | 1217.62 | 1220.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 1222.40 | 1217.62 | 1220.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 1222.40 | 1217.62 | 1220.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 1222.40 | 1217.62 | 1220.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 1222.43 | 1218.58 | 1221.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 1231.47 | 1218.58 | 1221.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1241.85 | 1223.23 | 1222.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 1244.70 | 1227.53 | 1224.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 14:15:00 | 1258.78 | 1260.39 | 1253.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 15:00:00 | 1258.78 | 1260.39 | 1253.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 1244.30 | 1257.15 | 1253.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:45:00 | 1243.85 | 1257.15 | 1253.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 1242.25 | 1254.17 | 1252.15 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 1241.95 | 1249.60 | 1250.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 1234.58 | 1245.26 | 1248.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 1232.80 | 1230.89 | 1235.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 1232.80 | 1230.89 | 1235.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 1232.80 | 1230.89 | 1235.00 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 1239.68 | 1236.03 | 1235.84 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 1233.47 | 1235.32 | 1235.54 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 14:15:00 | 1240.18 | 1236.02 | 1235.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 15:15:00 | 1241.80 | 1237.18 | 1236.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 14:15:00 | 1248.72 | 1248.83 | 1244.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 14:45:00 | 1246.13 | 1248.83 | 1244.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 1248.13 | 1249.27 | 1245.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 1247.13 | 1249.27 | 1245.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 1245.63 | 1248.83 | 1246.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:30:00 | 1247.60 | 1248.83 | 1246.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 1241.28 | 1247.32 | 1246.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 15:00:00 | 1241.28 | 1247.32 | 1246.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 1237.25 | 1244.28 | 1244.96 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 1252.50 | 1245.28 | 1244.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 11:15:00 | 1257.35 | 1247.69 | 1245.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 09:15:00 | 1280.00 | 1283.49 | 1277.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 09:30:00 | 1282.38 | 1283.49 | 1277.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 1274.60 | 1281.71 | 1277.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:00:00 | 1274.60 | 1281.71 | 1277.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 1275.45 | 1280.46 | 1277.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 12:15:00 | 1277.35 | 1280.46 | 1277.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 13:15:00 | 1272.50 | 1278.03 | 1276.52 | SL hit (close<static) qty=1.00 sl=1273.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 1272.10 | 1275.22 | 1275.57 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 15:15:00 | 1280.75 | 1276.11 | 1275.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 1284.10 | 1277.71 | 1276.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 10:15:00 | 1279.03 | 1281.14 | 1279.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 1279.03 | 1281.14 | 1279.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1279.03 | 1281.14 | 1279.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 1279.03 | 1281.14 | 1279.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 1274.70 | 1279.85 | 1278.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 1274.70 | 1279.85 | 1278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 1271.50 | 1278.18 | 1278.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 14:15:00 | 1267.85 | 1275.02 | 1276.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 09:15:00 | 1257.00 | 1250.26 | 1253.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 1257.00 | 1250.26 | 1253.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1257.00 | 1250.26 | 1253.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:45:00 | 1256.50 | 1250.26 | 1253.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 1261.95 | 1252.59 | 1254.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 11:00:00 | 1261.95 | 1252.59 | 1254.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 12:15:00 | 1266.85 | 1257.24 | 1256.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 1269.18 | 1262.78 | 1259.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 1290.00 | 1293.83 | 1282.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 10:00:00 | 1290.00 | 1293.83 | 1282.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 1293.38 | 1294.12 | 1290.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 15:00:00 | 1293.38 | 1294.12 | 1290.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1321.65 | 1299.61 | 1293.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 09:15:00 | 1323.95 | 1311.33 | 1302.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 11:45:00 | 1322.50 | 1317.68 | 1308.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 13:15:00 | 1322.10 | 1318.30 | 1309.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:15:00 | 1324.90 | 1318.97 | 1310.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 1372.38 | 1385.92 | 1379.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 15:00:00 | 1372.38 | 1385.92 | 1379.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 15:15:00 | 1374.50 | 1383.63 | 1379.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:15:00 | 1368.78 | 1383.63 | 1379.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 1372.73 | 1378.34 | 1377.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 11:15:00 | 1377.15 | 1378.34 | 1377.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 12:15:00 | 1367.73 | 1375.12 | 1376.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 12:15:00 | 1367.73 | 1375.12 | 1376.00 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 1400.65 | 1380.44 | 1377.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 10:15:00 | 1412.25 | 1396.11 | 1387.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 11:15:00 | 1402.45 | 1406.93 | 1398.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 11:15:00 | 1402.45 | 1406.93 | 1398.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 1402.45 | 1406.93 | 1398.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:00:00 | 1402.45 | 1406.93 | 1398.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 1421.00 | 1409.46 | 1401.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 15:15:00 | 1425.73 | 1409.46 | 1401.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 09:15:00 | 1308.18 | 1391.80 | 1395.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 09:15:00 | 1308.18 | 1391.80 | 1395.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 1289.20 | 1322.86 | 1351.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 1272.00 | 1252.50 | 1265.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 1272.00 | 1252.50 | 1265.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 1272.00 | 1252.50 | 1265.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 1272.00 | 1252.50 | 1265.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 1269.85 | 1255.97 | 1266.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 13:00:00 | 1263.72 | 1259.73 | 1266.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 14:30:00 | 1264.72 | 1261.43 | 1265.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:00:00 | 1264.65 | 1258.09 | 1261.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 09:15:00 | 1266.43 | 1261.74 | 1261.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1266.43 | 1261.74 | 1261.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 11:15:00 | 1268.83 | 1263.98 | 1262.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 1268.30 | 1268.65 | 1265.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 09:30:00 | 1268.18 | 1268.65 | 1265.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1262.93 | 1267.51 | 1265.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:45:00 | 1263.60 | 1267.51 | 1265.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 1261.65 | 1266.34 | 1265.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:00:00 | 1261.65 | 1266.34 | 1265.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 1263.70 | 1265.81 | 1265.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 13:15:00 | 1263.95 | 1265.81 | 1265.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 14:15:00 | 1257.50 | 1263.70 | 1264.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 14:15:00 | 1257.50 | 1263.70 | 1264.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 1247.72 | 1259.73 | 1262.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 1242.47 | 1239.07 | 1244.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 1242.47 | 1239.07 | 1244.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1242.47 | 1239.07 | 1244.75 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 14:15:00 | 1257.18 | 1247.87 | 1247.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 1258.38 | 1250.96 | 1248.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 1258.00 | 1258.64 | 1254.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 1258.00 | 1258.64 | 1254.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 1256.50 | 1257.87 | 1255.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:00:00 | 1256.50 | 1257.87 | 1255.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 1254.25 | 1257.15 | 1254.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:00:00 | 1254.25 | 1257.15 | 1254.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 1254.40 | 1256.60 | 1254.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:30:00 | 1253.13 | 1256.60 | 1254.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 1251.88 | 1255.66 | 1254.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 1245.85 | 1255.66 | 1254.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 1245.05 | 1253.53 | 1253.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 10:15:00 | 1240.88 | 1251.00 | 1252.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 13:15:00 | 1250.38 | 1249.23 | 1251.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-09 14:00:00 | 1250.38 | 1249.23 | 1251.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 1263.75 | 1252.14 | 1252.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 1263.75 | 1252.14 | 1252.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 1261.00 | 1253.91 | 1253.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 11:15:00 | 1267.58 | 1258.58 | 1255.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 1263.63 | 1264.97 | 1260.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 09:45:00 | 1265.70 | 1264.97 | 1260.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 1264.70 | 1264.92 | 1260.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:00:00 | 1264.70 | 1264.92 | 1260.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 1262.50 | 1264.43 | 1261.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:30:00 | 1258.85 | 1264.43 | 1261.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 1264.95 | 1264.54 | 1261.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 12:45:00 | 1258.68 | 1264.54 | 1261.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1279.50 | 1270.25 | 1265.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 13:45:00 | 1285.47 | 1277.51 | 1270.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 09:30:00 | 1284.60 | 1281.48 | 1274.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 10:15:00 | 1285.03 | 1281.48 | 1274.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 11:45:00 | 1284.22 | 1282.64 | 1276.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 1283.35 | 1284.51 | 1279.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-17 14:15:00 | 1268.83 | 1276.43 | 1277.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 1268.83 | 1276.43 | 1277.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 1257.00 | 1271.11 | 1274.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 13:15:00 | 1281.80 | 1269.04 | 1272.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 13:15:00 | 1281.80 | 1269.04 | 1272.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 1281.80 | 1269.04 | 1272.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:00:00 | 1281.80 | 1269.04 | 1272.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 1277.22 | 1270.68 | 1272.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:15:00 | 1276.00 | 1270.68 | 1272.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 09:15:00 | 1212.20 | 1227.39 | 1236.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-30 09:15:00 | 1217.00 | 1215.20 | 1224.31 | SL hit (close>ema200) qty=0.50 sl=1215.20 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 1214.97 | 1208.75 | 1208.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 1217.25 | 1213.60 | 1212.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 1224.10 | 1230.27 | 1224.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 1224.10 | 1230.27 | 1224.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1224.10 | 1230.27 | 1224.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1227.95 | 1230.27 | 1224.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1225.88 | 1229.40 | 1225.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 12:00:00 | 1227.47 | 1229.01 | 1225.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 12:15:00 | 1220.43 | 1227.29 | 1224.79 | SL hit (close<static) qty=1.00 sl=1220.55 alert=retest2 |

### Cycle 26 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 1218.50 | 1223.25 | 1223.37 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 10:15:00 | 1226.00 | 1223.83 | 1223.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 11:15:00 | 1227.43 | 1224.55 | 1223.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 14:15:00 | 1225.65 | 1226.17 | 1224.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 1225.65 | 1226.17 | 1224.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 1225.65 | 1226.17 | 1224.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 1225.65 | 1226.17 | 1224.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 1225.05 | 1225.94 | 1224.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 09:15:00 | 1231.50 | 1225.94 | 1224.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 11:45:00 | 1227.68 | 1226.40 | 1225.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 14:45:00 | 1227.13 | 1226.11 | 1225.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 15:15:00 | 1228.00 | 1226.11 | 1225.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 1227.58 | 1228.43 | 1227.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 13:45:00 | 1227.68 | 1228.43 | 1227.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 1228.28 | 1228.40 | 1227.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:15:00 | 1226.25 | 1228.40 | 1227.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 1226.25 | 1227.97 | 1227.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 1222.63 | 1227.97 | 1227.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 1224.33 | 1227.24 | 1226.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-18 10:15:00 | 1220.75 | 1225.94 | 1226.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 1220.75 | 1225.94 | 1226.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 1217.75 | 1222.60 | 1224.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 14:15:00 | 1171.72 | 1171.34 | 1176.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 1171.72 | 1171.34 | 1176.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 1173.25 | 1171.53 | 1175.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 1175.05 | 1171.53 | 1175.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 1178.72 | 1172.97 | 1175.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:00:00 | 1178.72 | 1172.97 | 1175.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1177.72 | 1173.92 | 1175.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:30:00 | 1178.40 | 1173.92 | 1175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 1183.85 | 1177.68 | 1177.21 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 1170.53 | 1178.08 | 1178.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 1167.72 | 1176.01 | 1177.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1175.78 | 1174.92 | 1176.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 1175.78 | 1174.92 | 1176.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 1175.78 | 1174.92 | 1176.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 1175.78 | 1174.92 | 1176.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 1174.00 | 1174.74 | 1176.22 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 1183.38 | 1177.33 | 1177.11 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 14:15:00 | 1173.22 | 1176.51 | 1176.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 15:15:00 | 1171.50 | 1175.51 | 1176.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 1158.55 | 1155.80 | 1161.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-04 15:00:00 | 1158.55 | 1155.80 | 1161.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1159.75 | 1156.79 | 1161.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 10:00:00 | 1159.75 | 1156.79 | 1161.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 1155.70 | 1157.77 | 1160.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:45:00 | 1162.00 | 1157.77 | 1160.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 1158.58 | 1157.72 | 1159.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 12:00:00 | 1156.50 | 1157.88 | 1159.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 1151.25 | 1158.92 | 1159.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:45:00 | 1157.13 | 1153.72 | 1155.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 1162.50 | 1155.79 | 1155.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1162.50 | 1155.79 | 1155.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 1163.58 | 1157.34 | 1156.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 1166.47 | 1172.06 | 1168.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1166.47 | 1172.06 | 1168.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1166.47 | 1172.06 | 1168.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:00:00 | 1166.47 | 1172.06 | 1168.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 1166.43 | 1170.93 | 1168.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 11:00:00 | 1166.43 | 1170.93 | 1168.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 1174.50 | 1171.22 | 1168.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:15:00 | 1176.75 | 1171.22 | 1168.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 14:15:00 | 1176.40 | 1171.92 | 1169.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 12:45:00 | 1176.13 | 1173.36 | 1171.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 10:45:00 | 1176.40 | 1173.90 | 1172.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 1172.00 | 1173.59 | 1172.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:00:00 | 1172.00 | 1173.59 | 1172.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 1177.60 | 1174.39 | 1172.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 15:15:00 | 1178.50 | 1174.39 | 1172.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:30:00 | 1181.10 | 1176.57 | 1174.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 1170.18 | 1174.93 | 1173.90 | SL hit (close<static) qty=1.00 sl=1171.68 alert=retest2 |

### Cycle 34 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 1165.90 | 1173.12 | 1173.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1160.53 | 1169.60 | 1171.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1129.68 | 1120.33 | 1127.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1129.68 | 1120.33 | 1127.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1129.68 | 1120.33 | 1127.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 1129.68 | 1120.33 | 1127.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1130.50 | 1122.36 | 1128.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 1133.50 | 1122.36 | 1128.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 1152.72 | 1134.00 | 1131.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 10:15:00 | 1158.83 | 1138.97 | 1134.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 1147.10 | 1151.05 | 1145.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 12:00:00 | 1147.10 | 1151.05 | 1145.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 1149.97 | 1150.83 | 1145.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 14:15:00 | 1150.95 | 1150.39 | 1145.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 15:15:00 | 1144.50 | 1148.45 | 1145.69 | SL hit (close<static) qty=1.00 sl=1145.05 alert=retest2 |

### Cycle 36 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 11:15:00 | 1159.70 | 1164.31 | 1164.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 12:15:00 | 1157.10 | 1162.87 | 1163.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 14:15:00 | 1158.00 | 1156.13 | 1158.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 15:00:00 | 1158.00 | 1156.13 | 1158.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1165.45 | 1158.28 | 1159.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:30:00 | 1165.80 | 1158.28 | 1159.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 1157.55 | 1158.07 | 1158.99 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1172.28 | 1160.52 | 1159.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 1173.00 | 1163.02 | 1160.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 1177.05 | 1179.52 | 1173.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 1177.05 | 1179.52 | 1173.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 1179.08 | 1181.56 | 1177.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 14:45:00 | 1178.50 | 1181.56 | 1177.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 1177.43 | 1180.74 | 1177.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:15:00 | 1175.95 | 1180.74 | 1177.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 1171.47 | 1178.88 | 1177.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:45:00 | 1172.85 | 1178.88 | 1177.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1169.22 | 1176.95 | 1176.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 1169.22 | 1176.95 | 1176.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 1172.30 | 1176.02 | 1176.08 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 1186.25 | 1176.91 | 1176.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 11:15:00 | 1190.97 | 1181.34 | 1178.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 13:15:00 | 1189.28 | 1189.44 | 1185.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 14:00:00 | 1189.28 | 1189.44 | 1185.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1189.55 | 1196.12 | 1194.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 1189.55 | 1196.12 | 1194.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 1189.35 | 1194.77 | 1194.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 1189.35 | 1194.77 | 1194.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 11:15:00 | 1190.05 | 1193.83 | 1193.85 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 1198.47 | 1194.29 | 1193.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 1203.63 | 1196.16 | 1194.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 15:15:00 | 1200.00 | 1200.43 | 1198.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 09:15:00 | 1194.10 | 1200.43 | 1198.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1192.45 | 1198.83 | 1197.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 1192.45 | 1198.83 | 1197.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 1194.00 | 1197.87 | 1197.20 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 12:15:00 | 1192.47 | 1195.93 | 1196.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 14:15:00 | 1189.50 | 1193.93 | 1195.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 1193.50 | 1193.14 | 1194.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 1193.50 | 1193.14 | 1194.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1193.50 | 1193.14 | 1194.70 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 1209.20 | 1197.77 | 1196.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 09:15:00 | 1230.10 | 1218.54 | 1211.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 1223.53 | 1227.23 | 1220.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 10:00:00 | 1223.53 | 1227.23 | 1220.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 1224.93 | 1226.77 | 1221.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 12:00:00 | 1226.93 | 1226.80 | 1221.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 13:30:00 | 1226.50 | 1227.20 | 1222.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 14:30:00 | 1227.00 | 1228.43 | 1225.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 15:00:00 | 1228.80 | 1228.43 | 1225.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 1225.50 | 1227.84 | 1225.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:45:00 | 1230.47 | 1228.26 | 1226.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 10:45:00 | 1229.50 | 1228.97 | 1226.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 13:00:00 | 1229.90 | 1229.36 | 1227.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:45:00 | 1229.53 | 1229.52 | 1228.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 1229.75 | 1229.57 | 1228.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-12 13:15:00 | 1218.08 | 1226.18 | 1226.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 13:15:00 | 1218.08 | 1226.18 | 1226.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 14:15:00 | 1211.38 | 1223.22 | 1225.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 1217.50 | 1214.05 | 1218.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 14:15:00 | 1217.50 | 1214.05 | 1218.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1217.50 | 1214.05 | 1218.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 1217.50 | 1214.05 | 1218.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 1216.03 | 1214.45 | 1218.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 1224.75 | 1214.45 | 1218.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1232.88 | 1218.14 | 1219.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 1232.88 | 1218.14 | 1219.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 1231.18 | 1220.74 | 1220.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 1235.43 | 1227.29 | 1224.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 1267.68 | 1281.95 | 1271.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 1267.68 | 1281.95 | 1271.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 1267.68 | 1281.95 | 1271.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 1267.68 | 1281.95 | 1271.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 1264.30 | 1278.42 | 1271.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 1264.30 | 1278.42 | 1271.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 10:15:00 | 1273.13 | 1274.85 | 1271.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 10:30:00 | 1274.00 | 1274.85 | 1271.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 1275.75 | 1275.03 | 1271.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 1277.90 | 1275.03 | 1271.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 12:15:00 | 1279.08 | 1278.30 | 1275.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 1291.68 | 1293.70 | 1293.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 10:15:00 | 1291.68 | 1293.70 | 1293.77 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 13:15:00 | 1297.47 | 1294.20 | 1293.96 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 1289.25 | 1293.90 | 1293.95 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 14:15:00 | 1306.10 | 1296.24 | 1294.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 09:15:00 | 1308.00 | 1304.12 | 1301.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 1303.70 | 1304.04 | 1302.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 10:15:00 | 1303.70 | 1304.04 | 1302.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 1303.70 | 1304.04 | 1302.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 1303.70 | 1304.04 | 1302.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1292.60 | 1301.75 | 1301.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 1289.00 | 1301.75 | 1301.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 1293.97 | 1300.19 | 1300.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 1289.85 | 1296.53 | 1297.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 10:15:00 | 1295.18 | 1295.18 | 1296.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 10:15:00 | 1295.18 | 1295.18 | 1296.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 1295.18 | 1295.18 | 1296.83 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 1325.35 | 1301.85 | 1299.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 15:15:00 | 1327.10 | 1306.90 | 1301.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 1384.40 | 1386.53 | 1373.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 1384.40 | 1386.53 | 1373.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 1375.18 | 1383.85 | 1374.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 15:00:00 | 1375.18 | 1383.85 | 1374.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 1373.40 | 1381.76 | 1374.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 1376.73 | 1381.76 | 1374.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1370.30 | 1379.47 | 1373.78 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 14:15:00 | 1363.45 | 1370.76 | 1371.21 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 1373.43 | 1367.26 | 1366.77 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 14:15:00 | 1356.85 | 1365.92 | 1366.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 10:15:00 | 1337.63 | 1357.83 | 1362.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 1343.98 | 1336.09 | 1343.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 1343.98 | 1336.09 | 1343.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1343.98 | 1336.09 | 1343.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 1343.98 | 1336.09 | 1343.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 1347.45 | 1338.36 | 1343.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 09:45:00 | 1340.93 | 1339.00 | 1343.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:30:00 | 1343.10 | 1340.33 | 1342.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 13:15:00 | 1351.55 | 1342.57 | 1343.62 | SL hit (close>static) qty=1.00 sl=1349.50 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 15:15:00 | 1355.50 | 1346.33 | 1345.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 1378.20 | 1352.71 | 1348.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 1407.15 | 1419.60 | 1401.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 1407.15 | 1419.60 | 1401.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1422.30 | 1418.46 | 1404.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 10:00:00 | 1439.48 | 1425.94 | 1415.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 11:00:00 | 1434.15 | 1427.58 | 1416.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 11:45:00 | 1431.95 | 1427.80 | 1417.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:45:00 | 1431.70 | 1427.40 | 1420.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1439.00 | 1454.03 | 1446.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 1439.00 | 1454.03 | 1446.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 1439.73 | 1451.17 | 1445.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 1430.55 | 1451.17 | 1445.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-06 11:15:00 | 1435.23 | 1441.83 | 1442.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 11:15:00 | 1435.23 | 1441.83 | 1442.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 14:15:00 | 1426.43 | 1435.42 | 1438.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1445.33 | 1436.11 | 1438.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1445.33 | 1436.11 | 1438.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1445.33 | 1436.11 | 1438.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:00:00 | 1445.33 | 1436.11 | 1438.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 1437.45 | 1436.38 | 1438.44 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 1439.55 | 1438.80 | 1438.73 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 1431.65 | 1437.37 | 1438.09 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 13:15:00 | 1441.58 | 1438.82 | 1438.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 14:15:00 | 1449.48 | 1440.95 | 1439.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 15:15:00 | 1458.65 | 1458.73 | 1451.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 09:15:00 | 1452.00 | 1458.73 | 1451.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1446.03 | 1456.19 | 1451.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:30:00 | 1444.85 | 1456.19 | 1451.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 1451.50 | 1455.25 | 1451.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:45:00 | 1447.78 | 1455.25 | 1451.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 1445.63 | 1453.33 | 1450.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 12:00:00 | 1445.63 | 1453.33 | 1450.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 1448.63 | 1452.39 | 1450.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 13:15:00 | 1452.23 | 1452.39 | 1450.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 15:15:00 | 1452.00 | 1452.94 | 1451.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 14:15:00 | 1460.60 | 1467.13 | 1467.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 14:15:00 | 1460.60 | 1467.13 | 1467.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 1455.03 | 1463.97 | 1466.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 12:15:00 | 1465.00 | 1463.59 | 1465.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 12:15:00 | 1465.00 | 1463.59 | 1465.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 1465.00 | 1463.59 | 1465.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:00:00 | 1465.00 | 1463.59 | 1465.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 1469.20 | 1464.72 | 1465.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:45:00 | 1469.40 | 1464.72 | 1465.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1473.55 | 1466.48 | 1466.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 1473.55 | 1466.48 | 1466.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 15:15:00 | 1469.50 | 1467.09 | 1466.80 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 1466.40 | 1466.58 | 1466.59 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 11:15:00 | 1470.00 | 1467.26 | 1466.90 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 1463.50 | 1466.51 | 1466.59 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 13:15:00 | 1469.35 | 1467.08 | 1466.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 09:15:00 | 1473.50 | 1469.53 | 1468.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 1471.45 | 1474.22 | 1471.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 13:15:00 | 1471.45 | 1474.22 | 1471.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 1471.45 | 1474.22 | 1471.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 1471.45 | 1474.22 | 1471.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 1473.25 | 1474.03 | 1471.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 1460.10 | 1474.03 | 1471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 1466.03 | 1472.43 | 1471.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 1459.18 | 1472.43 | 1471.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1464.50 | 1470.84 | 1470.42 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 1468.33 | 1469.97 | 1470.07 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 1473.50 | 1470.20 | 1470.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 1481.43 | 1472.45 | 1471.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 10:15:00 | 1485.18 | 1487.63 | 1482.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 10:45:00 | 1482.93 | 1487.63 | 1482.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1480.50 | 1485.72 | 1483.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 11:15:00 | 1488.18 | 1485.57 | 1483.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:00:00 | 1487.43 | 1485.94 | 1484.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 1490.60 | 1488.31 | 1485.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 14:45:00 | 1487.35 | 1487.97 | 1485.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 1485.50 | 1487.47 | 1485.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 1487.38 | 1487.47 | 1485.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1485.23 | 1487.03 | 1485.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 1464.88 | 1482.60 | 1483.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1464.88 | 1482.60 | 1483.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1457.63 | 1474.46 | 1479.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 09:15:00 | 1470.38 | 1467.21 | 1473.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 1470.38 | 1467.21 | 1473.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 1470.38 | 1467.21 | 1473.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:30:00 | 1469.90 | 1467.21 | 1473.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 1468.73 | 1467.52 | 1473.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:30:00 | 1472.50 | 1467.52 | 1473.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 1460.93 | 1463.58 | 1469.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 14:45:00 | 1467.13 | 1463.58 | 1469.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1474.08 | 1465.97 | 1469.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 1474.08 | 1465.97 | 1469.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 1477.88 | 1468.35 | 1470.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:45:00 | 1478.50 | 1468.35 | 1470.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 1485.50 | 1471.78 | 1471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 1490.00 | 1478.99 | 1475.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 1493.08 | 1498.78 | 1491.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:00:00 | 1493.08 | 1498.78 | 1491.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 1492.48 | 1497.52 | 1491.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 1491.73 | 1497.52 | 1491.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1494.78 | 1496.97 | 1491.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:45:00 | 1498.90 | 1497.29 | 1492.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 10:30:00 | 1496.53 | 1495.87 | 1493.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 1480.55 | 1492.81 | 1492.55 | SL hit (close<static) qty=1.00 sl=1489.10 alert=retest2 |

### Cycle 70 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 1484.28 | 1491.10 | 1491.80 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 14:15:00 | 1504.25 | 1494.54 | 1493.29 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 12:15:00 | 1487.10 | 1492.75 | 1493.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 13:15:00 | 1485.08 | 1491.22 | 1492.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 1484.45 | 1475.95 | 1481.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 1484.45 | 1475.95 | 1481.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1484.45 | 1475.95 | 1481.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 1484.45 | 1475.95 | 1481.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1480.60 | 1476.88 | 1481.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 11:30:00 | 1477.40 | 1477.67 | 1481.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 14:15:00 | 1476.40 | 1478.21 | 1481.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 11:15:00 | 1438.80 | 1430.24 | 1430.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 11:15:00 | 1438.80 | 1430.24 | 1430.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 14:15:00 | 1444.25 | 1436.28 | 1433.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 1442.05 | 1450.82 | 1447.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 1442.05 | 1450.82 | 1447.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1442.05 | 1450.82 | 1447.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:45:00 | 1444.23 | 1450.82 | 1447.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1443.20 | 1449.29 | 1447.33 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 13:15:00 | 1442.53 | 1446.08 | 1446.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 14:15:00 | 1440.58 | 1444.98 | 1445.67 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 09:15:00 | 1475.05 | 1450.73 | 1448.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 10:15:00 | 1480.65 | 1456.72 | 1451.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 1481.55 | 1491.06 | 1479.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 15:00:00 | 1481.55 | 1491.06 | 1479.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 1485.00 | 1489.85 | 1480.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 1491.58 | 1489.85 | 1480.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:45:00 | 1490.00 | 1489.76 | 1481.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 10:45:00 | 1488.83 | 1489.36 | 1481.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 12:15:00 | 1488.58 | 1488.83 | 1482.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 1485.15 | 1486.92 | 1482.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:45:00 | 1484.55 | 1486.92 | 1482.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 1487.40 | 1487.02 | 1483.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:15:00 | 1488.30 | 1487.02 | 1483.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1479.55 | 1485.52 | 1482.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 1479.55 | 1485.52 | 1482.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 1478.10 | 1484.04 | 1482.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-02 10:15:00 | 1478.10 | 1484.04 | 1482.44 | SL hit (close<static) qty=1.00 sl=1478.65 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 13:15:00 | 1478.13 | 1481.27 | 1481.44 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 14:15:00 | 1488.25 | 1482.67 | 1482.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 15:15:00 | 1488.50 | 1483.84 | 1482.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 09:15:00 | 1477.90 | 1482.65 | 1482.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 1477.90 | 1482.65 | 1482.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1477.90 | 1482.65 | 1482.21 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 10:15:00 | 1476.98 | 1481.51 | 1481.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 13:15:00 | 1475.08 | 1479.22 | 1480.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 13:15:00 | 1469.45 | 1466.33 | 1472.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-04 14:00:00 | 1469.45 | 1466.33 | 1472.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1459.68 | 1464.47 | 1469.74 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 1482.35 | 1470.56 | 1469.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 1487.50 | 1473.95 | 1471.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 1478.00 | 1480.94 | 1476.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 10:45:00 | 1480.00 | 1480.94 | 1476.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 11:15:00 | 1474.25 | 1479.60 | 1476.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:00:00 | 1474.25 | 1479.60 | 1476.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 1470.00 | 1477.68 | 1476.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:00:00 | 1470.00 | 1477.68 | 1476.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 1464.53 | 1472.96 | 1474.11 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 1477.90 | 1475.19 | 1474.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 14:15:00 | 1479.60 | 1476.82 | 1475.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 1473.68 | 1478.37 | 1477.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 1473.68 | 1478.37 | 1477.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 1473.68 | 1478.37 | 1477.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 1473.68 | 1478.37 | 1477.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 1473.53 | 1477.40 | 1476.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:30:00 | 1475.40 | 1477.40 | 1476.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 1463.88 | 1474.70 | 1475.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 1458.95 | 1470.57 | 1473.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 10:15:00 | 1476.15 | 1471.68 | 1473.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 10:15:00 | 1476.15 | 1471.68 | 1473.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 1476.15 | 1471.68 | 1473.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:00:00 | 1476.15 | 1471.68 | 1473.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 1479.95 | 1473.34 | 1474.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:00:00 | 1479.95 | 1473.34 | 1474.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1476.88 | 1474.05 | 1474.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:45:00 | 1481.30 | 1474.05 | 1474.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 1464.05 | 1472.31 | 1473.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 11:15:00 | 1461.00 | 1467.97 | 1471.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:30:00 | 1460.90 | 1464.18 | 1468.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 14:30:00 | 1461.63 | 1464.90 | 1468.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 12:15:00 | 1476.98 | 1470.21 | 1469.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 1476.98 | 1470.21 | 1469.80 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 1460.45 | 1467.86 | 1468.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 1448.88 | 1463.21 | 1466.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 1469.75 | 1462.20 | 1464.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 12:15:00 | 1469.75 | 1462.20 | 1464.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 1469.75 | 1462.20 | 1464.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:00:00 | 1469.75 | 1462.20 | 1464.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 1470.55 | 1463.87 | 1465.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 1470.55 | 1463.87 | 1465.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1470.10 | 1465.11 | 1465.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:45:00 | 1472.20 | 1465.11 | 1465.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 1471.53 | 1466.40 | 1466.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 1473.55 | 1467.83 | 1467.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 1469.23 | 1475.53 | 1472.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 1469.23 | 1475.53 | 1472.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1469.23 | 1475.53 | 1472.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:00:00 | 1469.23 | 1475.53 | 1472.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 1472.43 | 1474.91 | 1472.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:30:00 | 1467.13 | 1474.91 | 1472.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 1467.55 | 1473.44 | 1472.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:00:00 | 1467.55 | 1473.44 | 1472.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 1469.10 | 1472.57 | 1471.76 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 1459.43 | 1469.94 | 1470.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 1450.48 | 1461.11 | 1465.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 1460.50 | 1459.33 | 1463.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:00:00 | 1460.50 | 1459.33 | 1463.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 1463.28 | 1458.33 | 1461.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 1463.28 | 1458.33 | 1461.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 1458.73 | 1458.41 | 1461.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:30:00 | 1465.90 | 1458.41 | 1461.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1454.78 | 1457.76 | 1460.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 14:15:00 | 1452.45 | 1457.04 | 1459.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 15:00:00 | 1452.93 | 1456.22 | 1458.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 11:15:00 | 1463.33 | 1459.72 | 1459.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 1463.33 | 1459.72 | 1459.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 14:15:00 | 1466.55 | 1462.00 | 1460.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 1466.53 | 1471.81 | 1467.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 1466.53 | 1471.81 | 1467.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 1466.53 | 1471.81 | 1467.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 1466.53 | 1471.81 | 1467.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 1465.98 | 1470.64 | 1467.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 1474.13 | 1470.64 | 1467.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 10:00:00 | 1472.15 | 1470.94 | 1467.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 1469.05 | 1470.82 | 1469.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 09:15:00 | 1460.08 | 1468.39 | 1468.31 | SL hit (close<static) qty=1.00 sl=1462.88 alert=retest2 |

### Cycle 88 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 1453.08 | 1465.33 | 1466.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1438.55 | 1459.97 | 1464.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1407.50 | 1406.97 | 1419.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 11:15:00 | 1421.05 | 1410.10 | 1418.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1421.05 | 1410.10 | 1418.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 1421.05 | 1410.10 | 1418.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1426.75 | 1413.43 | 1419.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 1426.75 | 1413.43 | 1419.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 1417.83 | 1415.17 | 1419.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:45:00 | 1417.98 | 1415.17 | 1419.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 1417.13 | 1415.57 | 1419.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 1408.83 | 1415.57 | 1419.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1404.63 | 1413.38 | 1417.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 1402.60 | 1413.38 | 1417.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 10:00:00 | 1398.20 | 1400.17 | 1407.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 1397.00 | 1399.60 | 1403.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 15:00:00 | 1402.68 | 1398.44 | 1400.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1402.50 | 1399.25 | 1400.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 1410.43 | 1399.25 | 1400.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1407.40 | 1400.88 | 1401.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 1413.48 | 1403.40 | 1402.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1413.48 | 1403.40 | 1402.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 1419.00 | 1408.14 | 1404.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1415.25 | 1419.12 | 1414.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1415.25 | 1419.12 | 1414.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1415.50 | 1418.40 | 1414.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 1422.53 | 1418.40 | 1414.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 1407.13 | 1416.33 | 1414.24 | SL hit (close<static) qty=1.00 sl=1413.03 alert=retest2 |

### Cycle 90 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 1400.95 | 1412.39 | 1412.93 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 1424.53 | 1414.82 | 1413.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 1428.43 | 1419.25 | 1416.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1431.45 | 1432.17 | 1426.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1431.45 | 1432.17 | 1426.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1431.45 | 1432.17 | 1426.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:45:00 | 1440.83 | 1433.41 | 1428.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 13:45:00 | 1437.70 | 1434.11 | 1429.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 1460.65 | 1434.49 | 1430.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 1462.53 | 1468.80 | 1469.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 1462.53 | 1468.80 | 1469.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 1457.08 | 1466.45 | 1468.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 1431.85 | 1431.31 | 1440.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 1431.85 | 1431.31 | 1440.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1431.85 | 1431.31 | 1440.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:45:00 | 1435.18 | 1431.31 | 1440.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1477.25 | 1439.68 | 1439.73 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1489.95 | 1449.73 | 1444.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 1507.40 | 1471.41 | 1456.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1472.15 | 1483.55 | 1466.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1472.15 | 1483.55 | 1466.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1472.15 | 1483.55 | 1466.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1450.43 | 1483.55 | 1466.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1403.15 | 1467.47 | 1460.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1403.15 | 1467.47 | 1460.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1374.93 | 1448.96 | 1453.17 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 1441.18 | 1431.29 | 1430.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1445.55 | 1434.14 | 1431.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1470.63 | 1471.89 | 1459.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1474.75 | 1471.89 | 1459.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1466.00 | 1470.74 | 1460.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1460.50 | 1470.74 | 1460.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1462.33 | 1468.40 | 1462.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:00:00 | 1462.33 | 1468.40 | 1462.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 1462.33 | 1467.18 | 1462.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:15:00 | 1462.48 | 1467.18 | 1462.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1455.58 | 1464.86 | 1461.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 1455.58 | 1464.86 | 1461.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1457.53 | 1463.40 | 1461.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1466.88 | 1463.40 | 1461.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 1459.35 | 1470.34 | 1471.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 1459.35 | 1470.34 | 1471.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1457.83 | 1467.84 | 1470.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1467.50 | 1466.28 | 1468.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1467.50 | 1466.28 | 1468.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1467.50 | 1466.28 | 1468.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1467.50 | 1466.28 | 1468.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1475.50 | 1468.12 | 1469.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 1480.75 | 1468.12 | 1469.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1475.13 | 1469.52 | 1470.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 1475.13 | 1469.52 | 1470.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 1478.18 | 1471.25 | 1470.78 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 1461.63 | 1470.68 | 1470.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 1458.10 | 1468.16 | 1469.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 1447.50 | 1445.96 | 1451.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 13:00:00 | 1447.50 | 1445.96 | 1451.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1454.70 | 1448.28 | 1451.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 1454.70 | 1448.28 | 1451.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1452.53 | 1449.13 | 1451.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1457.33 | 1449.13 | 1451.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1467.00 | 1452.70 | 1452.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 1467.80 | 1452.70 | 1452.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 1477.60 | 1457.68 | 1455.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 1483.23 | 1462.79 | 1457.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 1561.50 | 1562.45 | 1551.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:45:00 | 1559.63 | 1562.45 | 1551.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1555.13 | 1561.99 | 1554.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 1555.13 | 1561.99 | 1554.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1549.83 | 1559.56 | 1554.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:45:00 | 1551.25 | 1559.56 | 1554.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 1545.00 | 1556.64 | 1553.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:00:00 | 1545.00 | 1556.64 | 1553.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1554.50 | 1553.18 | 1552.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 1556.93 | 1553.18 | 1552.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 1563.78 | 1559.66 | 1556.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 1576.43 | 1582.69 | 1582.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 1576.43 | 1582.69 | 1582.95 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 1602.85 | 1585.62 | 1583.58 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 12:15:00 | 1580.73 | 1588.92 | 1589.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 1576.40 | 1586.42 | 1588.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 1576.83 | 1575.22 | 1580.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:00:00 | 1576.83 | 1575.22 | 1580.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1592.63 | 1578.70 | 1582.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1592.63 | 1578.70 | 1582.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1584.98 | 1579.96 | 1582.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1572.23 | 1581.03 | 1582.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:15:00 | 1493.62 | 1511.29 | 1534.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 1501.43 | 1491.86 | 1509.15 | SL hit (close>ema200) qty=0.50 sl=1491.86 alert=retest2 |

### Cycle 103 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 1506.48 | 1501.42 | 1500.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1514.28 | 1506.01 | 1503.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1514.70 | 1515.63 | 1510.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 1514.70 | 1515.63 | 1510.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1513.13 | 1516.78 | 1513.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 1513.13 | 1516.78 | 1513.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1514.00 | 1516.22 | 1513.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:30:00 | 1514.50 | 1516.22 | 1513.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1511.50 | 1515.28 | 1512.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 1509.10 | 1515.28 | 1512.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1505.43 | 1513.31 | 1512.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 1504.00 | 1513.31 | 1512.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 1506.68 | 1510.69 | 1511.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 1503.00 | 1508.32 | 1509.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 1512.10 | 1508.23 | 1509.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1512.10 | 1508.23 | 1509.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1512.10 | 1508.23 | 1509.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 1512.58 | 1508.23 | 1509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1515.18 | 1509.62 | 1509.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 1515.18 | 1509.62 | 1509.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1511.93 | 1510.07 | 1510.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 1511.93 | 1510.07 | 1510.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 13:15:00 | 1510.50 | 1510.16 | 1510.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 14:15:00 | 1515.15 | 1511.15 | 1510.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1503.45 | 1510.70 | 1510.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1503.45 | 1510.70 | 1510.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1503.45 | 1510.70 | 1510.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 1499.25 | 1510.70 | 1510.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1502.23 | 1509.01 | 1509.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 1501.00 | 1507.41 | 1509.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1469.35 | 1461.44 | 1477.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1469.35 | 1461.44 | 1477.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1469.35 | 1461.44 | 1477.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 1454.50 | 1462.45 | 1466.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 1454.28 | 1459.97 | 1465.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:45:00 | 1453.00 | 1458.17 | 1463.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 1473.48 | 1464.00 | 1463.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 1473.48 | 1464.00 | 1463.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 1475.18 | 1466.24 | 1464.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1464.23 | 1467.99 | 1465.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1464.23 | 1467.99 | 1465.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1464.23 | 1467.99 | 1465.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 1460.40 | 1467.99 | 1465.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1470.30 | 1468.46 | 1466.37 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 1461.50 | 1465.04 | 1465.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 1459.03 | 1462.12 | 1463.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 1466.40 | 1462.98 | 1463.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 11:15:00 | 1466.40 | 1462.98 | 1463.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1466.40 | 1462.98 | 1463.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 1466.40 | 1462.98 | 1463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1469.48 | 1464.28 | 1464.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 1469.48 | 1464.28 | 1464.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 1465.33 | 1464.49 | 1464.48 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 1462.00 | 1463.99 | 1464.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 1460.05 | 1463.20 | 1463.87 | Break + close below crossover candle low |

### Cycle 111 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 1470.78 | 1464.72 | 1464.50 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 1459.43 | 1463.66 | 1464.04 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1472.65 | 1465.46 | 1464.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 1478.13 | 1469.44 | 1466.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 1493.25 | 1495.37 | 1488.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 1496.25 | 1495.37 | 1488.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1505.45 | 1497.38 | 1490.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:15:00 | 1506.70 | 1497.38 | 1490.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 1509.00 | 1497.82 | 1495.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 1505.70 | 1500.48 | 1498.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 11:00:00 | 1505.60 | 1508.97 | 1506.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1502.00 | 1506.53 | 1505.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 1502.00 | 1506.53 | 1505.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 1500.23 | 1504.72 | 1504.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 1500.23 | 1504.72 | 1504.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 1497.33 | 1501.13 | 1502.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 1501.25 | 1500.41 | 1501.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 1501.25 | 1500.41 | 1501.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1501.25 | 1500.41 | 1501.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:15:00 | 1505.15 | 1500.41 | 1501.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1503.90 | 1501.11 | 1502.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 1505.00 | 1501.11 | 1502.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1502.68 | 1501.42 | 1502.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:30:00 | 1505.88 | 1501.42 | 1502.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1507.50 | 1502.64 | 1502.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:45:00 | 1509.00 | 1502.64 | 1502.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 13:15:00 | 1531.85 | 1508.48 | 1505.31 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 1509.45 | 1512.56 | 1512.58 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 1514.80 | 1512.61 | 1512.57 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1506.88 | 1512.70 | 1512.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 1504.00 | 1510.96 | 1512.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 1465.00 | 1464.21 | 1473.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 10:45:00 | 1462.03 | 1464.21 | 1473.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 1464.50 | 1456.82 | 1461.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 1464.50 | 1456.82 | 1461.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1480.45 | 1461.54 | 1462.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 1480.45 | 1461.54 | 1462.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 1476.75 | 1464.58 | 1464.08 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 1463.23 | 1470.51 | 1470.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1457.38 | 1464.66 | 1467.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 13:15:00 | 1471.43 | 1465.71 | 1467.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 13:15:00 | 1471.43 | 1465.71 | 1467.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1471.43 | 1465.71 | 1467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:45:00 | 1473.93 | 1465.71 | 1467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1470.18 | 1466.60 | 1467.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:45:00 | 1471.30 | 1466.60 | 1467.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 1477.50 | 1469.33 | 1468.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 1487.48 | 1472.96 | 1470.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 1491.13 | 1492.50 | 1487.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 11:30:00 | 1490.28 | 1492.50 | 1487.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1489.58 | 1492.59 | 1488.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1489.58 | 1492.59 | 1488.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1490.25 | 1492.12 | 1489.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 1485.50 | 1492.12 | 1489.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1484.98 | 1490.69 | 1488.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 1482.03 | 1490.69 | 1488.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1483.73 | 1489.30 | 1488.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 1483.25 | 1489.30 | 1488.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1495.98 | 1489.64 | 1488.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:00:00 | 1497.18 | 1492.35 | 1490.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 1496.63 | 1496.32 | 1493.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 1497.00 | 1497.65 | 1494.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 12:00:00 | 1497.20 | 1497.20 | 1494.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1510.43 | 1499.89 | 1496.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:15:00 | 1517.65 | 1499.89 | 1496.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 1490.10 | 1503.41 | 1500.12 | SL hit (close<static) qty=1.00 sl=1496.10 alert=retest2 |

### Cycle 122 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 1487.20 | 1497.93 | 1498.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1480.78 | 1494.50 | 1496.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1383.88 | 1379.52 | 1396.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 1383.88 | 1379.52 | 1396.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1396.25 | 1384.73 | 1395.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 1396.25 | 1384.73 | 1395.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1397.23 | 1387.23 | 1395.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1397.23 | 1387.23 | 1395.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1399.48 | 1389.68 | 1395.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 1390.75 | 1389.68 | 1395.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 1365.50 | 1359.39 | 1358.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1365.50 | 1359.39 | 1358.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 12:15:00 | 1368.00 | 1363.04 | 1360.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 1365.05 | 1365.80 | 1363.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 1365.05 | 1365.80 | 1363.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1365.00 | 1365.64 | 1363.25 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 1354.35 | 1360.79 | 1361.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1341.45 | 1356.92 | 1359.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 1338.85 | 1337.17 | 1343.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 14:15:00 | 1338.85 | 1337.17 | 1343.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1338.85 | 1337.17 | 1343.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:30:00 | 1342.15 | 1337.17 | 1343.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1328.00 | 1335.63 | 1341.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 1325.43 | 1334.30 | 1340.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 1326.53 | 1333.21 | 1339.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:30:00 | 1326.25 | 1331.32 | 1337.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 15:00:00 | 1326.90 | 1331.32 | 1337.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1332.00 | 1330.93 | 1335.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:15:00 | 1340.05 | 1330.93 | 1335.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1347.70 | 1334.28 | 1336.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1347.70 | 1334.28 | 1336.89 | SL hit (close>static) qty=1.00 sl=1344.35 alert=retest2 |

### Cycle 125 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1345.35 | 1337.69 | 1336.84 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 1334.20 | 1338.28 | 1338.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 15:15:00 | 1332.00 | 1336.36 | 1337.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1338.55 | 1336.80 | 1337.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1338.55 | 1336.80 | 1337.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1338.55 | 1336.80 | 1337.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1338.55 | 1336.80 | 1337.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1338.40 | 1337.12 | 1337.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 1308.00 | 1337.12 | 1337.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 1325.05 | 1312.00 | 1311.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1325.05 | 1312.00 | 1311.12 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1305.50 | 1311.10 | 1311.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 1292.35 | 1304.91 | 1307.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 1284.10 | 1278.28 | 1285.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 1284.10 | 1278.28 | 1285.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1284.10 | 1278.28 | 1285.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1284.10 | 1278.28 | 1285.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1280.70 | 1278.76 | 1284.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 1282.65 | 1278.76 | 1284.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1265.70 | 1259.99 | 1266.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 1264.90 | 1259.99 | 1266.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1267.25 | 1261.44 | 1266.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:30:00 | 1264.40 | 1261.44 | 1266.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1269.75 | 1263.10 | 1267.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 1267.90 | 1263.10 | 1267.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1268.00 | 1264.08 | 1267.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 1271.45 | 1264.08 | 1267.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 1266.75 | 1264.62 | 1267.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 1263.20 | 1264.62 | 1267.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1251.20 | 1261.93 | 1265.73 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1276.80 | 1265.85 | 1265.11 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1239.00 | 1262.10 | 1263.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1237.75 | 1257.23 | 1261.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1238.50 | 1231.59 | 1242.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1238.50 | 1231.59 | 1242.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1233.50 | 1231.97 | 1241.61 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 1264.00 | 1248.87 | 1246.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1290.65 | 1257.23 | 1250.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 11:15:00 | 1291.20 | 1291.69 | 1283.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 11:45:00 | 1290.70 | 1291.69 | 1283.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1292.40 | 1293.56 | 1287.79 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 1269.70 | 1282.80 | 1284.29 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1295.90 | 1286.30 | 1285.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 1299.70 | 1292.43 | 1289.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 1317.75 | 1318.24 | 1308.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:45:00 | 1317.75 | 1318.24 | 1308.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1310.40 | 1315.68 | 1309.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1309.00 | 1315.68 | 1309.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1309.40 | 1314.43 | 1309.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 1308.90 | 1314.43 | 1309.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1311.40 | 1313.82 | 1309.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 1307.95 | 1313.82 | 1309.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1307.85 | 1312.63 | 1309.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 1306.00 | 1312.63 | 1309.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1309.50 | 1312.00 | 1309.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 1313.50 | 1312.00 | 1309.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1307.95 | 1311.19 | 1309.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:15:00 | 1310.95 | 1311.19 | 1309.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1310.60 | 1311.07 | 1309.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 1321.55 | 1313.17 | 1310.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1323.40 | 1317.88 | 1313.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 1307.00 | 1313.12 | 1313.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 1307.00 | 1313.12 | 1313.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 1299.00 | 1308.08 | 1310.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 1261.95 | 1259.94 | 1269.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 1261.95 | 1259.94 | 1269.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1272.55 | 1262.46 | 1270.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 1272.55 | 1262.46 | 1270.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1272.80 | 1264.53 | 1270.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1272.80 | 1264.53 | 1270.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1274.45 | 1266.52 | 1270.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1278.55 | 1266.52 | 1270.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1270.05 | 1269.11 | 1271.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:45:00 | 1271.10 | 1269.11 | 1271.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1271.45 | 1269.58 | 1271.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 1271.45 | 1269.58 | 1271.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1268.95 | 1269.45 | 1270.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 1267.45 | 1269.45 | 1270.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 1257.20 | 1269.13 | 1270.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 1204.08 | 1222.20 | 1233.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 1223.85 | 1219.82 | 1229.63 | SL hit (close>ema200) qty=0.50 sl=1219.82 alert=retest2 |

### Cycle 135 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 1222.15 | 1217.91 | 1217.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 1228.85 | 1223.00 | 1220.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1234.15 | 1245.66 | 1239.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 1234.15 | 1245.66 | 1239.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1234.15 | 1245.66 | 1239.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1234.15 | 1245.66 | 1239.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1230.75 | 1242.68 | 1238.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1225.55 | 1242.68 | 1238.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 1230.65 | 1240.27 | 1237.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 1230.65 | 1240.27 | 1237.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 1215.80 | 1233.04 | 1234.74 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1240.60 | 1235.60 | 1235.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 1268.65 | 1243.88 | 1239.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 10:15:00 | 1257.25 | 1259.79 | 1252.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 1257.25 | 1259.79 | 1252.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1257.25 | 1259.79 | 1252.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 1252.70 | 1259.79 | 1252.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 1255.55 | 1259.17 | 1253.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:00:00 | 1255.55 | 1259.17 | 1253.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 1251.65 | 1257.67 | 1253.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:45:00 | 1249.65 | 1257.67 | 1253.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1254.40 | 1257.01 | 1253.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:30:00 | 1251.65 | 1257.01 | 1253.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1243.25 | 1254.23 | 1252.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:30:00 | 1240.80 | 1254.23 | 1252.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1252.95 | 1253.97 | 1252.64 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 1243.25 | 1250.41 | 1251.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1241.20 | 1248.57 | 1250.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 1238.55 | 1235.91 | 1240.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:45:00 | 1237.30 | 1235.91 | 1240.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1249.00 | 1239.10 | 1240.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:30:00 | 1251.75 | 1239.10 | 1240.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 1254.10 | 1243.59 | 1242.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 1263.00 | 1252.12 | 1247.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1286.80 | 1299.47 | 1290.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1286.80 | 1299.47 | 1290.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1286.80 | 1299.47 | 1290.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1286.80 | 1299.47 | 1290.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1285.60 | 1296.70 | 1289.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 1287.40 | 1295.39 | 1289.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 13:00:00 | 1291.00 | 1294.51 | 1290.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1272.25 | 1288.83 | 1288.16 | SL hit (close<static) qty=1.00 sl=1277.55 alert=retest2 |

### Cycle 140 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 1277.00 | 1286.46 | 1287.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1270.70 | 1280.66 | 1284.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 1278.35 | 1277.87 | 1281.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1272.15 | 1277.87 | 1281.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1258.20 | 1266.35 | 1272.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:45:00 | 1256.65 | 1263.58 | 1270.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 1252.50 | 1261.75 | 1268.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 1240.50 | 1236.93 | 1240.38 | SL hit (close>ema400) qty=1.00 sl=1240.38 alert=retest1 |

### Cycle 141 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1252.75 | 1242.23 | 1241.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 1258.95 | 1251.03 | 1246.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1248.95 | 1259.18 | 1255.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1248.95 | 1259.18 | 1255.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1248.95 | 1259.18 | 1255.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1251.15 | 1259.18 | 1255.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1256.05 | 1258.55 | 1255.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1258.95 | 1258.55 | 1255.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1240.80 | 1256.70 | 1255.36 | SL hit (close<static) qty=1.00 sl=1244.70 alert=retest2 |

### Cycle 142 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1247.45 | 1254.19 | 1254.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 14:15:00 | 1245.75 | 1251.17 | 1252.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1259.15 | 1252.10 | 1252.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1259.15 | 1252.10 | 1252.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1259.15 | 1252.10 | 1252.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 1259.15 | 1252.10 | 1252.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1257.15 | 1253.11 | 1253.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:15:00 | 1261.35 | 1253.11 | 1253.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 1266.75 | 1255.84 | 1254.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 12:15:00 | 1276.20 | 1259.91 | 1256.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 1278.85 | 1278.98 | 1271.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 1278.85 | 1278.98 | 1271.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1274.45 | 1277.83 | 1273.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1274.45 | 1277.83 | 1273.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1279.20 | 1278.10 | 1274.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:00:00 | 1282.50 | 1278.98 | 1274.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 1271.65 | 1277.09 | 1275.05 | SL hit (close<static) qty=1.00 sl=1272.50 alert=retest2 |

### Cycle 144 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1264.95 | 1273.23 | 1273.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1246.90 | 1266.19 | 1270.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 10:15:00 | 1254.85 | 1254.75 | 1260.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 10:30:00 | 1256.90 | 1254.75 | 1260.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 1255.20 | 1254.84 | 1259.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:45:00 | 1256.85 | 1254.84 | 1259.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 1217.60 | 1214.09 | 1220.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 15:00:00 | 1217.60 | 1214.09 | 1220.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1217.00 | 1215.27 | 1219.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 1223.55 | 1215.27 | 1219.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1220.00 | 1216.21 | 1219.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:45:00 | 1221.75 | 1216.21 | 1219.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1222.15 | 1217.40 | 1219.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 1222.15 | 1217.40 | 1219.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1220.25 | 1217.97 | 1219.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1225.50 | 1217.97 | 1219.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1225.45 | 1220.58 | 1220.71 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 1223.00 | 1221.07 | 1220.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1230.85 | 1226.46 | 1224.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 1227.00 | 1227.32 | 1225.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 09:15:00 | 1231.75 | 1227.32 | 1225.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1235.55 | 1228.97 | 1226.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:45:00 | 1238.80 | 1232.91 | 1230.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1214.45 | 1226.76 | 1228.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1214.45 | 1226.76 | 1228.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 1210.10 | 1215.56 | 1220.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 13:15:00 | 1209.20 | 1207.11 | 1211.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 14:00:00 | 1209.20 | 1207.11 | 1211.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1212.90 | 1208.29 | 1211.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:45:00 | 1211.55 | 1208.29 | 1211.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 1213.30 | 1209.29 | 1211.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:30:00 | 1213.30 | 1209.29 | 1211.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 1205.85 | 1208.61 | 1210.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:00:00 | 1201.45 | 1206.92 | 1209.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 1198.00 | 1181.00 | 1178.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 1198.00 | 1181.00 | 1178.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 1200.00 | 1187.20 | 1182.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 1242.30 | 1242.39 | 1224.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 1242.30 | 1242.39 | 1224.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1238.60 | 1240.38 | 1230.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:45:00 | 1245.15 | 1242.08 | 1234.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:45:00 | 1249.35 | 1243.09 | 1235.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 12:00:00 | 1246.45 | 1246.43 | 1239.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 1247.70 | 1253.07 | 1248.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1246.65 | 1251.79 | 1248.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1249.05 | 1251.79 | 1248.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1243.80 | 1250.19 | 1248.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:15:00 | 1241.35 | 1250.19 | 1248.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1239.60 | 1248.07 | 1247.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 1241.30 | 1248.07 | 1247.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-17 11:15:00 | 1239.00 | 1246.26 | 1246.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 1239.00 | 1246.26 | 1246.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 1235.45 | 1244.10 | 1245.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 1239.75 | 1239.58 | 1241.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 15:00:00 | 1239.75 | 1239.58 | 1241.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1248.40 | 1241.41 | 1242.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:45:00 | 1248.10 | 1241.41 | 1242.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 1245.85 | 1242.30 | 1242.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 11:15:00 | 1244.20 | 1242.30 | 1242.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 1251.00 | 1244.29 | 1243.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 12:15:00 | 1251.00 | 1244.29 | 1243.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 1256.35 | 1248.19 | 1245.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1293.80 | 1294.96 | 1283.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 14:15:00 | 1284.55 | 1290.78 | 1285.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1284.55 | 1290.78 | 1285.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 1284.85 | 1290.78 | 1285.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1285.25 | 1289.67 | 1285.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1288.75 | 1289.67 | 1285.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 1283.40 | 1288.42 | 1285.23 | SL hit (close<static) qty=1.00 sl=1283.65 alert=retest2 |

### Cycle 150 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 1279.80 | 1283.35 | 1283.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1274.00 | 1281.48 | 1282.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 1282.65 | 1280.05 | 1281.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 1282.65 | 1280.05 | 1281.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1282.65 | 1280.05 | 1281.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 1282.65 | 1280.05 | 1281.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1281.00 | 1280.24 | 1281.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:45:00 | 1276.00 | 1279.32 | 1281.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 1291.65 | 1282.11 | 1281.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1291.65 | 1282.11 | 1281.51 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 1275.35 | 1280.82 | 1281.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1259.70 | 1273.93 | 1277.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1171.25 | 1170.10 | 1193.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1178.35 | 1170.10 | 1193.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1189.60 | 1177.15 | 1189.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1193.45 | 1177.15 | 1189.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1185.90 | 1178.90 | 1189.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1182.25 | 1178.90 | 1189.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1182.85 | 1179.69 | 1188.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1175.80 | 1179.99 | 1188.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 1178.30 | 1180.26 | 1187.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1212.80 | 1190.26 | 1189.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1212.80 | 1190.26 | 1189.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1221.40 | 1200.23 | 1194.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 1233.70 | 1234.32 | 1221.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:30:00 | 1232.70 | 1234.32 | 1221.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1230.00 | 1235.00 | 1228.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1246.80 | 1236.42 | 1229.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-28 13:15:00 | 1371.48 | 1342.11 | 1320.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 14:15:00 | 1405.70 | 1414.86 | 1415.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1398.00 | 1410.49 | 1412.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1423.80 | 1393.96 | 1399.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1423.80 | 1393.96 | 1399.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1423.80 | 1393.96 | 1399.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 1425.50 | 1393.96 | 1399.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1422.10 | 1404.15 | 1403.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1430.70 | 1412.45 | 1407.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1422.60 | 1424.45 | 1417.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1422.60 | 1424.45 | 1417.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1420.00 | 1423.56 | 1417.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1416.60 | 1423.56 | 1417.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1413.80 | 1421.61 | 1417.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1413.80 | 1421.61 | 1417.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1416.20 | 1420.53 | 1417.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1426.20 | 1420.53 | 1417.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1429.90 | 1421.93 | 1420.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 1435.70 | 1441.48 | 1441.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1435.70 | 1441.48 | 1441.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1432.40 | 1439.66 | 1440.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1434.00 | 1432.43 | 1436.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:30:00 | 1434.10 | 1432.43 | 1436.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1422.20 | 1430.39 | 1434.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1415.00 | 1428.76 | 1432.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 1426.00 | 1423.44 | 1423.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1426.00 | 1423.44 | 1423.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1438.50 | 1426.86 | 1424.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1418.30 | 1430.81 | 1429.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1424.80 | 1429.61 | 1428.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 1416.10 | 1429.61 | 1428.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1430.30 | 1429.75 | 1428.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:15:00 | 1423.60 | 1429.75 | 1428.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1421.10 | 1428.02 | 1428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 1417.70 | 1425.95 | 1427.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1419.00 | 1414.97 | 1417.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1418.40 | 1415.66 | 1417.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1421.10 | 1415.66 | 1417.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1422.00 | 1416.93 | 1418.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1427.80 | 1416.93 | 1418.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1417.50 | 1417.04 | 1418.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:30:00 | 1413.60 | 1416.17 | 1417.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 1413.40 | 1415.58 | 1417.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1398.90 | 1417.02 | 1417.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 1411.80 | 1411.77 | 1414.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1412.30 | 1411.87 | 1414.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 1415.60 | 1411.87 | 1414.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1414.00 | 1412.30 | 1414.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1414.00 | 1412.30 | 1414.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1412.60 | 1412.36 | 1414.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1411.90 | 1412.36 | 1414.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1415.20 | 1412.93 | 1414.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 1408.40 | 1410.87 | 1412.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1408.10 | 1409.44 | 1411.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 1408.00 | 1409.16 | 1411.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 13:15:00 | 1418.40 | 1412.46 | 1412.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 1418.40 | 1412.46 | 1412.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 1424.00 | 1414.77 | 1413.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 1442.10 | 1443.02 | 1435.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 14:00:00 | 1442.10 | 1443.02 | 1435.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1442.90 | 1446.63 | 1443.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 1442.90 | 1446.63 | 1443.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1440.60 | 1445.42 | 1443.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 1440.60 | 1445.42 | 1443.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1440.60 | 1444.46 | 1442.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 1442.10 | 1444.46 | 1442.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1438.10 | 1443.19 | 1442.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1438.10 | 1443.19 | 1442.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1437.30 | 1442.01 | 1441.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1456.70 | 1442.01 | 1441.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 14:15:00 | 1442.40 | 1446.78 | 1447.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1442.40 | 1446.78 | 1447.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1417.30 | 1439.72 | 1443.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1438.50 | 1432.53 | 1436.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1436.50 | 1433.33 | 1436.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 1435.30 | 1433.33 | 1436.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1438.20 | 1434.30 | 1436.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1439.70 | 1434.30 | 1436.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1440.00 | 1435.44 | 1437.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 1441.20 | 1435.44 | 1437.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1438.00 | 1435.95 | 1437.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 1436.90 | 1435.95 | 1437.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 1437.00 | 1432.92 | 1434.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1436.00 | 1432.92 | 1434.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:00:00 | 1434.40 | 1432.67 | 1432.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 13:15:00 | 1437.30 | 1433.60 | 1433.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 13:15:00 | 1437.30 | 1433.60 | 1433.23 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 1428.00 | 1432.48 | 1432.79 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1447.80 | 1435.54 | 1434.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1454.70 | 1439.37 | 1436.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1446.70 | 1451.75 | 1444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1449.80 | 1459.50 | 1454.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1449.80 | 1459.50 | 1454.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1452.80 | 1458.16 | 1454.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 1456.80 | 1458.16 | 1454.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1520.50 | 1535.50 | 1535.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 15:15:00 | 1520.50 | 1535.50 | 1535.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 1515.20 | 1531.44 | 1533.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1493.40 | 1489.94 | 1497.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1493.40 | 1489.94 | 1497.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1485.40 | 1484.33 | 1488.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1483.40 | 1483.74 | 1487.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1483.60 | 1483.24 | 1486.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:45:00 | 1483.50 | 1483.03 | 1486.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.23 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.42 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.33 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1399.10 | 1397.03 | 1405.42 | SL hit (close>ema200) qty=0.50 sl=1397.03 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 1416.60 | 1402.56 | 1401.26 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1392.90 | 1402.33 | 1403.08 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1413.30 | 1402.04 | 1400.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 1414.80 | 1407.89 | 1404.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1397.20 | 1406.51 | 1404.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1397.50 | 1404.70 | 1403.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 1396.80 | 1404.70 | 1403.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 1396.40 | 1403.04 | 1403.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 1392.50 | 1400.94 | 1402.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 1394.50 | 1392.39 | 1395.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1388.10 | 1383.37 | 1388.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 1386.70 | 1383.37 | 1388.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1391.00 | 1384.90 | 1388.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1388.50 | 1384.90 | 1388.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1383.30 | 1384.58 | 1388.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1376.70 | 1382.46 | 1387.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1397.20 | 1382.05 | 1380.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1397.20 | 1382.05 | 1380.96 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1375.00 | 1382.12 | 1382.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 1373.90 | 1377.35 | 1379.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 1389.90 | 1378.64 | 1379.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 1385.30 | 1380.91 | 1380.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1413.30 | 1388.11 | 1384.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 1413.00 | 1416.18 | 1407.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 1413.00 | 1416.18 | 1407.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1420.50 | 1423.51 | 1417.39 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1410.00 | 1414.87 | 1415.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1408.70 | 1413.63 | 1414.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 1416.10 | 1413.32 | 1414.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1415.50 | 1413.75 | 1414.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 1417.80 | 1413.75 | 1414.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1413.70 | 1413.17 | 1413.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1408.00 | 1413.17 | 1413.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 1374.40 | 1367.55 | 1367.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1374.40 | 1367.55 | 1367.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1383.90 | 1375.70 | 1372.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 1376.00 | 1377.00 | 1374.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1373.80 | 1376.36 | 1374.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1371.60 | 1376.36 | 1374.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1372.80 | 1375.65 | 1373.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1372.80 | 1375.65 | 1373.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1377.50 | 1376.02 | 1374.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1381.40 | 1376.44 | 1374.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1380.70 | 1378.74 | 1377.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1404.70 | 1408.56 | 1408.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1404.70 | 1408.56 | 1408.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 1403.50 | 1406.77 | 1407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1392.20 | 1391.09 | 1397.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1392.20 | 1391.09 | 1397.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1395.00 | 1390.39 | 1394.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1393.80 | 1390.39 | 1394.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1393.70 | 1391.05 | 1394.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 1389.70 | 1390.50 | 1393.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1374.40 | 1369.02 | 1368.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1374.40 | 1369.02 | 1368.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1387.90 | 1373.74 | 1370.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1381.00 | 1382.60 | 1377.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 1380.30 | 1382.60 | 1377.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1371.00 | 1380.28 | 1377.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1371.00 | 1380.28 | 1377.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1369.30 | 1378.09 | 1376.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 1366.70 | 1378.09 | 1376.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 1369.00 | 1374.90 | 1375.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1366.50 | 1373.22 | 1374.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1377.00 | 1375.03 | 1374.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1382.60 | 1377.17 | 1375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 1380.70 | 1381.57 | 1379.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:15:00 | 1372.90 | 1381.57 | 1379.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1371.50 | 1379.56 | 1378.42 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1370.00 | 1377.65 | 1377.65 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 1380.70 | 1376.78 | 1376.78 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1373.60 | 1376.28 | 1376.56 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1378.30 | 1376.74 | 1376.56 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 1374.70 | 1376.33 | 1376.39 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1377.90 | 1376.35 | 1376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1383.40 | 1377.76 | 1376.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1460.60 | 1463.66 | 1448.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 1460.60 | 1463.66 | 1448.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1446.30 | 1460.18 | 1448.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1446.30 | 1460.18 | 1448.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1447.00 | 1457.55 | 1448.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1445.40 | 1457.55 | 1448.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1448.00 | 1455.64 | 1448.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1454.60 | 1454.61 | 1448.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 1452.40 | 1453.85 | 1450.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1486.50 | 1489.10 | 1489.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1486.50 | 1489.10 | 1489.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 1484.60 | 1487.74 | 1488.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:00:00 | 1480.50 | 1486.21 | 1487.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:00:00 | 1479.80 | 1484.93 | 1486.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1493.20 | 1482.70 | 1484.99 | SL hit (close>static) qty=1.00 sl=1492.80 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 1494.80 | 1487.41 | 1486.88 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1480.40 | 1487.54 | 1488.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 1476.90 | 1485.41 | 1487.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1492.40 | 1486.04 | 1487.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1492.60 | 1487.35 | 1487.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1492.60 | 1487.35 | 1487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1494.90 | 1488.86 | 1488.20 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1481.80 | 1487.58 | 1488.20 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1494.00 | 1488.86 | 1488.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1504.00 | 1492.68 | 1490.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1509.90 | 1512.84 | 1506.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1509.90 | 1512.84 | 1506.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1515.00 | 1512.98 | 1508.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1517.90 | 1513.55 | 1508.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1517.50 | 1513.88 | 1509.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 1516.80 | 1514.28 | 1509.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 1520.10 | 1515.38 | 1511.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1514.50 | 1516.27 | 1513.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 1513.50 | 1516.27 | 1513.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1513.00 | 1515.61 | 1513.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 1513.00 | 1515.61 | 1513.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1516.00 | 1515.69 | 1513.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1518.00 | 1516.15 | 1513.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:45:00 | 1521.40 | 1516.44 | 1514.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 1520.00 | 1518.66 | 1516.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:45:00 | 1517.70 | 1519.18 | 1516.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1514.50 | 1518.25 | 1516.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1514.50 | 1518.25 | 1516.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1517.20 | 1518.04 | 1516.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 1518.70 | 1518.04 | 1516.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:45:00 | 1520.00 | 1518.76 | 1517.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 1553.40 | 1564.42 | 1565.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1553.40 | 1564.42 | 1565.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 1548.20 | 1559.30 | 1562.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1550.80 | 1543.78 | 1549.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1559.90 | 1547.00 | 1550.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1559.90 | 1547.00 | 1550.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1549.20 | 1547.44 | 1550.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 1548.60 | 1547.44 | 1550.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1546.40 | 1539.24 | 1538.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1546.40 | 1539.24 | 1538.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1551.30 | 1543.36 | 1540.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 1548.00 | 1552.75 | 1547.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1540.80 | 1551.78 | 1550.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1540.80 | 1551.78 | 1550.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1541.30 | 1549.68 | 1549.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 1540.70 | 1549.68 | 1549.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1543.20 | 1548.39 | 1548.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1539.30 | 1543.04 | 1544.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1549.70 | 1544.35 | 1545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1548.50 | 1545.18 | 1545.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 1551.00 | 1545.18 | 1545.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1544.70 | 1544.94 | 1545.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 1543.10 | 1544.94 | 1545.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1564.90 | 1548.70 | 1546.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 1572.00 | 1553.36 | 1549.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 1571.80 | 1572.94 | 1568.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1569.10 | 1571.54 | 1568.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1569.10 | 1571.54 | 1568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1566.60 | 1570.56 | 1568.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1566.60 | 1570.56 | 1568.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1563.70 | 1569.18 | 1567.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1563.70 | 1569.18 | 1567.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 1557.50 | 1566.85 | 1566.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1551.40 | 1558.38 | 1561.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1550.50 | 1549.98 | 1554.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:30:00 | 1551.20 | 1549.98 | 1554.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1548.00 | 1544.93 | 1549.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1546.90 | 1544.93 | 1549.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1555.00 | 1546.94 | 1549.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1555.00 | 1546.94 | 1549.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1567.80 | 1551.11 | 1551.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1567.80 | 1551.11 | 1551.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1572.50 | 1555.39 | 1553.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1582.20 | 1566.33 | 1559.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1572.90 | 1573.43 | 1566.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1585.00 | 1573.43 | 1566.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1579.70 | 1588.49 | 1582.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 1579.70 | 1588.49 | 1582.80 | SL hit (close<ema400) qty=1.00 sl=1582.80 alert=retest1 |

### Cycle 196 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 1526.20 | 1572.79 | 1576.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 1510.50 | 1560.33 | 1570.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1476.00 | 1474.68 | 1488.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 15:00:00 | 1476.00 | 1474.68 | 1488.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1478.70 | 1470.54 | 1479.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1478.70 | 1470.54 | 1479.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1483.40 | 1473.11 | 1480.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:15:00 | 1484.00 | 1473.11 | 1480.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1484.00 | 1475.29 | 1480.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1468.60 | 1475.29 | 1480.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1463.50 | 1461.18 | 1466.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1464.50 | 1461.18 | 1466.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1466.80 | 1461.13 | 1464.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 1470.30 | 1461.13 | 1464.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1473.10 | 1463.52 | 1465.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1476.00 | 1463.52 | 1465.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1458.70 | 1463.71 | 1465.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1421.60 | 1461.34 | 1463.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 1398.50 | 1394.30 | 1393.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 1398.50 | 1394.30 | 1393.99 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1391.30 | 1393.70 | 1393.74 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1395.60 | 1394.08 | 1393.91 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 1392.40 | 1393.75 | 1393.78 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1394.80 | 1393.96 | 1393.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1398.60 | 1394.89 | 1394.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 1391.10 | 1394.13 | 1394.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 15:15:00 | 1388.80 | 1393.06 | 1393.53 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 1397.00 | 1393.05 | 1392.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1404.80 | 1395.40 | 1393.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1378.40 | 1394.86 | 1394.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1370.70 | 1390.03 | 1392.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1349.80 | 1381.98 | 1388.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 1378.30 | 1371.60 | 1380.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:00:00 | 1378.30 | 1371.60 | 1380.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1380.20 | 1373.32 | 1380.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1380.20 | 1373.32 | 1380.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1379.20 | 1374.50 | 1380.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1379.80 | 1374.50 | 1380.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1391.00 | 1377.80 | 1381.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1391.00 | 1377.80 | 1381.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1388.00 | 1379.84 | 1381.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1445.20 | 1379.84 | 1381.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1443.90 | 1392.65 | 1387.39 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 1450.00 | 1458.91 | 1459.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1447.70 | 1456.67 | 1458.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 1426.00 | 1425.08 | 1433.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 1424.10 | 1425.08 | 1433.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1438.20 | 1427.71 | 1434.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 1436.00 | 1427.71 | 1434.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1435.80 | 1429.32 | 1434.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1424.10 | 1429.32 | 1434.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:15:00 | 1433.00 | 1426.49 | 1428.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 1441.90 | 1430.87 | 1430.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1441.90 | 1430.87 | 1430.02 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1421.10 | 1429.25 | 1429.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 1418.30 | 1427.06 | 1428.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1420.30 | 1417.66 | 1422.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:00:00 | 1420.30 | 1417.66 | 1422.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1422.40 | 1418.61 | 1422.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 1422.20 | 1418.61 | 1422.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1421.90 | 1419.27 | 1422.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 1425.20 | 1419.27 | 1422.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1422.40 | 1419.89 | 1422.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 1420.00 | 1419.89 | 1422.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1418.70 | 1419.66 | 1422.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1418.70 | 1419.66 | 1422.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1426.00 | 1420.84 | 1422.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 1422.70 | 1422.34 | 1422.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 1419.80 | 1422.02 | 1422.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1428.80 | 1423.38 | 1423.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1428.80 | 1423.38 | 1423.13 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1416.70 | 1422.30 | 1422.70 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1426.70 | 1423.26 | 1422.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 1429.30 | 1424.47 | 1423.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1412.40 | 1425.09 | 1424.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 1410.90 | 1422.26 | 1423.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 14:15:00 | 1395.40 | 1416.88 | 1420.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 1409.00 | 1406.83 | 1412.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 1392.40 | 1406.83 | 1412.01 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1405.50 | 1404.91 | 1409.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 1405.20 | 1404.91 | 1409.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1407.80 | 1405.48 | 1409.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 1407.80 | 1405.48 | 1409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1391.20 | 1402.63 | 1407.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1367.50 | 1401.06 | 1406.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1322.78 | 1360.44 | 1378.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1384.00 | 1355.59 | 1365.44 | SL hit (close>ema200) qty=0.50 sl=1355.59 alert=retest1 |

### Cycle 213 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 1377.90 | 1370.82 | 1370.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1387.80 | 1374.22 | 1372.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1414.50 | 1407.57 | 1398.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 1414.30 | 1410.48 | 1401.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 13:30:00 | 1413.40 | 1411.17 | 1404.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:00:00 | 1414.00 | 1411.17 | 1404.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1410.90 | 1410.69 | 1405.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1419.80 | 1410.69 | 1405.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1403.70 | 1409.29 | 1405.52 | SL hit (close<static) qty=1.00 sl=1405.70 alert=retest2 |

### Cycle 214 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1398.30 | 1403.27 | 1403.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1389.60 | 1400.54 | 1402.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 1406.90 | 1398.97 | 1400.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1409.70 | 1401.11 | 1401.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 1409.70 | 1401.11 | 1401.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1407.50 | 1402.39 | 1402.14 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1390.60 | 1400.32 | 1401.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1380.20 | 1392.66 | 1396.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 1383.70 | 1381.69 | 1387.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 1383.70 | 1381.69 | 1387.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1396.80 | 1384.71 | 1388.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1396.80 | 1384.71 | 1388.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1394.00 | 1386.57 | 1388.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1398.20 | 1386.57 | 1388.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1397.90 | 1391.03 | 1390.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 1403.60 | 1395.34 | 1392.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 1406.50 | 1396.15 | 1393.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 1404.10 | 1406.13 | 1401.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 1391.10 | 1400.86 | 1400.42 | SL hit (close<static) qty=1.00 sl=1392.30 alert=retest2 |

### Cycle 218 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1385.70 | 1397.83 | 1399.08 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1422.30 | 1401.79 | 1400.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 1428.90 | 1407.21 | 1403.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 15:15:00 | 1414.00 | 1415.45 | 1409.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 09:15:00 | 1404.50 | 1415.45 | 1409.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1400.70 | 1412.50 | 1408.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 1393.20 | 1412.50 | 1408.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1396.00 | 1409.20 | 1407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 1396.50 | 1409.20 | 1407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1400.40 | 1405.95 | 1406.37 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 1407.20 | 1406.72 | 1406.68 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1404.00 | 1406.18 | 1406.43 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 1413.80 | 1407.70 | 1407.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1427.60 | 1416.25 | 1412.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 1412.00 | 1417.71 | 1414.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1413.80 | 1416.93 | 1414.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1386.50 | 1416.93 | 1414.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1375.20 | 1408.58 | 1410.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1370.90 | 1401.05 | 1407.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 1364.20 | 1367.05 | 1369.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1350.80 | 1367.62 | 1369.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:15:00 | 1295.99 | 1336.47 | 1349.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 1303.90 | 1302.64 | 1317.37 | SL hit (close>ema200) qty=0.50 sl=1302.64 alert=retest2 |

### Cycle 225 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1345.00 | 1323.67 | 1323.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 1349.20 | 1337.39 | 1330.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1336.70 | 1337.25 | 1331.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1336.70 | 1337.25 | 1331.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1336.90 | 1337.18 | 1331.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:45:00 | 1334.80 | 1337.18 | 1331.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1332.60 | 1335.77 | 1332.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 1332.30 | 1335.77 | 1332.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1330.70 | 1334.76 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 1330.70 | 1334.76 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1329.10 | 1333.62 | 1332.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1340.00 | 1333.62 | 1332.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1321.00 | 1340.72 | 1338.38 | SL hit (close<static) qty=1.00 sl=1327.30 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1320.00 | 1336.57 | 1336.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 11:15:00 | 1316.00 | 1332.46 | 1334.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:30:00 | 1338.90 | 1325.17 | 1329.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 1339.40 | 1331.89 | 1331.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1344.40 | 1335.48 | 1333.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1334.00 | 1336.37 | 1334.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1338.10 | 1336.72 | 1334.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1341.00 | 1337.32 | 1335.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1346.70 | 1356.93 | 1357.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1346.70 | 1356.93 | 1357.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1341.90 | 1351.56 | 1354.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1339.00 | 1335.35 | 1342.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1341.60 | 1336.83 | 1341.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1341.60 | 1336.83 | 1341.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1347.60 | 1338.99 | 1342.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 1349.60 | 1338.99 | 1342.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1356.00 | 1342.39 | 1343.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1356.00 | 1342.39 | 1343.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1365.00 | 1346.91 | 1345.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1370.00 | 1351.53 | 1347.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1398.60 | 1410.06 | 1392.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 1398.60 | 1410.06 | 1392.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1400.20 | 1408.09 | 1393.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1398.80 | 1408.09 | 1393.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1447.80 | 1456.70 | 1446.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:45:00 | 1445.10 | 1456.70 | 1446.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1446.40 | 1454.64 | 1446.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 1446.40 | 1454.64 | 1446.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1433.00 | 1450.31 | 1445.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 1432.80 | 1450.31 | 1445.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1436.80 | 1447.61 | 1444.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1450.00 | 1447.61 | 1444.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:30:00 | 1442.30 | 1443.16 | 1443.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 1435.30 | 1441.59 | 1442.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 1435.30 | 1441.59 | 1442.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 1428.40 | 1436.89 | 1439.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 1438.50 | 1430.91 | 1435.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1436.00 | 1431.93 | 1435.19 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 09:15:00 | 1246.90 | 2023-05-16 10:15:00 | 1241.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-05-16 10:00:00 | 1246.50 | 2023-05-16 10:15:00 | 1241.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-06-19 12:15:00 | 1277.35 | 2023-06-19 13:15:00 | 1272.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-06-19 14:30:00 | 1276.47 | 2023-06-20 09:15:00 | 1271.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-07-07 09:15:00 | 1323.95 | 2023-07-14 12:15:00 | 1367.73 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2023-07-07 11:45:00 | 1322.50 | 2023-07-14 12:15:00 | 1367.73 | STOP_HIT | 1.00 | 3.42% |
| BUY | retest2 | 2023-07-07 13:15:00 | 1322.10 | 2023-07-14 12:15:00 | 1367.73 | STOP_HIT | 1.00 | 3.45% |
| BUY | retest2 | 2023-07-07 14:15:00 | 1324.90 | 2023-07-14 12:15:00 | 1367.73 | STOP_HIT | 1.00 | 3.23% |
| BUY | retest2 | 2023-07-14 11:15:00 | 1377.15 | 2023-07-14 12:15:00 | 1367.73 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-07-19 15:15:00 | 1425.73 | 2023-07-20 09:15:00 | 1308.18 | STOP_HIT | 1.00 | -8.24% |
| SELL | retest2 | 2023-07-26 13:00:00 | 1263.72 | 2023-07-31 09:15:00 | 1266.43 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-07-26 14:30:00 | 1264.72 | 2023-07-31 09:15:00 | 1266.43 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-07-28 10:00:00 | 1264.65 | 2023-07-31 09:15:00 | 1266.43 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-08-01 13:15:00 | 1263.95 | 2023-08-01 14:15:00 | 1257.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-08-14 13:45:00 | 1285.47 | 2023-08-17 14:15:00 | 1268.83 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-08-16 09:30:00 | 1284.60 | 2023-08-17 14:15:00 | 1268.83 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-08-16 10:15:00 | 1285.03 | 2023-08-17 14:15:00 | 1268.83 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-16 11:45:00 | 1284.22 | 2023-08-17 14:15:00 | 1268.83 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-08-18 15:15:00 | 1276.00 | 2023-08-29 09:15:00 | 1212.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-18 15:15:00 | 1276.00 | 2023-08-30 09:15:00 | 1217.00 | STOP_HIT | 0.50 | 4.62% |
| BUY | retest2 | 2023-09-12 12:00:00 | 1227.47 | 2023-09-12 12:15:00 | 1220.43 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-09-14 09:15:00 | 1231.50 | 2023-09-18 10:15:00 | 1220.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-09-14 11:45:00 | 1227.68 | 2023-09-18 10:15:00 | 1220.75 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-09-14 14:45:00 | 1227.13 | 2023-09-18 10:15:00 | 1220.75 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-09-14 15:15:00 | 1228.00 | 2023-09-18 10:15:00 | 1220.75 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-10-06 12:00:00 | 1156.50 | 2023-10-11 09:15:00 | 1162.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-10-09 09:15:00 | 1151.25 | 2023-10-11 09:15:00 | 1162.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-10-10 12:45:00 | 1157.13 | 2023-10-11 09:15:00 | 1162.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-10-13 13:15:00 | 1176.75 | 2023-10-18 11:15:00 | 1170.18 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-10-13 14:15:00 | 1176.40 | 2023-10-18 11:15:00 | 1170.18 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-10-16 12:45:00 | 1176.13 | 2023-10-18 12:15:00 | 1165.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-10-17 10:45:00 | 1176.40 | 2023-10-18 12:15:00 | 1165.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-10-17 15:15:00 | 1178.50 | 2023-10-18 12:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-10-18 09:30:00 | 1181.10 | 2023-10-18 12:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-10-31 14:15:00 | 1150.95 | 2023-10-31 15:15:00 | 1144.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-11-01 10:45:00 | 1153.65 | 2023-11-09 11:15:00 | 1159.70 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2023-11-02 09:15:00 | 1158.20 | 2023-11-09 11:15:00 | 1159.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2023-12-07 12:00:00 | 1226.93 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-12-07 13:30:00 | 1226.50 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-12-08 14:30:00 | 1227.00 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-08 15:00:00 | 1228.80 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-12-11 09:45:00 | 1230.47 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-12-11 10:45:00 | 1229.50 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-11 13:00:00 | 1229.90 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-12-12 09:45:00 | 1229.53 | 2023-12-12 13:15:00 | 1218.08 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-21 12:15:00 | 1277.90 | 2024-01-01 10:15:00 | 1291.68 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2023-12-22 12:15:00 | 1279.08 | 2024-01-01 10:15:00 | 1291.68 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-01-25 09:45:00 | 1340.93 | 2024-01-25 13:15:00 | 1351.55 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-01-25 12:30:00 | 1343.10 | 2024-01-25 13:15:00 | 1351.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-02-01 10:00:00 | 1439.48 | 2024-02-06 11:15:00 | 1435.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-02-01 11:00:00 | 1434.15 | 2024-02-06 11:15:00 | 1435.23 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-02-01 11:45:00 | 1431.95 | 2024-02-06 11:15:00 | 1435.23 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-02-01 14:45:00 | 1431.70 | 2024-02-06 11:15:00 | 1435.23 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-02-12 13:15:00 | 1452.23 | 2024-02-16 14:15:00 | 1460.60 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-02-12 15:15:00 | 1452.00 | 2024-02-16 14:15:00 | 1460.60 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-02-27 11:15:00 | 1488.18 | 2024-02-28 10:15:00 | 1464.88 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-02-27 12:00:00 | 1487.43 | 2024-02-28 10:15:00 | 1464.88 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-02-27 12:30:00 | 1490.60 | 2024-02-28 10:15:00 | 1464.88 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-02-27 14:45:00 | 1487.35 | 2024-02-28 10:15:00 | 1464.88 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-03-05 12:45:00 | 1498.90 | 2024-03-06 11:15:00 | 1480.55 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-03-06 10:30:00 | 1496.53 | 2024-03-06 11:15:00 | 1480.55 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-03-12 11:30:00 | 1477.40 | 2024-03-20 11:15:00 | 1438.80 | STOP_HIT | 1.00 | 2.61% |
| SELL | retest2 | 2024-03-12 14:15:00 | 1476.40 | 2024-03-20 11:15:00 | 1438.80 | STOP_HIT | 1.00 | 2.55% |
| BUY | retest2 | 2024-04-01 09:15:00 | 1491.58 | 2024-04-02 10:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-04-01 09:45:00 | 1490.00 | 2024-04-02 10:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-04-01 10:45:00 | 1488.83 | 2024-04-02 10:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-04-01 12:15:00 | 1488.58 | 2024-04-02 10:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-04-16 11:15:00 | 1461.00 | 2024-04-18 12:15:00 | 1476.98 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-04-16 12:30:00 | 1460.90 | 2024-04-18 12:15:00 | 1476.98 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-04-16 14:30:00 | 1461.63 | 2024-04-18 12:15:00 | 1476.98 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-04-26 14:15:00 | 1452.45 | 2024-04-29 11:15:00 | 1463.33 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-04-26 15:00:00 | 1452.93 | 2024-04-29 11:15:00 | 1463.33 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-05-02 09:15:00 | 1474.13 | 2024-05-03 09:15:00 | 1460.08 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-05-02 10:00:00 | 1472.15 | 2024-05-03 09:15:00 | 1460.08 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-05-02 15:15:00 | 1469.05 | 2024-05-03 09:15:00 | 1460.08 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1402.60 | 2024-05-14 10:15:00 | 1413.48 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-10 10:00:00 | 1398.20 | 2024-05-14 10:15:00 | 1413.48 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-05-13 09:15:00 | 1397.00 | 2024-05-14 10:15:00 | 1413.48 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-05-13 15:00:00 | 1402.68 | 2024-05-14 10:15:00 | 1413.48 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-05-16 09:15:00 | 1422.53 | 2024-05-16 10:15:00 | 1407.13 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-05-16 12:00:00 | 1418.40 | 2024-05-16 12:15:00 | 1409.30 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-05-21 12:45:00 | 1440.83 | 2024-05-28 13:15:00 | 1462.53 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-05-21 13:45:00 | 1437.70 | 2024-05-28 13:15:00 | 1462.53 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2024-05-22 09:15:00 | 1460.65 | 2024-05-28 13:15:00 | 1462.53 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1466.88 | 2024-06-19 13:15:00 | 1459.35 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-07-04 09:15:00 | 1556.93 | 2024-07-10 11:15:00 | 1576.43 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2024-07-05 09:30:00 | 1563.78 | 2024-07-10 11:15:00 | 1576.43 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1572.23 | 2024-07-23 09:15:00 | 1493.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1572.23 | 2024-07-24 10:15:00 | 1501.43 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2024-08-08 10:00:00 | 1454.50 | 2024-08-09 12:15:00 | 1473.48 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-08-08 10:30:00 | 1454.28 | 2024-08-09 12:15:00 | 1473.48 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-08 12:45:00 | 1453.00 | 2024-08-09 12:15:00 | 1473.48 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-08-21 10:15:00 | 1506.70 | 2024-08-27 14:15:00 | 1500.23 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-08-23 09:15:00 | 1509.00 | 2024-08-27 14:15:00 | 1500.23 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-08-26 09:15:00 | 1505.70 | 2024-08-27 14:15:00 | 1500.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-08-27 11:00:00 | 1505.60 | 2024-08-27 14:15:00 | 1500.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-09-26 10:00:00 | 1497.18 | 2024-09-30 10:15:00 | 1490.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-09-26 14:30:00 | 1496.63 | 2024-09-30 12:15:00 | 1487.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-27 09:45:00 | 1497.00 | 2024-09-30 12:15:00 | 1487.20 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-09-27 12:00:00 | 1497.20 | 2024-09-30 12:15:00 | 1487.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-09-27 14:15:00 | 1517.65 | 2024-09-30 12:15:00 | 1487.20 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-10-09 09:15:00 | 1390.75 | 2024-10-18 13:15:00 | 1365.50 | STOP_HIT | 1.00 | 1.82% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1325.43 | 2024-10-28 10:15:00 | 1347.70 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-25 12:15:00 | 1326.53 | 2024-10-28 10:15:00 | 1347.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-10-25 14:30:00 | 1326.25 | 2024-10-28 10:15:00 | 1347.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-10-25 15:00:00 | 1326.90 | 2024-10-28 10:15:00 | 1347.70 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-10-29 09:15:00 | 1332.75 | 2024-10-29 14:15:00 | 1340.75 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-29 13:15:00 | 1332.05 | 2024-10-29 14:15:00 | 1340.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-11-04 09:15:00 | 1308.00 | 2024-11-06 14:15:00 | 1325.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-12-05 12:00:00 | 1321.55 | 2024-12-09 09:15:00 | 1307.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-05 15:00:00 | 1323.40 | 2024-12-09 09:15:00 | 1307.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-12-16 14:15:00 | 1267.45 | 2024-12-20 14:15:00 | 1204.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 1267.45 | 2024-12-23 10:15:00 | 1223.85 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-12-17 09:15:00 | 1257.20 | 2025-01-01 12:15:00 | 1222.15 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-01-21 11:30:00 | 1287.40 | 2025-01-21 14:15:00 | 1272.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-01-21 13:00:00 | 1291.00 | 2025-01-21 14:15:00 | 1272.25 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest1 | 2025-01-23 09:15:00 | 1272.15 | 2025-01-29 11:15:00 | 1240.50 | STOP_HIT | 1.00 | 2.49% |
| SELL | retest2 | 2025-01-24 11:45:00 | 1256.65 | 2025-01-30 11:15:00 | 1252.75 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1252.50 | 2025-01-30 11:15:00 | 1252.75 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1258.95 | 2025-02-03 09:15:00 | 1240.80 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-06 15:00:00 | 1282.50 | 2025-02-07 10:15:00 | 1271.65 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-02-07 11:30:00 | 1279.60 | 2025-02-07 12:15:00 | 1271.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-02-21 09:45:00 | 1238.80 | 2025-02-24 09:15:00 | 1214.45 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-02-28 15:00:00 | 1201.45 | 2025-03-06 10:15:00 | 1198.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-03-11 13:45:00 | 1245.15 | 2025-03-17 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-03-11 14:45:00 | 1249.35 | 2025-03-17 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-03-12 12:00:00 | 1246.45 | 2025-03-17 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-03-13 15:00:00 | 1247.70 | 2025-03-17 11:15:00 | 1239.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-03-19 11:15:00 | 1244.20 | 2025-03-19 12:15:00 | 1251.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1288.75 | 2025-03-26 09:15:00 | 1283.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-03-27 11:45:00 | 1276.00 | 2025-03-28 10:15:00 | 1291.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1175.80 | 2025-04-11 09:15:00 | 1212.80 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-04-09 09:45:00 | 1178.30 | 2025-04-11 09:15:00 | 1212.80 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1246.80 | 2025-04-28 13:15:00 | 1371.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1426.20 | 2025-05-20 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-05-15 10:30:00 | 1429.90 | 2025-05-20 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1415.00 | 2025-05-23 14:15:00 | 1426.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-30 11:30:00 | 1413.60 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-05-30 14:00:00 | 1413.40 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1398.90 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-02 12:45:00 | 1411.80 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-03 14:45:00 | 1408.40 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-04 10:00:00 | 1408.10 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-04 11:00:00 | 1408.00 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1456.70 | 2025-06-12 14:15:00 | 1442.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-06-16 15:15:00 | 1436.90 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-06-18 09:45:00 | 1437.00 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-18 10:15:00 | 1436.00 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-19 13:00:00 | 1434.40 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-06-24 15:15:00 | 1456.80 | 2025-07-09 15:15:00 | 1520.50 | STOP_HIT | 1.00 | 4.37% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1483.40 | 2025-07-24 11:15:00 | 1409.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1483.60 | 2025-07-24 11:15:00 | 1409.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:45:00 | 1483.50 | 2025-07-24 11:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1483.40 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.68% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1483.60 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2025-07-17 12:45:00 | 1483.50 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.69% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1376.70 | 2025-08-12 09:15:00 | 1397.20 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1408.00 | 2025-09-05 13:15:00 | 1374.40 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1381.40 | 2025-09-22 10:15:00 | 1404.70 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1380.70 | 2025-09-22 10:15:00 | 1404.70 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2025-09-24 13:45:00 | 1389.70 | 2025-10-06 14:15:00 | 1374.40 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1454.60 | 2025-10-31 14:15:00 | 1486.50 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-10-24 15:00:00 | 1452.40 | 2025-10-31 14:15:00 | 1486.50 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-11-04 13:00:00 | 1480.50 | 2025-11-06 09:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-04 14:00:00 | 1479.80 | 2025-11-06 09:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1517.90 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1517.50 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-11-14 13:15:00 | 1516.80 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2025-11-14 15:00:00 | 1520.10 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-11-17 15:00:00 | 1518.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-11-18 11:45:00 | 1521.40 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-11-18 15:15:00 | 1520.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-11-19 09:45:00 | 1517.70 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-11-19 12:15:00 | 1518.70 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-11-19 13:45:00 | 1520.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1548.60 | 2025-12-11 12:15:00 | 1546.40 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2026-01-02 09:15:00 | 1585.00 | 2026-01-05 13:15:00 | 1579.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1421.60 | 2026-01-28 15:15:00 | 1398.50 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1424.10 | 2026-02-18 14:15:00 | 1441.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-02-18 13:15:00 | 1433.00 | 2026-02-18 14:15:00 | 1441.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1422.70 | 2026-02-23 14:15:00 | 1428.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-02-23 13:45:00 | 1419.80 | 2026-02-23 14:15:00 | 1428.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2026-02-27 09:15:00 | 1392.40 | 2026-03-04 09:15:00 | 1322.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-27 09:15:00 | 1392.40 | 2026-03-05 09:15:00 | 1384.00 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1367.50 | 2026-03-05 13:15:00 | 1377.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-03-09 14:30:00 | 1414.50 | 2026-03-11 09:15:00 | 1403.70 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-03-10 10:15:00 | 1414.30 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-10 13:30:00 | 1413.40 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-03-10 14:00:00 | 1414.00 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-11 09:15:00 | 1419.80 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-03-18 09:15:00 | 1406.50 | 2026-03-19 13:15:00 | 1391.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-03-19 09:45:00 | 1404.10 | 2026-03-19 13:15:00 | 1391.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1364.20 | 2026-04-06 10:15:00 | 1295.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1364.20 | 2026-04-07 14:15:00 | 1303.90 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1350.80 | 2026-04-08 11:15:00 | 1345.00 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1340.00 | 2026-04-13 09:15:00 | 1321.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1341.00 | 2026-04-23 10:15:00 | 1346.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1450.00 | 2026-05-07 10:15:00 | 1435.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-05-07 09:30:00 | 1442.30 | 2026-05-07 10:15:00 | 1435.30 | STOP_HIT | 1.00 | -0.49% |
