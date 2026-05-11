# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1493.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 224 |
| ALERT1 | 156 |
| ALERT2 | 155 |
| ALERT2_SKIP | 84 |
| ALERT3 | 393 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 21 |
| ENTRY2 | 168 |
| PARTIAL | 33 |
| TARGET_HIT | 14 |
| STOP_HIT | 179 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 222 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 92 / 130
- **Target hits / Stop hits / Partials:** 10 / 179 / 33
- **Avg / median % per leg:** 0.84% / -0.61%
- **Sum % (uncompounded):** 187.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 98 | 22 | 22.4% | 9 | 89 | 0 | -0.31% | -30.0% |
| BUY @ 2nd Alert (retest1) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.17% | -28.6% |
| BUY @ 3rd Alert (retest2) | 89 | 22 | 24.7% | 9 | 80 | 0 | -0.02% | -1.5% |
| SELL (all) | 124 | 70 | 56.5% | 1 | 90 | 33 | 1.75% | 217.1% |
| SELL @ 2nd Alert (retest1) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.46% | -6.0% |
| SELL @ 3rd Alert (retest2) | 111 | 68 | 61.3% | 1 | 78 | 32 | 2.01% | 223.1% |
| retest1 (combined) | 22 | 2 | 9.1% | 0 | 21 | 1 | -1.57% | -34.6% |
| retest2 (combined) | 200 | 90 | 45.0% | 10 | 158 | 32 | 1.11% | 221.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 1215.05 | 1224.33 | 1224.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 11:15:00 | 1205.15 | 1217.81 | 1221.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 14:15:00 | 1213.40 | 1212.28 | 1217.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 14:15:00 | 1213.40 | 1212.28 | 1217.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 1213.40 | 1212.28 | 1217.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 14:45:00 | 1213.95 | 1212.28 | 1217.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 1207.65 | 1211.31 | 1216.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 09:45:00 | 1200.60 | 1210.90 | 1214.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 10:15:00 | 1200.65 | 1210.90 | 1214.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 11:00:00 | 1201.45 | 1209.01 | 1212.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 12:15:00 | 1202.00 | 1207.90 | 1212.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 1213.95 | 1205.62 | 1208.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:00:00 | 1213.95 | 1205.62 | 1208.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 1205.75 | 1205.65 | 1208.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-05-31 13:15:00 | 1241.10 | 1212.74 | 1211.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 13:15:00 | 1241.10 | 1212.74 | 1211.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 15:15:00 | 1247.95 | 1222.69 | 1216.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 1239.20 | 1239.68 | 1229.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 12:15:00 | 1228.00 | 1237.07 | 1230.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 1228.00 | 1237.07 | 1230.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 12:45:00 | 1230.00 | 1237.07 | 1230.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 1230.90 | 1235.84 | 1230.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 12:45:00 | 1241.30 | 1235.70 | 1232.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 13:45:00 | 1240.95 | 1236.57 | 1233.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 15:15:00 | 1240.50 | 1237.26 | 1234.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 13:15:00 | 1233.00 | 1246.68 | 1247.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 1233.00 | 1246.68 | 1247.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 1226.75 | 1238.55 | 1243.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 1254.10 | 1237.69 | 1239.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 1254.10 | 1237.69 | 1239.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 1254.10 | 1237.69 | 1239.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:00:00 | 1254.10 | 1237.69 | 1239.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 1248.35 | 1239.82 | 1240.58 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 1250.60 | 1241.98 | 1241.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 1254.10 | 1247.14 | 1244.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 14:15:00 | 1241.50 | 1248.74 | 1246.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 14:15:00 | 1241.50 | 1248.74 | 1246.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 1241.50 | 1248.74 | 1246.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 15:00:00 | 1241.50 | 1248.74 | 1246.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 1240.00 | 1246.99 | 1246.20 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 1221.90 | 1241.97 | 1243.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 10:15:00 | 1204.95 | 1222.13 | 1230.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-19 09:15:00 | 1259.25 | 1202.55 | 1206.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 09:15:00 | 1259.25 | 1202.55 | 1206.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 1259.25 | 1202.55 | 1206.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:30:00 | 1254.95 | 1202.55 | 1206.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 10:15:00 | 1245.60 | 1211.16 | 1210.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 14:15:00 | 1275.00 | 1236.88 | 1223.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 1294.90 | 1301.09 | 1281.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 10:00:00 | 1294.90 | 1301.09 | 1281.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 1321.55 | 1343.96 | 1332.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:00:00 | 1321.55 | 1343.96 | 1332.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 1328.00 | 1340.77 | 1331.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:30:00 | 1330.00 | 1340.77 | 1331.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 1335.70 | 1339.75 | 1332.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:15:00 | 1325.50 | 1339.75 | 1332.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 1325.50 | 1336.90 | 1331.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 10:00:00 | 1371.35 | 1343.79 | 1335.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 15:15:00 | 1325.00 | 1342.39 | 1343.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 15:15:00 | 1325.00 | 1342.39 | 1343.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 09:15:00 | 1270.20 | 1327.95 | 1336.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 11:15:00 | 1272.00 | 1271.62 | 1293.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-03 11:45:00 | 1271.25 | 1271.62 | 1293.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 1285.00 | 1271.70 | 1284.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 14:15:00 | 1259.70 | 1270.22 | 1280.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 09:30:00 | 1255.05 | 1256.82 | 1263.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 13:15:00 | 1279.00 | 1269.55 | 1268.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 13:15:00 | 1279.00 | 1269.55 | 1268.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 09:15:00 | 1295.60 | 1277.04 | 1272.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 1273.00 | 1276.23 | 1272.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 1273.00 | 1276.23 | 1272.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 1273.00 | 1276.23 | 1272.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:00:00 | 1273.00 | 1276.23 | 1272.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 1290.75 | 1279.13 | 1274.05 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 1267.95 | 1274.20 | 1274.98 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 1290.90 | 1276.55 | 1275.85 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 1279.40 | 1283.96 | 1284.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 1268.60 | 1280.88 | 1283.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 1279.90 | 1277.00 | 1280.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 1279.90 | 1277.00 | 1280.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 1279.90 | 1277.00 | 1280.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 1278.75 | 1277.00 | 1280.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 1283.30 | 1278.57 | 1280.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 1283.50 | 1278.57 | 1280.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 1279.30 | 1278.72 | 1280.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:45:00 | 1280.80 | 1278.72 | 1280.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 1275.00 | 1277.97 | 1279.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:30:00 | 1278.00 | 1277.97 | 1279.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 1282.00 | 1277.82 | 1279.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 1288.55 | 1277.82 | 1279.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1290.00 | 1280.26 | 1280.43 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 1290.00 | 1282.21 | 1281.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 14:15:00 | 1293.25 | 1283.72 | 1282.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 1286.05 | 1291.40 | 1287.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 1286.05 | 1291.40 | 1287.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 1286.05 | 1291.40 | 1287.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 1286.05 | 1291.40 | 1287.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 1283.60 | 1289.84 | 1286.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:30:00 | 1283.60 | 1289.84 | 1286.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 13:15:00 | 1286.00 | 1289.07 | 1286.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 14:30:00 | 1300.65 | 1292.87 | 1288.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 11:15:00 | 1276.90 | 1308.74 | 1306.97 | SL hit (close<static) qty=1.00 sl=1282.60 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 1290.45 | 1305.08 | 1305.47 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 15:15:00 | 1315.00 | 1306.41 | 1305.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 1379.40 | 1321.01 | 1312.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 15:15:00 | 1360.55 | 1364.59 | 1351.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:15:00 | 1395.90 | 1364.59 | 1351.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 1377.10 | 1397.14 | 1387.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-27 13:15:00 | 1377.10 | 1397.14 | 1387.11 | SL hit (close<ema400) qty=1.00 sl=1387.11 alert=retest1 |

### Cycle 15 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 1443.15 | 1458.39 | 1459.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 10:15:00 | 1435.55 | 1447.26 | 1452.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 14:15:00 | 1448.20 | 1441.92 | 1447.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 14:15:00 | 1448.20 | 1441.92 | 1447.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 1448.20 | 1441.92 | 1447.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 1448.20 | 1441.92 | 1447.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 1440.00 | 1441.54 | 1446.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 09:15:00 | 1426.90 | 1441.54 | 1446.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 09:15:00 | 1429.10 | 1404.02 | 1403.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 1429.10 | 1404.02 | 1403.56 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 1380.70 | 1402.65 | 1405.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 1377.00 | 1397.52 | 1402.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 1381.55 | 1380.35 | 1390.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 10:45:00 | 1383.00 | 1380.35 | 1390.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 1391.00 | 1382.48 | 1390.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:00:00 | 1391.00 | 1382.48 | 1390.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 1392.00 | 1384.38 | 1390.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:30:00 | 1393.55 | 1384.38 | 1390.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 1394.25 | 1386.35 | 1390.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 14:00:00 | 1394.25 | 1386.35 | 1390.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 1399.95 | 1389.07 | 1391.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 14:45:00 | 1396.00 | 1389.07 | 1391.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 1400.65 | 1393.14 | 1393.05 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 09:15:00 | 1383.00 | 1394.26 | 1394.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 15:15:00 | 1372.05 | 1380.90 | 1386.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 1400.00 | 1384.72 | 1387.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 1400.00 | 1384.72 | 1387.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 1400.00 | 1384.72 | 1387.99 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 12:15:00 | 1400.00 | 1391.64 | 1390.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 14:15:00 | 1423.00 | 1400.66 | 1396.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 15:15:00 | 1405.05 | 1408.13 | 1403.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 15:15:00 | 1405.05 | 1408.13 | 1403.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 1405.05 | 1408.13 | 1403.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:15:00 | 1415.60 | 1408.13 | 1403.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 11:15:00 | 1399.10 | 1404.74 | 1403.01 | SL hit (close<static) qty=1.00 sl=1400.55 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 1394.45 | 1401.31 | 1401.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 12:15:00 | 1383.95 | 1393.67 | 1397.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 14:15:00 | 1396.95 | 1393.15 | 1396.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 1396.95 | 1393.15 | 1396.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 1396.95 | 1393.15 | 1396.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 1396.95 | 1393.15 | 1396.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 1401.00 | 1394.72 | 1396.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 1404.20 | 1394.72 | 1396.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1393.65 | 1394.51 | 1396.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 1396.50 | 1394.51 | 1396.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 1399.55 | 1395.62 | 1396.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:00:00 | 1399.55 | 1395.62 | 1396.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 12:15:00 | 1406.35 | 1397.77 | 1397.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 13:15:00 | 1410.20 | 1400.25 | 1398.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 1391.50 | 1400.77 | 1399.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 1391.50 | 1400.77 | 1399.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 1391.50 | 1400.77 | 1399.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 09:30:00 | 1440.90 | 1413.56 | 1406.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 14:15:00 | 1440.95 | 1446.55 | 1447.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 14:15:00 | 1440.95 | 1446.55 | 1447.08 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 1478.15 | 1447.10 | 1446.11 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 1415.80 | 1446.40 | 1449.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 1386.05 | 1434.33 | 1443.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 1411.35 | 1409.30 | 1422.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 15:00:00 | 1411.35 | 1409.30 | 1422.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 1420.05 | 1408.09 | 1414.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 1419.55 | 1408.09 | 1414.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 1416.60 | 1409.79 | 1414.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:15:00 | 1406.35 | 1412.86 | 1415.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 13:00:00 | 1409.75 | 1412.24 | 1414.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 13:30:00 | 1409.15 | 1410.00 | 1413.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 12:15:00 | 1336.03 | 1367.27 | 1388.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 12:15:00 | 1339.26 | 1367.27 | 1388.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 12:15:00 | 1338.69 | 1367.27 | 1388.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-20 14:15:00 | 1344.60 | 1338.77 | 1356.25 | SL hit (close>ema200) qty=0.50 sl=1338.77 alert=retest2 |

### Cycle 26 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 1341.90 | 1315.70 | 1314.69 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 1302.00 | 1322.93 | 1324.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 1299.90 | 1318.33 | 1321.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1314.95 | 1312.01 | 1317.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 09:45:00 | 1309.00 | 1312.01 | 1317.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 1303.90 | 1310.38 | 1316.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 15:15:00 | 1290.00 | 1311.51 | 1314.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 11:15:00 | 1328.30 | 1313.64 | 1314.23 | SL hit (close>static) qty=1.00 sl=1319.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 12:15:00 | 1323.00 | 1315.51 | 1315.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 14:15:00 | 1335.80 | 1319.50 | 1316.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 1309.90 | 1318.47 | 1316.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 1309.90 | 1318.47 | 1316.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 1309.90 | 1318.47 | 1316.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 1309.90 | 1318.47 | 1316.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 1313.95 | 1317.57 | 1316.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 12:30:00 | 1324.00 | 1317.73 | 1316.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 09:15:00 | 1307.80 | 1315.85 | 1316.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 09:15:00 | 1307.80 | 1315.85 | 1316.34 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 13:15:00 | 1327.10 | 1318.14 | 1317.15 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 10:15:00 | 1301.00 | 1314.28 | 1315.71 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 1332.55 | 1314.72 | 1313.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 1336.40 | 1323.81 | 1318.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 13:15:00 | 1389.25 | 1393.14 | 1383.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 14:00:00 | 1389.25 | 1393.14 | 1383.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 1381.35 | 1390.78 | 1383.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 14:30:00 | 1381.80 | 1390.78 | 1383.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 1381.65 | 1388.96 | 1383.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 1386.60 | 1388.96 | 1383.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 1380.05 | 1388.57 | 1386.50 | SL hit (close<static) qty=1.00 sl=1381.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 1378.00 | 1384.87 | 1385.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 1375.00 | 1382.90 | 1384.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 1382.55 | 1381.95 | 1383.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 1382.55 | 1381.95 | 1383.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 1382.55 | 1381.95 | 1383.73 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 1394.90 | 1385.63 | 1384.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 15:15:00 | 1399.80 | 1389.37 | 1387.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-25 12:15:00 | 1475.00 | 1488.89 | 1461.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-25 12:45:00 | 1475.40 | 1488.89 | 1461.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 1475.00 | 1485.62 | 1468.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:45:00 | 1472.40 | 1485.62 | 1468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 1588.50 | 1600.41 | 1582.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:30:00 | 1581.25 | 1600.80 | 1584.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 1630.00 | 1626.15 | 1613.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 1680.00 | 1626.15 | 1613.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 13:15:00 | 1655.00 | 1670.88 | 1671.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 1655.00 | 1670.88 | 1671.70 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 12:15:00 | 1688.95 | 1672.94 | 1671.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 15:15:00 | 1693.00 | 1680.71 | 1675.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 11:15:00 | 1687.00 | 1692.03 | 1683.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 12:00:00 | 1687.00 | 1692.03 | 1683.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 1683.65 | 1690.36 | 1683.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:00:00 | 1683.65 | 1690.36 | 1683.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 1684.30 | 1689.15 | 1683.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:45:00 | 1682.00 | 1689.15 | 1683.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 1709.30 | 1693.18 | 1685.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 1715.50 | 1691.74 | 1685.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 1724.20 | 1693.39 | 1687.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 14:15:00 | 1715.85 | 1712.05 | 1700.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 14:15:00 | 1680.00 | 1696.18 | 1698.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 1680.00 | 1696.18 | 1698.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 1663.75 | 1685.98 | 1692.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 15:15:00 | 1679.10 | 1675.68 | 1683.24 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 09:15:00 | 1670.00 | 1675.68 | 1683.24 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 14:15:00 | 1586.50 | 1645.47 | 1663.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-11-20 11:15:00 | 1630.30 | 1623.66 | 1645.36 | SL hit (close>ema200) qty=0.50 sl=1623.66 alert=retest1 |

### Cycle 38 — BUY (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 11:15:00 | 1711.10 | 1656.06 | 1651.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 12:15:00 | 1732.45 | 1671.34 | 1658.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 10:15:00 | 1693.00 | 1700.93 | 1681.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 11:00:00 | 1693.00 | 1700.93 | 1681.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 1678.15 | 1696.38 | 1680.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 1678.15 | 1696.38 | 1680.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 1668.30 | 1690.76 | 1679.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:00:00 | 1668.30 | 1690.76 | 1679.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 1672.90 | 1687.19 | 1679.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:00:00 | 1672.90 | 1687.19 | 1679.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 1675.00 | 1686.06 | 1680.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:15:00 | 1687.50 | 1686.06 | 1680.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1682.60 | 1685.37 | 1680.25 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 13:15:00 | 1653.30 | 1674.27 | 1676.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 14:15:00 | 1651.75 | 1669.77 | 1674.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1693.00 | 1673.17 | 1674.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1693.00 | 1673.17 | 1674.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1693.00 | 1673.17 | 1674.85 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 12:15:00 | 1679.75 | 1676.45 | 1676.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 14:15:00 | 1687.35 | 1679.06 | 1677.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 15:15:00 | 1675.00 | 1678.25 | 1677.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 15:15:00 | 1675.00 | 1678.25 | 1677.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 1675.00 | 1678.25 | 1677.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 09:30:00 | 1699.85 | 1681.66 | 1678.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 1689.30 | 1688.81 | 1684.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 10:00:00 | 1699.45 | 1690.94 | 1686.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 12:15:00 | 1671.80 | 1685.27 | 1684.80 | SL hit (close<static) qty=1.00 sl=1675.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 13:15:00 | 1671.95 | 1682.61 | 1683.63 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 1694.25 | 1685.68 | 1684.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 1696.55 | 1687.85 | 1685.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 1678.60 | 1687.38 | 1686.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 12:15:00 | 1678.60 | 1687.38 | 1686.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 1678.60 | 1687.38 | 1686.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:00:00 | 1678.60 | 1687.38 | 1686.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 1691.70 | 1688.24 | 1686.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 14:30:00 | 1694.45 | 1690.55 | 1688.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 15:00:00 | 1699.75 | 1690.55 | 1688.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 10:00:00 | 1698.95 | 1693.42 | 1689.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 1696.00 | 1702.06 | 1696.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 1696.00 | 1700.85 | 1696.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1740.10 | 1700.85 | 1696.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 15:15:00 | 1714.70 | 1716.62 | 1708.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 1713.75 | 1738.31 | 1736.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 14:15:00 | 1722.00 | 1733.47 | 1734.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 1722.00 | 1733.47 | 1734.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 14:15:00 | 1703.05 | 1720.75 | 1727.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 1778.05 | 1729.05 | 1729.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1778.05 | 1729.05 | 1729.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1778.05 | 1729.05 | 1729.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:00:00 | 1778.05 | 1729.05 | 1729.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 1775.00 | 1738.24 | 1733.75 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 15:15:00 | 1749.50 | 1758.87 | 1758.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 09:15:00 | 1732.90 | 1753.68 | 1756.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 1699.20 | 1695.96 | 1711.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 12:15:00 | 1686.00 | 1697.31 | 1709.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 13:45:00 | 1688.40 | 1695.24 | 1706.29 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 14:15:00 | 1689.45 | 1695.24 | 1706.29 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1680.10 | 1687.50 | 1699.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 1702.45 | 1689.73 | 1697.60 | SL hit (close>ema400) qty=1.00 sl=1697.60 alert=retest1 |

### Cycle 46 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 1588.50 | 1583.28 | 1583.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 11:15:00 | 1594.80 | 1587.07 | 1585.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 15:15:00 | 1702.00 | 1706.69 | 1678.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 09:15:00 | 1774.10 | 1706.69 | 1678.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:00:00 | 1765.10 | 1718.38 | 1685.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:15:00 | 1754.25 | 1750.01 | 1719.54 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 12:00:00 | 1766.40 | 1755.41 | 1729.95 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 1743.80 | 1750.09 | 1737.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:45:00 | 1739.65 | 1750.09 | 1737.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 1735.40 | 1747.16 | 1737.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 1735.40 | 1747.16 | 1737.71 | SL hit (close<ema400) qty=1.00 sl=1737.71 alert=retest1 |

### Cycle 47 — SELL (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 09:15:00 | 1718.20 | 1731.82 | 1732.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 10:15:00 | 1700.00 | 1725.46 | 1729.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 1708.40 | 1689.79 | 1701.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 1708.40 | 1689.79 | 1701.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 1708.40 | 1689.79 | 1701.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:45:00 | 1672.50 | 1694.15 | 1699.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 11:15:00 | 1659.70 | 1689.95 | 1697.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 12:00:00 | 1670.25 | 1655.75 | 1662.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 09:15:00 | 1683.60 | 1665.24 | 1664.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 09:15:00 | 1683.60 | 1665.24 | 1664.72 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 1659.85 | 1666.58 | 1667.07 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 11:15:00 | 1672.10 | 1667.69 | 1667.52 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 1647.65 | 1663.68 | 1665.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 1635.00 | 1657.94 | 1662.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 12:15:00 | 1559.90 | 1552.92 | 1586.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 13:00:00 | 1559.90 | 1552.92 | 1586.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 1597.00 | 1563.23 | 1585.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:45:00 | 1595.00 | 1563.23 | 1585.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1599.00 | 1570.39 | 1586.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 1612.50 | 1570.39 | 1586.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 1562.50 | 1572.46 | 1582.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 14:00:00 | 1562.50 | 1572.46 | 1582.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 1578.80 | 1568.23 | 1576.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 10:30:00 | 1577.05 | 1568.23 | 1576.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 1578.00 | 1570.19 | 1576.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 12:00:00 | 1578.00 | 1570.19 | 1576.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 1568.15 | 1569.78 | 1576.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 14:30:00 | 1564.00 | 1569.15 | 1574.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 1608.30 | 1575.03 | 1576.27 | SL hit (close>static) qty=1.00 sl=1584.35 alert=retest2 |

### Cycle 52 — BUY (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 10:15:00 | 1601.35 | 1580.30 | 1578.55 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 1550.25 | 1578.96 | 1579.35 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 13:15:00 | 1594.10 | 1580.39 | 1579.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 15:15:00 | 1600.05 | 1585.07 | 1581.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 09:15:00 | 1577.15 | 1583.49 | 1581.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 1577.15 | 1583.49 | 1581.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1577.15 | 1583.49 | 1581.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 1577.15 | 1583.49 | 1581.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1576.45 | 1582.08 | 1581.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:15:00 | 1588.20 | 1582.08 | 1581.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 12:30:00 | 1585.65 | 1583.47 | 1581.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 13:15:00 | 1591.00 | 1583.47 | 1581.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 1599.85 | 1590.94 | 1589.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 1598.15 | 1606.10 | 1601.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:45:00 | 1599.85 | 1606.10 | 1601.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 1617.10 | 1608.30 | 1602.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:30:00 | 1595.55 | 1608.30 | 1602.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 1606.00 | 1611.82 | 1606.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:15:00 | 1596.45 | 1611.82 | 1606.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1588.40 | 1607.14 | 1604.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 1588.40 | 1607.14 | 1604.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-08 10:15:00 | 1581.25 | 1601.96 | 1602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 1581.25 | 1601.96 | 1602.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 1567.10 | 1594.99 | 1599.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 13:15:00 | 1567.45 | 1563.55 | 1575.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 13:15:00 | 1567.45 | 1563.55 | 1575.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 1567.45 | 1563.55 | 1575.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 13:45:00 | 1565.00 | 1563.55 | 1575.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 1596.00 | 1570.04 | 1577.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 15:00:00 | 1596.00 | 1570.04 | 1577.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 15:15:00 | 1589.70 | 1573.97 | 1578.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 1652.45 | 1573.97 | 1578.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 1639.05 | 1586.99 | 1583.91 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 13:15:00 | 1604.80 | 1613.29 | 1613.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 14:15:00 | 1587.00 | 1608.04 | 1611.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 09:15:00 | 1593.25 | 1579.54 | 1591.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 1593.25 | 1579.54 | 1591.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1593.25 | 1579.54 | 1591.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:00:00 | 1593.25 | 1579.54 | 1591.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1568.65 | 1577.36 | 1589.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:30:00 | 1593.20 | 1577.36 | 1589.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1475.00 | 1468.79 | 1476.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 1488.75 | 1468.79 | 1476.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1481.00 | 1471.23 | 1476.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 14:30:00 | 1469.35 | 1476.13 | 1477.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 09:15:00 | 1491.95 | 1479.11 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 1491.95 | 1479.11 | 1478.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 10:15:00 | 1498.50 | 1482.99 | 1480.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 1484.35 | 1485.22 | 1482.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 13:15:00 | 1484.35 | 1485.22 | 1482.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 1484.35 | 1485.22 | 1482.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 1484.35 | 1485.22 | 1482.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1484.15 | 1485.01 | 1482.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:45:00 | 1482.15 | 1485.01 | 1482.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1489.40 | 1486.69 | 1483.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:45:00 | 1485.70 | 1486.69 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 1466.10 | 1482.57 | 1482.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 1466.10 | 1482.57 | 1482.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1469.35 | 1479.92 | 1481.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 13:15:00 | 1464.45 | 1474.39 | 1478.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 1470.00 | 1465.45 | 1470.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 1470.00 | 1465.45 | 1470.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 1470.00 | 1465.45 | 1470.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 1470.00 | 1465.45 | 1470.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1469.55 | 1466.27 | 1470.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 1473.35 | 1466.27 | 1470.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1485.15 | 1470.04 | 1471.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:15:00 | 1494.65 | 1470.04 | 1471.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 1488.05 | 1473.64 | 1473.32 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 1466.95 | 1476.82 | 1477.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 12:15:00 | 1446.85 | 1470.83 | 1474.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 11:15:00 | 1451.10 | 1450.68 | 1460.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-05 11:30:00 | 1451.90 | 1450.68 | 1460.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1444.95 | 1449.53 | 1459.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 15:00:00 | 1434.35 | 1445.80 | 1455.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 09:15:00 | 1362.63 | 1390.48 | 1415.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-07 12:15:00 | 1401.40 | 1390.68 | 1409.41 | SL hit (close>ema200) qty=0.50 sl=1390.68 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 13:15:00 | 1380.50 | 1332.98 | 1331.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 1396.75 | 1374.10 | 1356.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 1389.90 | 1393.50 | 1371.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 1389.90 | 1393.50 | 1371.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1389.90 | 1393.50 | 1371.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 1376.55 | 1393.50 | 1371.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1383.30 | 1393.99 | 1382.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1383.30 | 1393.99 | 1382.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1386.05 | 1392.41 | 1382.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 1372.05 | 1392.41 | 1382.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1393.65 | 1399.65 | 1391.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:30:00 | 1389.00 | 1399.65 | 1391.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 1400.10 | 1399.74 | 1391.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 11:45:00 | 1408.60 | 1401.43 | 1393.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 12:45:00 | 1405.95 | 1401.72 | 1394.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 14:15:00 | 1380.30 | 1397.16 | 1393.49 | SL hit (close<static) qty=1.00 sl=1391.75 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 1378.90 | 1392.31 | 1394.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 1350.20 | 1383.89 | 1390.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 12:15:00 | 1379.75 | 1377.55 | 1385.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 13:00:00 | 1379.75 | 1377.55 | 1385.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 1390.00 | 1380.04 | 1385.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:00:00 | 1390.00 | 1380.04 | 1385.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 1398.00 | 1383.63 | 1386.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 15:00:00 | 1398.00 | 1383.63 | 1386.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 1388.00 | 1384.50 | 1386.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:15:00 | 1413.50 | 1384.50 | 1386.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1401.60 | 1387.92 | 1388.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:30:00 | 1403.25 | 1387.92 | 1388.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 10:15:00 | 1408.55 | 1392.05 | 1389.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 12:15:00 | 1413.00 | 1398.88 | 1393.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 12:15:00 | 1406.55 | 1415.81 | 1406.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 12:15:00 | 1406.55 | 1415.81 | 1406.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 1406.55 | 1415.81 | 1406.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 13:00:00 | 1406.55 | 1415.81 | 1406.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 1410.00 | 1414.64 | 1407.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 13:30:00 | 1406.05 | 1414.64 | 1407.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1430.00 | 1417.72 | 1409.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 1412.15 | 1417.72 | 1409.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 1411.10 | 1421.74 | 1414.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 12:00:00 | 1411.10 | 1421.74 | 1414.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 12:15:00 | 1407.50 | 1418.89 | 1413.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 12:45:00 | 1406.90 | 1418.89 | 1413.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 10:15:00 | 1405.20 | 1410.64 | 1411.01 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 12:15:00 | 1417.45 | 1410.65 | 1409.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 14:15:00 | 1423.15 | 1413.88 | 1411.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 10:15:00 | 1478.80 | 1479.87 | 1457.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 11:00:00 | 1478.80 | 1479.87 | 1457.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 1473.00 | 1477.61 | 1471.09 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 1454.55 | 1467.67 | 1468.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 10:15:00 | 1441.00 | 1458.09 | 1463.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 1487.00 | 1447.49 | 1453.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 1487.00 | 1447.49 | 1453.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1487.00 | 1447.49 | 1453.22 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 1496.95 | 1457.38 | 1457.19 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 10:15:00 | 1456.50 | 1467.34 | 1467.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 14:15:00 | 1453.65 | 1461.59 | 1464.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 09:15:00 | 1438.50 | 1436.13 | 1446.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 1438.50 | 1436.13 | 1446.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1438.50 | 1436.13 | 1446.41 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 1465.25 | 1448.27 | 1447.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 1477.00 | 1455.65 | 1450.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 10:15:00 | 1486.85 | 1491.16 | 1477.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 10:45:00 | 1485.90 | 1491.16 | 1477.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1495.00 | 1494.87 | 1485.52 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 09:15:00 | 1470.00 | 1485.11 | 1485.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 1467.90 | 1479.16 | 1482.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 09:15:00 | 1489.95 | 1475.76 | 1478.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 1489.95 | 1475.76 | 1478.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1489.95 | 1475.76 | 1478.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 1489.95 | 1475.76 | 1478.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 1486.25 | 1477.86 | 1479.22 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 12:15:00 | 1486.50 | 1481.10 | 1480.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 11:15:00 | 1490.50 | 1485.22 | 1482.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 1488.00 | 1496.07 | 1490.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 1488.00 | 1496.07 | 1490.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1488.00 | 1496.07 | 1490.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:15:00 | 1482.00 | 1496.07 | 1490.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 1477.50 | 1492.36 | 1489.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:30:00 | 1475.85 | 1492.36 | 1489.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 1469.15 | 1484.14 | 1485.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 1450.05 | 1464.68 | 1471.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 1465.65 | 1450.87 | 1459.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 1465.65 | 1450.87 | 1459.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 1465.65 | 1450.87 | 1459.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:30:00 | 1474.00 | 1450.87 | 1459.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1482.95 | 1457.28 | 1461.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 1482.95 | 1457.28 | 1461.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 1460.00 | 1457.83 | 1461.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 1477.25 | 1457.83 | 1461.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1449.85 | 1456.23 | 1460.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 1459.80 | 1456.23 | 1460.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1442.90 | 1442.20 | 1451.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:45:00 | 1429.60 | 1440.15 | 1447.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:45:00 | 1431.25 | 1437.73 | 1446.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:30:00 | 1430.00 | 1426.83 | 1436.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:00:00 | 1397.20 | 1426.83 | 1436.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 1433.00 | 1415.40 | 1422.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 1402.05 | 1415.40 | 1422.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 15:00:00 | 1395.55 | 1401.74 | 1411.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 11:15:00 | 1416.95 | 1402.20 | 1402.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1416.95 | 1402.20 | 1402.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 1430.00 | 1415.93 | 1413.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1400.40 | 1415.66 | 1413.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1400.40 | 1415.66 | 1413.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1400.40 | 1415.66 | 1413.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1400.40 | 1415.66 | 1413.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1403.00 | 1413.12 | 1412.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1401.10 | 1413.12 | 1412.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 1398.05 | 1410.11 | 1411.50 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 1425.40 | 1411.74 | 1411.73 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 1410.65 | 1411.52 | 1411.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 1396.00 | 1408.42 | 1410.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 1407.75 | 1406.17 | 1408.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 1407.75 | 1406.17 | 1408.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1407.75 | 1406.17 | 1408.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1407.75 | 1406.17 | 1408.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1405.00 | 1405.94 | 1407.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 1401.35 | 1405.94 | 1407.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1414.35 | 1407.62 | 1408.45 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1415.10 | 1409.12 | 1409.06 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 1405.00 | 1409.06 | 1409.12 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 1417.50 | 1410.14 | 1409.56 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 1389.95 | 1405.59 | 1407.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 1376.65 | 1393.06 | 1399.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 1319.60 | 1318.09 | 1329.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:45:00 | 1323.50 | 1318.09 | 1329.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1327.25 | 1320.32 | 1328.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 1330.90 | 1320.32 | 1328.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1324.10 | 1321.07 | 1328.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 1329.50 | 1321.07 | 1328.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1310.00 | 1318.86 | 1326.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1323.65 | 1318.86 | 1326.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1316.60 | 1318.41 | 1325.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1285.10 | 1318.29 | 1322.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 1261.00 | 1300.80 | 1313.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1220.84 | 1287.40 | 1304.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1197.95 | 1287.40 | 1304.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1303.35 | 1283.11 | 1296.61 | SL hit (close>ema200) qty=0.50 sl=1283.11 alert=retest2 |

### Cycle 82 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1389.95 | 1320.97 | 1311.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 1410.00 | 1338.78 | 1320.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 1488.90 | 1493.16 | 1450.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 1488.90 | 1493.16 | 1450.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1492.65 | 1497.46 | 1487.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 1481.25 | 1497.46 | 1487.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1494.60 | 1496.89 | 1488.13 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 1478.05 | 1485.45 | 1485.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 1470.00 | 1477.50 | 1480.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 1491.65 | 1480.33 | 1481.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1491.65 | 1480.33 | 1481.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1491.65 | 1480.33 | 1481.47 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1490.00 | 1482.26 | 1482.24 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 1470.90 | 1484.27 | 1484.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 1466.00 | 1475.64 | 1479.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 1481.50 | 1473.33 | 1476.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 14:15:00 | 1481.50 | 1473.33 | 1476.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1481.50 | 1473.33 | 1476.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1465.00 | 1474.41 | 1476.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 1468.25 | 1471.82 | 1474.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1466.40 | 1471.59 | 1473.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:15:00 | 1391.75 | 1416.83 | 1438.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:15:00 | 1394.84 | 1416.83 | 1438.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:15:00 | 1393.08 | 1416.83 | 1438.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1371.10 | 1367.75 | 1386.47 | SL hit (close>ema200) qty=0.50 sl=1367.75 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1374.70 | 1360.00 | 1358.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 1385.15 | 1370.63 | 1364.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 1366.20 | 1369.74 | 1364.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 10:15:00 | 1366.20 | 1369.74 | 1364.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1366.20 | 1369.74 | 1364.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 1366.20 | 1369.74 | 1364.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1363.95 | 1368.58 | 1364.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 1363.95 | 1368.58 | 1364.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1358.70 | 1366.61 | 1364.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1358.70 | 1366.61 | 1364.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1348.95 | 1363.08 | 1362.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 1349.80 | 1363.08 | 1362.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 1339.80 | 1358.42 | 1360.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 1336.70 | 1349.27 | 1355.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 1334.70 | 1333.99 | 1343.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 1334.70 | 1333.99 | 1343.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1334.70 | 1333.99 | 1343.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1329.00 | 1333.19 | 1342.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:15:00 | 1328.90 | 1332.56 | 1341.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 1326.10 | 1331.79 | 1339.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 1329.30 | 1331.37 | 1338.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1339.80 | 1332.84 | 1338.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:00:00 | 1339.80 | 1332.84 | 1338.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1339.30 | 1334.13 | 1338.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 1339.30 | 1334.13 | 1338.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1338.05 | 1334.91 | 1338.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:45:00 | 1342.65 | 1334.91 | 1338.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1329.85 | 1333.90 | 1337.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1326.65 | 1331.82 | 1335.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1262.55 | 1280.61 | 1297.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1262.45 | 1280.61 | 1297.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1259.79 | 1280.61 | 1297.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1262.83 | 1280.61 | 1297.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:15:00 | 1260.32 | 1280.61 | 1297.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 1286.75 | 1269.34 | 1283.97 | SL hit (close>ema200) qty=0.50 sl=1269.34 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 1296.35 | 1284.98 | 1283.88 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 1281.15 | 1285.09 | 1285.19 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1299.80 | 1287.30 | 1286.13 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 1277.25 | 1285.05 | 1285.78 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 1299.95 | 1286.95 | 1286.45 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 1284.30 | 1288.17 | 1288.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 1247.50 | 1280.17 | 1284.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 1270.00 | 1265.55 | 1273.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 15:15:00 | 1270.00 | 1265.55 | 1273.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1270.00 | 1265.55 | 1273.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1273.60 | 1265.55 | 1273.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1270.95 | 1266.63 | 1273.24 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 1295.00 | 1277.25 | 1276.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 1299.75 | 1281.75 | 1279.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 1296.15 | 1298.02 | 1291.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 15:00:00 | 1296.15 | 1298.02 | 1291.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1294.65 | 1297.66 | 1292.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:45:00 | 1298.55 | 1297.64 | 1292.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 13:15:00 | 1275.00 | 1288.30 | 1289.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 1275.00 | 1288.30 | 1289.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 1270.20 | 1284.68 | 1287.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 11:15:00 | 1297.80 | 1284.97 | 1286.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 11:15:00 | 1297.80 | 1284.97 | 1286.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 1297.80 | 1284.97 | 1286.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 1297.80 | 1284.97 | 1286.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 1301.95 | 1288.37 | 1287.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1325.65 | 1302.62 | 1295.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 1316.00 | 1319.28 | 1307.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:30:00 | 1314.00 | 1319.28 | 1307.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1311.25 | 1317.17 | 1309.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:45:00 | 1324.85 | 1316.63 | 1311.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:00:00 | 1321.85 | 1319.48 | 1313.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:00:00 | 1319.95 | 1319.51 | 1314.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:45:00 | 1323.90 | 1320.87 | 1316.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1325.55 | 1321.81 | 1317.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1325.55 | 1321.81 | 1317.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1315.30 | 1322.62 | 1318.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 1314.50 | 1318.32 | 1318.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 1314.50 | 1318.32 | 1318.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 1313.00 | 1317.25 | 1317.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 1285.00 | 1283.04 | 1295.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:15:00 | 1290.50 | 1283.04 | 1295.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1307.40 | 1287.91 | 1296.65 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 13:15:00 | 1309.00 | 1301.48 | 1301.12 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1295.05 | 1300.95 | 1301.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 12:15:00 | 1288.65 | 1298.08 | 1299.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 1264.40 | 1264.05 | 1276.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 1264.40 | 1264.05 | 1276.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1264.40 | 1264.05 | 1276.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:15:00 | 1253.80 | 1262.23 | 1274.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:00:00 | 1253.65 | 1254.93 | 1266.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1191.11 | 1206.30 | 1226.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 1190.97 | 1206.30 | 1226.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 1199.45 | 1178.61 | 1193.37 | SL hit (close>ema200) qty=0.50 sl=1178.61 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1227.60 | 1203.64 | 1201.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 1235.00 | 1213.51 | 1206.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 1226.70 | 1227.82 | 1221.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:30:00 | 1225.45 | 1227.82 | 1221.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1222.90 | 1225.60 | 1222.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 1222.90 | 1225.60 | 1222.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 1223.70 | 1225.22 | 1222.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 1248.70 | 1225.22 | 1222.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 12:45:00 | 1226.50 | 1228.61 | 1225.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 1201.50 | 1220.23 | 1222.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1201.50 | 1220.23 | 1222.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 1195.00 | 1199.71 | 1204.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1199.05 | 1197.10 | 1201.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1199.05 | 1197.10 | 1201.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1199.05 | 1197.10 | 1201.31 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1218.65 | 1205.67 | 1204.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1221.20 | 1208.77 | 1205.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 1209.30 | 1212.15 | 1208.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 1209.30 | 1212.15 | 1208.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1209.30 | 1212.15 | 1208.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 1209.90 | 1212.15 | 1208.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1207.15 | 1211.15 | 1208.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 1206.30 | 1211.15 | 1208.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1206.95 | 1210.31 | 1208.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 1203.70 | 1210.31 | 1208.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1209.00 | 1210.05 | 1208.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1210.00 | 1209.99 | 1208.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 10:30:00 | 1209.85 | 1209.80 | 1208.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:15:00 | 1210.00 | 1209.80 | 1208.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:30:00 | 1210.00 | 1209.82 | 1208.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 1209.45 | 1209.75 | 1209.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:15:00 | 1209.45 | 1209.75 | 1209.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1209.30 | 1209.66 | 1209.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:30:00 | 1209.45 | 1209.66 | 1209.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1210.20 | 1209.77 | 1209.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1204.45 | 1209.77 | 1209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1201.45 | 1208.10 | 1208.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 1201.45 | 1208.10 | 1208.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 1185.50 | 1197.22 | 1202.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 09:15:00 | 1183.80 | 1183.22 | 1190.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1183.80 | 1183.22 | 1190.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1183.80 | 1183.22 | 1190.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 1177.50 | 1184.09 | 1189.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1212.10 | 1190.55 | 1191.02 | SL hit (close>static) qty=1.00 sl=1192.85 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 15:15:00 | 1207.90 | 1194.02 | 1192.56 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 1188.65 | 1191.22 | 1191.52 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 1208.60 | 1194.22 | 1192.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1216.05 | 1199.36 | 1195.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 13:15:00 | 1205.10 | 1205.58 | 1200.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 14:00:00 | 1205.10 | 1205.58 | 1200.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 1203.40 | 1207.86 | 1203.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:00:00 | 1203.40 | 1207.86 | 1203.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1202.55 | 1206.80 | 1203.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:30:00 | 1205.25 | 1206.80 | 1203.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1197.85 | 1205.01 | 1203.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1197.85 | 1205.01 | 1203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1194.60 | 1202.93 | 1202.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 1192.30 | 1202.93 | 1202.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1239.45 | 1255.28 | 1248.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 1240.15 | 1255.28 | 1248.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1238.25 | 1251.87 | 1247.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 13:00:00 | 1241.60 | 1247.65 | 1246.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 1243.95 | 1246.40 | 1245.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 1237.20 | 1245.78 | 1246.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1237.20 | 1245.78 | 1246.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1232.80 | 1242.22 | 1244.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1266.75 | 1241.39 | 1242.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 1266.75 | 1241.39 | 1242.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1266.75 | 1241.39 | 1242.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:30:00 | 1265.35 | 1241.39 | 1242.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 1269.00 | 1246.92 | 1244.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 1285.55 | 1254.64 | 1248.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 1262.00 | 1269.84 | 1261.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 1262.00 | 1269.84 | 1261.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1262.00 | 1269.84 | 1261.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 1262.00 | 1269.84 | 1261.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1265.90 | 1269.05 | 1262.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:15:00 | 1250.00 | 1269.05 | 1262.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 1248.35 | 1264.91 | 1261.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:30:00 | 1249.00 | 1264.91 | 1261.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 1248.55 | 1261.64 | 1259.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:00:00 | 1248.55 | 1261.64 | 1259.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 14:15:00 | 1245.00 | 1256.22 | 1257.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 15:15:00 | 1238.00 | 1252.58 | 1255.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 15:15:00 | 1221.00 | 1219.09 | 1227.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:15:00 | 1222.00 | 1219.09 | 1227.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1221.55 | 1219.58 | 1226.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 1211.65 | 1217.49 | 1222.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 1200.05 | 1213.34 | 1218.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1151.07 | 1170.70 | 1184.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1140.05 | 1150.51 | 1165.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 1145.00 | 1144.28 | 1156.21 | SL hit (close>ema200) qty=0.50 sl=1144.28 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1159.95 | 1154.32 | 1153.73 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 1149.50 | 1153.75 | 1153.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 1124.95 | 1147.54 | 1150.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 09:15:00 | 1046.25 | 1039.63 | 1052.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1046.25 | 1039.63 | 1052.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1046.25 | 1039.63 | 1052.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 1056.00 | 1039.63 | 1052.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1049.40 | 1041.96 | 1050.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 1049.40 | 1041.96 | 1050.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1046.85 | 1042.93 | 1049.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 1048.00 | 1042.93 | 1049.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1049.35 | 1044.22 | 1049.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 1045.90 | 1044.22 | 1049.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1049.45 | 1045.26 | 1049.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 1026.85 | 1045.26 | 1049.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 975.51 | 1003.70 | 1013.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 1003.00 | 997.29 | 1004.68 | SL hit (close>ema200) qty=0.50 sl=997.29 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 992.30 | 961.89 | 961.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 998.30 | 969.17 | 964.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 960.80 | 975.12 | 969.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 960.80 | 975.12 | 969.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 960.80 | 975.12 | 969.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 960.80 | 975.12 | 969.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 963.70 | 972.84 | 969.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 958.50 | 972.84 | 969.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 976.60 | 971.78 | 969.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 979.40 | 973.50 | 970.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 980.40 | 973.50 | 970.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 948.00 | 975.85 | 973.32 | SL hit (close<static) qty=1.00 sl=965.10 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 955.05 | 968.39 | 970.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 939.20 | 956.78 | 963.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 937.45 | 936.52 | 945.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 12:15:00 | 937.45 | 936.52 | 945.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 937.45 | 936.52 | 945.16 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 12:15:00 | 951.45 | 948.51 | 948.13 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 945.00 | 947.56 | 947.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 934.35 | 944.92 | 946.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 15:15:00 | 890.90 | 888.95 | 897.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:15:00 | 875.45 | 888.95 | 897.93 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 14:00:00 | 879.35 | 879.09 | 888.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 15:00:00 | 879.95 | 879.26 | 888.02 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 09:15:00 | 875.60 | 879.77 | 887.45 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 888.50 | 881.52 | 887.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 888.50 | 881.52 | 887.55 | SL hit (close>ema400) qty=1.00 sl=887.55 alert=retest1 |

### Cycle 116 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 890.80 | 881.34 | 881.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 903.15 | 888.43 | 884.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 987.50 | 987.55 | 971.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:30:00 | 987.20 | 987.55 | 971.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 924.70 | 973.78 | 968.18 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 905.10 | 960.04 | 962.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 877.00 | 913.21 | 934.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 890.65 | 887.76 | 908.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 890.65 | 887.76 | 908.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 909.50 | 887.90 | 891.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 909.50 | 887.90 | 891.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 915.25 | 893.37 | 893.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:00:00 | 915.25 | 893.37 | 893.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 12:15:00 | 929.95 | 900.69 | 897.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 941.20 | 926.68 | 913.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 929.60 | 931.24 | 921.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 09:30:00 | 934.05 | 931.24 | 921.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 928.55 | 932.43 | 928.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 928.55 | 932.43 | 928.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 922.00 | 930.34 | 928.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 922.00 | 930.34 | 928.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 920.00 | 928.27 | 927.50 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 918.65 | 926.35 | 926.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 905.95 | 919.64 | 923.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 15:15:00 | 900.95 | 896.57 | 905.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:45:00 | 886.60 | 894.57 | 903.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 897.85 | 892.51 | 898.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 897.85 | 892.51 | 898.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 900.00 | 894.01 | 898.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-13 15:15:00 | 900.00 | 894.01 | 898.70 | SL hit (close>ema400) qty=1.00 sl=898.70 alert=retest1 |

### Cycle 120 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 838.00 | 830.13 | 829.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 13:15:00 | 840.30 | 835.73 | 832.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 11:15:00 | 883.00 | 885.30 | 868.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:30:00 | 881.80 | 885.30 | 868.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 918.85 | 933.91 | 926.28 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 916.70 | 922.25 | 922.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 904.00 | 917.42 | 920.07 | Break + close below crossover candle low |

### Cycle 122 — BUY (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 09:15:00 | 1003.30 | 928.55 | 922.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 1071.25 | 982.36 | 951.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 12:15:00 | 998.40 | 998.50 | 975.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 13:00:00 | 998.40 | 998.50 | 975.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 963.95 | 991.97 | 979.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 963.95 | 991.97 | 979.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 963.00 | 986.17 | 978.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 963.75 | 986.17 | 978.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 952.00 | 971.91 | 973.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 950.85 | 965.59 | 969.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 945.30 | 939.26 | 952.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 945.30 | 939.26 | 952.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 945.30 | 939.26 | 952.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 949.35 | 939.26 | 952.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 945.85 | 940.39 | 950.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:30:00 | 945.75 | 940.39 | 950.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 945.60 | 941.43 | 950.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:30:00 | 948.80 | 941.43 | 950.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 951.20 | 943.39 | 950.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:45:00 | 951.80 | 943.39 | 950.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 958.40 | 946.39 | 950.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:15:00 | 960.40 | 946.39 | 950.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 952.15 | 946.02 | 949.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 950.55 | 946.02 | 949.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 950.00 | 946.81 | 949.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 945.85 | 947.12 | 949.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 14:15:00 | 898.56 | 937.78 | 944.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 13:15:00 | 911.35 | 910.55 | 918.69 | SL hit (close>ema200) qty=0.50 sl=910.55 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 968.10 | 927.12 | 923.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 1013.50 | 944.40 | 932.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 968.00 | 976.70 | 957.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 968.00 | 976.70 | 957.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 959.15 | 969.85 | 960.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 958.15 | 969.85 | 960.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 961.35 | 968.15 | 960.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 936.65 | 968.15 | 960.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 929.65 | 960.45 | 957.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 929.65 | 960.45 | 957.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 934.55 | 955.27 | 955.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 921.95 | 948.61 | 952.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 958.90 | 948.79 | 951.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 958.90 | 948.79 | 951.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 958.90 | 948.79 | 951.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 958.90 | 948.79 | 951.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 970.00 | 953.04 | 953.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 946.00 | 953.04 | 953.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 956.40 | 953.71 | 953.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 956.40 | 953.71 | 953.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 975.30 | 958.73 | 955.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 936.05 | 958.00 | 956.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 936.05 | 958.00 | 956.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 936.05 | 958.00 | 956.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 933.35 | 958.00 | 956.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 929.90 | 952.38 | 954.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 916.20 | 934.33 | 944.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 13:15:00 | 891.70 | 881.28 | 907.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 14:00:00 | 891.70 | 881.28 | 907.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 916.20 | 888.26 | 907.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 916.20 | 888.26 | 907.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 894.00 | 889.41 | 906.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 870.45 | 889.41 | 906.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 953.20 | 919.33 | 916.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 953.20 | 919.33 | 916.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1015.65 | 949.35 | 931.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 10:15:00 | 1062.60 | 1062.92 | 1034.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 11:00:00 | 1062.60 | 1062.92 | 1034.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1045.00 | 1060.56 | 1047.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1045.00 | 1060.56 | 1047.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1041.90 | 1056.83 | 1047.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1031.95 | 1056.83 | 1047.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1049.90 | 1055.44 | 1047.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1043.20 | 1055.44 | 1047.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1049.90 | 1054.33 | 1047.73 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 1016.00 | 1042.80 | 1043.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 1005.20 | 1017.12 | 1024.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1031.10 | 1015.70 | 1021.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1031.10 | 1015.70 | 1021.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1031.10 | 1015.70 | 1021.75 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 1050.60 | 1025.58 | 1025.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 1072.65 | 1039.63 | 1032.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 1068.65 | 1068.82 | 1056.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1068.65 | 1068.82 | 1056.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1068.65 | 1068.82 | 1056.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 1091.90 | 1074.06 | 1060.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 1079.00 | 1076.54 | 1065.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1039.60 | 1067.00 | 1063.33 | SL hit (close<static) qty=1.00 sl=1056.05 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1033.00 | 1060.20 | 1060.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1020.30 | 1045.07 | 1051.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1004.95 | 1002.89 | 1020.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 1014.75 | 1002.89 | 1020.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1011.45 | 1002.93 | 1013.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1013.55 | 1002.93 | 1013.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1019.10 | 1006.16 | 1014.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1019.10 | 1006.16 | 1014.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1015.45 | 1008.02 | 1014.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 1019.65 | 1008.02 | 1014.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1017.60 | 1009.93 | 1014.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 1027.45 | 1009.93 | 1014.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 1019.00 | 1011.75 | 1014.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:15:00 | 1016.15 | 1011.75 | 1014.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 1044.65 | 1018.33 | 1017.61 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 1001.05 | 1016.42 | 1016.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 993.35 | 1011.81 | 1014.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 988.80 | 979.27 | 993.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 988.80 | 979.27 | 993.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 988.80 | 979.27 | 993.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 988.05 | 979.27 | 993.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 974.25 | 978.27 | 991.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 973.95 | 978.27 | 991.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 996.15 | 982.63 | 991.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1000.45 | 982.63 | 991.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1016.75 | 989.45 | 993.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1016.75 | 989.45 | 993.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 984.45 | 992.74 | 994.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 979.10 | 988.34 | 992.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 11:15:00 | 930.14 | 958.26 | 974.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-20 09:15:00 | 881.19 | 911.36 | 942.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 973.40 | 892.28 | 887.50 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 901.10 | 919.19 | 919.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 14:15:00 | 892.85 | 909.34 | 914.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 908.80 | 907.74 | 912.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 904.30 | 907.74 | 912.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 922.50 | 910.69 | 913.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 922.50 | 910.69 | 913.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 917.55 | 912.06 | 913.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 909.50 | 912.06 | 913.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 917.80 | 912.85 | 912.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 917.80 | 912.85 | 912.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 949.65 | 922.77 | 917.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 974.05 | 974.82 | 953.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 974.05 | 974.82 | 953.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 962.00 | 969.87 | 960.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 956.65 | 969.87 | 960.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 939.40 | 963.77 | 958.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 939.40 | 963.77 | 958.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 940.20 | 959.06 | 956.86 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 938.05 | 954.86 | 955.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 934.85 | 950.86 | 953.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 946.75 | 931.12 | 934.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 946.75 | 931.12 | 934.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 946.75 | 931.12 | 934.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 946.75 | 931.12 | 934.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 929.70 | 930.84 | 933.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 924.30 | 927.86 | 931.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:30:00 | 923.80 | 926.41 | 929.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:15:00 | 923.65 | 926.41 | 929.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:15:00 | 924.30 | 926.40 | 929.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 924.00 | 922.47 | 925.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 936.20 | 927.65 | 927.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 936.20 | 927.65 | 927.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 953.25 | 932.77 | 929.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 1000.00 | 1000.57 | 981.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:15:00 | 1025.95 | 1000.57 | 981.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 997.00 | 1003.84 | 993.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 993.95 | 1003.84 | 993.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 988.00 | 1000.68 | 993.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 988.00 | 1000.68 | 993.01 | SL hit (close<ema400) qty=1.00 sl=993.01 alert=retest1 |

### Cycle 139 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 962.55 | 984.22 | 986.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 941.50 | 969.26 | 978.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 938.00 | 933.92 | 947.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 938.00 | 933.92 | 947.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 950.50 | 937.23 | 948.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 953.00 | 937.23 | 948.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 980.55 | 945.90 | 950.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 980.55 | 945.90 | 950.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 980.00 | 952.72 | 953.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 966.35 | 952.72 | 953.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 960.45 | 954.66 | 954.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 960.45 | 954.66 | 954.36 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 945.00 | 954.18 | 954.61 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 964.70 | 956.28 | 955.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 981.50 | 972.39 | 965.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 15:15:00 | 971.95 | 972.30 | 966.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:15:00 | 994.45 | 972.30 | 966.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 987.00 | 990.63 | 981.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 1002.10 | 988.97 | 983.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 941.50 | 981.07 | 980.77 | SL hit (close<static) qty=1.00 sl=976.80 alert=retest2 |

### Cycle 143 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 952.25 | 975.31 | 978.18 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 994.95 | 974.35 | 973.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 1020.05 | 991.81 | 982.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 1007.80 | 1014.31 | 1004.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 1007.80 | 1014.31 | 1004.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 1023.95 | 1016.24 | 1006.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1035.70 | 1017.28 | 1008.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 09:15:00 | 1139.27 | 1105.22 | 1080.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1115.15 | 1136.13 | 1136.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 1100.25 | 1122.53 | 1129.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 10:15:00 | 1112.30 | 1107.23 | 1116.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 11:00:00 | 1112.30 | 1107.23 | 1116.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1129.15 | 1111.61 | 1117.21 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1148.75 | 1125.18 | 1122.56 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 1103.00 | 1119.37 | 1121.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 1087.80 | 1113.06 | 1118.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1178.00 | 1117.31 | 1118.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1178.00 | 1117.31 | 1118.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1178.00 | 1117.31 | 1118.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1178.00 | 1117.31 | 1118.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 1153.10 | 1124.47 | 1121.25 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1119.80 | 1141.90 | 1143.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1107.10 | 1134.94 | 1140.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1126.70 | 1121.20 | 1130.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:45:00 | 1121.80 | 1121.20 | 1130.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1131.20 | 1123.20 | 1130.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1130.00 | 1123.20 | 1130.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1137.40 | 1126.04 | 1131.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1137.40 | 1126.04 | 1131.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1139.40 | 1128.71 | 1132.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1139.40 | 1128.71 | 1132.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1142.90 | 1132.87 | 1133.59 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 1144.70 | 1136.08 | 1134.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 1151.50 | 1139.16 | 1136.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 1130.50 | 1142.44 | 1138.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 1130.50 | 1142.44 | 1138.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1130.50 | 1142.44 | 1138.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1130.50 | 1142.44 | 1138.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1137.00 | 1141.35 | 1138.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1110.50 | 1141.35 | 1138.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1088.70 | 1130.82 | 1134.05 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1148.50 | 1126.91 | 1126.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1152.30 | 1131.98 | 1128.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1180.10 | 1185.77 | 1174.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1216.00 | 1185.77 | 1174.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1193.90 | 1193.75 | 1182.60 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 13:30:00 | 1191.90 | 1192.80 | 1183.18 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1198.50 | 1193.94 | 1184.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 1202.80 | 1192.55 | 1185.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:00:00 | 1206.00 | 1193.72 | 1187.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 1133.20 | 1184.44 | 1185.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 153 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1133.20 | 1184.44 | 1185.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 10:15:00 | 1128.30 | 1173.21 | 1180.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1118.90 | 1117.50 | 1117.40 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1114.50 | 1116.90 | 1117.14 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1152.00 | 1123.92 | 1120.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1195.10 | 1151.19 | 1137.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 14:15:00 | 1172.00 | 1178.08 | 1159.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 1172.00 | 1178.08 | 1159.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1170.10 | 1176.48 | 1160.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1185.50 | 1176.48 | 1160.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1158.80 | 1168.32 | 1168.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1158.80 | 1168.32 | 1168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1157.00 | 1166.06 | 1167.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 1163.80 | 1153.80 | 1157.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1155.10 | 1154.06 | 1157.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:15:00 | 1155.00 | 1154.06 | 1157.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1155.00 | 1154.25 | 1157.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 1157.70 | 1153.82 | 1156.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1163.90 | 1155.96 | 1157.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:45:00 | 1162.00 | 1155.96 | 1157.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1159.40 | 1156.65 | 1157.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 1164.60 | 1156.65 | 1157.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 1169.20 | 1159.16 | 1158.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 1171.80 | 1164.40 | 1161.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1198.20 | 1207.40 | 1192.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 1198.20 | 1207.40 | 1192.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1205.70 | 1206.07 | 1200.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1213.70 | 1199.92 | 1199.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 1212.00 | 1212.18 | 1208.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 1211.30 | 1212.18 | 1208.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1196.40 | 1204.51 | 1205.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1196.40 | 1204.51 | 1205.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1181.30 | 1199.87 | 1203.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1183.40 | 1181.15 | 1189.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1183.40 | 1181.15 | 1189.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1182.10 | 1181.46 | 1188.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1168.00 | 1189.54 | 1189.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 1168.50 | 1182.22 | 1186.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 12:15:00 | 1109.60 | 1122.23 | 1134.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 12:15:00 | 1110.08 | 1122.23 | 1134.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1124.90 | 1119.59 | 1128.68 | SL hit (close>ema200) qty=0.50 sl=1119.59 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1163.50 | 1134.66 | 1131.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1174.80 | 1142.69 | 1135.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1196.30 | 1198.32 | 1179.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 1196.50 | 1198.32 | 1179.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1189.70 | 1192.63 | 1183.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1196.50 | 1192.63 | 1183.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 1316.15 | 1264.77 | 1250.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1280.40 | 1282.54 | 1282.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1273.30 | 1280.34 | 1281.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1296.20 | 1279.59 | 1280.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 14:15:00 | 1307.40 | 1285.15 | 1283.01 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 1298.00 | 1300.94 | 1301.22 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 1309.00 | 1302.55 | 1301.93 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 1287.00 | 1301.58 | 1302.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1268.30 | 1293.36 | 1298.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1272.00 | 1271.86 | 1282.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1266.10 | 1271.86 | 1282.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1276.10 | 1273.11 | 1281.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1277.40 | 1273.11 | 1281.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1291.10 | 1276.34 | 1281.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 1291.10 | 1276.34 | 1281.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1290.80 | 1279.23 | 1282.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 1290.80 | 1279.23 | 1282.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1274.80 | 1276.65 | 1279.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1281.70 | 1276.65 | 1279.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1281.00 | 1277.52 | 1279.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 1279.80 | 1277.52 | 1279.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1281.00 | 1278.21 | 1280.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1305.70 | 1278.21 | 1280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1342.10 | 1290.99 | 1285.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 1369.00 | 1314.10 | 1297.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 1336.90 | 1354.78 | 1339.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1328.90 | 1349.60 | 1338.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1328.90 | 1349.60 | 1338.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 1310.40 | 1329.76 | 1331.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 1301.50 | 1324.11 | 1329.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 1256.50 | 1246.06 | 1259.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1262.10 | 1249.27 | 1259.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 1265.90 | 1249.27 | 1259.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1263.90 | 1252.19 | 1260.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1264.30 | 1252.19 | 1260.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1260.60 | 1255.59 | 1260.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 1255.20 | 1255.59 | 1260.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1251.40 | 1254.75 | 1259.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1267.30 | 1255.25 | 1258.77 | SL hit (close>static) qty=1.00 sl=1265.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1245.00 | 1234.38 | 1233.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1256.10 | 1243.85 | 1238.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1297.90 | 1298.81 | 1282.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 1297.90 | 1298.81 | 1282.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1308.40 | 1300.73 | 1285.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1316.70 | 1305.20 | 1288.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 1318.80 | 1322.76 | 1311.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1365.50 | 1381.47 | 1382.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1365.50 | 1381.47 | 1382.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1362.00 | 1377.58 | 1380.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 1354.90 | 1353.81 | 1363.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:00:00 | 1354.90 | 1353.81 | 1363.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1355.20 | 1343.13 | 1353.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1355.70 | 1343.13 | 1353.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1358.40 | 1346.18 | 1354.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 1358.00 | 1346.18 | 1354.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1384.00 | 1353.75 | 1356.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:00:00 | 1384.00 | 1353.75 | 1356.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 1391.60 | 1361.32 | 1360.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1410.90 | 1382.81 | 1371.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 1378.00 | 1384.57 | 1385.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 1368.60 | 1381.37 | 1383.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1311.80 | 1307.32 | 1318.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 15:15:00 | 1295.20 | 1304.85 | 1313.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:30:00 | 1292.90 | 1298.55 | 1307.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 14:45:00 | 1295.30 | 1295.74 | 1303.64 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1313.10 | 1294.47 | 1300.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 1313.10 | 1294.47 | 1300.11 | SL hit (close>ema400) qty=1.00 sl=1300.11 alert=retest1 |

### Cycle 172 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1322.00 | 1305.51 | 1304.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 1335.80 | 1315.52 | 1309.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1356.70 | 1365.18 | 1365.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1343.70 | 1358.61 | 1362.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 1358.50 | 1356.46 | 1360.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1356.50 | 1356.46 | 1360.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1366.00 | 1358.37 | 1360.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1368.00 | 1358.37 | 1360.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 1381.40 | 1362.98 | 1362.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1416.00 | 1388.03 | 1376.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 1403.60 | 1407.49 | 1394.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1389.00 | 1405.14 | 1397.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 1392.50 | 1405.14 | 1397.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1375.90 | 1399.29 | 1395.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1375.90 | 1399.29 | 1395.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 09:15:00 | 1377.10 | 1390.96 | 1392.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1353.30 | 1383.43 | 1388.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1361.60 | 1355.27 | 1368.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1354.60 | 1354.41 | 1363.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 1364.80 | 1354.41 | 1363.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1365.90 | 1351.64 | 1358.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 1365.90 | 1351.64 | 1358.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1380.60 | 1357.43 | 1360.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 1377.80 | 1357.43 | 1360.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1401.90 | 1366.32 | 1363.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 1405.50 | 1387.73 | 1376.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 1398.00 | 1398.18 | 1385.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 1393.50 | 1398.18 | 1385.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1392.00 | 1396.78 | 1391.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 1390.00 | 1396.78 | 1391.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1393.00 | 1396.02 | 1392.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1393.90 | 1396.02 | 1392.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1400.00 | 1396.82 | 1392.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1387.00 | 1396.82 | 1392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1384.70 | 1394.40 | 1392.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1384.70 | 1394.40 | 1392.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1392.00 | 1393.92 | 1392.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 1395.00 | 1393.92 | 1392.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 1373.10 | 1388.83 | 1390.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1373.10 | 1388.83 | 1390.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1369.90 | 1385.05 | 1388.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1377.40 | 1377.40 | 1382.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 1377.40 | 1377.40 | 1382.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1378.50 | 1376.28 | 1380.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1378.50 | 1376.28 | 1380.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1380.30 | 1377.08 | 1380.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1381.00 | 1377.08 | 1380.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1376.60 | 1376.99 | 1379.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 1375.00 | 1376.81 | 1379.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1375.40 | 1376.45 | 1379.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1385.60 | 1380.15 | 1380.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 1385.60 | 1380.15 | 1380.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 1403.30 | 1384.78 | 1382.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 1398.90 | 1403.51 | 1395.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1392.30 | 1401.27 | 1394.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1392.30 | 1401.27 | 1394.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1392.60 | 1401.27 | 1394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1382.90 | 1397.59 | 1393.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 1384.60 | 1397.59 | 1393.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1376.10 | 1389.13 | 1390.36 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1392.90 | 1386.08 | 1385.70 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1354.00 | 1383.07 | 1385.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 1346.20 | 1375.70 | 1381.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1302.20 | 1297.13 | 1315.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 10:15:00 | 1309.00 | 1297.13 | 1315.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1319.50 | 1301.60 | 1315.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 1319.50 | 1301.60 | 1315.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1331.40 | 1307.56 | 1317.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 1331.40 | 1307.56 | 1317.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 1329.60 | 1323.10 | 1322.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 1387.10 | 1335.90 | 1328.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 15:15:00 | 1469.90 | 1471.87 | 1446.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 1446.00 | 1471.87 | 1446.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1426.70 | 1462.84 | 1444.32 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 1387.50 | 1427.50 | 1430.97 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 1481.90 | 1434.32 | 1430.88 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1410.30 | 1437.81 | 1439.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1405.60 | 1422.88 | 1431.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1417.90 | 1414.69 | 1422.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1354.10 | 1326.12 | 1349.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1354.10 | 1326.12 | 1349.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1360.40 | 1332.97 | 1350.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 1344.70 | 1334.94 | 1349.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1377.00 | 1343.35 | 1351.92 | SL hit (close>static) qty=1.00 sl=1365.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1391.50 | 1359.16 | 1358.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1401.60 | 1367.65 | 1361.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 11:15:00 | 1393.00 | 1398.53 | 1381.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 1393.00 | 1398.53 | 1381.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1388.00 | 1396.42 | 1382.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 1385.00 | 1396.42 | 1382.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1384.30 | 1394.00 | 1382.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1377.70 | 1394.00 | 1382.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1369.80 | 1389.16 | 1381.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 1369.80 | 1389.16 | 1381.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1370.00 | 1385.33 | 1380.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1391.00 | 1385.33 | 1380.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 1367.60 | 1378.22 | 1379.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 14:15:00 | 1367.60 | 1378.22 | 1379.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 1362.80 | 1373.82 | 1376.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1361.10 | 1359.24 | 1366.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 1361.10 | 1359.24 | 1366.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1360.50 | 1359.96 | 1365.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:45:00 | 1366.70 | 1359.96 | 1365.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1371.80 | 1362.39 | 1365.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1371.80 | 1362.39 | 1365.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1370.90 | 1364.09 | 1365.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1370.50 | 1364.09 | 1365.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1368.40 | 1366.39 | 1366.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1371.90 | 1366.39 | 1366.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1309.20 | 1318.81 | 1329.87 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 1346.40 | 1331.64 | 1330.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 1353.50 | 1336.01 | 1332.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 15:15:00 | 1380.10 | 1388.94 | 1377.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 1385.00 | 1388.94 | 1377.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1369.90 | 1385.13 | 1377.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 1368.90 | 1385.13 | 1377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1357.00 | 1379.51 | 1375.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1359.90 | 1379.51 | 1375.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 1358.70 | 1371.87 | 1372.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 1350.60 | 1364.22 | 1368.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1368.90 | 1363.82 | 1367.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1369.90 | 1365.04 | 1367.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1369.90 | 1365.04 | 1367.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1372.90 | 1366.61 | 1367.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1353.00 | 1366.61 | 1367.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 1285.35 | 1306.75 | 1317.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1267.20 | 1261.93 | 1279.31 | SL hit (close>ema200) qty=0.50 sl=1261.93 alert=retest2 |

### Cycle 190 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1291.20 | 1274.28 | 1274.15 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 1257.00 | 1276.91 | 1277.61 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 1288.40 | 1279.00 | 1277.97 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1266.70 | 1278.44 | 1278.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1260.00 | 1274.75 | 1276.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1281.10 | 1267.20 | 1270.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1278.20 | 1269.40 | 1271.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1262.00 | 1269.40 | 1271.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 1265.00 | 1267.24 | 1269.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 1265.90 | 1267.24 | 1269.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 1279.00 | 1270.95 | 1270.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1279.00 | 1270.95 | 1270.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1285.10 | 1277.33 | 1273.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1292.40 | 1296.41 | 1288.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:45:00 | 1293.40 | 1296.41 | 1288.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1302.00 | 1298.99 | 1291.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1301.80 | 1298.99 | 1291.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1313.10 | 1302.14 | 1294.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1314.10 | 1302.14 | 1294.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 1316.90 | 1305.87 | 1300.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 1288.10 | 1300.30 | 1299.09 | SL hit (close<static) qty=1.00 sl=1294.10 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1261.50 | 1292.54 | 1295.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1243.20 | 1267.44 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1250.00 | 1249.83 | 1264.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 1250.00 | 1249.83 | 1264.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1267.00 | 1253.44 | 1262.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1267.00 | 1253.44 | 1262.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1269.50 | 1256.65 | 1262.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 1269.90 | 1256.65 | 1262.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1270.70 | 1266.27 | 1266.05 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 1256.30 | 1264.28 | 1265.16 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1293.40 | 1269.72 | 1267.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1298.90 | 1279.47 | 1272.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1304.80 | 1311.39 | 1298.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1315.80 | 1312.28 | 1299.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 1321.30 | 1314.28 | 1301.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 1322.30 | 1314.28 | 1301.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:45:00 | 1320.80 | 1322.17 | 1310.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:00:00 | 1319.20 | 1321.58 | 1311.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1303.50 | 1317.96 | 1310.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 1304.10 | 1317.96 | 1310.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1322.20 | 1318.81 | 1311.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1307.50 | 1318.81 | 1311.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1325.00 | 1333.95 | 1327.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1320.00 | 1333.95 | 1327.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1334.00 | 1333.96 | 1327.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 1337.30 | 1332.12 | 1328.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1301.40 | 1323.72 | 1325.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1301.40 | 1323.72 | 1325.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 1297.60 | 1315.13 | 1321.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:45:00 | 1304.20 | 1300.43 | 1309.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 1306.30 | 1301.60 | 1308.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 1317.30 | 1301.60 | 1308.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1283.40 | 1297.96 | 1306.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 1308.60 | 1297.96 | 1306.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1298.50 | 1298.17 | 1304.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 1312.40 | 1298.17 | 1304.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1296.00 | 1297.92 | 1303.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 1299.60 | 1297.92 | 1303.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1290.00 | 1295.29 | 1301.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 1300.00 | 1295.29 | 1301.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1299.20 | 1293.72 | 1298.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1299.20 | 1293.72 | 1298.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1300.00 | 1294.98 | 1298.87 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 1308.20 | 1301.56 | 1300.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 12:15:00 | 1312.30 | 1304.59 | 1302.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1298.00 | 1303.27 | 1301.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1283.20 | 1299.26 | 1300.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 1280.00 | 1295.41 | 1298.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 1356.30 | 1274.59 | 1280.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 10:15:00 | 1370.60 | 1293.79 | 1288.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 12:15:00 | 1384.00 | 1324.44 | 1303.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1409.90 | 1432.74 | 1396.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1409.90 | 1432.74 | 1396.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1383.00 | 1422.79 | 1394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1383.00 | 1422.79 | 1394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1403.90 | 1419.01 | 1395.65 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 1348.20 | 1382.34 | 1385.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1328.10 | 1366.04 | 1377.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1340.60 | 1284.86 | 1301.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1331.40 | 1294.17 | 1304.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1321.00 | 1294.17 | 1304.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1335.00 | 1311.46 | 1310.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1335.00 | 1311.46 | 1310.63 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 1300.70 | 1310.21 | 1310.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1282.10 | 1301.83 | 1306.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1318.00 | 1289.65 | 1286.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1340.20 | 1311.16 | 1298.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1314.50 | 1328.82 | 1318.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1299.20 | 1322.89 | 1316.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 1298.00 | 1322.89 | 1316.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 1288.70 | 1310.60 | 1311.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1285.10 | 1302.99 | 1307.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 1297.00 | 1282.27 | 1288.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1293.90 | 1284.59 | 1288.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 1281.10 | 1284.84 | 1288.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1277.00 | 1283.01 | 1286.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1276.20 | 1261.93 | 1262.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1282.10 | 1265.96 | 1264.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 1282.10 | 1265.96 | 1264.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1285.10 | 1269.79 | 1266.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1280.00 | 1280.15 | 1275.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 1270.70 | 1280.15 | 1275.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1277.10 | 1279.54 | 1276.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1286.60 | 1281.23 | 1277.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:30:00 | 1281.40 | 1282.54 | 1279.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1284.90 | 1293.18 | 1289.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1281.10 | 1290.76 | 1288.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1289.50 | 1290.51 | 1288.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1278.50 | 1286.05 | 1286.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1278.50 | 1286.05 | 1286.90 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1302.50 | 1289.46 | 1287.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1307.50 | 1293.07 | 1289.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1282.10 | 1318.14 | 1309.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1265.10 | 1307.53 | 1305.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 1265.10 | 1307.53 | 1305.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1251.50 | 1296.32 | 1300.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1231.50 | 1268.08 | 1283.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 1256.90 | 1229.44 | 1249.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1236.00 | 1230.75 | 1248.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:45:00 | 1228.00 | 1230.98 | 1247.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:15:00 | 1228.10 | 1230.98 | 1247.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:30:00 | 1229.90 | 1230.91 | 1244.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 1229.50 | 1230.01 | 1240.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1241.70 | 1232.35 | 1240.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 1238.60 | 1232.35 | 1240.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1235.70 | 1233.02 | 1240.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:00:00 | 1220.00 | 1230.41 | 1238.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1166.60 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1166.69 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1168.40 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1168.02 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1159.00 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1193.60 | 1169.76 | 1185.82 | SL hit (close>ema200) qty=0.50 sl=1169.76 alert=retest2 |

### Cycle 212 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1214.60 | 1194.97 | 1194.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1222.20 | 1206.76 | 1200.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1206.10 | 1206.63 | 1200.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 1206.10 | 1206.63 | 1200.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1200.70 | 1206.73 | 1201.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1200.70 | 1206.73 | 1201.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1210.60 | 1207.51 | 1202.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 14:15:00 | 1212.10 | 1207.51 | 1202.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1195.30 | 1205.07 | 1202.02 | SL hit (close<static) qty=1.00 sl=1199.70 alert=retest2 |

### Cycle 213 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1166.00 | 1195.48 | 1198.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1147.00 | 1166.43 | 1179.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 10:15:00 | 1165.00 | 1163.54 | 1172.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:00:00 | 1165.00 | 1163.54 | 1172.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1168.50 | 1164.70 | 1171.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1168.50 | 1164.70 | 1171.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1169.10 | 1165.58 | 1171.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:15:00 | 1169.60 | 1165.58 | 1171.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1176.30 | 1167.72 | 1171.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1177.50 | 1167.72 | 1171.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1172.00 | 1168.58 | 1171.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 1166.70 | 1168.58 | 1171.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1204.00 | 1174.19 | 1172.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1204.00 | 1174.19 | 1172.00 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1161.10 | 1173.24 | 1173.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1150.40 | 1162.25 | 1167.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:30:00 | 1149.60 | 1153.29 | 1160.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1147.00 | 1143.18 | 1145.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1169.20 | 1148.99 | 1147.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1169.20 | 1148.99 | 1147.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1179.60 | 1155.11 | 1150.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1158.00 | 1165.70 | 1158.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1161.30 | 1164.82 | 1158.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:30:00 | 1164.10 | 1165.64 | 1159.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 1165.00 | 1165.73 | 1160.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 1166.70 | 1166.14 | 1161.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:30:00 | 1164.30 | 1165.05 | 1162.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1168.60 | 1165.76 | 1162.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:00:00 | 1174.60 | 1167.53 | 1163.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 1160.80 | 1170.23 | 1166.32 | SL hit (close<static) qty=1.00 sl=1162.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1228.60 | 1229.90 | 1229.91 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 1232.20 | 1230.36 | 1230.12 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 1225.30 | 1229.42 | 1229.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 1220.00 | 1227.54 | 1228.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1234.90 | 1230.65 | 1230.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1238.10 | 1232.14 | 1230.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 14:15:00 | 1235.00 | 1235.36 | 1232.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:45:00 | 1235.10 | 1235.36 | 1232.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1234.00 | 1235.09 | 1232.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1241.00 | 1235.09 | 1232.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 13:15:00 | 1224.60 | 1237.04 | 1235.66 | SL hit (close<static) qty=1.00 sl=1230.30 alert=retest2 |

### Cycle 221 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1247.90 | 1260.91 | 1261.48 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1269.90 | 1261.06 | 1260.79 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1246.30 | 1257.82 | 1259.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1229.70 | 1245.63 | 1252.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1236.00 | 1232.84 | 1242.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 1236.00 | 1232.84 | 1242.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1249.00 | 1236.74 | 1242.55 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1258.50 | 1245.31 | 1244.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1266.00 | 1249.45 | 1246.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1463.10 | 1466.36 | 1429.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 1454.80 | 1466.36 | 1429.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 09:15:00 | 1054.30 | 2023-05-16 11:15:00 | 1101.82 | TARGET_HIT | 1.00 | 4.51% |
| SELL | retest2 | 2023-05-30 09:45:00 | 1200.60 | 2023-05-31 13:15:00 | 1241.10 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2023-05-30 10:15:00 | 1200.65 | 2023-05-31 13:15:00 | 1241.10 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2023-05-30 11:00:00 | 1201.45 | 2023-05-31 13:15:00 | 1241.10 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2023-05-30 12:15:00 | 1202.00 | 2023-05-31 13:15:00 | 1241.10 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2023-06-06 12:45:00 | 1241.30 | 2023-06-08 13:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-06-06 13:45:00 | 1240.95 | 2023-06-08 13:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-06-06 15:15:00 | 1240.50 | 2023-06-08 13:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-27 10:00:00 | 1371.35 | 2023-06-28 15:15:00 | 1325.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2023-07-04 14:15:00 | 1259.70 | 2023-07-06 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-07-06 09:30:00 | 1255.05 | 2023-07-06 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-07-18 14:30:00 | 1300.65 | 2023-07-21 11:15:00 | 1276.90 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2023-07-26 09:15:00 | 1395.90 | 2023-07-27 13:15:00 | 1377.10 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-08-07 09:15:00 | 1473.55 | 2023-08-08 09:15:00 | 1449.75 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-08-07 15:00:00 | 1464.25 | 2023-08-08 09:15:00 | 1449.75 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-08-10 09:15:00 | 1426.90 | 2023-08-17 09:15:00 | 1429.10 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2023-08-29 09:15:00 | 1415.60 | 2023-08-29 11:15:00 | 1399.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-09-04 09:30:00 | 1440.90 | 2023-09-07 14:15:00 | 1440.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-09-15 12:15:00 | 1406.35 | 2023-09-18 12:15:00 | 1336.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 13:00:00 | 1409.75 | 2023-09-18 12:15:00 | 1339.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 13:30:00 | 1409.15 | 2023-09-18 12:15:00 | 1338.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 12:15:00 | 1406.35 | 2023-09-20 14:15:00 | 1344.60 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2023-09-15 13:00:00 | 1409.75 | 2023-09-20 14:15:00 | 1344.60 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-09-15 13:30:00 | 1409.15 | 2023-09-20 14:15:00 | 1344.60 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2023-09-29 15:15:00 | 1290.00 | 2023-10-03 11:15:00 | 1328.30 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2023-10-04 12:30:00 | 1324.00 | 2023-10-05 09:15:00 | 1307.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1386.60 | 2023-10-18 09:15:00 | 1380.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-10-18 11:30:00 | 1388.50 | 2023-10-18 14:15:00 | 1380.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-10-18 12:00:00 | 1387.95 | 2023-10-18 14:15:00 | 1380.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-11-03 09:15:00 | 1680.00 | 2023-11-08 13:15:00 | 1655.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-11-12 18:15:00 | 1715.50 | 2023-11-15 14:15:00 | 1680.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-11-13 09:15:00 | 1724.20 | 2023-11-15 14:15:00 | 1680.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2023-11-13 14:15:00 | 1715.85 | 2023-11-15 14:15:00 | 1680.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest1 | 2023-11-17 09:15:00 | 1670.00 | 2023-11-17 14:15:00 | 1586.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-11-17 09:15:00 | 1670.00 | 2023-11-20 11:15:00 | 1630.30 | STOP_HIT | 0.50 | 2.38% |
| BUY | retest2 | 2023-11-28 09:30:00 | 1699.85 | 2023-11-29 12:15:00 | 1671.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-11-29 09:15:00 | 1689.30 | 2023-11-29 12:15:00 | 1671.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-11-29 10:00:00 | 1699.45 | 2023-11-29 12:15:00 | 1671.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-11-30 14:30:00 | 1694.45 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2023-11-30 15:00:00 | 1699.75 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2023-12-01 10:00:00 | 1698.95 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2023-12-01 15:15:00 | 1696.00 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1740.10 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-12-04 15:15:00 | 1714.70 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2023-12-07 09:30:00 | 1713.75 | 2023-12-07 14:15:00 | 1722.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest1 | 2023-12-19 12:15:00 | 1686.00 | 2023-12-20 12:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest1 | 2023-12-19 13:45:00 | 1688.40 | 2023-12-20 12:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2023-12-19 14:15:00 | 1689.45 | 2023-12-20 12:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-12-20 14:45:00 | 1663.95 | 2023-12-28 09:15:00 | 1580.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 14:45:00 | 1663.95 | 2023-12-29 12:15:00 | 1590.35 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2023-12-22 12:45:00 | 1639.70 | 2024-01-04 09:15:00 | 1588.50 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest1 | 2024-01-09 09:15:00 | 1774.10 | 2024-01-11 11:15:00 | 1735.40 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest1 | 2024-01-09 10:00:00 | 1765.10 | 2024-01-11 11:15:00 | 1735.40 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2024-01-10 09:15:00 | 1754.25 | 2024-01-11 11:15:00 | 1735.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2024-01-10 12:00:00 | 1766.40 | 2024-01-11 11:15:00 | 1735.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-01-17 09:45:00 | 1672.50 | 2024-01-20 09:15:00 | 1683.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-01-17 11:15:00 | 1659.70 | 2024-01-20 09:15:00 | 1683.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-01-19 12:00:00 | 1670.25 | 2024-01-20 09:15:00 | 1683.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-01-30 14:30:00 | 1564.00 | 2024-01-31 09:15:00 | 1608.30 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-02-02 11:15:00 | 1588.20 | 2024-02-08 10:15:00 | 1581.25 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-02-02 12:30:00 | 1585.65 | 2024-02-08 10:15:00 | 1581.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-02-02 13:15:00 | 1591.00 | 2024-02-08 10:15:00 | 1581.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-02-06 09:15:00 | 1599.85 | 2024-02-08 10:15:00 | 1581.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-02-26 14:30:00 | 1469.35 | 2024-02-27 09:15:00 | 1491.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-03-05 15:00:00 | 1434.35 | 2024-03-07 09:15:00 | 1362.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 15:00:00 | 1434.35 | 2024-03-07 12:15:00 | 1401.40 | STOP_HIT | 0.50 | 2.30% |
| BUY | retest2 | 2024-03-20 11:45:00 | 1408.60 | 2024-03-20 14:15:00 | 1380.30 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-03-20 12:45:00 | 1405.95 | 2024-03-20 14:15:00 | 1380.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-05-08 13:45:00 | 1429.60 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-05-08 14:45:00 | 1431.25 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2024-05-09 14:30:00 | 1430.00 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-05-09 15:00:00 | 1397.20 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-05-13 09:15:00 | 1402.05 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-05-13 15:00:00 | 1395.55 | 2024-05-15 11:15:00 | 1416.95 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1285.10 | 2024-06-04 12:15:00 | 1220.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1261.00 | 2024-06-04 12:15:00 | 1197.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1285.10 | 2024-06-05 09:15:00 | 1303.35 | STOP_HIT | 0.50 | -1.42% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1261.00 | 2024-06-05 09:15:00 | 1303.35 | STOP_HIT | 0.50 | -3.36% |
| SELL | retest2 | 2024-06-21 10:15:00 | 1465.00 | 2024-06-25 10:15:00 | 1391.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 13:00:00 | 1468.25 | 2024-06-25 10:15:00 | 1394.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1466.40 | 2024-06-25 10:15:00 | 1393.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 10:15:00 | 1465.00 | 2024-06-27 09:15:00 | 1371.10 | STOP_HIT | 0.50 | 6.41% |
| SELL | retest2 | 2024-06-21 13:00:00 | 1468.25 | 2024-06-27 09:15:00 | 1371.10 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1466.40 | 2024-06-27 09:15:00 | 1371.10 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2024-07-04 10:30:00 | 1329.00 | 2024-07-10 11:15:00 | 1262.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 12:15:00 | 1328.90 | 2024-07-10 11:15:00 | 1262.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 13:30:00 | 1326.10 | 2024-07-10 11:15:00 | 1259.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 14:30:00 | 1329.30 | 2024-07-10 11:15:00 | 1262.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 09:15:00 | 1326.65 | 2024-07-10 11:15:00 | 1260.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 10:30:00 | 1329.00 | 2024-07-11 09:15:00 | 1286.75 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2024-07-04 12:15:00 | 1328.90 | 2024-07-11 09:15:00 | 1286.75 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2024-07-04 13:30:00 | 1326.10 | 2024-07-11 09:15:00 | 1286.75 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2024-07-04 14:30:00 | 1329.30 | 2024-07-11 09:15:00 | 1286.75 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-07-08 09:15:00 | 1326.65 | 2024-07-11 09:15:00 | 1286.75 | STOP_HIT | 0.50 | 3.01% |
| BUY | retest2 | 2024-07-25 10:45:00 | 1298.55 | 2024-07-25 13:15:00 | 1275.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-07-30 14:45:00 | 1324.85 | 2024-08-02 11:15:00 | 1314.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-31 10:00:00 | 1321.85 | 2024-08-02 11:15:00 | 1314.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-07-31 12:00:00 | 1319.95 | 2024-08-02 11:15:00 | 1314.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-07-31 13:45:00 | 1323.90 | 2024-08-02 11:15:00 | 1314.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-08-09 11:15:00 | 1253.80 | 2024-08-14 09:15:00 | 1191.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 15:00:00 | 1253.65 | 2024-08-14 09:15:00 | 1190.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:15:00 | 1253.80 | 2024-08-16 13:15:00 | 1199.45 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2024-08-09 15:00:00 | 1253.65 | 2024-08-16 13:15:00 | 1199.45 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest2 | 2024-08-22 09:15:00 | 1248.70 | 2024-08-23 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-08-22 12:45:00 | 1226.50 | 2024-08-23 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1210.00 | 2024-09-04 09:15:00 | 1201.45 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-03 10:30:00 | 1209.85 | 2024-09-04 09:15:00 | 1201.45 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-03 11:15:00 | 1210.00 | 2024-09-04 09:15:00 | 1201.45 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-03 12:30:00 | 1210.00 | 2024-09-04 09:15:00 | 1201.45 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-09-06 11:30:00 | 1177.50 | 2024-09-06 14:15:00 | 1212.10 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-09-17 13:00:00 | 1241.60 | 2024-09-18 14:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-17 14:30:00 | 1243.95 | 2024-09-18 14:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1211.65 | 2024-10-04 09:15:00 | 1151.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 15:00:00 | 1200.05 | 2024-10-07 09:15:00 | 1140.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1211.65 | 2024-10-07 14:15:00 | 1145.00 | STOP_HIT | 0.50 | 5.50% |
| SELL | retest2 | 2024-09-27 15:00:00 | 1200.05 | 2024-10-07 14:15:00 | 1145.00 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2024-10-18 09:15:00 | 1026.85 | 2024-10-23 09:15:00 | 975.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 09:15:00 | 1026.85 | 2024-10-24 09:15:00 | 1003.00 | STOP_HIT | 0.50 | 2.32% |
| BUY | retest2 | 2024-10-31 14:30:00 | 979.40 | 2024-11-04 09:15:00 | 948.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-10-31 15:00:00 | 980.40 | 2024-11-04 09:15:00 | 948.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest1 | 2024-11-18 09:15:00 | 875.45 | 2024-11-19 09:15:00 | 888.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2024-11-18 14:00:00 | 879.35 | 2024-11-19 09:15:00 | 888.50 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2024-11-18 15:00:00 | 879.95 | 2024-11-19 09:15:00 | 888.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest1 | 2024-11-19 09:15:00 | 875.60 | 2024-11-19 09:15:00 | 888.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-11-21 09:30:00 | 881.65 | 2024-11-22 13:15:00 | 890.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-11-21 10:15:00 | 882.00 | 2024-11-22 13:15:00 | 890.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest1 | 2024-12-13 09:45:00 | 886.60 | 2024-12-13 15:15:00 | 900.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-12-16 14:15:00 | 890.75 | 2024-12-19 09:15:00 | 846.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 15:15:00 | 890.90 | 2024-12-19 09:15:00 | 846.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 890.75 | 2024-12-23 10:15:00 | 826.50 | STOP_HIT | 0.50 | 7.21% |
| SELL | retest2 | 2024-12-16 15:15:00 | 890.90 | 2024-12-23 10:15:00 | 826.50 | STOP_HIT | 0.50 | 7.23% |
| SELL | retest2 | 2025-01-15 13:30:00 | 945.85 | 2025-01-15 14:15:00 | 898.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 13:30:00 | 945.85 | 2025-01-17 13:15:00 | 911.35 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-01-23 09:15:00 | 946.00 | 2025-01-23 09:15:00 | 956.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-28 09:15:00 | 870.45 | 2025-01-28 13:15:00 | 953.20 | STOP_HIT | 1.00 | -9.51% |
| BUY | retest2 | 2025-02-07 10:45:00 | 1091.90 | 2025-02-10 09:15:00 | 1039.60 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2025-02-07 15:00:00 | 1079.00 | 2025-02-10 09:15:00 | 1039.60 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-02-18 12:45:00 | 979.10 | 2025-02-19 11:15:00 | 930.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 12:45:00 | 979.10 | 2025-02-20 09:15:00 | 881.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 12:15:00 | 909.50 | 2025-03-05 11:15:00 | 917.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-03-13 15:00:00 | 924.30 | 2025-03-18 15:15:00 | 936.20 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-03-17 09:30:00 | 923.80 | 2025-03-18 15:15:00 | 936.20 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-03-17 10:15:00 | 923.65 | 2025-03-18 15:15:00 | 936.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-17 11:15:00 | 924.30 | 2025-03-18 15:15:00 | 936.20 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest1 | 2025-03-24 09:15:00 | 1025.95 | 2025-03-25 09:15:00 | 988.00 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-03-28 09:15:00 | 966.35 | 2025-03-28 10:15:00 | 960.45 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-04-04 15:00:00 | 1002.10 | 2025-04-07 09:15:00 | 941.50 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest2 | 2025-04-15 09:15:00 | 1035.70 | 2025-04-21 09:15:00 | 1139.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-05-15 09:15:00 | 1216.00 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -6.81% |
| BUY | retest1 | 2025-05-15 13:00:00 | 1193.90 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest1 | 2025-05-15 13:30:00 | 1191.90 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2025-05-16 09:30:00 | 1202.80 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -5.79% |
| BUY | retest2 | 2025-05-16 13:00:00 | 1206.00 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -6.04% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1185.50 | 2025-05-30 09:15:00 | 1158.80 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1213.70 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-12 09:45:00 | 1212.00 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-12 10:15:00 | 1211.30 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1168.00 | 2025-06-20 12:15:00 | 1109.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:00:00 | 1168.50 | 2025-06-20 12:15:00 | 1110.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1168.00 | 2025-06-23 09:15:00 | 1124.90 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-06-17 11:00:00 | 1168.50 | 2025-06-23 09:15:00 | 1124.90 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1196.50 | 2025-07-04 09:15:00 | 1316.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1255.20 | 2025-08-01 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1251.40 | 2025-08-01 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-01 11:30:00 | 1254.60 | 2025-08-11 09:15:00 | 1245.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-08-14 09:45:00 | 1316.70 | 2025-08-26 11:15:00 | 1365.50 | STOP_HIT | 1.00 | 3.71% |
| BUY | retest2 | 2025-08-18 11:30:00 | 1318.80 | 2025-08-26 11:15:00 | 1365.50 | STOP_HIT | 1.00 | 3.54% |
| SELL | retest1 | 2025-09-11 15:15:00 | 1295.20 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest1 | 2025-09-12 11:30:00 | 1292.90 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest1 | 2025-09-12 14:45:00 | 1295.30 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-07 15:15:00 | 1395.00 | 2025-10-08 10:15:00 | 1373.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-10-10 10:45:00 | 1375.00 | 2025-10-10 15:15:00 | 1385.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1375.40 | 2025-10-10 15:15:00 | 1385.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1344.70 | 2025-11-10 10:15:00 | 1377.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1391.00 | 2025-11-12 14:15:00 | 1367.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1353.00 | 2025-12-05 15:15:00 | 1285.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1353.00 | 2025-12-09 11:15:00 | 1267.20 | STOP_HIT | 0.50 | 6.34% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1262.00 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-18 11:30:00 | 1265.00 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-18 12:15:00 | 1265.90 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1314.10 | 2025-12-26 13:15:00 | 1288.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-26 09:45:00 | 1316.90 | 2025-12-26 13:15:00 | 1288.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-01-06 11:30:00 | 1321.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-06 12:00:00 | 1322.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-07 09:45:00 | 1320.80 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-07 11:00:00 | 1319.20 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-09 13:30:00 | 1337.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1321.00 | 2026-01-30 13:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-10 15:15:00 | 1281.10 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-02-11 09:30:00 | 1277.00 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-18 09:30:00 | 1276.20 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-20 10:30:00 | 1286.60 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-20 14:30:00 | 1281.40 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1284.90 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-02-24 11:00:00 | 1281.10 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-03-04 11:45:00 | 1228.00 | 2026-03-09 09:15:00 | 1166.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1228.10 | 2026-03-09 09:15:00 | 1166.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1229.90 | 2026-03-09 09:15:00 | 1168.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 09:30:00 | 1229.50 | 2026-03-09 09:15:00 | 1168.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:00:00 | 1220.00 | 2026-03-09 09:15:00 | 1159.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 11:45:00 | 1228.00 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1228.10 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1229.90 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-03-05 09:30:00 | 1229.50 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-03-05 13:00:00 | 1220.00 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest2 | 2026-03-11 14:15:00 | 1212.10 | 2026-03-11 14:15:00 | 1195.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-17 09:15:00 | 1166.70 | 2026-03-18 09:15:00 | 1204.00 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-03-23 09:30:00 | 1149.60 | 2026-03-25 09:15:00 | 1169.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1147.00 | 2026-03-25 09:15:00 | 1169.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-03-27 10:30:00 | 1164.10 | 2026-03-30 14:15:00 | 1160.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-03-27 13:00:00 | 1165.00 | 2026-04-07 09:15:00 | 1280.51 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2026-03-27 14:45:00 | 1166.70 | 2026-04-07 09:15:00 | 1281.50 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2026-03-30 09:30:00 | 1164.30 | 2026-04-07 09:15:00 | 1283.37 | TARGET_HIT | 1.00 | 10.23% |
| BUY | retest2 | 2026-03-30 12:00:00 | 1174.60 | 2026-04-07 09:15:00 | 1280.73 | TARGET_HIT | 1.00 | 9.04% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1187.80 | 2026-04-07 09:15:00 | 1292.28 | TARGET_HIT | 1.00 | 8.80% |
| BUY | retest2 | 2026-04-01 09:45:00 | 1174.80 | 2026-04-07 09:15:00 | 1295.91 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2026-04-01 10:30:00 | 1178.10 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2026-04-06 12:15:00 | 1191.00 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1189.70 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2026-04-07 09:15:00 | 1257.00 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1241.00 | 2026-04-16 13:15:00 | 1224.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-04-16 15:00:00 | 1241.80 | 2026-04-22 10:15:00 | 1247.90 | STOP_HIT | 1.00 | 0.49% |
