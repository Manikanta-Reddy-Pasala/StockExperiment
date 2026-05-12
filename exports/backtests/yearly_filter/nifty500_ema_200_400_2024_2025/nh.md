# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 64 |
| PARTIAL | 6 |
| TARGET_HIT | 8 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 55
- **Target hits / Stop hits / Partials:** 8 / 56 / 6
- **Avg / median % per leg:** -0.35% / -1.52%
- **Sum % (uncompounded):** -24.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 7 | 21.2% | 7 | 26 | 0 | 0.27% | 9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 7 | 21.2% | 7 | 26 | 0 | 0.27% | 9.1% |
| SELL (all) | 37 | 8 | 21.6% | 1 | 30 | 6 | -0.90% | -33.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 8 | 21.6% | 1 | 30 | 6 | -0.90% | -33.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 15 | 21.4% | 8 | 56 | 6 | -0.35% | -24.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 10:15:00 | 1278.65 | 1272.06 | 1272.03 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 1266.00 | 1271.95 | 1271.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 1255.10 | 1271.67 | 1271.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1267.20 | 1230.95 | 1248.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1267.20 | 1230.95 | 1248.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1267.20 | 1230.95 | 1248.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 10:00:00 | 1234.60 | 1241.55 | 1251.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 14:45:00 | 1234.95 | 1240.86 | 1250.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 14:00:00 | 1234.20 | 1241.17 | 1250.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 14:45:00 | 1233.10 | 1241.16 | 1250.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1248.10 | 1225.52 | 1237.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 1248.10 | 1225.52 | 1237.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1242.75 | 1225.69 | 1238.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 1236.50 | 1226.10 | 1238.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1237.40 | 1226.42 | 1238.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:45:00 | 1237.00 | 1226.57 | 1238.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 1252.95 | 1228.02 | 1238.44 | SL hit (close>static) qty=1.00 sl=1250.85 alert=retest2 |

### Cycle 3 — BUY (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 14:15:00 | 1274.95 | 1242.55 | 1242.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 12:15:00 | 1226.00 | 1242.48 | 1242.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 1222.00 | 1242.27 | 1242.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 11:15:00 | 1245.80 | 1239.70 | 1241.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 11:15:00 | 1245.80 | 1239.70 | 1241.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1245.80 | 1239.70 | 1241.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 1245.80 | 1239.70 | 1241.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 1240.00 | 1239.70 | 1241.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:00:00 | 1237.85 | 1239.68 | 1241.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 1228.95 | 1239.59 | 1240.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 09:15:00 | 1175.96 | 1228.27 | 1234.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 1232.90 | 1222.51 | 1231.31 | SL hit (close>ema200) qty=0.50 sl=1222.51 alert=retest2 |

### Cycle 5 — BUY (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 15:15:00 | 1275.50 | 1238.59 | 1238.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 1280.20 | 1239.01 | 1238.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1284.55 | 1284.95 | 1266.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:30:00 | 1285.70 | 1284.95 | 1266.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1272.20 | 1284.20 | 1266.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 1265.70 | 1284.20 | 1266.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1274.45 | 1284.10 | 1266.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:30:00 | 1267.55 | 1284.10 | 1266.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1254.00 | 1283.66 | 1266.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1254.00 | 1283.66 | 1266.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1238.40 | 1283.21 | 1266.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:30:00 | 1234.05 | 1283.21 | 1266.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1265.05 | 1281.19 | 1266.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 1268.00 | 1281.19 | 1266.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 1256.05 | 1280.77 | 1266.05 | SL hit (close<static) qty=1.00 sl=1260.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1206.20 | 1255.79 | 1255.95 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1287.55 | 1255.03 | 1254.91 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 1230.65 | 1254.87 | 1254.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1216.50 | 1253.25 | 1254.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 1251.45 | 1250.06 | 1252.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 10:15:00 | 1251.45 | 1250.06 | 1252.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1251.45 | 1250.06 | 1252.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 1251.45 | 1250.06 | 1252.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1257.60 | 1250.13 | 1252.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 1256.50 | 1250.13 | 1252.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1266.00 | 1250.29 | 1252.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1270.90 | 1250.29 | 1252.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1235.00 | 1250.82 | 1252.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 1213.45 | 1250.82 | 1252.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:45:00 | 1229.80 | 1244.27 | 1249.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 11:45:00 | 1231.15 | 1244.16 | 1249.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 12:15:00 | 1229.90 | 1244.16 | 1249.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1245.65 | 1244.11 | 1248.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 1245.65 | 1244.11 | 1248.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 1249.75 | 1244.16 | 1248.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 1253.25 | 1244.16 | 1248.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1260.60 | 1244.33 | 1249.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1260.60 | 1244.33 | 1249.05 | SL hit (close>static) qty=1.00 sl=1253.75 alert=retest2 |

### Cycle 9 — BUY (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 12:15:00 | 1292.30 | 1253.48 | 1253.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 1318.10 | 1259.99 | 1257.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 1273.00 | 1276.37 | 1266.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 12:00:00 | 1273.00 | 1276.37 | 1266.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1285.65 | 1281.74 | 1271.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:30:00 | 1289.40 | 1281.81 | 1271.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:15:00 | 1286.90 | 1281.91 | 1271.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 14:00:00 | 1288.75 | 1281.98 | 1271.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 14:45:00 | 1286.90 | 1282.05 | 1271.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1279.50 | 1282.70 | 1272.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1275.90 | 1282.70 | 1272.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1293.50 | 1288.23 | 1276.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:30:00 | 1309.00 | 1287.85 | 1277.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 15:00:00 | 1311.60 | 1287.85 | 1277.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:30:00 | 1309.70 | 1288.30 | 1278.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:00:00 | 1310.15 | 1288.30 | 1278.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1297.20 | 1304.38 | 1289.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:45:00 | 1290.00 | 1304.38 | 1289.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 1289.10 | 1304.07 | 1289.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 13:00:00 | 1289.10 | 1304.07 | 1289.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 1271.60 | 1303.75 | 1289.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 13:15:00 | 1271.60 | 1303.75 | 1289.22 | SL hit (close<static) qty=1.00 sl=1276.10 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1817.50 | 1869.56 | 1869.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1803.60 | 1865.56 | 1867.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 13:15:00 | 1836.00 | 1835.15 | 1850.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:45:00 | 1833.10 | 1835.15 | 1850.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1806.70 | 1807.64 | 1830.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1798.10 | 1807.60 | 1830.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1800.00 | 1807.60 | 1830.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 1794.50 | 1807.46 | 1829.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1795.80 | 1807.34 | 1829.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1708.19 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1710.00 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1704.77 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1706.01 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1804.40 | 1774.90 | 1803.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1804.40 | 1774.90 | 1803.39 | SL hit (close>ema200) qty=0.50 sl=1774.90 alert=retest2 |

### Cycle 11 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1953.50 | 1796.49 | 1796.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 1996.70 | 1801.44 | 1798.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1893.80 | 1893.99 | 1858.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1893.80 | 1893.99 | 1858.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1860.90 | 1893.85 | 1860.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 1860.90 | 1893.85 | 1860.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1857.20 | 1893.48 | 1860.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 1858.70 | 1893.48 | 1860.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1858.40 | 1893.13 | 1860.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1886.30 | 1892.78 | 1860.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 1863.20 | 1889.74 | 1864.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 1838.60 | 1888.91 | 1864.19 | SL hit (close<static) qty=1.00 sl=1850.00 alert=retest2 |

### Cycle 12 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1723.50 | 1866.96 | 1867.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1713.60 | 1863.97 | 1865.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 1804.70 | 1803.23 | 1829.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 1804.70 | 1803.23 | 1829.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1836.90 | 1803.34 | 1828.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 1843.00 | 1803.34 | 1828.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1842.10 | 1803.73 | 1828.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1842.20 | 1803.73 | 1828.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1820.40 | 1812.58 | 1831.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1803.40 | 1812.67 | 1831.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 1815.40 | 1812.70 | 1830.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1833.00 | 1812.90 | 1830.99 | SL hit (close>static) qty=1.00 sl=1831.20 alert=retest2 |

### Cycle 13 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 1865.30 | 1767.05 | 1767.04 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-13 10:00:00 | 1234.60 | 2024-07-08 09:15:00 | 1252.95 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-06-14 14:45:00 | 1234.95 | 2024-07-08 09:15:00 | 1252.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-06-18 14:00:00 | 1234.20 | 2024-07-08 09:15:00 | 1252.95 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-06-18 14:45:00 | 1233.10 | 2024-07-15 13:15:00 | 1251.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-07-04 13:45:00 | 1236.50 | 2024-07-15 13:15:00 | 1251.60 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1237.40 | 2024-07-15 13:15:00 | 1251.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-05 09:45:00 | 1237.00 | 2024-07-15 13:15:00 | 1251.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-07-08 11:30:00 | 1236.55 | 2024-07-30 14:15:00 | 1274.95 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-07-09 10:15:00 | 1220.90 | 2024-07-30 14:15:00 | 1274.95 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-07-10 10:00:00 | 1220.35 | 2024-07-30 14:15:00 | 1274.95 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2024-07-12 13:00:00 | 1220.35 | 2024-07-30 14:15:00 | 1274.95 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2024-08-09 14:00:00 | 1237.85 | 2024-08-19 09:15:00 | 1175.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 14:00:00 | 1237.85 | 2024-08-21 10:15:00 | 1232.90 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2024-08-09 14:30:00 | 1228.95 | 2024-08-21 14:15:00 | 1258.95 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-08-21 11:45:00 | 1237.65 | 2024-08-21 14:15:00 | 1258.95 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-20 11:15:00 | 1268.00 | 2024-09-20 12:15:00 | 1256.05 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-20 12:30:00 | 1267.15 | 2024-09-20 13:15:00 | 1248.10 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-11-04 09:15:00 | 1213.45 | 2024-11-07 09:15:00 | 1260.60 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-11-06 10:45:00 | 1229.80 | 2024-11-07 09:15:00 | 1260.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-11-06 11:45:00 | 1231.15 | 2024-11-07 09:15:00 | 1260.60 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-11-06 12:15:00 | 1229.90 | 2024-11-07 09:15:00 | 1260.60 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-12-19 10:30:00 | 1289.40 | 2025-01-13 13:15:00 | 1271.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-19 13:15:00 | 1286.90 | 2025-01-13 13:15:00 | 1271.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-12-19 14:00:00 | 1288.75 | 2025-01-13 13:15:00 | 1271.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-19 14:45:00 | 1286.90 | 2025-01-13 13:15:00 | 1271.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-01-01 14:30:00 | 1309.00 | 2025-01-27 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-01-01 15:00:00 | 1311.60 | 2025-01-27 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-01-02 09:30:00 | 1309.70 | 2025-01-27 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-01-02 10:00:00 | 1310.15 | 2025-01-27 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-01-14 14:15:00 | 1292.35 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-01-15 14:45:00 | 1293.40 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1295.45 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-01-16 13:30:00 | 1304.95 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-01-27 11:15:00 | 1283.75 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-01-27 13:00:00 | 1291.90 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-01-28 10:30:00 | 1283.30 | 2025-01-28 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-01-29 12:00:00 | 1283.65 | 2025-02-10 09:15:00 | 1412.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1312.50 | 2025-02-25 09:15:00 | 1443.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-31 09:45:00 | 1317.55 | 2025-02-25 09:15:00 | 1449.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-14 14:30:00 | 1327.30 | 2025-02-25 09:15:00 | 1442.54 | TARGET_HIT | 1.00 | 8.68% |
| BUY | retest2 | 2025-02-17 12:15:00 | 1311.40 | 2025-02-25 09:15:00 | 1451.67 | TARGET_HIT | 1.00 | 10.70% |
| BUY | retest2 | 2025-02-17 13:30:00 | 1319.70 | 2025-02-25 09:15:00 | 1451.84 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-02-17 14:00:00 | 1319.85 | 2025-02-25 11:15:00 | 1460.03 | TARGET_HIT | 1.00 | 10.62% |
| SELL | retest2 | 2025-09-17 12:30:00 | 1798.10 | 2025-09-26 09:15:00 | 1708.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 13:15:00 | 1800.00 | 2025-09-26 09:15:00 | 1710.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 1794.50 | 2025-09-26 09:15:00 | 1704.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1795.80 | 2025-09-26 09:15:00 | 1706.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 1798.10 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.35% |
| SELL | retest2 | 2025-09-17 13:15:00 | 1800.00 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.24% |
| SELL | retest2 | 2025-09-18 10:00:00 | 1794.50 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.55% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1795.80 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.48% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1767.80 | 2025-11-04 09:15:00 | 1835.70 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-11-03 13:15:00 | 1775.00 | 2025-11-04 09:15:00 | 1835.70 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1762.90 | 2025-11-13 11:15:00 | 1794.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-11-13 12:15:00 | 1781.10 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -7.91% |
| SELL | retest2 | 2025-11-13 15:15:00 | 1773.00 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -8.40% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1886.30 | 2025-12-18 13:15:00 | 1838.60 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-12-18 11:45:00 | 1863.20 | 2025-12-18 13:15:00 | 1838.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-19 10:00:00 | 1873.80 | 2025-12-30 09:15:00 | 1851.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-12-26 15:15:00 | 1863.00 | 2025-12-30 10:15:00 | 1808.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-12-29 14:15:00 | 1866.30 | 2025-12-30 10:15:00 | 1808.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-12-30 14:30:00 | 1868.90 | 2025-12-30 15:15:00 | 1840.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-31 10:30:00 | 1873.60 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-01-01 09:30:00 | 1870.00 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-01-14 09:15:00 | 1933.00 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1803.40 | 2026-02-16 10:15:00 | 1833.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-16 10:00:00 | 1815.40 | 2026-02-16 10:15:00 | 1833.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1814.20 | 2026-02-25 11:15:00 | 1862.20 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 09:15:00 | 1708.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-16 12:15:00 | 1618.11 | TARGET_HIT | 0.50 | 10.00% |
