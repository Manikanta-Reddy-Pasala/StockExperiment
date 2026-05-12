# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1323.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 51 |
| ALERT2 | 51 |
| ALERT2_SKIP | 23 |
| ALERT3 | 144 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 43 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 34
- **Target hits / Stop hits / Partials:** 3 / 44 / 8
- **Avg / median % per leg:** 0.97% / -0.73%
- **Sum % (uncompounded):** 53.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 4 | 16.0% | 0 | 25 | 0 | -0.66% | -16.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.49% | -1.5% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 0 | 24 | 0 | -0.62% | -15.0% |
| SELL (all) | 30 | 17 | 56.7% | 3 | 19 | 8 | 2.33% | 70.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.37% | -4.1% |
| SELL @ 3rd Alert (retest2) | 27 | 17 | 63.0% | 3 | 16 | 8 | 2.74% | 74.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| retest2 (combined) | 51 | 21 | 41.2% | 3 | 40 | 8 | 1.16% | 59.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1259.70 | 1236.92 | 1234.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1266.70 | 1246.44 | 1239.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1262.60 | 1262.77 | 1254.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 1263.00 | 1262.77 | 1254.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1258.00 | 1260.48 | 1255.45 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 1248.00 | 1253.07 | 1253.44 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 1259.30 | 1254.15 | 1253.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1272.80 | 1262.11 | 1258.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1257.00 | 1267.66 | 1264.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1266.90 | 1267.50 | 1264.77 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1239.80 | 1259.24 | 1261.62 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 1266.80 | 1262.15 | 1261.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 1268.00 | 1263.32 | 1262.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 1265.80 | 1265.81 | 1264.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 1265.80 | 1265.81 | 1264.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1271.90 | 1269.19 | 1266.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 1268.20 | 1269.19 | 1266.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1267.50 | 1269.41 | 1267.12 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1262.00 | 1266.52 | 1266.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1254.00 | 1264.01 | 1265.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 1267.30 | 1261.12 | 1260.50 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1253.60 | 1261.51 | 1261.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 13:15:00 | 1246.20 | 1255.64 | 1258.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 1236.60 | 1231.45 | 1237.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1236.50 | 1232.46 | 1236.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1236.50 | 1232.46 | 1236.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1245.80 | 1235.13 | 1237.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 1245.80 | 1235.13 | 1237.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1242.80 | 1236.66 | 1238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 1241.40 | 1236.66 | 1238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 1251.00 | 1240.64 | 1239.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1257.50 | 1245.80 | 1242.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1305.50 | 1306.53 | 1291.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 1303.00 | 1306.53 | 1291.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1302.60 | 1309.50 | 1302.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:30:00 | 1302.30 | 1309.50 | 1302.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1303.70 | 1308.34 | 1302.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1303.70 | 1308.34 | 1302.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1302.40 | 1307.15 | 1302.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1302.40 | 1307.15 | 1302.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1283.80 | 1302.48 | 1300.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1283.80 | 1302.48 | 1300.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1291.50 | 1300.29 | 1299.70 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1287.90 | 1297.81 | 1298.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 1280.90 | 1292.73 | 1296.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1286.00 | 1283.12 | 1288.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1286.00 | 1283.12 | 1288.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1288.90 | 1284.28 | 1288.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 1288.90 | 1284.28 | 1288.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1307.60 | 1288.94 | 1290.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1307.60 | 1288.94 | 1290.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1309.40 | 1293.03 | 1291.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1313.90 | 1297.21 | 1293.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 1290.00 | 1306.92 | 1302.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1290.00 | 1303.53 | 1301.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1283.10 | 1303.53 | 1301.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1279.80 | 1298.79 | 1299.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1260.00 | 1272.15 | 1281.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1298.50 | 1273.26 | 1277.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1290.00 | 1276.61 | 1278.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1274.40 | 1276.61 | 1278.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1288.00 | 1279.49 | 1279.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1288.00 | 1279.49 | 1279.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1294.30 | 1284.65 | 1281.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1306.30 | 1308.45 | 1298.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1306.30 | 1308.45 | 1298.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1303.30 | 1306.25 | 1300.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1310.00 | 1305.86 | 1301.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 1310.60 | 1312.08 | 1310.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1315.90 | 1311.45 | 1310.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 1302.60 | 1309.68 | 1309.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 1302.60 | 1309.68 | 1309.94 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1320.00 | 1311.29 | 1310.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 1322.30 | 1314.52 | 1312.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1359.50 | 1366.83 | 1353.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 1365.30 | 1366.83 | 1353.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1357.60 | 1364.20 | 1354.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 1364.70 | 1364.52 | 1355.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 1350.90 | 1361.71 | 1355.91 | SL hit (close<static) qty=1.00 sl=1353.30 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1360.70 | 1365.67 | 1366.25 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 12:15:00 | 1370.10 | 1367.25 | 1366.91 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1363.30 | 1367.22 | 1367.39 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1370.60 | 1367.89 | 1367.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1375.40 | 1369.39 | 1368.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1385.80 | 1389.11 | 1381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1390.10 | 1389.45 | 1383.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 1391.20 | 1389.98 | 1383.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 1392.10 | 1389.98 | 1383.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1391.10 | 1391.87 | 1386.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:15:00 | 1391.40 | 1391.13 | 1387.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1390.70 | 1391.42 | 1388.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1391.90 | 1391.42 | 1388.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1390.00 | 1391.13 | 1388.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 1382.10 | 1391.13 | 1388.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1381.00 | 1389.11 | 1388.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1381.00 | 1389.11 | 1388.21 | SL hit (close<static) qty=1.00 sl=1382.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1377.20 | 1385.68 | 1386.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 1370.50 | 1381.45 | 1384.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 1386.50 | 1382.24 | 1384.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1387.70 | 1383.33 | 1384.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1387.70 | 1383.33 | 1384.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1388.90 | 1384.45 | 1385.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1391.50 | 1384.45 | 1385.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1386.30 | 1384.82 | 1385.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1357.00 | 1385.05 | 1385.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 1337.10 | 1329.63 | 1329.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1337.10 | 1329.63 | 1329.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 1343.40 | 1333.76 | 1331.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1324.10 | 1335.16 | 1332.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1320.20 | 1332.16 | 1331.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1320.30 | 1332.16 | 1331.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1314.60 | 1328.65 | 1330.22 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1340.60 | 1328.72 | 1327.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 1342.90 | 1331.56 | 1328.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 1330.40 | 1331.44 | 1328.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1333.60 | 1331.88 | 1329.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1333.50 | 1331.88 | 1329.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1327.30 | 1330.96 | 1329.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1327.30 | 1330.96 | 1329.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1324.90 | 1329.75 | 1328.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1324.30 | 1329.75 | 1328.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 1316.40 | 1327.08 | 1327.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1314.00 | 1323.17 | 1325.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1312.30 | 1316.73 | 1319.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1319.40 | 1317.27 | 1319.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 1306.30 | 1314.79 | 1318.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1240.98 | 1293.94 | 1303.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 1257.90 | 1250.11 | 1265.94 | SL hit (close>ema200) qty=0.50 sl=1250.11 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 1273.70 | 1266.63 | 1266.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1356.40 | 1288.85 | 1277.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 1367.00 | 1367.74 | 1348.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 1367.00 | 1367.74 | 1348.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1352.50 | 1364.77 | 1354.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1353.60 | 1364.77 | 1354.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1360.00 | 1363.82 | 1355.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:00:00 | 1366.00 | 1362.46 | 1356.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 1365.40 | 1362.41 | 1356.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1364.90 | 1363.87 | 1359.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1366.10 | 1360.07 | 1358.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1362.00 | 1362.94 | 1360.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1362.60 | 1362.94 | 1360.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1355.90 | 1361.53 | 1360.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1355.90 | 1361.53 | 1360.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1358.20 | 1360.86 | 1360.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1350.30 | 1360.86 | 1360.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1348.00 | 1358.29 | 1359.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1348.00 | 1358.29 | 1359.05 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 1361.60 | 1357.01 | 1356.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1373.30 | 1360.49 | 1358.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1417.80 | 1426.36 | 1411.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1417.80 | 1426.36 | 1411.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1421.40 | 1424.60 | 1416.77 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1399.10 | 1413.34 | 1414.26 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1422.50 | 1415.17 | 1415.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 1434.90 | 1421.55 | 1418.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 1416.80 | 1426.06 | 1422.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1424.10 | 1425.67 | 1422.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 1420.30 | 1425.67 | 1422.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1423.40 | 1425.21 | 1422.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 1422.40 | 1425.21 | 1422.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1416.50 | 1423.47 | 1421.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 1416.50 | 1423.47 | 1421.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1419.60 | 1422.70 | 1421.75 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1410.50 | 1419.15 | 1420.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1408.70 | 1413.58 | 1416.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 1394.70 | 1391.83 | 1399.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 1394.70 | 1391.83 | 1399.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1402.80 | 1395.17 | 1399.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 1399.70 | 1395.17 | 1399.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1400.70 | 1396.28 | 1399.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 1398.00 | 1400.31 | 1400.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1409.60 | 1401.80 | 1401.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1409.60 | 1401.80 | 1401.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1415.80 | 1407.38 | 1404.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 1417.40 | 1418.74 | 1413.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 1417.40 | 1418.74 | 1413.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1413.80 | 1417.75 | 1413.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 1414.00 | 1417.75 | 1413.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1417.00 | 1417.60 | 1413.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1414.20 | 1417.60 | 1413.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1408.80 | 1415.84 | 1413.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 1409.90 | 1415.84 | 1413.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1408.40 | 1414.35 | 1412.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 1407.10 | 1414.35 | 1412.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1406.10 | 1410.94 | 1411.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 15:15:00 | 1404.20 | 1409.41 | 1410.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 1412.00 | 1408.81 | 1410.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1426.80 | 1412.41 | 1411.56 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1393.30 | 1415.06 | 1415.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1386.30 | 1409.31 | 1413.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1375.40 | 1375.21 | 1387.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1375.40 | 1375.21 | 1387.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1370.50 | 1370.35 | 1376.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1357.60 | 1372.43 | 1375.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1360.00 | 1353.54 | 1359.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 1359.20 | 1355.31 | 1360.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 1360.20 | 1355.31 | 1360.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1373.80 | 1359.01 | 1361.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1373.80 | 1359.01 | 1361.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1374.50 | 1362.11 | 1362.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1375.90 | 1362.11 | 1362.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 1381.90 | 1366.06 | 1364.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 1381.90 | 1366.06 | 1364.36 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 1359.20 | 1363.42 | 1363.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 14:15:00 | 1352.00 | 1361.13 | 1362.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 1354.30 | 1353.97 | 1357.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1352.60 | 1353.97 | 1357.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1351.80 | 1353.53 | 1356.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 1355.60 | 1353.53 | 1356.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 1350.50 | 1352.86 | 1355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 1350.50 | 1352.86 | 1355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 1355.00 | 1353.29 | 1355.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 1355.40 | 1353.29 | 1355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1355.10 | 1353.65 | 1355.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 1355.30 | 1353.65 | 1355.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 1358.40 | 1354.60 | 1355.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 1362.90 | 1354.60 | 1355.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 1358.00 | 1355.28 | 1356.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 1354.90 | 1355.28 | 1356.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1351.80 | 1353.90 | 1355.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 1348.50 | 1353.90 | 1355.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 1357.70 | 1352.33 | 1353.88 | SL hit (close>static) qty=1.00 sl=1356.80 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1366.20 | 1356.86 | 1355.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 09:15:00 | 1386.50 | 1370.88 | 1366.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 15:15:00 | 1377.30 | 1378.47 | 1373.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:15:00 | 1412.00 | 1378.47 | 1373.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1391.00 | 1408.18 | 1396.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1391.00 | 1408.18 | 1396.22 | SL hit (close<ema400) qty=1.00 sl=1396.22 alert=retest1 |

### Cycle 38 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 1371.30 | 1387.32 | 1388.80 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 1395.10 | 1387.91 | 1387.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1396.90 | 1391.58 | 1389.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:00:00 | 1423.30 | 1422.14 | 1416.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 1427.10 | 1438.52 | 1436.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 1432.80 | 1435.92 | 1435.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 1432.80 | 1435.92 | 1435.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 1415.70 | 1430.32 | 1433.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1427.50 | 1421.72 | 1426.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1426.00 | 1422.57 | 1426.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1410.90 | 1422.57 | 1426.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1340.36 | 1348.44 | 1364.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 1316.00 | 1315.25 | 1336.00 | SL hit (close>ema200) qty=0.50 sl=1315.25 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1348.20 | 1337.68 | 1337.26 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1320.20 | 1336.28 | 1337.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1310.90 | 1327.48 | 1332.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1321.60 | 1316.26 | 1324.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 1321.60 | 1316.26 | 1324.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1330.90 | 1319.19 | 1324.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 1330.90 | 1319.19 | 1324.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1322.90 | 1319.93 | 1324.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:45:00 | 1334.40 | 1319.93 | 1324.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1323.40 | 1320.62 | 1324.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 1320.40 | 1320.62 | 1324.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1335.40 | 1323.58 | 1325.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 1335.40 | 1323.58 | 1325.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1339.90 | 1326.84 | 1326.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 1353.30 | 1335.74 | 1331.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1349.40 | 1350.19 | 1341.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 1349.40 | 1350.19 | 1341.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1397.80 | 1405.87 | 1399.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 1397.10 | 1405.87 | 1399.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1393.10 | 1403.32 | 1398.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1388.40 | 1403.32 | 1398.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1392.20 | 1401.09 | 1398.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1386.40 | 1401.09 | 1398.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1382.60 | 1394.50 | 1395.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1366.50 | 1388.90 | 1393.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 1365.50 | 1360.59 | 1369.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 1367.80 | 1360.59 | 1369.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1374.50 | 1363.38 | 1369.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1374.50 | 1363.38 | 1369.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1379.40 | 1366.58 | 1370.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1379.40 | 1366.58 | 1370.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1382.90 | 1369.84 | 1371.91 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1387.70 | 1373.42 | 1373.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1390.20 | 1376.77 | 1374.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1386.70 | 1379.54 | 1377.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1365.70 | 1376.67 | 1377.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1365.70 | 1376.67 | 1377.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 1360.40 | 1369.29 | 1372.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1327.50 | 1326.60 | 1333.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:15:00 | 1318.10 | 1324.60 | 1331.08 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:45:00 | 1318.10 | 1323.16 | 1329.84 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 15:15:00 | 1318.00 | 1320.41 | 1326.71 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1321.80 | 1320.43 | 1325.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1336.10 | 1324.18 | 1326.04 | SL hit (close>ema400) qty=1.00 sl=1326.04 alert=retest1 |

### Cycle 47 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1335.20 | 1328.82 | 1327.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 1339.90 | 1332.26 | 1329.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1338.70 | 1335.65 | 1332.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1344.40 | 1337.40 | 1333.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1346.80 | 1337.40 | 1333.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:00:00 | 1351.50 | 1341.39 | 1336.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1368.90 | 1387.47 | 1389.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 13:15:00 | 1368.90 | 1387.47 | 1389.04 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1406.80 | 1389.94 | 1387.68 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1384.50 | 1391.02 | 1391.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1364.30 | 1382.04 | 1386.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 1364.00 | 1360.93 | 1369.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 1364.00 | 1360.93 | 1369.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1366.80 | 1362.11 | 1369.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1369.40 | 1362.11 | 1369.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1361.60 | 1362.00 | 1368.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1361.60 | 1362.00 | 1368.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1377.30 | 1365.03 | 1368.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1382.90 | 1365.03 | 1368.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1376.30 | 1367.29 | 1369.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 1377.10 | 1367.29 | 1369.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1381.20 | 1372.30 | 1371.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 1385.00 | 1376.57 | 1373.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1475.30 | 1482.17 | 1458.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1475.30 | 1482.17 | 1458.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1479.70 | 1496.77 | 1482.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1479.70 | 1496.77 | 1482.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1473.00 | 1492.02 | 1481.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1473.00 | 1492.02 | 1481.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1478.60 | 1489.33 | 1481.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 1473.70 | 1489.33 | 1481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1474.30 | 1484.26 | 1480.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1473.60 | 1484.26 | 1480.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1466.00 | 1478.40 | 1478.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 1460.80 | 1472.32 | 1475.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 1475.30 | 1471.02 | 1473.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1470.00 | 1470.82 | 1473.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1473.00 | 1470.82 | 1473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1451.80 | 1465.81 | 1469.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1446.40 | 1459.66 | 1466.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1447.00 | 1455.70 | 1463.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1445.50 | 1452.53 | 1460.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1374.65 | 1397.75 | 1413.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1374.08 | 1389.32 | 1408.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1373.22 | 1389.32 | 1408.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 1301.76 | 1342.48 | 1375.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1340.80 | 1317.96 | 1317.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1345.30 | 1330.22 | 1323.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1336.50 | 1349.96 | 1338.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1349.90 | 1351.55 | 1343.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 1310.20 | 1351.55 | 1343.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1351.90 | 1351.62 | 1344.64 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 1314.70 | 1339.31 | 1340.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1301.50 | 1320.11 | 1328.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 1312.60 | 1311.17 | 1321.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 1309.00 | 1311.17 | 1321.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1330.00 | 1308.25 | 1315.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1330.00 | 1308.25 | 1315.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1327.00 | 1312.00 | 1316.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1356.50 | 1312.00 | 1316.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1360.90 | 1321.78 | 1320.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1406.60 | 1364.65 | 1345.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1409.10 | 1418.62 | 1400.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:30:00 | 1408.60 | 1418.62 | 1400.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1467.70 | 1483.58 | 1462.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1467.70 | 1483.58 | 1462.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1522.60 | 1525.55 | 1517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:45:00 | 1521.60 | 1525.55 | 1517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1521.00 | 1526.90 | 1521.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 1520.00 | 1526.90 | 1521.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1517.40 | 1525.00 | 1521.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1517.40 | 1525.00 | 1521.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1524.30 | 1524.86 | 1521.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 1527.90 | 1525.28 | 1522.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 1528.00 | 1526.26 | 1523.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1527.60 | 1528.18 | 1524.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 1510.90 | 1529.75 | 1529.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1510.90 | 1529.75 | 1529.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1504.00 | 1524.60 | 1527.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 1536.00 | 1526.58 | 1527.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1543.90 | 1530.05 | 1529.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 14:15:00 | 1546.70 | 1535.91 | 1532.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 1539.40 | 1540.46 | 1535.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 1539.40 | 1540.46 | 1535.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1535.30 | 1539.43 | 1535.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 1533.90 | 1539.43 | 1535.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1536.50 | 1538.84 | 1535.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 1535.70 | 1538.84 | 1535.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1539.00 | 1538.88 | 1535.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:15:00 | 1535.60 | 1538.88 | 1535.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1535.70 | 1538.24 | 1535.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1531.40 | 1538.24 | 1535.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1537.00 | 1537.99 | 1536.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1545.90 | 1539.67 | 1536.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 1535.00 | 1541.53 | 1539.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 1532.70 | 1541.53 | 1539.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1534.90 | 1540.21 | 1538.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1540.20 | 1542.57 | 1540.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 1522.90 | 1537.07 | 1538.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 1522.90 | 1537.07 | 1538.06 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1549.60 | 1536.99 | 1536.91 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1535.40 | 1536.67 | 1536.77 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 1537.50 | 1536.84 | 1536.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 1547.40 | 1539.36 | 1538.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 1557.50 | 1557.76 | 1550.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 1505.70 | 1557.76 | 1550.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1522.90 | 1550.79 | 1547.67 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1509.10 | 1538.65 | 1542.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1496.60 | 1530.24 | 1538.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 1455.70 | 1454.51 | 1480.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 1455.70 | 1454.51 | 1480.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1478.90 | 1460.76 | 1478.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 1478.90 | 1460.76 | 1478.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1476.30 | 1463.87 | 1478.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 1479.20 | 1463.87 | 1478.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1480.80 | 1467.25 | 1478.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1480.80 | 1467.25 | 1478.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1484.70 | 1470.74 | 1479.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1485.90 | 1470.74 | 1479.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1482.30 | 1473.05 | 1479.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 1485.00 | 1473.05 | 1479.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1469.50 | 1472.34 | 1478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 1464.00 | 1470.43 | 1477.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1420.30 | 1476.07 | 1478.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1490.00 | 1462.53 | 1459.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1490.00 | 1462.53 | 1459.96 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1426.50 | 1460.21 | 1461.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1396.80 | 1442.25 | 1451.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1415.40 | 1398.19 | 1413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1413.40 | 1402.48 | 1413.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1415.00 | 1402.48 | 1413.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1404.50 | 1402.89 | 1412.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1398.40 | 1402.89 | 1412.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1428.20 | 1416.51 | 1415.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1428.20 | 1416.51 | 1415.10 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1372.70 | 1411.11 | 1413.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 1361.00 | 1401.09 | 1409.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1265.20 | 1263.24 | 1297.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 1267.70 | 1263.24 | 1297.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1301.50 | 1272.95 | 1296.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1300.00 | 1272.95 | 1296.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1304.00 | 1279.16 | 1297.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1303.80 | 1279.16 | 1297.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1360.60 | 1313.00 | 1308.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1370.30 | 1337.31 | 1321.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1331.90 | 1346.78 | 1330.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1332.20 | 1343.86 | 1331.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1326.00 | 1343.86 | 1331.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1326.20 | 1340.33 | 1330.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1326.20 | 1340.33 | 1330.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1332.40 | 1338.75 | 1330.73 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1296.20 | 1324.79 | 1326.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1277.70 | 1315.37 | 1321.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1283.20 | 1290.13 | 1303.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1263.90 | 1282.78 | 1299.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 14:15:00 | 1219.04 | 1260.34 | 1282.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1200.70 | 1244.48 | 1271.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1231.30 | 1229.49 | 1254.01 | SL hit (close>ema200) qty=0.50 sl=1229.49 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 1271.30 | 1239.80 | 1239.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 1282.20 | 1263.47 | 1255.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 1389.90 | 1392.73 | 1369.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 1389.90 | 1392.73 | 1369.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1435.40 | 1469.53 | 1460.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 1435.40 | 1469.53 | 1460.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1448.50 | 1465.33 | 1459.01 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 1447.10 | 1454.50 | 1455.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1436.00 | 1447.55 | 1451.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1459.00 | 1448.44 | 1450.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1465.70 | 1451.90 | 1452.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1465.70 | 1451.90 | 1452.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 1455.00 | 1452.52 | 1452.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1493.90 | 1460.79 | 1456.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 1503.80 | 1505.87 | 1492.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:45:00 | 1506.00 | 1505.87 | 1492.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1489.60 | 1505.38 | 1497.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 1489.60 | 1505.38 | 1497.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1497.30 | 1503.77 | 1497.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 1489.60 | 1503.77 | 1497.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1484.50 | 1499.91 | 1496.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1484.50 | 1499.91 | 1496.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1478.20 | 1495.57 | 1494.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1478.20 | 1495.57 | 1494.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 1471.00 | 1490.66 | 1492.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 1423.90 | 1477.31 | 1486.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 1450.00 | 1442.77 | 1458.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 1450.00 | 1442.77 | 1458.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1446.00 | 1443.41 | 1457.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 1454.30 | 1443.41 | 1457.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 1454.10 | 1448.54 | 1456.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1424.80 | 1449.31 | 1455.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1353.56 | 1374.37 | 1394.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1373.10 | 1371.45 | 1389.41 | SL hit (close>ema200) qty=0.50 sl=1371.45 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-23 09:15:00 | 1274.40 | 2025-06-23 10:15:00 | 1288.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1310.00 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-27 15:15:00 | 1310.60 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-30 09:30:00 | 1315.90 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-04 14:30:00 | 1364.70 | 2025-07-07 09:15:00 | 1350.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-07 14:45:00 | 1363.00 | 2025-07-08 09:15:00 | 1353.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-08 11:30:00 | 1364.40 | 2025-07-11 10:15:00 | 1360.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-07-16 12:30:00 | 1391.20 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-16 13:00:00 | 1392.10 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-17 09:45:00 | 1391.10 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-17 13:15:00 | 1391.40 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-18 12:45:00 | 1386.50 | 2025-07-18 13:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1357.00 | 2025-07-30 14:15:00 | 1337.10 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2025-08-07 11:30:00 | 1306.30 | 2025-08-11 09:15:00 | 1240.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 11:30:00 | 1306.30 | 2025-08-12 12:15:00 | 1257.90 | STOP_HIT | 0.50 | 3.71% |
| BUY | retest2 | 2025-08-21 14:00:00 | 1366.00 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-21 15:15:00 | 1365.40 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-22 10:45:00 | 1364.90 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1366.10 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-12 15:15:00 | 1398.00 | 2025-09-15 09:15:00 | 1409.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1357.60 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1360.00 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-29 12:30:00 | 1359.20 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-29 13:15:00 | 1360.20 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-10-06 11:15:00 | 1348.50 | 2025-10-06 13:15:00 | 1357.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2025-10-10 09:15:00 | 1412.00 | 2025-10-13 09:15:00 | 1391.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-20 11:00:00 | 1423.30 | 2025-10-27 11:15:00 | 1432.80 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-10-27 09:45:00 | 1427.10 | 2025-10-27 11:15:00 | 1432.80 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1410.90 | 2025-11-06 11:15:00 | 1340.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1410.90 | 2025-11-07 11:15:00 | 1316.00 | STOP_HIT | 0.50 | 6.73% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1386.70 | 2025-11-28 11:15:00 | 1365.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest1 | 2025-12-08 11:15:00 | 1318.10 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2025-12-08 11:45:00 | 1318.10 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2025-12-08 15:15:00 | 1318.00 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1346.80 | 2025-12-19 13:15:00 | 1368.90 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-12-11 12:00:00 | 1351.50 | 2025-12-19 13:15:00 | 1368.90 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1446.40 | 2026-01-20 10:15:00 | 1374.65 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1447.00 | 2026-01-20 11:15:00 | 1374.08 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1445.50 | 2026-01-20 11:15:00 | 1373.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1446.40 | 2026-01-21 09:15:00 | 1301.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1447.00 | 2026-01-21 09:15:00 | 1302.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1445.50 | 2026-01-21 09:15:00 | 1300.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-17 15:00:00 | 1527.90 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-18 10:15:00 | 1528.00 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1527.60 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-25 09:30:00 | 1540.20 | 2026-02-25 12:15:00 | 1522.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-06 11:30:00 | 1464.00 | 2026-03-11 10:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1420.30 | 2026-03-11 10:15:00 | 1490.00 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1398.40 | 2026-03-18 10:15:00 | 1428.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1283.20 | 2026-04-01 14:15:00 | 1219.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1263.90 | 2026-04-02 09:15:00 | 1200.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1283.20 | 2026-04-02 13:15:00 | 1231.30 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1263.90 | 2026-04-02 13:15:00 | 1231.30 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1424.80 | 2026-05-07 09:15:00 | 1353.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1424.80 | 2026-05-07 11:15:00 | 1373.10 | STOP_HIT | 0.50 | 3.63% |
