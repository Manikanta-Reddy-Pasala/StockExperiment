# INFY (INFY)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1179.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 213 |
| ALERT1 | 147 |
| ALERT2 | 143 |
| ALERT2_SKIP | 73 |
| ALERT3 | 369 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 143 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 146 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 100
- **Target hits / Stop hits / Partials:** 1 / 145 / 8
- **Avg / median % per leg:** 0.20% / -0.59%
- **Sum % (uncompounded):** 30.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 27 | 40.9% | 0 | 66 | 0 | 0.27% | 18.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.73% | -0.7% |
| BUY @ 3rd Alert (retest2) | 65 | 27 | 41.5% | 0 | 65 | 0 | 0.29% | 18.7% |
| SELL (all) | 88 | 27 | 30.7% | 1 | 79 | 8 | 0.14% | 12.4% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.59% | 9.2% |
| SELL @ 3rd Alert (retest2) | 86 | 26 | 30.2% | 0 | 78 | 8 | 0.04% | 3.2% |
| retest1 (combined) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.82% | 8.5% |
| retest2 (combined) | 151 | 53 | 35.1% | 0 | 143 | 8 | 0.14% | 21.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 1267.00 | 1259.88 | 1259.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 1269.60 | 1261.22 | 1259.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 14:15:00 | 1263.65 | 1264.85 | 1262.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 15:00:00 | 1263.65 | 1264.85 | 1262.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 1255.70 | 1262.88 | 1262.02 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 1250.05 | 1260.32 | 1260.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 1243.80 | 1257.01 | 1259.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 1252.25 | 1251.20 | 1255.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 1252.25 | 1251.20 | 1255.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1252.25 | 1251.20 | 1255.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 14:45:00 | 1245.15 | 1251.85 | 1254.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 09:15:00 | 1268.60 | 1254.90 | 1255.09 | SL hit (close>static) qty=1.00 sl=1259.65 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 1266.60 | 1257.24 | 1256.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 1270.35 | 1259.86 | 1257.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 1299.50 | 1300.69 | 1289.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 15:00:00 | 1299.50 | 1300.69 | 1289.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 1294.00 | 1298.81 | 1294.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 1292.05 | 1298.81 | 1294.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 1293.85 | 1297.82 | 1294.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:15:00 | 1292.00 | 1297.82 | 1294.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 1297.20 | 1297.70 | 1294.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 14:15:00 | 1298.50 | 1297.68 | 1295.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 09:15:00 | 1303.00 | 1317.64 | 1318.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 09:15:00 | 1303.00 | 1317.64 | 1318.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 14:15:00 | 1299.20 | 1306.89 | 1312.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 10:15:00 | 1306.25 | 1305.18 | 1310.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-05 11:00:00 | 1306.25 | 1305.18 | 1310.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 1287.85 | 1285.08 | 1293.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 10:30:00 | 1284.00 | 1285.08 | 1292.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 14:30:00 | 1284.00 | 1286.13 | 1289.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 10:00:00 | 1284.20 | 1273.21 | 1278.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 12:15:00 | 1291.60 | 1281.83 | 1281.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 1291.60 | 1281.83 | 1281.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 13:15:00 | 1294.80 | 1284.42 | 1282.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 1295.25 | 1299.61 | 1294.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 1295.25 | 1299.61 | 1294.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 1295.25 | 1299.61 | 1294.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 09:30:00 | 1295.35 | 1299.61 | 1294.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 1287.35 | 1296.97 | 1295.35 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 1286.35 | 1293.11 | 1293.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 10:15:00 | 1283.30 | 1288.17 | 1290.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 13:15:00 | 1290.90 | 1287.84 | 1289.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 13:15:00 | 1290.90 | 1287.84 | 1289.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 13:15:00 | 1290.90 | 1287.84 | 1289.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:00:00 | 1290.90 | 1287.84 | 1289.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 1289.20 | 1288.11 | 1289.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 15:00:00 | 1289.20 | 1288.11 | 1289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 1289.10 | 1288.31 | 1289.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 1296.00 | 1288.31 | 1289.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 1292.00 | 1289.05 | 1289.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 10:45:00 | 1289.35 | 1289.13 | 1289.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 12:15:00 | 1292.70 | 1290.58 | 1290.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 12:15:00 | 1292.70 | 1290.58 | 1290.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 14:15:00 | 1295.20 | 1291.52 | 1290.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 1297.70 | 1299.26 | 1296.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 11:00:00 | 1297.70 | 1299.26 | 1296.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 1298.50 | 1299.59 | 1297.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 1284.55 | 1299.59 | 1297.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 1281.90 | 1296.05 | 1296.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1268.45 | 1282.64 | 1288.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 1271.10 | 1270.93 | 1275.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:45:00 | 1270.10 | 1270.93 | 1275.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1278.00 | 1272.31 | 1275.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:45:00 | 1275.65 | 1273.01 | 1275.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 1283.60 | 1278.05 | 1277.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 1283.60 | 1278.05 | 1277.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 10:15:00 | 1290.35 | 1280.51 | 1278.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 09:15:00 | 1343.90 | 1344.89 | 1338.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 09:30:00 | 1343.20 | 1344.89 | 1338.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 1339.60 | 1343.09 | 1339.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 12:30:00 | 1339.75 | 1343.09 | 1339.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 1338.40 | 1342.15 | 1339.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:00:00 | 1338.40 | 1342.15 | 1339.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 1344.80 | 1342.68 | 1339.56 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 1327.15 | 1337.89 | 1338.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 1323.80 | 1331.19 | 1334.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 11:15:00 | 1334.10 | 1330.98 | 1333.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 11:15:00 | 1334.10 | 1330.98 | 1333.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 1334.10 | 1330.98 | 1333.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 1334.10 | 1330.98 | 1333.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 1338.10 | 1332.40 | 1334.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:30:00 | 1340.15 | 1332.40 | 1334.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 13:15:00 | 1336.05 | 1333.13 | 1334.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 14:30:00 | 1332.10 | 1332.17 | 1333.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 09:15:00 | 1344.65 | 1334.08 | 1334.34 | SL hit (close>static) qty=1.00 sl=1339.85 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 1344.55 | 1336.17 | 1335.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 1347.60 | 1339.96 | 1337.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 09:15:00 | 1334.00 | 1341.68 | 1339.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 1334.00 | 1341.68 | 1339.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1334.00 | 1341.68 | 1339.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 1334.00 | 1341.68 | 1339.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 1336.30 | 1340.61 | 1338.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:45:00 | 1331.85 | 1340.61 | 1338.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 12:15:00 | 1334.65 | 1337.72 | 1337.85 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 1367.00 | 1341.99 | 1339.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 10:15:00 | 1369.50 | 1347.49 | 1342.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 12:15:00 | 1425.60 | 1427.32 | 1401.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 13:00:00 | 1425.60 | 1427.32 | 1401.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 1446.05 | 1465.08 | 1452.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:00:00 | 1446.05 | 1465.08 | 1452.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 1445.20 | 1461.11 | 1451.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:15:00 | 1443.05 | 1461.11 | 1451.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 1446.00 | 1458.09 | 1451.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 14:30:00 | 1450.70 | 1453.50 | 1450.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 09:15:00 | 1342.45 | 1429.57 | 1440.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 1342.45 | 1429.57 | 1440.02 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 12:15:00 | 1357.70 | 1352.31 | 1351.66 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 1344.25 | 1350.67 | 1351.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 11:15:00 | 1338.40 | 1348.22 | 1350.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 1344.80 | 1343.95 | 1346.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 1344.80 | 1343.95 | 1346.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1344.80 | 1343.95 | 1346.87 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 1354.50 | 1349.01 | 1348.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 1355.25 | 1350.26 | 1349.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 1357.95 | 1362.13 | 1357.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 1357.95 | 1362.13 | 1357.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 1357.95 | 1362.13 | 1357.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 1357.95 | 1362.13 | 1357.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 1355.00 | 1360.71 | 1357.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 1355.85 | 1360.71 | 1357.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 1357.10 | 1359.98 | 1357.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 1359.35 | 1357.63 | 1356.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:45:00 | 1358.65 | 1357.92 | 1357.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 13:15:00 | 1351.60 | 1357.01 | 1357.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 13:15:00 | 1351.60 | 1357.01 | 1357.10 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 1365.05 | 1358.62 | 1357.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 1381.80 | 1364.11 | 1360.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 1388.45 | 1388.53 | 1383.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-09 10:15:00 | 1384.00 | 1388.53 | 1383.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 1390.55 | 1388.93 | 1384.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 1388.90 | 1388.93 | 1384.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 1389.25 | 1390.91 | 1388.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 1389.25 | 1390.91 | 1388.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 1391.00 | 1390.93 | 1388.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 1392.40 | 1390.93 | 1388.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 09:15:00 | 1380.05 | 1388.76 | 1388.11 | SL hit (close<static) qty=1.00 sl=1387.55 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 1378.45 | 1386.69 | 1387.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 11:15:00 | 1371.00 | 1383.56 | 1385.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 09:15:00 | 1378.45 | 1377.58 | 1381.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 09:45:00 | 1378.80 | 1377.58 | 1381.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 1384.40 | 1377.80 | 1380.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:00:00 | 1384.40 | 1377.80 | 1380.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 1384.45 | 1379.13 | 1381.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:00:00 | 1384.45 | 1379.13 | 1381.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 14:15:00 | 1394.00 | 1384.03 | 1383.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 09:15:00 | 1411.00 | 1390.85 | 1386.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 1409.80 | 1410.36 | 1400.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 10:00:00 | 1409.80 | 1410.36 | 1400.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 1394.30 | 1408.43 | 1404.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:45:00 | 1394.55 | 1408.43 | 1404.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 1390.45 | 1404.83 | 1403.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:45:00 | 1389.95 | 1404.83 | 1403.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 1393.65 | 1402.59 | 1402.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 1388.00 | 1399.68 | 1401.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 1398.70 | 1395.48 | 1398.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 1398.70 | 1395.48 | 1398.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1398.70 | 1395.48 | 1398.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 1398.70 | 1395.48 | 1398.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 1399.85 | 1396.36 | 1398.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:45:00 | 1400.20 | 1396.36 | 1398.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 1402.35 | 1397.55 | 1398.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:45:00 | 1402.90 | 1397.55 | 1398.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 1406.95 | 1400.07 | 1399.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 1410.75 | 1403.55 | 1401.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 1416.00 | 1417.92 | 1412.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 1416.00 | 1417.92 | 1412.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1416.00 | 1417.92 | 1412.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 14:45:00 | 1420.85 | 1417.94 | 1414.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 1422.10 | 1417.60 | 1416.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 10:45:00 | 1420.50 | 1418.24 | 1416.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 12:45:00 | 1420.95 | 1419.05 | 1417.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 1417.30 | 1418.78 | 1417.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 14:45:00 | 1415.95 | 1418.78 | 1417.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 1418.00 | 1418.63 | 1417.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 1427.50 | 1418.63 | 1417.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-18 12:15:00 | 1494.50 | 1501.30 | 1501.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 1494.50 | 1501.30 | 1501.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 1490.65 | 1499.17 | 1500.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 14:15:00 | 1489.50 | 1488.83 | 1493.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-20 15:00:00 | 1489.50 | 1488.83 | 1493.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 1492.25 | 1489.51 | 1493.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 1483.90 | 1489.51 | 1493.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 12:15:00 | 1498.45 | 1488.97 | 1491.38 | SL hit (close>static) qty=1.00 sl=1493.15 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 14:15:00 | 1502.00 | 1493.58 | 1493.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 11:15:00 | 1503.15 | 1495.93 | 1494.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 13:15:00 | 1495.00 | 1497.14 | 1495.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 13:15:00 | 1495.00 | 1497.14 | 1495.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 13:15:00 | 1495.00 | 1497.14 | 1495.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:00:00 | 1495.00 | 1497.14 | 1495.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 1495.75 | 1496.86 | 1495.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 1495.75 | 1496.86 | 1495.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 1495.00 | 1496.49 | 1495.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 1486.90 | 1496.49 | 1495.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 09:15:00 | 1476.65 | 1492.52 | 1493.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 10:15:00 | 1472.90 | 1488.60 | 1491.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 11:15:00 | 1463.60 | 1462.11 | 1470.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-27 12:00:00 | 1463.60 | 1462.11 | 1470.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 1465.70 | 1464.18 | 1469.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 15:00:00 | 1465.70 | 1464.18 | 1469.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 1461.35 | 1464.38 | 1468.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:00:00 | 1452.75 | 1462.06 | 1467.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 09:15:00 | 1457.60 | 1440.70 | 1438.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 1457.60 | 1440.70 | 1438.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 10:15:00 | 1470.70 | 1446.70 | 1441.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 13:15:00 | 1477.70 | 1477.95 | 1469.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 14:00:00 | 1477.70 | 1477.95 | 1469.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1487.05 | 1479.14 | 1472.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 1475.70 | 1479.14 | 1472.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 1492.15 | 1495.46 | 1487.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:45:00 | 1488.80 | 1495.46 | 1487.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 1490.55 | 1494.58 | 1488.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:15:00 | 1488.05 | 1494.58 | 1488.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 1479.30 | 1491.53 | 1488.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:00:00 | 1479.30 | 1491.53 | 1488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 1480.10 | 1489.24 | 1487.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:30:00 | 1481.70 | 1489.24 | 1487.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 1480.90 | 1487.29 | 1486.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 13:00:00 | 1480.90 | 1487.29 | 1486.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 1477.35 | 1485.31 | 1485.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 14:15:00 | 1466.25 | 1481.49 | 1484.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 1440.00 | 1439.18 | 1454.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 10:15:00 | 1443.90 | 1438.55 | 1445.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 1443.90 | 1438.55 | 1445.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 1443.95 | 1438.55 | 1445.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 1445.25 | 1439.89 | 1445.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:00:00 | 1445.25 | 1439.89 | 1445.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 1451.05 | 1442.12 | 1446.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:45:00 | 1453.00 | 1442.12 | 1446.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 1441.00 | 1441.90 | 1445.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 14:15:00 | 1439.50 | 1441.90 | 1445.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 13:15:00 | 1451.80 | 1447.26 | 1446.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 13:15:00 | 1451.80 | 1447.26 | 1446.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 14:15:00 | 1439.65 | 1445.74 | 1446.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 1432.05 | 1442.38 | 1444.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 10:15:00 | 1436.00 | 1435.64 | 1439.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-20 11:00:00 | 1436.00 | 1435.64 | 1439.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1379.25 | 1369.00 | 1381.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 10:30:00 | 1363.10 | 1372.37 | 1376.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 12:45:00 | 1363.40 | 1367.51 | 1371.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 11:15:00 | 1361.95 | 1363.33 | 1367.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 1383.30 | 1370.32 | 1369.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 1383.30 | 1370.32 | 1369.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 1389.70 | 1378.65 | 1373.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 1399.75 | 1401.28 | 1395.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 11:00:00 | 1399.75 | 1401.28 | 1395.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 1395.00 | 1400.02 | 1396.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:00:00 | 1395.00 | 1400.02 | 1396.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 1392.00 | 1398.42 | 1396.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:45:00 | 1387.35 | 1398.42 | 1396.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 1384.30 | 1394.33 | 1394.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 1382.00 | 1391.86 | 1393.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 1389.00 | 1374.28 | 1378.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 1389.00 | 1374.28 | 1378.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1389.00 | 1374.28 | 1378.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:45:00 | 1388.75 | 1374.28 | 1378.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 1375.80 | 1375.72 | 1378.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 14:45:00 | 1373.80 | 1375.68 | 1377.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 1401.20 | 1380.47 | 1379.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1401.20 | 1380.47 | 1379.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 1412.05 | 1396.44 | 1388.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 13:15:00 | 1438.30 | 1438.34 | 1424.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 13:45:00 | 1436.35 | 1438.34 | 1424.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1441.80 | 1439.92 | 1436.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 10:15:00 | 1445.50 | 1439.92 | 1436.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 12:30:00 | 1446.05 | 1442.79 | 1438.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 13:30:00 | 1445.35 | 1443.43 | 1439.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 12:15:00 | 1439.35 | 1445.86 | 1446.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1439.35 | 1445.86 | 1446.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 13:15:00 | 1437.35 | 1444.16 | 1445.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 1443.20 | 1437.10 | 1439.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 1443.20 | 1437.10 | 1439.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1443.20 | 1437.10 | 1439.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 1443.20 | 1437.10 | 1439.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1443.00 | 1438.28 | 1440.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 1455.50 | 1438.28 | 1440.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 1459.00 | 1442.42 | 1441.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 14:15:00 | 1460.00 | 1452.88 | 1447.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 1452.00 | 1453.71 | 1449.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 09:45:00 | 1452.35 | 1453.71 | 1449.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 1450.85 | 1454.34 | 1451.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:00:00 | 1450.85 | 1454.34 | 1451.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 1455.65 | 1454.60 | 1451.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1465.15 | 1451.75 | 1451.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 12:30:00 | 1459.15 | 1457.34 | 1454.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 13:15:00 | 1460.75 | 1457.34 | 1454.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 1446.50 | 1455.14 | 1455.10 | SL hit (close<static) qty=1.00 sl=1450.70 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 1446.20 | 1453.35 | 1454.29 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 1462.00 | 1454.78 | 1454.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 14:15:00 | 1475.50 | 1464.75 | 1459.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 1464.85 | 1466.41 | 1461.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 10:15:00 | 1464.45 | 1466.41 | 1461.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 1469.60 | 1467.05 | 1462.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:30:00 | 1473.15 | 1468.65 | 1465.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 14:45:00 | 1470.10 | 1479.99 | 1479.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 15:15:00 | 1477.00 | 1479.39 | 1479.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 1477.00 | 1479.39 | 1479.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 1454.15 | 1474.34 | 1477.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 1483.15 | 1458.62 | 1464.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 1483.15 | 1458.62 | 1464.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1483.15 | 1458.62 | 1464.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:30:00 | 1480.95 | 1458.62 | 1464.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 1488.50 | 1464.59 | 1466.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 1493.70 | 1464.59 | 1466.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 1489.75 | 1469.62 | 1468.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 1496.90 | 1478.40 | 1473.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 15:15:00 | 1563.00 | 1564.78 | 1544.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:15:00 | 1567.35 | 1564.78 | 1544.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1548.20 | 1561.47 | 1545.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 1548.20 | 1561.47 | 1545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1547.25 | 1558.62 | 1545.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 1547.40 | 1558.62 | 1545.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 1552.55 | 1557.41 | 1545.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:30:00 | 1558.65 | 1558.27 | 1547.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 1575.00 | 1557.58 | 1549.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 1555.95 | 1562.85 | 1556.23 | SL hit (close<ema400) qty=1.00 sl=1556.23 alert=retest1 |

### Cycle 40 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 1539.85 | 1550.06 | 1551.34 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 1560.70 | 1546.15 | 1545.93 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 12:15:00 | 1544.40 | 1546.17 | 1546.24 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 1561.05 | 1548.44 | 1547.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 14:15:00 | 1568.40 | 1557.15 | 1552.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 11:15:00 | 1558.15 | 1559.72 | 1555.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 11:15:00 | 1558.15 | 1559.72 | 1555.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 1558.15 | 1559.72 | 1555.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 12:00:00 | 1558.15 | 1559.72 | 1555.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 1562.35 | 1560.80 | 1557.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:45:00 | 1558.65 | 1560.80 | 1557.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1548.00 | 1559.07 | 1556.96 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 11:15:00 | 1545.00 | 1553.85 | 1554.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 14:15:00 | 1540.90 | 1548.44 | 1551.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-01 10:15:00 | 1547.40 | 1546.24 | 1549.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-01 11:00:00 | 1547.40 | 1546.24 | 1549.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 1545.10 | 1546.01 | 1549.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 1539.75 | 1548.71 | 1549.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 11:15:00 | 1524.60 | 1519.15 | 1518.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 1524.60 | 1519.15 | 1518.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 1547.75 | 1527.46 | 1524.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 13:15:00 | 1533.85 | 1535.97 | 1530.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 14:00:00 | 1533.85 | 1535.97 | 1530.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 1527.35 | 1534.24 | 1529.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 15:00:00 | 1527.35 | 1534.24 | 1529.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 1530.05 | 1533.40 | 1529.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 1524.10 | 1533.40 | 1529.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1522.70 | 1531.26 | 1529.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:00:00 | 1522.70 | 1531.26 | 1529.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 1524.90 | 1529.99 | 1528.91 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 1517.55 | 1527.50 | 1527.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 12:15:00 | 1512.45 | 1524.49 | 1526.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 15:15:00 | 1520.30 | 1506.11 | 1513.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 15:15:00 | 1520.30 | 1506.11 | 1513.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 1520.30 | 1506.11 | 1513.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 1580.90 | 1506.11 | 1513.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 1598.40 | 1524.57 | 1520.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 11:15:00 | 1610.85 | 1554.02 | 1535.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 1636.95 | 1638.52 | 1614.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1641.85 | 1635.90 | 1620.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1641.85 | 1635.90 | 1620.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:30:00 | 1644.35 | 1637.51 | 1624.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 12:15:00 | 1644.40 | 1637.51 | 1624.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 15:15:00 | 1646.65 | 1640.70 | 1634.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:15:00 | 1660.10 | 1652.21 | 1651.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 1649.95 | 1652.63 | 1651.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:45:00 | 1649.10 | 1652.63 | 1651.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 1650.60 | 1652.22 | 1651.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:30:00 | 1647.50 | 1652.22 | 1651.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1657.05 | 1653.19 | 1652.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:30:00 | 1645.15 | 1653.19 | 1652.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1658.50 | 1654.25 | 1652.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:30:00 | 1655.70 | 1654.25 | 1652.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1666.00 | 1665.74 | 1661.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 1667.55 | 1665.74 | 1661.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1657.15 | 1664.02 | 1661.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:45:00 | 1662.95 | 1664.02 | 1661.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 1651.55 | 1661.53 | 1660.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:00:00 | 1651.55 | 1661.53 | 1660.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-29 13:15:00 | 1655.80 | 1659.35 | 1659.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 13:15:00 | 1655.80 | 1659.35 | 1659.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 14:15:00 | 1655.00 | 1658.48 | 1659.16 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 1685.65 | 1663.24 | 1661.16 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 1653.50 | 1661.62 | 1661.92 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 09:15:00 | 1672.20 | 1661.80 | 1661.40 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 1655.20 | 1660.59 | 1661.11 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 1694.50 | 1666.48 | 1663.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 10:15:00 | 1699.60 | 1673.10 | 1666.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 1684.90 | 1691.22 | 1684.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 14:15:00 | 1684.90 | 1691.22 | 1684.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1684.90 | 1691.22 | 1684.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 1685.30 | 1691.22 | 1684.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 1683.00 | 1689.58 | 1684.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 1694.70 | 1689.58 | 1684.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1714.70 | 1694.60 | 1687.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 12:00:00 | 1720.80 | 1703.04 | 1692.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 1695.30 | 1699.73 | 1700.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 1695.30 | 1699.73 | 1700.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 1684.85 | 1696.75 | 1698.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 1682.10 | 1675.48 | 1682.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 1682.10 | 1675.48 | 1682.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1682.10 | 1675.48 | 1682.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:30:00 | 1687.60 | 1675.48 | 1682.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 1681.40 | 1676.67 | 1682.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 11:15:00 | 1684.90 | 1676.67 | 1682.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 1685.10 | 1678.35 | 1682.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 12:00:00 | 1685.10 | 1678.35 | 1682.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 1690.30 | 1680.74 | 1683.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 1690.30 | 1680.74 | 1683.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1674.50 | 1679.52 | 1682.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 1645.50 | 1679.16 | 1680.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 12:30:00 | 1662.00 | 1666.51 | 1668.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 14:15:00 | 1676.30 | 1669.70 | 1669.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 1676.30 | 1669.70 | 1669.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 15:15:00 | 1680.00 | 1671.76 | 1670.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 15:15:00 | 1699.00 | 1699.20 | 1692.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 09:15:00 | 1692.50 | 1699.20 | 1692.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1679.70 | 1695.30 | 1691.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 1679.70 | 1695.30 | 1691.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1673.80 | 1691.00 | 1689.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 1673.80 | 1691.00 | 1689.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 1681.00 | 1687.78 | 1688.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 1671.90 | 1684.60 | 1686.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 14:15:00 | 1687.40 | 1685.16 | 1686.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 14:15:00 | 1687.40 | 1685.16 | 1686.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 1687.40 | 1685.16 | 1686.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:00:00 | 1687.40 | 1685.16 | 1686.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 1688.80 | 1685.89 | 1687.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 1665.15 | 1685.89 | 1687.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 10:15:00 | 1682.35 | 1672.46 | 1671.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 1682.35 | 1672.46 | 1671.56 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 1661.25 | 1672.11 | 1672.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 1655.40 | 1668.77 | 1670.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 14:15:00 | 1661.70 | 1661.49 | 1665.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 1656.25 | 1660.20 | 1664.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1656.25 | 1660.20 | 1664.52 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 1679.50 | 1665.05 | 1664.60 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 13:15:00 | 1660.00 | 1666.13 | 1666.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 14:15:00 | 1652.90 | 1663.48 | 1665.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 1662.00 | 1661.64 | 1664.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 1662.00 | 1661.64 | 1664.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1662.00 | 1661.64 | 1664.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-02 12:00:00 | 1655.55 | 1660.42 | 1663.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 11:15:00 | 1623.10 | 1610.36 | 1610.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 11:15:00 | 1623.10 | 1610.36 | 1610.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 09:15:00 | 1636.75 | 1617.30 | 1614.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 1626.00 | 1637.87 | 1628.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 1626.00 | 1637.87 | 1628.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1626.00 | 1637.87 | 1628.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 10:00:00 | 1626.00 | 1637.87 | 1628.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 1627.35 | 1635.77 | 1628.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 10:45:00 | 1623.55 | 1635.77 | 1628.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 11:15:00 | 1630.70 | 1634.75 | 1629.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:30:00 | 1628.15 | 1634.75 | 1629.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 1634.25 | 1634.65 | 1629.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:00:00 | 1634.25 | 1634.65 | 1629.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 1632.00 | 1634.12 | 1629.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:45:00 | 1628.95 | 1634.12 | 1629.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1637.35 | 1634.77 | 1630.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:15:00 | 1631.45 | 1634.77 | 1630.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1631.45 | 1634.10 | 1630.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:15:00 | 1623.65 | 1634.10 | 1630.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1620.75 | 1631.43 | 1629.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:00:00 | 1620.75 | 1631.43 | 1629.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 1617.70 | 1628.69 | 1628.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:30:00 | 1618.20 | 1628.69 | 1628.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 11:15:00 | 1617.30 | 1626.41 | 1627.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 14:15:00 | 1603.70 | 1619.97 | 1624.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 1504.50 | 1494.63 | 1505.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 1504.50 | 1494.63 | 1505.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1504.50 | 1494.63 | 1505.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 1504.50 | 1494.63 | 1505.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1505.90 | 1496.89 | 1505.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:30:00 | 1509.80 | 1496.89 | 1505.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 1504.35 | 1498.38 | 1505.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:30:00 | 1507.35 | 1498.38 | 1505.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 1508.00 | 1500.30 | 1505.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:45:00 | 1507.45 | 1500.30 | 1505.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 1504.95 | 1501.23 | 1505.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 13:45:00 | 1505.85 | 1501.23 | 1505.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1490.40 | 1499.07 | 1503.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:45:00 | 1505.70 | 1499.07 | 1503.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 1509.05 | 1500.45 | 1503.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 10:15:00 | 1507.75 | 1500.45 | 1503.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 09:15:00 | 1507.25 | 1484.90 | 1483.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 09:15:00 | 1507.25 | 1484.90 | 1483.95 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 1485.65 | 1493.57 | 1494.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 1475.35 | 1488.53 | 1491.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 11:15:00 | 1436.00 | 1429.74 | 1446.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 12:00:00 | 1436.00 | 1429.74 | 1446.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1411.15 | 1412.66 | 1425.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 1416.05 | 1412.66 | 1425.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1420.00 | 1414.02 | 1423.68 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 1440.25 | 1427.79 | 1426.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 13:15:00 | 1445.20 | 1435.99 | 1431.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 09:15:00 | 1435.00 | 1437.36 | 1433.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 10:00:00 | 1435.00 | 1437.36 | 1433.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 1436.00 | 1437.09 | 1433.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:45:00 | 1437.60 | 1436.68 | 1434.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 10:45:00 | 1439.30 | 1437.84 | 1436.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:30:00 | 1437.20 | 1437.53 | 1436.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 14:15:00 | 1430.05 | 1435.10 | 1435.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 1430.05 | 1435.10 | 1435.26 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 12:15:00 | 1438.05 | 1435.03 | 1435.02 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 09:15:00 | 1432.10 | 1434.90 | 1435.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 1419.40 | 1430.10 | 1432.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 1415.70 | 1413.99 | 1419.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-03 15:00:00 | 1415.70 | 1413.99 | 1419.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 1433.10 | 1418.09 | 1420.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:45:00 | 1439.75 | 1418.09 | 1420.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 10:15:00 | 1435.05 | 1421.49 | 1421.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 11:15:00 | 1449.15 | 1435.02 | 1431.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 15:15:00 | 1434.70 | 1437.17 | 1434.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 15:15:00 | 1434.70 | 1437.17 | 1434.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 1434.70 | 1437.17 | 1434.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 1418.40 | 1437.17 | 1434.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1425.40 | 1434.81 | 1433.38 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1420.85 | 1432.02 | 1432.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1415.00 | 1423.94 | 1427.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 13:15:00 | 1421.40 | 1420.87 | 1424.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 14:00:00 | 1421.40 | 1420.87 | 1424.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1423.75 | 1421.95 | 1424.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 1426.10 | 1421.95 | 1424.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1419.15 | 1421.39 | 1423.97 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 1443.45 | 1426.52 | 1425.26 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 1434.70 | 1438.00 | 1438.39 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 1448.65 | 1439.74 | 1439.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 10:15:00 | 1451.85 | 1442.16 | 1440.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 10:15:00 | 1467.30 | 1467.38 | 1459.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:45:00 | 1466.15 | 1467.38 | 1459.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1462.60 | 1466.02 | 1462.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 1470.25 | 1466.16 | 1462.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 1469.70 | 1466.87 | 1463.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 1459.15 | 1467.30 | 1467.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 1459.15 | 1467.30 | 1467.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 1456.00 | 1465.04 | 1466.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1434.20 | 1429.15 | 1438.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 1434.20 | 1429.15 | 1438.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1409.20 | 1418.55 | 1430.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1387.60 | 1410.49 | 1420.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 1405.55 | 1406.24 | 1416.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:30:00 | 1401.20 | 1401.29 | 1413.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 11:15:00 | 1437.40 | 1413.54 | 1413.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1437.40 | 1413.54 | 1413.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1438.25 | 1425.82 | 1419.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1502.85 | 1512.46 | 1485.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 1502.85 | 1512.46 | 1485.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1495.20 | 1501.50 | 1496.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 1495.20 | 1501.50 | 1496.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1495.90 | 1500.38 | 1496.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1504.00 | 1500.38 | 1496.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 12:15:00 | 1497.05 | 1498.35 | 1496.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 1492.90 | 1497.26 | 1495.86 | SL hit (close<static) qty=1.00 sl=1493.95 alert=retest2 |

### Cycle 76 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 1485.30 | 1493.35 | 1494.23 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1500.25 | 1493.51 | 1493.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 11:15:00 | 1503.00 | 1495.41 | 1494.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 1506.55 | 1506.67 | 1502.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1506.55 | 1506.67 | 1502.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1506.55 | 1506.67 | 1502.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1548.20 | 1511.02 | 1506.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 12:15:00 | 1646.55 | 1652.88 | 1652.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 1646.55 | 1652.88 | 1652.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 1640.70 | 1650.45 | 1651.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 1650.30 | 1649.48 | 1650.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1650.30 | 1649.48 | 1650.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1650.30 | 1649.48 | 1650.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 1659.00 | 1649.48 | 1650.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1653.60 | 1650.31 | 1651.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 1653.60 | 1650.31 | 1651.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1653.55 | 1650.95 | 1651.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:15:00 | 1657.45 | 1650.95 | 1651.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1653.10 | 1651.38 | 1651.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 1658.50 | 1651.38 | 1651.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 1652.85 | 1651.80 | 1651.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 1688.00 | 1659.07 | 1655.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 1707.15 | 1709.48 | 1694.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:45:00 | 1705.95 | 1709.48 | 1694.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1805.40 | 1808.00 | 1794.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1804.40 | 1808.00 | 1794.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1824.00 | 1828.15 | 1817.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:45:00 | 1829.15 | 1826.40 | 1819.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1858.50 | 1825.77 | 1820.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 1857.95 | 1865.45 | 1865.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 1857.95 | 1865.45 | 1865.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 1851.70 | 1860.00 | 1863.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1785.45 | 1770.25 | 1796.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1785.45 | 1770.25 | 1796.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1785.45 | 1770.25 | 1796.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 1788.15 | 1770.25 | 1796.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1784.05 | 1763.22 | 1778.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:15:00 | 1777.30 | 1766.80 | 1778.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:00:00 | 1773.95 | 1768.23 | 1778.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1766.10 | 1777.96 | 1780.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 1773.30 | 1766.40 | 1766.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 1773.30 | 1766.40 | 1766.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 1791.05 | 1771.33 | 1768.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 1796.25 | 1796.57 | 1788.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:45:00 | 1796.50 | 1796.57 | 1788.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1877.50 | 1876.15 | 1870.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 1870.85 | 1876.15 | 1870.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1865.50 | 1875.18 | 1871.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 1865.30 | 1875.18 | 1871.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1864.90 | 1873.13 | 1871.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 1864.90 | 1873.13 | 1871.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1865.50 | 1871.60 | 1870.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 1865.30 | 1871.60 | 1870.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1861.80 | 1868.38 | 1869.21 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1884.45 | 1871.21 | 1870.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 1896.95 | 1881.04 | 1876.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 13:15:00 | 1928.15 | 1934.65 | 1921.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 13:15:00 | 1928.15 | 1934.65 | 1921.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 1928.15 | 1934.65 | 1921.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 1928.15 | 1934.65 | 1921.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1941.80 | 1935.43 | 1928.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 1929.05 | 1935.43 | 1928.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1958.60 | 1957.21 | 1946.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 1951.50 | 1957.21 | 1946.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1940.80 | 1953.38 | 1948.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1940.80 | 1953.38 | 1948.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1940.90 | 1950.88 | 1948.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1908.15 | 1950.88 | 1948.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 1914.55 | 1943.62 | 1945.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1900.50 | 1909.85 | 1918.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1918.40 | 1910.30 | 1916.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 1918.40 | 1910.30 | 1916.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1918.40 | 1910.30 | 1916.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 1918.40 | 1910.30 | 1916.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1916.40 | 1911.52 | 1916.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:45:00 | 1913.50 | 1912.24 | 1916.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 14:15:00 | 1905.55 | 1912.62 | 1915.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 1922.80 | 1910.71 | 1913.36 | SL hit (close>static) qty=1.00 sl=1920.55 alert=retest2 |

### Cycle 85 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 1933.75 | 1918.09 | 1916.42 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 1910.00 | 1916.14 | 1916.15 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 1923.00 | 1916.89 | 1916.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 1924.50 | 1918.41 | 1917.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1913.05 | 1917.34 | 1916.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 1913.05 | 1917.34 | 1916.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1913.05 | 1917.34 | 1916.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1913.05 | 1917.34 | 1916.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1907.25 | 1915.32 | 1915.94 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 1922.90 | 1916.46 | 1916.33 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 1913.85 | 1916.00 | 1916.16 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 1932.60 | 1919.32 | 1917.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 1955.50 | 1926.55 | 1921.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 1943.50 | 1945.29 | 1935.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 1943.50 | 1945.29 | 1935.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1939.10 | 1943.35 | 1936.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 1945.05 | 1942.18 | 1937.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 1898.55 | 1939.92 | 1941.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 1898.55 | 1939.92 | 1941.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 1895.25 | 1930.99 | 1937.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1895.75 | 1894.52 | 1907.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 1895.75 | 1894.52 | 1907.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1892.70 | 1894.39 | 1905.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 1872.25 | 1898.28 | 1904.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 15:15:00 | 1910.00 | 1901.36 | 1904.52 | SL hit (close>static) qty=1.00 sl=1908.95 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 1914.50 | 1896.43 | 1894.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 1931.20 | 1906.40 | 1901.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 13:15:00 | 1914.00 | 1915.71 | 1908.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 14:00:00 | 1914.00 | 1915.71 | 1908.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1906.55 | 1913.88 | 1907.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1906.55 | 1913.88 | 1907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1908.00 | 1912.70 | 1907.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 1897.00 | 1912.70 | 1907.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1891.30 | 1908.42 | 1906.41 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1884.05 | 1901.05 | 1903.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1878.20 | 1893.29 | 1899.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1899.95 | 1889.92 | 1895.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1899.95 | 1889.92 | 1895.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1899.95 | 1889.92 | 1895.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 1899.95 | 1889.92 | 1895.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1885.90 | 1889.12 | 1894.82 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 15:15:00 | 1906.00 | 1898.82 | 1898.04 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1883.95 | 1895.30 | 1896.65 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 1907.55 | 1896.98 | 1896.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 1938.40 | 1905.26 | 1900.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 1918.35 | 1918.60 | 1909.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 1918.35 | 1918.60 | 1909.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1924.95 | 1928.07 | 1921.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:30:00 | 1914.85 | 1928.07 | 1921.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 1939.00 | 1930.03 | 1923.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:45:00 | 1925.80 | 1930.03 | 1923.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1952.50 | 1955.45 | 1945.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 1943.15 | 1955.45 | 1945.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1943.05 | 1952.97 | 1945.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:45:00 | 1948.10 | 1952.97 | 1945.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1940.65 | 1950.50 | 1945.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:45:00 | 1942.95 | 1950.50 | 1945.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 1919.10 | 1938.91 | 1940.78 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1952.95 | 1940.04 | 1938.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 1963.00 | 1944.63 | 1941.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 11:15:00 | 1955.05 | 1957.63 | 1951.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 12:00:00 | 1955.05 | 1957.63 | 1951.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1957.90 | 1958.01 | 1952.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 1957.90 | 1958.01 | 1952.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1957.45 | 1958.22 | 1953.87 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 1944.10 | 1950.15 | 1950.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 1920.55 | 1942.58 | 1947.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 09:15:00 | 1946.70 | 1939.49 | 1944.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1946.70 | 1939.49 | 1944.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1946.70 | 1939.49 | 1944.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 1954.70 | 1939.49 | 1944.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1939.75 | 1939.54 | 1944.28 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 12:15:00 | 1974.95 | 1948.86 | 1947.83 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 1885.40 | 1944.26 | 1947.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 10:15:00 | 1875.80 | 1930.57 | 1940.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 12:15:00 | 1859.75 | 1859.17 | 1876.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 12:30:00 | 1861.25 | 1859.17 | 1876.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1874.80 | 1861.32 | 1870.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1874.80 | 1861.32 | 1870.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1880.10 | 1865.08 | 1871.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1880.70 | 1865.08 | 1871.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 1868.30 | 1870.28 | 1872.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 1872.35 | 1870.28 | 1872.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1865.10 | 1869.24 | 1871.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:30:00 | 1860.50 | 1866.71 | 1869.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:45:00 | 1859.95 | 1864.55 | 1867.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 1878.60 | 1867.42 | 1866.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 1878.60 | 1867.42 | 1866.99 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1837.40 | 1861.81 | 1864.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 09:15:00 | 1826.20 | 1841.27 | 1851.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 1764.40 | 1755.72 | 1772.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 14:45:00 | 1761.35 | 1755.72 | 1772.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1800.25 | 1763.24 | 1765.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 1800.25 | 1763.24 | 1765.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1810.95 | 1772.79 | 1769.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1818.85 | 1788.60 | 1778.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1796.00 | 1803.92 | 1790.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1796.00 | 1803.92 | 1790.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1793.95 | 1801.92 | 1790.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:45:00 | 1798.45 | 1801.07 | 1791.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 1817.80 | 1855.86 | 1858.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1817.80 | 1855.86 | 1858.04 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 1857.60 | 1840.19 | 1837.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 1883.20 | 1854.51 | 1845.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 14:15:00 | 1922.80 | 1924.15 | 1910.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:00:00 | 1922.80 | 1924.15 | 1910.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1903.55 | 1920.48 | 1911.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 1897.45 | 1920.48 | 1911.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1871.40 | 1910.66 | 1907.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1869.55 | 1910.66 | 1907.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 1871.80 | 1902.89 | 1904.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 1861.00 | 1894.51 | 1900.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 1865.80 | 1863.30 | 1873.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:15:00 | 1868.95 | 1863.30 | 1873.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1864.00 | 1863.44 | 1872.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 1870.05 | 1863.44 | 1872.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1870.90 | 1864.93 | 1872.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:45:00 | 1871.90 | 1864.93 | 1872.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1871.90 | 1866.32 | 1872.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 1871.90 | 1866.32 | 1872.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1880.00 | 1869.06 | 1872.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1880.00 | 1869.06 | 1872.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1877.45 | 1870.74 | 1873.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1875.85 | 1870.74 | 1873.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1893.30 | 1877.09 | 1875.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1895.65 | 1880.80 | 1877.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1888.15 | 1891.28 | 1885.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1888.15 | 1891.28 | 1885.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1888.15 | 1891.28 | 1885.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1884.10 | 1891.28 | 1885.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1896.05 | 1892.23 | 1886.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:30:00 | 1900.00 | 1893.53 | 1887.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 1911.25 | 1892.39 | 1888.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 1919.60 | 1899.00 | 1892.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 1970.35 | 1979.13 | 1979.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 1970.35 | 1979.13 | 1979.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1948.40 | 1970.32 | 1973.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1952.00 | 1951.27 | 1959.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 10:15:00 | 1952.00 | 1951.27 | 1959.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1952.00 | 1951.27 | 1959.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 1942.20 | 1951.27 | 1959.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 15:15:00 | 1845.09 | 1917.86 | 1926.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1920.90 | 1912.29 | 1917.91 | SL hit (close>ema200) qty=0.50 sl=1912.29 alert=retest2 |

### Cycle 111 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1935.70 | 1899.20 | 1895.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 1937.50 | 1906.86 | 1898.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 1934.95 | 1935.25 | 1918.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:45:00 | 1933.00 | 1935.25 | 1918.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1935.65 | 1939.17 | 1930.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1928.50 | 1939.17 | 1930.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 1948.25 | 1940.99 | 1932.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 1951.75 | 1939.98 | 1933.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 1928.55 | 1939.14 | 1935.10 | SL hit (close<static) qty=1.00 sl=1931.45 alert=retest2 |

### Cycle 112 — SELL (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 15:15:00 | 1929.00 | 1932.76 | 1932.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 1911.95 | 1928.60 | 1931.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1929.00 | 1919.55 | 1924.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 1929.00 | 1919.55 | 1924.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1929.00 | 1919.55 | 1924.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:45:00 | 1929.95 | 1919.55 | 1924.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1937.00 | 1923.04 | 1925.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1937.00 | 1923.04 | 1925.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1932.75 | 1924.98 | 1926.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1941.80 | 1924.98 | 1926.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1945.40 | 1929.07 | 1928.28 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 1917.70 | 1926.79 | 1927.45 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 1955.00 | 1931.84 | 1929.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 11:15:00 | 1970.00 | 1939.47 | 1932.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 15:15:00 | 1963.00 | 1963.94 | 1954.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:15:00 | 1941.05 | 1963.94 | 1954.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1949.00 | 1960.96 | 1954.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1935.70 | 1960.96 | 1954.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1957.85 | 1960.33 | 1954.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1957.85 | 1960.33 | 1954.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1945.05 | 1957.28 | 1953.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 1945.05 | 1957.28 | 1953.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1933.95 | 1952.61 | 1952.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 1933.95 | 1952.61 | 1952.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 1938.70 | 1949.83 | 1950.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 14:15:00 | 1921.05 | 1939.87 | 1944.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 11:15:00 | 1826.40 | 1818.57 | 1840.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-21 12:00:00 | 1826.40 | 1818.57 | 1840.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1827.90 | 1816.73 | 1830.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1828.80 | 1816.73 | 1830.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1834.00 | 1820.18 | 1831.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:45:00 | 1832.00 | 1820.18 | 1831.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1846.30 | 1825.41 | 1832.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1846.30 | 1825.41 | 1832.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 13:15:00 | 1861.50 | 1836.79 | 1836.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1868.65 | 1848.67 | 1842.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 1875.00 | 1876.77 | 1866.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:45:00 | 1874.20 | 1876.77 | 1866.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1857.85 | 1872.70 | 1866.25 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 1834.35 | 1856.84 | 1859.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 1819.60 | 1849.39 | 1856.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1845.00 | 1840.48 | 1849.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 1845.00 | 1840.48 | 1849.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1845.00 | 1840.48 | 1849.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:30:00 | 1859.60 | 1840.48 | 1849.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 1847.40 | 1841.87 | 1849.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 1847.40 | 1841.87 | 1849.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1849.55 | 1843.40 | 1849.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:45:00 | 1848.60 | 1843.40 | 1849.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1841.25 | 1842.97 | 1848.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 13:15:00 | 1835.65 | 1842.97 | 1848.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 1828.80 | 1839.81 | 1846.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1855.30 | 1841.50 | 1845.81 | SL hit (close>static) qty=1.00 sl=1852.35 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1859.85 | 1848.77 | 1848.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1886.40 | 1856.30 | 1851.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 1862.95 | 1865.01 | 1858.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 10:15:00 | 1862.95 | 1865.01 | 1858.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1862.95 | 1865.01 | 1858.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 1862.95 | 1865.01 | 1858.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1864.50 | 1864.91 | 1858.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 1858.05 | 1864.91 | 1858.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1848.15 | 1861.01 | 1858.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1848.15 | 1861.01 | 1858.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1860.20 | 1860.85 | 1858.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 1847.45 | 1860.85 | 1858.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1868.55 | 1877.85 | 1872.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 1868.55 | 1877.85 | 1872.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1857.00 | 1873.68 | 1870.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 1857.50 | 1873.68 | 1870.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1858.50 | 1870.64 | 1869.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 1859.40 | 1870.64 | 1869.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 1847.95 | 1866.11 | 1867.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1842.75 | 1860.94 | 1865.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 1871.40 | 1859.09 | 1862.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 1871.40 | 1859.09 | 1862.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 1871.40 | 1859.09 | 1862.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 1871.40 | 1859.09 | 1862.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 1868.45 | 1860.97 | 1863.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 14:30:00 | 1859.95 | 1861.53 | 1863.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1894.60 | 1868.56 | 1866.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 1894.60 | 1868.56 | 1866.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 14:15:00 | 1914.70 | 1905.87 | 1897.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 1905.20 | 1907.00 | 1899.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1905.20 | 1907.00 | 1899.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1905.20 | 1907.00 | 1899.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 1899.40 | 1907.00 | 1899.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1899.65 | 1904.50 | 1900.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 1899.65 | 1904.50 | 1900.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1893.15 | 1902.23 | 1899.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 1893.15 | 1902.23 | 1899.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1905.30 | 1902.85 | 1899.96 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1894.65 | 1898.47 | 1898.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1883.75 | 1895.53 | 1897.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 09:15:00 | 1896.00 | 1888.81 | 1892.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 1896.00 | 1888.81 | 1892.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1896.00 | 1888.81 | 1892.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 1901.45 | 1888.81 | 1892.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 1897.05 | 1890.46 | 1892.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:30:00 | 1898.60 | 1890.46 | 1892.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 1889.85 | 1890.34 | 1892.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:45:00 | 1885.95 | 1886.60 | 1890.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:45:00 | 1883.15 | 1881.10 | 1886.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 1791.65 | 1808.88 | 1818.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 1788.99 | 1808.88 | 1818.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 1776.85 | 1774.59 | 1792.22 | SL hit (close>ema200) qty=0.50 sl=1774.59 alert=retest2 |

### Cycle 123 — BUY (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 13:15:00 | 1710.95 | 1709.60 | 1709.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 14:15:00 | 1717.20 | 1711.12 | 1710.28 | Break + close above crossover candle high |

### Cycle 124 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 1697.60 | 1708.56 | 1709.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 10:15:00 | 1685.50 | 1703.95 | 1707.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 1700.35 | 1692.13 | 1698.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1700.35 | 1692.13 | 1698.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1700.35 | 1692.13 | 1698.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 1707.75 | 1692.13 | 1698.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1714.05 | 1696.51 | 1699.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 1712.55 | 1696.51 | 1699.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1714.40 | 1700.09 | 1700.94 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 12:15:00 | 1713.80 | 1702.83 | 1702.11 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1651.55 | 1692.30 | 1697.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1585.80 | 1651.34 | 1671.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1585.80 | 1584.29 | 1604.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 1585.80 | 1584.29 | 1604.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1607.90 | 1592.00 | 1598.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1607.90 | 1592.00 | 1598.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1607.30 | 1595.06 | 1599.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 1608.65 | 1595.06 | 1599.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1598.55 | 1597.76 | 1599.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 1598.55 | 1597.76 | 1599.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 1610.35 | 1600.27 | 1600.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 1610.35 | 1600.27 | 1600.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 1608.70 | 1601.96 | 1601.52 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 09:15:00 | 1580.05 | 1597.58 | 1599.57 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 1610.55 | 1595.25 | 1595.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 12:15:00 | 1617.95 | 1602.20 | 1598.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 1592.90 | 1604.88 | 1601.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 1592.90 | 1604.88 | 1601.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1592.90 | 1604.88 | 1601.54 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 12:15:00 | 1595.25 | 1599.27 | 1599.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 09:15:00 | 1577.25 | 1592.65 | 1596.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 13:15:00 | 1606.05 | 1592.76 | 1594.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 13:15:00 | 1606.05 | 1592.76 | 1594.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 1606.05 | 1592.76 | 1594.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 1606.05 | 1592.76 | 1594.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1593.40 | 1592.89 | 1594.56 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 09:15:00 | 1622.00 | 1598.88 | 1597.00 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1595.60 | 1606.51 | 1607.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 1575.15 | 1598.99 | 1603.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1542.70 | 1539.66 | 1558.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1512.50 | 1546.13 | 1554.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1361.25 | 1448.53 | 1481.85 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 133 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 1424.10 | 1419.78 | 1419.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 1430.40 | 1421.91 | 1420.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 1416.10 | 1420.74 | 1420.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 1416.10 | 1420.74 | 1420.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 1416.10 | 1420.74 | 1420.11 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 10:15:00 | 1406.00 | 1417.80 | 1418.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1380.80 | 1407.58 | 1413.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 12:15:00 | 1414.80 | 1405.03 | 1410.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 12:15:00 | 1414.80 | 1405.03 | 1410.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1414.80 | 1405.03 | 1410.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 1414.80 | 1405.03 | 1410.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1415.70 | 1407.16 | 1410.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:30:00 | 1418.10 | 1407.16 | 1410.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1418.70 | 1409.47 | 1411.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1418.70 | 1409.47 | 1411.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 1427.70 | 1413.12 | 1412.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 1449.60 | 1420.41 | 1416.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 1426.10 | 1441.48 | 1432.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 1426.10 | 1441.48 | 1432.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1426.10 | 1441.48 | 1432.42 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 15:15:00 | 1423.00 | 1427.89 | 1428.49 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 09:15:00 | 1462.30 | 1434.77 | 1431.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 10:15:00 | 1480.90 | 1444.00 | 1436.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 13:15:00 | 1465.90 | 1469.43 | 1459.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 14:00:00 | 1465.90 | 1469.43 | 1459.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1476.40 | 1480.07 | 1472.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 1469.60 | 1480.07 | 1472.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1472.80 | 1478.61 | 1472.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 1471.10 | 1478.61 | 1472.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 1478.70 | 1478.63 | 1473.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 12:15:00 | 1483.50 | 1478.63 | 1473.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 1481.60 | 1479.49 | 1475.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:45:00 | 1480.40 | 1479.90 | 1476.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 11:00:00 | 1481.40 | 1480.20 | 1476.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1491.50 | 1496.32 | 1491.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1511.60 | 1496.32 | 1491.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 11:00:00 | 1507.10 | 1502.12 | 1494.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 11:45:00 | 1504.00 | 1501.82 | 1495.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:30:00 | 1504.90 | 1502.43 | 1496.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1507.40 | 1507.26 | 1503.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 1501.30 | 1507.26 | 1503.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1502.50 | 1508.43 | 1506.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 1510.60 | 1508.12 | 1506.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 1511.00 | 1508.12 | 1506.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 15:00:00 | 1511.70 | 1508.33 | 1507.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1500.70 | 1506.43 | 1507.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1500.70 | 1506.43 | 1507.05 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1567.80 | 1519.19 | 1512.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1573.70 | 1536.99 | 1522.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1585.20 | 1586.68 | 1559.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 1582.70 | 1586.68 | 1559.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1574.40 | 1584.96 | 1576.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 1574.40 | 1584.96 | 1576.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1575.00 | 1582.97 | 1576.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1598.60 | 1585.06 | 1578.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:15:00 | 1590.10 | 1592.80 | 1587.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 1590.20 | 1591.55 | 1587.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 1572.00 | 1587.29 | 1586.51 | SL hit (close<static) qty=1.00 sl=1572.50 alert=retest2 |

### Cycle 140 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 1567.50 | 1583.33 | 1584.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 11:15:00 | 1564.20 | 1579.51 | 1582.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 1576.50 | 1570.55 | 1576.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1576.50 | 1570.55 | 1576.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1576.50 | 1570.55 | 1576.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1580.70 | 1570.55 | 1576.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1575.00 | 1571.44 | 1576.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 1576.50 | 1571.44 | 1576.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1569.00 | 1570.95 | 1575.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 12:45:00 | 1564.70 | 1569.82 | 1574.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1568.30 | 1566.41 | 1569.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1583.80 | 1559.70 | 1562.23 | SL hit (close>static) qty=1.00 sl=1575.70 alert=retest2 |

### Cycle 141 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1585.50 | 1564.86 | 1564.34 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 1565.40 | 1570.45 | 1570.72 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1575.40 | 1571.33 | 1571.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1592.40 | 1577.22 | 1574.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1561.10 | 1577.87 | 1576.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1561.10 | 1577.87 | 1576.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1561.10 | 1577.87 | 1576.72 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 1561.90 | 1574.68 | 1575.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1545.00 | 1563.03 | 1568.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1549.60 | 1547.20 | 1553.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1549.60 | 1547.20 | 1553.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1549.60 | 1547.20 | 1553.21 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1557.50 | 1554.29 | 1553.88 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1550.00 | 1553.43 | 1553.53 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 1555.50 | 1553.84 | 1553.71 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 1548.00 | 1552.84 | 1553.29 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1557.90 | 1553.94 | 1553.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1562.50 | 1555.96 | 1554.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1607.60 | 1612.97 | 1599.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1607.60 | 1612.97 | 1599.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1607.60 | 1612.97 | 1599.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1601.50 | 1612.97 | 1599.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1605.00 | 1612.43 | 1605.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1587.80 | 1612.43 | 1605.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1594.80 | 1608.90 | 1604.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1604.70 | 1608.90 | 1604.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 1617.30 | 1625.18 | 1625.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1617.30 | 1625.18 | 1625.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 1613.00 | 1619.98 | 1622.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1620.00 | 1619.24 | 1621.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 1620.00 | 1619.24 | 1621.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1619.80 | 1619.23 | 1621.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 1619.60 | 1619.23 | 1621.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1623.50 | 1620.08 | 1621.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1623.50 | 1620.08 | 1621.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1618.00 | 1619.67 | 1621.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1589.80 | 1619.67 | 1621.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 1614.30 | 1602.66 | 1601.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 1614.30 | 1602.66 | 1601.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 1620.00 | 1612.47 | 1607.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 1614.90 | 1615.51 | 1611.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 11:30:00 | 1615.30 | 1615.51 | 1611.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1610.40 | 1614.05 | 1611.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 1610.80 | 1614.05 | 1611.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1609.70 | 1613.18 | 1611.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1609.70 | 1613.18 | 1611.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1611.00 | 1612.75 | 1611.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1603.20 | 1612.75 | 1611.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1600.20 | 1610.24 | 1610.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1596.00 | 1610.24 | 1610.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 1602.40 | 1608.67 | 1609.44 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 1641.90 | 1612.85 | 1609.76 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1620.80 | 1630.09 | 1630.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 1613.60 | 1622.99 | 1626.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1601.00 | 1580.07 | 1589.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 1601.00 | 1580.07 | 1589.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1601.00 | 1580.07 | 1589.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1601.00 | 1580.07 | 1589.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1590.70 | 1582.19 | 1589.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 1587.00 | 1582.19 | 1589.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 1584.80 | 1583.79 | 1588.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 1608.40 | 1593.21 | 1591.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 1608.40 | 1593.21 | 1591.64 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 1583.70 | 1592.74 | 1593.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 1581.00 | 1590.39 | 1592.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1588.20 | 1587.62 | 1589.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:15:00 | 1573.00 | 1587.62 | 1589.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1581.00 | 1584.06 | 1587.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1579.00 | 1583.84 | 1586.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1579.00 | 1584.19 | 1586.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1579.80 | 1582.24 | 1585.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 1579.10 | 1581.81 | 1584.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1580.30 | 1576.90 | 1580.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1585.90 | 1579.08 | 1580.96 | SL hit (close>ema400) qty=1.00 sl=1580.96 alert=retest1 |

### Cycle 157 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 1449.50 | 1431.59 | 1430.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 10:15:00 | 1467.80 | 1438.83 | 1433.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 10:15:00 | 1440.20 | 1444.87 | 1440.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 1440.20 | 1444.87 | 1440.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1440.20 | 1444.87 | 1440.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 1441.70 | 1444.87 | 1440.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1431.90 | 1442.28 | 1439.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 1431.90 | 1442.28 | 1439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1437.60 | 1441.34 | 1439.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 1431.60 | 1441.34 | 1439.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1435.20 | 1439.50 | 1438.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1435.20 | 1439.50 | 1438.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1436.00 | 1438.80 | 1438.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1442.00 | 1438.80 | 1438.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1443.10 | 1440.30 | 1439.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 1439.80 | 1440.30 | 1439.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1439.90 | 1441.85 | 1440.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 1439.00 | 1441.85 | 1440.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1442.50 | 1441.98 | 1440.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1462.90 | 1441.98 | 1440.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 1505.20 | 1512.04 | 1512.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1505.20 | 1512.04 | 1512.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1500.60 | 1509.75 | 1511.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1497.20 | 1484.72 | 1494.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1497.20 | 1484.72 | 1494.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1497.20 | 1484.72 | 1494.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1495.30 | 1484.72 | 1494.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1498.70 | 1487.51 | 1494.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 1491.70 | 1490.82 | 1494.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 1494.40 | 1490.82 | 1494.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1490.40 | 1495.35 | 1496.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1507.20 | 1498.26 | 1497.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1507.20 | 1498.26 | 1497.33 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 1480.00 | 1494.70 | 1496.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 1475.00 | 1490.76 | 1494.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1487.00 | 1447.63 | 1451.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1487.00 | 1447.63 | 1451.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1487.00 | 1447.63 | 1451.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1487.50 | 1447.63 | 1451.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1496.70 | 1457.44 | 1455.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 11:15:00 | 1498.20 | 1465.59 | 1459.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1509.60 | 1519.82 | 1502.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 1509.60 | 1519.82 | 1502.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1506.60 | 1520.58 | 1515.51 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 1507.30 | 1513.03 | 1513.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 1506.80 | 1511.20 | 1512.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1513.00 | 1508.97 | 1510.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 14:15:00 | 1513.00 | 1508.97 | 1510.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1513.00 | 1508.97 | 1510.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1513.00 | 1508.97 | 1510.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1513.50 | 1509.87 | 1510.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1522.40 | 1509.87 | 1510.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1516.60 | 1511.22 | 1511.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 1518.60 | 1511.22 | 1511.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1519.50 | 1512.87 | 1512.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1522.80 | 1514.86 | 1512.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1529.00 | 1536.02 | 1529.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1529.00 | 1536.02 | 1529.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1529.00 | 1536.02 | 1529.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1529.00 | 1536.02 | 1529.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1525.00 | 1533.82 | 1528.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 1523.70 | 1533.82 | 1528.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1521.80 | 1531.41 | 1528.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1521.80 | 1531.41 | 1528.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1540.50 | 1531.96 | 1529.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 1526.80 | 1531.96 | 1529.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1507.50 | 1528.84 | 1528.30 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1501.00 | 1523.27 | 1525.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 1495.10 | 1511.03 | 1519.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1504.50 | 1503.56 | 1511.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 1504.50 | 1503.56 | 1511.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1500.20 | 1495.20 | 1499.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 1502.60 | 1495.20 | 1499.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1497.10 | 1495.58 | 1499.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 1494.40 | 1494.99 | 1499.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1462.90 | 1449.21 | 1447.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 1462.90 | 1449.21 | 1447.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 1463.40 | 1452.04 | 1449.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 1460.10 | 1463.67 | 1458.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 1460.10 | 1463.67 | 1458.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1464.80 | 1463.90 | 1458.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 1468.90 | 1464.90 | 1459.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 1454.70 | 1462.86 | 1459.25 | SL hit (close<static) qty=1.00 sl=1457.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 1490.20 | 1496.13 | 1496.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 09:15:00 | 1473.70 | 1487.95 | 1491.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1472.10 | 1471.39 | 1477.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 1472.10 | 1471.39 | 1477.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1465.20 | 1451.07 | 1460.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 1465.20 | 1451.07 | 1460.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1460.70 | 1453.00 | 1460.24 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1471.00 | 1463.34 | 1462.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1532.90 | 1477.25 | 1468.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 1519.10 | 1522.21 | 1507.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:45:00 | 1519.30 | 1522.21 | 1507.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1515.00 | 1521.53 | 1511.13 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1498.70 | 1507.18 | 1507.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 1493.20 | 1504.38 | 1506.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1501.80 | 1499.87 | 1503.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 1501.80 | 1499.87 | 1503.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1502.20 | 1500.34 | 1503.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1506.30 | 1500.34 | 1503.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1504.60 | 1501.19 | 1503.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 1513.90 | 1501.19 | 1503.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1514.60 | 1503.87 | 1504.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1514.60 | 1503.87 | 1504.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1513.60 | 1505.82 | 1505.09 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 1498.50 | 1504.46 | 1505.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 1491.10 | 1499.16 | 1502.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1485.50 | 1482.99 | 1487.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 1485.50 | 1482.99 | 1487.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1486.00 | 1483.59 | 1487.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1476.40 | 1483.59 | 1487.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1503.60 | 1477.69 | 1475.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1503.60 | 1477.69 | 1475.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 10:15:00 | 1518.20 | 1485.79 | 1479.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1540.50 | 1544.87 | 1530.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1540.50 | 1544.87 | 1530.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1540.50 | 1544.87 | 1530.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1534.20 | 1544.87 | 1530.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1504.50 | 1537.13 | 1534.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1504.50 | 1537.13 | 1534.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1510.30 | 1531.77 | 1532.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1493.60 | 1519.45 | 1526.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 1506.70 | 1504.60 | 1512.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 1506.70 | 1504.60 | 1512.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1527.90 | 1499.09 | 1503.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1527.90 | 1499.09 | 1503.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 1537.70 | 1506.81 | 1506.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 15:15:00 | 1542.00 | 1528.97 | 1518.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1532.70 | 1535.32 | 1528.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1532.70 | 1535.32 | 1528.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1532.70 | 1535.32 | 1528.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 1543.80 | 1537.53 | 1530.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 1530.70 | 1542.70 | 1543.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 11:15:00 | 1530.70 | 1542.70 | 1543.02 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1550.00 | 1542.38 | 1541.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1553.90 | 1544.68 | 1542.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1555.40 | 1557.04 | 1550.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 1555.40 | 1557.04 | 1550.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1562.30 | 1561.18 | 1558.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1575.60 | 1566.15 | 1563.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1576.70 | 1570.12 | 1565.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:30:00 | 1580.00 | 1572.13 | 1567.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1596.90 | 1602.54 | 1603.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 1596.90 | 1602.54 | 1603.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 1593.40 | 1596.78 | 1599.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1594.70 | 1591.81 | 1595.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1594.70 | 1591.81 | 1595.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1594.70 | 1591.81 | 1595.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 11:00:00 | 1582.60 | 1589.97 | 1594.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 1583.90 | 1590.71 | 1592.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1597.50 | 1594.31 | 1593.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1597.50 | 1594.31 | 1593.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1600.70 | 1595.59 | 1594.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1589.40 | 1600.67 | 1598.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1589.40 | 1600.67 | 1598.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1589.40 | 1600.67 | 1598.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1590.10 | 1600.67 | 1598.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1587.20 | 1597.98 | 1597.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1587.20 | 1597.98 | 1597.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1588.90 | 1596.16 | 1596.83 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 1599.70 | 1596.30 | 1596.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 1604.40 | 1597.92 | 1596.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1662.60 | 1671.64 | 1654.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1662.60 | 1667.17 | 1660.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1662.60 | 1667.17 | 1660.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 1655.50 | 1667.17 | 1660.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1659.70 | 1665.67 | 1660.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1659.70 | 1665.67 | 1660.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1659.60 | 1664.46 | 1660.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1659.10 | 1664.46 | 1660.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1660.30 | 1663.63 | 1660.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 1659.60 | 1663.63 | 1660.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1663.90 | 1663.68 | 1660.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 1659.90 | 1663.68 | 1660.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1665.00 | 1663.95 | 1660.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1665.00 | 1663.95 | 1660.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1667.00 | 1664.56 | 1661.39 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1655.30 | 1659.65 | 1660.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1645.80 | 1655.93 | 1658.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1622.40 | 1621.43 | 1629.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:15:00 | 1625.90 | 1621.43 | 1629.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1635.10 | 1624.93 | 1629.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1635.10 | 1624.93 | 1629.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1636.80 | 1627.30 | 1630.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:45:00 | 1638.90 | 1627.30 | 1630.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1629.50 | 1629.21 | 1630.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1637.70 | 1629.21 | 1630.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1634.00 | 1630.17 | 1631.04 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1639.00 | 1631.93 | 1631.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 1640.40 | 1635.73 | 1633.73 | Break + close above crossover candle high |

### Cycle 182 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1591.10 | 1628.29 | 1630.97 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 1632.80 | 1618.76 | 1618.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 1635.60 | 1623.78 | 1620.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1627.90 | 1631.03 | 1625.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1627.90 | 1631.03 | 1625.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1627.90 | 1631.03 | 1625.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 1627.50 | 1631.03 | 1625.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1625.90 | 1630.00 | 1625.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 1621.50 | 1630.00 | 1625.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1620.70 | 1628.14 | 1625.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1620.70 | 1628.14 | 1625.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1616.40 | 1625.79 | 1624.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 1617.50 | 1625.79 | 1624.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1613.00 | 1623.23 | 1623.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 1609.50 | 1616.13 | 1619.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 1616.90 | 1616.03 | 1618.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 1616.90 | 1616.03 | 1618.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1616.90 | 1616.03 | 1618.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1597.20 | 1616.03 | 1618.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1673.30 | 1613.52 | 1607.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1673.30 | 1613.52 | 1607.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1684.80 | 1627.77 | 1614.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1670.60 | 1676.20 | 1661.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 1673.60 | 1676.20 | 1661.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 1663.20 | 1670.43 | 1663.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 1663.20 | 1670.43 | 1663.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1657.70 | 1667.89 | 1662.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 1657.70 | 1667.89 | 1662.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1654.40 | 1665.19 | 1661.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1643.40 | 1665.19 | 1661.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1639.40 | 1657.09 | 1658.60 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1660.40 | 1657.53 | 1657.47 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 1655.20 | 1657.06 | 1657.27 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 1663.20 | 1658.18 | 1657.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 1670.30 | 1661.35 | 1659.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1663.70 | 1664.81 | 1661.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 1663.70 | 1664.81 | 1661.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1670.20 | 1665.89 | 1662.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:30:00 | 1676.50 | 1669.67 | 1664.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 1660.30 | 1672.51 | 1670.21 | SL hit (close<static) qty=1.00 sl=1662.20 alert=retest2 |

### Cycle 190 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 1654.90 | 1666.83 | 1667.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1648.70 | 1661.89 | 1665.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 1660.00 | 1657.95 | 1662.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 1660.00 | 1657.95 | 1662.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1661.50 | 1658.66 | 1662.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 1661.50 | 1658.66 | 1662.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1658.70 | 1658.67 | 1661.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 1661.20 | 1658.67 | 1661.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1661.00 | 1659.14 | 1661.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1629.60 | 1659.14 | 1661.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 1642.00 | 1645.71 | 1646.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 1672.10 | 1644.52 | 1642.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1672.10 | 1644.52 | 1642.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1673.90 | 1650.39 | 1645.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 1654.70 | 1655.57 | 1649.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 1556.10 | 1655.57 | 1649.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 192 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 1550.60 | 1634.58 | 1640.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 1536.10 | 1614.88 | 1631.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1507.00 | 1506.53 | 1526.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 1507.00 | 1506.53 | 1526.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1509.90 | 1502.46 | 1513.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 1510.00 | 1502.46 | 1513.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1405.30 | 1368.62 | 1381.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1409.60 | 1368.62 | 1381.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1404.20 | 1375.74 | 1383.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1403.10 | 1375.74 | 1383.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1417.50 | 1392.18 | 1389.96 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 1353.50 | 1384.77 | 1387.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 1333.60 | 1354.41 | 1364.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1319.90 | 1295.17 | 1314.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1319.90 | 1295.17 | 1314.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1319.90 | 1295.17 | 1314.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1319.90 | 1295.17 | 1314.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1307.10 | 1297.56 | 1313.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:30:00 | 1303.80 | 1299.65 | 1313.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 1301.50 | 1299.65 | 1313.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 1303.50 | 1297.55 | 1306.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:00:00 | 1303.20 | 1295.19 | 1300.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1305.70 | 1299.04 | 1301.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 1306.90 | 1299.04 | 1301.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1298.90 | 1301.14 | 1301.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1292.50 | 1301.14 | 1301.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 1293.90 | 1300.04 | 1301.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 11:00:00 | 1291.80 | 1298.39 | 1300.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1308.60 | 1293.17 | 1295.58 | SL hit (close>static) qty=1.00 sl=1302.60 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 11:15:00 | 1302.30 | 1297.45 | 1297.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1319.00 | 1304.85 | 1301.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 1308.00 | 1308.28 | 1304.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:00:00 | 1308.00 | 1308.28 | 1304.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1291.60 | 1305.77 | 1304.56 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1296.50 | 1302.47 | 1303.18 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 1315.70 | 1305.29 | 1304.28 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 10:15:00 | 1297.00 | 1303.90 | 1304.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 1290.70 | 1301.26 | 1302.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1252.10 | 1242.54 | 1253.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1252.10 | 1242.54 | 1253.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1252.10 | 1242.54 | 1253.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1251.80 | 1242.54 | 1253.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1274.60 | 1243.98 | 1246.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1271.60 | 1243.98 | 1246.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1282.10 | 1251.61 | 1249.93 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1231.00 | 1254.41 | 1254.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1229.70 | 1246.33 | 1250.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1248.00 | 1237.93 | 1244.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1248.00 | 1237.93 | 1244.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1248.00 | 1237.93 | 1244.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:15:00 | 1253.50 | 1237.93 | 1244.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1247.50 | 1239.84 | 1244.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 1253.10 | 1239.84 | 1244.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1255.40 | 1242.38 | 1244.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1255.40 | 1242.38 | 1244.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1256.00 | 1245.10 | 1245.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1238.80 | 1245.10 | 1245.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1248.50 | 1245.78 | 1245.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 09:15:00 | 1248.50 | 1245.78 | 1245.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-23 13:15:00 | 1256.30 | 1249.13 | 1247.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1280.00 | 1281.92 | 1273.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:30:00 | 1280.80 | 1281.92 | 1273.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1278.70 | 1280.82 | 1274.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1277.40 | 1280.82 | 1274.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1278.00 | 1280.26 | 1275.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 1271.70 | 1280.26 | 1275.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1280.10 | 1280.22 | 1275.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:15:00 | 1273.80 | 1280.22 | 1275.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1269.60 | 1278.10 | 1275.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 1269.60 | 1278.10 | 1275.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1270.00 | 1276.48 | 1274.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:15:00 | 1266.40 | 1276.48 | 1274.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1266.40 | 1274.46 | 1273.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1261.10 | 1274.46 | 1273.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1260.30 | 1271.63 | 1272.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 1245.60 | 1261.60 | 1267.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1298.00 | 1265.83 | 1267.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1298.00 | 1265.83 | 1267.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1298.00 | 1265.83 | 1267.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1298.00 | 1265.83 | 1267.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1296.00 | 1271.86 | 1270.05 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 1266.40 | 1272.18 | 1272.45 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1294.50 | 1276.87 | 1274.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1304.50 | 1282.40 | 1277.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1318.30 | 1336.18 | 1326.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1318.30 | 1336.18 | 1326.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1318.30 | 1336.18 | 1326.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 1315.20 | 1336.18 | 1326.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1324.20 | 1333.78 | 1326.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:15:00 | 1320.60 | 1333.78 | 1326.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 1330.50 | 1331.71 | 1326.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:30:00 | 1327.00 | 1331.71 | 1326.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1328.20 | 1331.00 | 1326.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 1334.60 | 1331.72 | 1327.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1289.20 | 1323.02 | 1324.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 09:15:00 | 1289.20 | 1323.02 | 1324.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1272.80 | 1294.04 | 1306.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1311.40 | 1286.08 | 1294.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1311.40 | 1286.08 | 1294.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1311.40 | 1286.08 | 1294.17 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 1305.30 | 1298.13 | 1297.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1320.70 | 1303.69 | 1300.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 1310.80 | 1312.68 | 1308.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 13:00:00 | 1310.80 | 1312.68 | 1308.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 1312.60 | 1312.66 | 1309.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:15:00 | 1308.60 | 1312.66 | 1309.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1318.70 | 1313.87 | 1310.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:30:00 | 1314.30 | 1313.87 | 1310.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1311.80 | 1314.44 | 1311.05 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1271.60 | 1305.74 | 1309.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 1264.30 | 1297.45 | 1305.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 1175.70 | 1161.35 | 1175.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 1175.70 | 1161.35 | 1175.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1175.70 | 1161.35 | 1175.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 1172.30 | 1161.35 | 1175.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1176.80 | 1164.44 | 1175.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 1176.80 | 1164.44 | 1175.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1177.20 | 1166.99 | 1175.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 1177.20 | 1166.99 | 1175.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1176.60 | 1168.91 | 1176.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 1176.60 | 1168.91 | 1176.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1170.50 | 1169.23 | 1175.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 1168.20 | 1169.23 | 1175.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1168.60 | 1168.89 | 1174.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:30:00 | 1169.00 | 1169.15 | 1173.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 1184.50 | 1173.30 | 1174.65 | SL hit (close>static) qty=1.00 sl=1178.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 1180.90 | 1176.61 | 1176.03 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 1169.90 | 1175.81 | 1176.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 14:15:00 | 1168.00 | 1173.22 | 1174.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 1172.60 | 1172.29 | 1174.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1172.60 | 1172.29 | 1174.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1172.60 | 1172.29 | 1174.06 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 1177.60 | 1174.76 | 1174.64 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 1171.70 | 1174.59 | 1174.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 1169.40 | 1173.55 | 1174.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 12:15:00 | 1172.30 | 1169.72 | 1171.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 12:15:00 | 1172.30 | 1169.72 | 1171.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1172.30 | 1169.72 | 1171.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 1172.30 | 1169.72 | 1171.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1168.10 | 1169.40 | 1171.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 1165.40 | 1168.12 | 1170.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 1175.50 | 1170.29 | 1170.54 | SL hit (close>static) qty=1.00 sl=1172.60 alert=retest2 |

### Cycle 213 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 1177.90 | 1171.81 | 1171.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 15:15:00 | 1179.50 | 1174.42 | 1172.56 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 14:45:00 | 1245.15 | 2023-05-19 09:15:00 | 1268.60 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-05-25 14:15:00 | 1298.50 | 2023-06-02 09:15:00 | 1303.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2023-06-07 10:30:00 | 1284.00 | 2023-06-12 12:15:00 | 1291.60 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-06-08 14:30:00 | 1284.00 | 2023-06-12 12:15:00 | 1291.60 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-06-12 10:00:00 | 1284.20 | 2023-06-12 12:15:00 | 1291.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-06-19 10:45:00 | 1289.35 | 2023-06-19 12:15:00 | 1292.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-06-27 10:45:00 | 1275.65 | 2023-06-28 09:15:00 | 1283.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-07-10 14:30:00 | 1332.10 | 2023-07-11 09:15:00 | 1344.65 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-07-20 14:30:00 | 1450.70 | 2023-07-21 09:15:00 | 1342.45 | STOP_HIT | 1.00 | -7.46% |
| BUY | retest2 | 2023-08-03 09:15:00 | 1359.35 | 2023-08-03 13:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-08-03 09:45:00 | 1358.65 | 2023-08-03 13:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-08-11 09:15:00 | 1392.40 | 2023-08-11 09:15:00 | 1380.05 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-08-25 14:45:00 | 1420.85 | 2023-09-18 12:15:00 | 1494.50 | STOP_HIT | 1.00 | 5.18% |
| BUY | retest2 | 2023-08-29 09:30:00 | 1422.10 | 2023-09-18 12:15:00 | 1494.50 | STOP_HIT | 1.00 | 5.09% |
| BUY | retest2 | 2023-08-29 10:45:00 | 1420.50 | 2023-09-18 12:15:00 | 1494.50 | STOP_HIT | 1.00 | 5.21% |
| BUY | retest2 | 2023-08-29 12:45:00 | 1420.95 | 2023-09-18 12:15:00 | 1494.50 | STOP_HIT | 1.00 | 5.18% |
| BUY | retest2 | 2023-08-30 09:15:00 | 1427.50 | 2023-09-18 12:15:00 | 1494.50 | STOP_HIT | 1.00 | 4.69% |
| SELL | retest2 | 2023-09-21 09:15:00 | 1483.90 | 2023-09-21 12:15:00 | 1498.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-28 11:00:00 | 1452.75 | 2023-10-05 09:15:00 | 1457.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-10-17 14:15:00 | 1439.50 | 2023-10-18 13:15:00 | 1451.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-10-31 10:30:00 | 1363.10 | 2023-11-03 09:15:00 | 1383.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-11-01 12:45:00 | 1363.40 | 2023-11-03 09:15:00 | 1383.30 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-11-02 11:15:00 | 1361.95 | 2023-11-03 09:15:00 | 1383.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-11-13 14:45:00 | 1373.80 | 2023-11-15 09:15:00 | 1401.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2023-11-22 10:15:00 | 1445.50 | 2023-11-24 12:15:00 | 1439.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-11-22 12:30:00 | 1446.05 | 2023-11-24 12:15:00 | 1439.35 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-11-22 13:30:00 | 1445.35 | 2023-11-24 12:15:00 | 1439.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1465.15 | 2023-12-05 11:15:00 | 1446.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-12-04 12:30:00 | 1459.15 | 2023-12-05 11:15:00 | 1446.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-12-04 13:15:00 | 1460.75 | 2023-12-05 11:15:00 | 1446.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-12-08 09:30:00 | 1473.15 | 2023-12-12 15:15:00 | 1477.00 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2023-12-12 14:45:00 | 1470.10 | 2023-12-12 15:15:00 | 1477.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest1 | 2023-12-19 09:15:00 | 1567.35 | 2023-12-20 13:15:00 | 1555.95 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-19 12:30:00 | 1558.65 | 2023-12-20 14:15:00 | 1538.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-12-20 09:15:00 | 1575.00 | 2023-12-20 14:15:00 | 1538.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-01-02 09:15:00 | 1539.75 | 2024-01-05 11:15:00 | 1524.60 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2024-01-17 11:30:00 | 1644.35 | 2024-01-29 13:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2024-01-17 12:15:00 | 1644.40 | 2024-01-29 13:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-01-18 15:15:00 | 1646.65 | 2024-01-29 13:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-01-24 09:15:00 | 1660.10 | 2024-01-29 13:15:00 | 1655.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-02-06 12:00:00 | 1720.80 | 2024-02-08 09:15:00 | 1695.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-02-14 09:15:00 | 1645.50 | 2024-02-15 14:15:00 | 1676.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-02-15 12:30:00 | 1662.00 | 2024-02-15 14:15:00 | 1676.30 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-02-21 09:15:00 | 1665.15 | 2024-02-23 10:15:00 | 1682.35 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-03-02 12:00:00 | 1655.55 | 2024-03-12 11:15:00 | 1623.10 | STOP_HIT | 1.00 | 1.96% |
| SELL | retest2 | 2024-04-01 10:15:00 | 1507.75 | 2024-04-09 09:15:00 | 1507.25 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-04-25 13:45:00 | 1437.60 | 2024-04-26 14:15:00 | 1430.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-04-26 10:45:00 | 1439.30 | 2024-04-26 14:15:00 | 1430.05 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-04-26 11:30:00 | 1437.20 | 2024-04-26 14:15:00 | 1430.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-05-27 10:15:00 | 1470.25 | 2024-05-29 09:15:00 | 1459.15 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-27 11:00:00 | 1469.70 | 2024-05-29 09:15:00 | 1459.15 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1387.60 | 2024-06-05 11:15:00 | 1437.40 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1405.55 | 2024-06-05 11:15:00 | 1437.40 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-06-04 11:30:00 | 1401.20 | 2024-06-05 11:15:00 | 1437.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1504.00 | 2024-06-12 12:15:00 | 1492.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-06-12 12:15:00 | 1497.05 | 2024-06-12 12:15:00 | 1492.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-06-21 09:15:00 | 1548.20 | 2024-07-10 12:15:00 | 1646.55 | STOP_HIT | 1.00 | 6.35% |
| BUY | retest2 | 2024-07-25 12:45:00 | 1829.15 | 2024-08-01 10:15:00 | 1857.95 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1858.50 | 2024-08-01 10:15:00 | 1857.95 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-08-07 11:15:00 | 1777.30 | 2024-08-12 09:15:00 | 1773.30 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-08-07 12:00:00 | 1773.95 | 2024-08-12 09:15:00 | 1773.30 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-08-08 09:15:00 | 1766.10 | 2024-08-12 09:15:00 | 1773.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-09-09 12:45:00 | 1913.50 | 2024-09-10 10:15:00 | 1922.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-09-09 14:15:00 | 1905.55 | 2024-09-10 10:15:00 | 1922.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-09-16 13:15:00 | 1945.05 | 2024-09-18 09:15:00 | 1898.55 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-09-20 13:30:00 | 1872.25 | 2024-09-20 15:15:00 | 1910.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-09-23 11:15:00 | 1882.40 | 2024-09-26 09:15:00 | 1914.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-23 12:00:00 | 1885.10 | 2024-09-26 09:15:00 | 1914.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-09-24 09:15:00 | 1877.90 | 2024-09-26 09:15:00 | 1914.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-09-25 09:15:00 | 1892.20 | 2024-09-26 09:15:00 | 1914.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-09-25 10:00:00 | 1892.25 | 2024-09-26 09:15:00 | 1914.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-10-24 14:30:00 | 1860.50 | 2024-10-28 11:15:00 | 1878.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-10-25 11:45:00 | 1859.95 | 2024-10-28 11:15:00 | 1878.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-11-07 11:45:00 | 1798.45 | 2024-11-18 09:15:00 | 1817.80 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2024-12-04 13:30:00 | 1900.00 | 2024-12-17 11:15:00 | 1970.35 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2024-12-05 09:15:00 | 1911.25 | 2024-12-17 11:15:00 | 1970.35 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2024-12-05 12:00:00 | 1919.60 | 2024-12-17 11:15:00 | 1970.35 | STOP_HIT | 1.00 | 2.64% |
| SELL | retest2 | 2024-12-20 11:15:00 | 1942.20 | 2024-12-24 15:15:00 | 1845.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 11:15:00 | 1942.20 | 2024-12-27 09:15:00 | 1920.90 | STOP_HIT | 0.50 | 1.10% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1951.75 | 2025-01-07 11:15:00 | 1928.55 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-28 13:15:00 | 1835.65 | 2025-01-29 09:15:00 | 1855.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-01-28 14:45:00 | 1828.80 | 2025-01-29 09:15:00 | 1855.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-02-03 14:30:00 | 1859.95 | 2025-02-04 09:15:00 | 1894.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-02-11 12:45:00 | 1885.95 | 2025-02-24 09:15:00 | 1791.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 09:45:00 | 1883.15 | 2025-02-24 09:15:00 | 1788.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 12:45:00 | 1885.95 | 2025-02-25 09:15:00 | 1776.85 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2025-02-12 09:45:00 | 1883.15 | 2025-02-25 09:15:00 | 1776.85 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest1 | 2025-04-03 09:15:00 | 1512.50 | 2025-04-07 09:15:00 | 1361.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 15:15:00 | 1428.00 | 2025-04-15 14:15:00 | 1424.10 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-04-11 10:30:00 | 1427.05 | 2025-04-15 14:15:00 | 1424.10 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-04-11 11:15:00 | 1422.25 | 2025-04-15 14:15:00 | 1424.10 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-04-15 09:45:00 | 1427.10 | 2025-04-15 14:15:00 | 1424.10 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-04-28 12:15:00 | 1483.50 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-04-28 14:45:00 | 1481.60 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-04-29 09:45:00 | 1480.40 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2025-04-29 11:00:00 | 1481.40 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1511.60 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-05-02 11:00:00 | 1507.10 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-05-02 11:45:00 | 1504.00 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-05-02 12:30:00 | 1504.90 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-07 11:30:00 | 1510.60 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-07 12:15:00 | 1511.00 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-05-08 15:00:00 | 1511.70 | 2025-05-09 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1598.60 | 2025-05-19 09:15:00 | 1572.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-16 13:15:00 | 1590.10 | 2025-05-19 09:15:00 | 1572.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-05-16 15:00:00 | 1590.20 | 2025-05-19 09:15:00 | 1572.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-05-20 12:45:00 | 1564.70 | 2025-05-23 09:15:00 | 1583.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-21 15:00:00 | 1568.30 | 2025-05-23 09:15:00 | 1583.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1604.70 | 2025-06-19 11:15:00 | 1617.30 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1589.80 | 2025-06-25 13:15:00 | 1614.30 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-15 12:15:00 | 1587.00 | 2025-07-16 12:15:00 | 1608.40 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-15 15:00:00 | 1584.80 | 2025-07-16 12:15:00 | 1608.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2025-07-21 09:15:00 | 1573.00 | 2025-07-23 11:15:00 | 1585.90 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-21 14:45:00 | 1579.00 | 2025-07-28 09:15:00 | 1500.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1579.00 | 2025-07-28 09:15:00 | 1500.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1579.80 | 2025-07-28 09:15:00 | 1500.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 1579.10 | 2025-07-28 09:15:00 | 1500.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:45:00 | 1579.00 | 2025-07-28 14:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1579.00 | 2025-07-28 14:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1579.80 | 2025-07-28 14:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-07-22 11:30:00 | 1579.10 | 2025-07-28 14:15:00 | 1518.50 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-07-23 15:15:00 | 1558.90 | 2025-08-01 11:15:00 | 1480.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 15:15:00 | 1558.90 | 2025-08-04 13:15:00 | 1479.30 | STOP_HIT | 0.50 | 5.11% |
| BUY | retest2 | 2025-08-20 09:15:00 | 1462.90 | 2025-08-28 13:15:00 | 1505.20 | STOP_HIT | 1.00 | 2.89% |
| SELL | retest2 | 2025-09-01 12:30:00 | 1491.70 | 2025-09-02 11:15:00 | 1507.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-01 13:15:00 | 1494.40 | 2025-09-02 11:15:00 | 1507.20 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1490.40 | 2025-09-02 11:15:00 | 1507.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-25 11:30:00 | 1494.40 | 2025-10-06 10:15:00 | 1462.90 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-10-07 14:00:00 | 1468.90 | 2025-10-07 14:15:00 | 1454.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1487.90 | 2025-10-13 13:15:00 | 1490.20 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1476.40 | 2025-11-10 09:15:00 | 1503.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-11-21 10:45:00 | 1543.80 | 2025-11-25 11:15:00 | 1530.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-03 10:45:00 | 1575.60 | 2025-12-09 12:15:00 | 1596.90 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-12-03 12:45:00 | 1576.70 | 2025-12-09 12:15:00 | 1596.90 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-12-03 14:30:00 | 1580.00 | 2025-12-09 12:15:00 | 1596.90 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-12-11 11:00:00 | 1582.60 | 2025-12-12 15:15:00 | 1597.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1583.90 | 2025-12-12 15:15:00 | 1597.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1597.20 | 2026-01-16 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2026-01-27 09:30:00 | 1676.50 | 2026-01-28 10:15:00 | 1660.30 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1629.60 | 2026-02-03 10:15:00 | 1672.10 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-01 15:15:00 | 1642.00 | 2026-02-03 10:15:00 | 1672.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-25 11:30:00 | 1303.80 | 2026-03-04 09:15:00 | 1308.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-02-25 12:15:00 | 1301.50 | 2026-03-04 09:15:00 | 1308.60 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-02-26 10:15:00 | 1303.50 | 2026-03-04 09:15:00 | 1308.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-02-27 10:00:00 | 1303.20 | 2026-03-04 11:15:00 | 1302.30 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1292.50 | 2026-03-04 11:15:00 | 1302.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-02 10:15:00 | 1293.90 | 2026-03-04 11:15:00 | 1302.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-03-02 11:00:00 | 1291.80 | 2026-03-04 11:15:00 | 1302.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1238.80 | 2026-03-23 09:15:00 | 1248.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-09 15:00:00 | 1334.60 | 2026-04-10 09:15:00 | 1289.20 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-04-29 14:15:00 | 1168.20 | 2026-04-30 12:15:00 | 1184.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1168.60 | 2026-04-30 12:15:00 | 1184.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-30 10:30:00 | 1169.00 | 2026-04-30 12:15:00 | 1184.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-05-07 14:30:00 | 1165.40 | 2026-05-08 12:15:00 | 1175.50 | STOP_HIT | 1.00 | -0.87% |
