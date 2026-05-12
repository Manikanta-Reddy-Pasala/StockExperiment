# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2024-07-10 09:15:00 → 2026-05-11 15:15:00 (3168 bars)
- **Last close:** 1674.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 140 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 53 |
| ALERT3 | 255 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 141 |
| PARTIAL | 21 |
| TARGET_HIT | 5 |
| STOP_HIT | 143 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 169 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 109
- **Target hits / Stop hits / Partials:** 5 / 143 / 21
- **Avg / median % per leg:** 0.16% / -0.77%
- **Sum % (uncompounded):** 26.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 14 | 19.2% | 2 | 68 | 3 | -0.92% | -67.1% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 0 | 7 | 3 | 2.42% | 24.2% |
| BUY @ 3rd Alert (retest2) | 63 | 8 | 12.7% | 2 | 61 | 0 | -1.45% | -91.4% |
| SELL (all) | 96 | 46 | 47.9% | 3 | 75 | 18 | 0.97% | 93.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 96 | 46 | 47.9% | 3 | 75 | 18 | 0.97% | 93.4% |
| retest1 (combined) | 10 | 6 | 60.0% | 0 | 7 | 3 | 2.42% | 24.2% |
| retest2 (combined) | 159 | 54 | 34.0% | 5 | 136 | 18 | 0.01% | 2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 1345.30 | 1358.78 | 1360.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 10:15:00 | 1340.00 | 1355.02 | 1358.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 1367.70 | 1353.71 | 1356.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 13:15:00 | 1367.70 | 1353.71 | 1356.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1367.70 | 1353.71 | 1356.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 1367.70 | 1353.71 | 1356.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1371.15 | 1357.20 | 1357.81 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 1375.00 | 1360.76 | 1359.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1385.00 | 1365.61 | 1361.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 1362.25 | 1368.26 | 1364.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 1362.25 | 1368.26 | 1364.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1362.25 | 1368.26 | 1364.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 1361.35 | 1368.26 | 1364.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1360.20 | 1366.65 | 1363.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:15:00 | 1360.70 | 1366.65 | 1363.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1360.10 | 1365.34 | 1363.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 1361.10 | 1365.34 | 1363.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1360.75 | 1364.42 | 1363.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 1355.05 | 1364.42 | 1363.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1361.00 | 1363.05 | 1362.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:30:00 | 1360.20 | 1363.05 | 1362.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1367.55 | 1364.49 | 1363.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 1365.25 | 1364.49 | 1363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1368.50 | 1365.76 | 1364.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:30:00 | 1364.75 | 1365.76 | 1364.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1365.05 | 1365.62 | 1364.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 1360.50 | 1365.62 | 1364.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1360.00 | 1364.50 | 1363.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 1360.00 | 1364.50 | 1363.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 1350.20 | 1361.64 | 1362.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 1341.90 | 1352.56 | 1357.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1310.00 | 1296.81 | 1312.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1310.00 | 1296.81 | 1312.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1310.00 | 1296.81 | 1312.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1310.00 | 1296.81 | 1312.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1303.00 | 1298.05 | 1311.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 1301.05 | 1299.24 | 1310.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:45:00 | 1300.60 | 1299.59 | 1310.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 1293.00 | 1302.00 | 1308.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:30:00 | 1300.15 | 1302.26 | 1307.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1305.15 | 1301.75 | 1305.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1305.15 | 1301.75 | 1305.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1309.00 | 1303.20 | 1305.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1308.55 | 1303.20 | 1305.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1296.00 | 1301.76 | 1305.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 1294.90 | 1301.76 | 1305.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1293.00 | 1295.91 | 1299.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:30:00 | 1291.00 | 1294.30 | 1298.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 15:00:00 | 1292.05 | 1291.83 | 1295.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1290.25 | 1290.59 | 1294.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 1298.30 | 1290.59 | 1294.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1291.55 | 1290.79 | 1294.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 1291.90 | 1290.79 | 1294.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1293.00 | 1291.10 | 1293.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 1293.00 | 1291.10 | 1293.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1295.75 | 1292.03 | 1293.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 1295.75 | 1292.03 | 1293.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1304.00 | 1294.43 | 1294.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 1304.00 | 1294.43 | 1294.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 1303.00 | 1296.14 | 1295.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 15:15:00 | 1303.00 | 1296.14 | 1295.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 1306.05 | 1298.12 | 1296.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 15:15:00 | 1299.95 | 1303.16 | 1300.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 15:15:00 | 1299.95 | 1303.16 | 1300.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1299.95 | 1303.16 | 1300.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1315.60 | 1303.16 | 1300.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1292.00 | 1309.68 | 1307.01 | SL hit (close<static) qty=1.00 sl=1299.95 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 1295.70 | 1304.13 | 1304.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1271.65 | 1294.54 | 1299.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 13:15:00 | 1272.05 | 1257.68 | 1269.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 13:15:00 | 1272.05 | 1257.68 | 1269.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 1272.05 | 1257.68 | 1269.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:45:00 | 1272.00 | 1257.68 | 1269.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 1300.00 | 1266.14 | 1272.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 1300.00 | 1266.14 | 1272.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1287.20 | 1270.35 | 1273.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:15:00 | 1301.05 | 1270.35 | 1273.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 1291.00 | 1278.24 | 1276.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 11:15:00 | 1307.60 | 1284.11 | 1279.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 11:15:00 | 1295.40 | 1298.18 | 1290.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 12:00:00 | 1295.40 | 1298.18 | 1290.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1287.00 | 1296.30 | 1291.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 1288.00 | 1296.30 | 1291.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1286.00 | 1294.24 | 1291.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 1280.00 | 1291.17 | 1290.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 1295.75 | 1292.09 | 1290.62 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 1285.00 | 1289.64 | 1289.91 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 1300.70 | 1291.35 | 1290.61 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 1288.35 | 1289.88 | 1290.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 13:15:00 | 1280.65 | 1288.04 | 1289.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 09:15:00 | 1260.30 | 1255.35 | 1267.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 1260.30 | 1255.35 | 1267.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1260.30 | 1255.35 | 1267.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 1260.30 | 1255.35 | 1267.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1265.35 | 1247.26 | 1255.68 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1286.15 | 1263.47 | 1260.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 1321.60 | 1275.09 | 1265.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 1323.05 | 1324.06 | 1306.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1335.15 | 1324.06 | 1306.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:15:00 | 1337.05 | 1325.85 | 1309.28 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 1339.50 | 1328.58 | 1312.02 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:15:00 | 1401.91 | 1365.84 | 1352.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:15:00 | 1403.90 | 1365.84 | 1352.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:15:00 | 1406.48 | 1365.84 | 1352.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 1403.70 | 1404.70 | 1389.60 | SL hit (close<ema200) qty=0.50 sl=1404.70 alert=retest1 |

### Cycle 11 — SELL (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 13:15:00 | 1385.25 | 1391.31 | 1391.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 1375.05 | 1387.21 | 1389.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1383.10 | 1379.97 | 1383.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1383.10 | 1379.97 | 1383.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1383.10 | 1379.97 | 1383.74 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 1400.65 | 1388.08 | 1386.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1414.65 | 1395.50 | 1390.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 1392.80 | 1398.11 | 1393.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 1392.80 | 1398.11 | 1393.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1392.80 | 1398.11 | 1393.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 1392.80 | 1398.11 | 1393.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1387.20 | 1395.93 | 1393.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 1387.20 | 1395.93 | 1393.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1387.95 | 1394.33 | 1392.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 1387.95 | 1394.33 | 1392.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 1387.25 | 1391.58 | 1391.58 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 1394.95 | 1392.25 | 1391.89 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 1386.85 | 1391.17 | 1391.43 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 1406.20 | 1393.70 | 1392.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 10:15:00 | 1416.50 | 1400.84 | 1396.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 1418.30 | 1419.59 | 1411.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:00:00 | 1418.30 | 1419.59 | 1411.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1408.50 | 1417.37 | 1411.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 1409.30 | 1417.37 | 1411.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 1401.05 | 1414.11 | 1410.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1412.30 | 1414.11 | 1410.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1392.70 | 1409.83 | 1408.61 | SL hit (close<static) qty=1.00 sl=1400.20 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1391.15 | 1406.09 | 1407.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 14:15:00 | 1389.95 | 1395.87 | 1399.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1418.35 | 1399.06 | 1400.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1418.35 | 1399.06 | 1400.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1418.35 | 1399.06 | 1400.18 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1431.60 | 1405.57 | 1403.04 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 14:15:00 | 1389.45 | 1400.54 | 1401.52 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 1423.50 | 1403.60 | 1402.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 1443.05 | 1414.70 | 1408.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 1420.60 | 1423.96 | 1416.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:00:00 | 1420.60 | 1423.96 | 1416.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1428.95 | 1434.40 | 1426.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 1422.20 | 1434.40 | 1426.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1437.00 | 1434.92 | 1427.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:45:00 | 1420.00 | 1434.92 | 1427.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1428.30 | 1433.26 | 1428.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 1428.30 | 1433.26 | 1428.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1428.05 | 1432.22 | 1428.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 1428.05 | 1432.22 | 1428.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1422.00 | 1430.17 | 1427.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 1422.00 | 1430.17 | 1427.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1426.80 | 1429.50 | 1427.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1436.95 | 1429.50 | 1427.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 1432.75 | 1427.20 | 1426.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-18 10:15:00 | 1576.03 | 1499.58 | 1470.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 1449.00 | 1481.06 | 1481.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 10:15:00 | 1448.00 | 1467.77 | 1474.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 1450.70 | 1449.01 | 1458.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 13:00:00 | 1450.70 | 1449.01 | 1458.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1441.85 | 1446.56 | 1454.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 1436.10 | 1445.02 | 1452.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 1433.35 | 1442.81 | 1451.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 1458.30 | 1440.01 | 1442.36 | SL hit (close>static) qty=1.00 sl=1454.25 alert=retest2 |

### Cycle 22 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 1461.45 | 1444.30 | 1444.09 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1429.00 | 1442.23 | 1443.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 13:15:00 | 1417.30 | 1437.25 | 1441.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 1438.45 | 1436.90 | 1439.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1438.45 | 1436.90 | 1439.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1438.45 | 1436.90 | 1439.86 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 1452.45 | 1442.44 | 1442.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 1471.95 | 1455.33 | 1449.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 1464.90 | 1467.17 | 1459.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 15:15:00 | 1464.90 | 1467.17 | 1459.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 1464.90 | 1467.17 | 1459.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 1473.95 | 1467.17 | 1459.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1482.40 | 1470.21 | 1461.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:45:00 | 1509.85 | 1478.70 | 1466.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 14:00:00 | 1494.65 | 1484.94 | 1471.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 09:30:00 | 1492.65 | 1491.07 | 1478.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 1446.10 | 1470.33 | 1473.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1446.10 | 1470.33 | 1473.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 1426.20 | 1450.30 | 1461.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1369.60 | 1359.91 | 1393.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:30:00 | 1372.55 | 1359.91 | 1393.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1397.65 | 1375.85 | 1393.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 1397.65 | 1375.85 | 1393.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1401.05 | 1380.89 | 1393.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1401.05 | 1380.89 | 1393.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1394.00 | 1383.51 | 1393.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1416.90 | 1383.51 | 1393.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1402.85 | 1387.38 | 1394.70 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1417.30 | 1399.19 | 1398.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 1428.75 | 1405.11 | 1401.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1462.05 | 1463.06 | 1444.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 1477.55 | 1463.06 | 1444.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1462.00 | 1470.16 | 1458.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1472.00 | 1470.16 | 1458.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1468.20 | 1485.99 | 1485.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 1468.20 | 1485.99 | 1485.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 1455.80 | 1479.95 | 1483.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 1406.85 | 1399.70 | 1418.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 1406.85 | 1399.70 | 1418.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1406.85 | 1399.70 | 1418.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:45:00 | 1411.95 | 1399.70 | 1418.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1405.60 | 1400.88 | 1416.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 1400.10 | 1400.88 | 1416.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 1399.95 | 1398.67 | 1411.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 1405.65 | 1394.21 | 1393.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 1405.65 | 1394.21 | 1393.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 15:15:00 | 1409.00 | 1397.17 | 1394.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1395.95 | 1396.93 | 1394.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 1395.95 | 1396.93 | 1394.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1395.95 | 1396.93 | 1394.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 1432.00 | 1404.99 | 1399.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 1431.05 | 1441.89 | 1442.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 1431.05 | 1441.89 | 1442.00 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 1447.70 | 1443.05 | 1442.52 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1420.00 | 1437.96 | 1440.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 1412.40 | 1432.85 | 1437.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1445.70 | 1422.17 | 1429.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1445.70 | 1422.17 | 1429.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1445.70 | 1422.17 | 1429.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 1422.85 | 1422.17 | 1429.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1458.50 | 1429.44 | 1432.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:30:00 | 1445.40 | 1433.55 | 1434.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 12:15:00 | 1456.70 | 1438.18 | 1436.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 1456.70 | 1438.18 | 1436.11 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 1422.00 | 1434.37 | 1435.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 1398.10 | 1423.50 | 1429.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1374.00 | 1360.80 | 1384.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1374.00 | 1360.80 | 1384.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1305.10 | 1295.36 | 1310.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 1329.75 | 1295.36 | 1310.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1307.00 | 1300.41 | 1308.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:45:00 | 1302.40 | 1300.41 | 1308.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1310.95 | 1302.52 | 1308.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 1309.20 | 1302.52 | 1308.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 1315.55 | 1305.12 | 1309.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 1322.20 | 1305.12 | 1309.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 1310.00 | 1308.41 | 1309.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 13:15:00 | 1293.55 | 1307.11 | 1309.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 1416.25 | 1328.59 | 1318.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 1416.25 | 1328.59 | 1318.07 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 1376.05 | 1379.57 | 1379.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 1370.40 | 1377.00 | 1378.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 1375.35 | 1375.28 | 1377.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 1375.35 | 1375.28 | 1377.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1380.60 | 1374.84 | 1376.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:45:00 | 1388.00 | 1374.84 | 1376.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1377.95 | 1375.47 | 1376.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1369.80 | 1375.47 | 1376.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:00:00 | 1369.35 | 1373.68 | 1375.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1380.10 | 1366.65 | 1365.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 1380.10 | 1366.65 | 1365.72 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 1359.35 | 1365.22 | 1365.40 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1373.00 | 1366.78 | 1366.09 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1343.00 | 1361.73 | 1363.89 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 1376.00 | 1360.30 | 1360.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 1397.05 | 1375.80 | 1369.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 1389.00 | 1394.91 | 1385.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 1389.00 | 1394.91 | 1385.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1383.70 | 1392.67 | 1385.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 1399.10 | 1392.67 | 1385.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 1395.05 | 1399.00 | 1393.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 1392.90 | 1396.98 | 1393.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 14:00:00 | 1393.10 | 1393.36 | 1392.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1390.95 | 1392.88 | 1392.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 1390.00 | 1392.88 | 1392.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 1385.00 | 1391.30 | 1391.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 1385.00 | 1391.30 | 1391.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1366.70 | 1386.38 | 1389.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 12:15:00 | 1381.50 | 1380.52 | 1385.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 13:00:00 | 1381.50 | 1380.52 | 1385.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1371.50 | 1378.72 | 1384.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 1380.00 | 1378.72 | 1384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1369.60 | 1373.11 | 1379.95 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 1401.80 | 1385.10 | 1383.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 1411.60 | 1390.40 | 1385.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 10:15:00 | 1433.50 | 1436.41 | 1422.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:30:00 | 1434.95 | 1436.41 | 1422.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1443.00 | 1436.36 | 1424.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 1427.80 | 1436.36 | 1424.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1441.00 | 1439.11 | 1428.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1447.25 | 1439.11 | 1428.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 12:00:00 | 1447.70 | 1444.09 | 1434.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:45:00 | 1442.10 | 1448.30 | 1441.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:00:00 | 1444.30 | 1447.50 | 1441.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1449.20 | 1447.84 | 1442.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 13:15:00 | 1458.35 | 1450.29 | 1446.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:30:00 | 1464.00 | 1453.47 | 1448.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:45:00 | 1458.30 | 1454.69 | 1450.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1458.05 | 1454.69 | 1450.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1466.45 | 1457.04 | 1451.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:15:00 | 1472.80 | 1457.04 | 1451.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:45:00 | 1476.80 | 1461.90 | 1455.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1445.00 | 1458.98 | 1458.56 | SL hit (close<static) qty=1.00 sl=1451.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1440.80 | 1455.34 | 1456.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1436.00 | 1449.04 | 1453.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1442.00 | 1440.99 | 1448.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1442.00 | 1440.99 | 1448.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1442.00 | 1440.99 | 1448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1446.30 | 1440.99 | 1448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1445.55 | 1441.17 | 1445.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1445.55 | 1441.17 | 1445.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1422.80 | 1437.49 | 1443.76 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1453.80 | 1442.65 | 1442.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 10:15:00 | 1460.00 | 1446.12 | 1444.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 1440.00 | 1447.22 | 1445.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 1440.00 | 1447.22 | 1445.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1440.00 | 1447.22 | 1445.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 1440.00 | 1447.22 | 1445.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1443.00 | 1446.37 | 1445.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 1418.00 | 1446.37 | 1445.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1396.35 | 1436.37 | 1440.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 1378.95 | 1418.33 | 1431.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1335.00 | 1331.67 | 1359.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1335.00 | 1331.67 | 1359.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1351.80 | 1342.25 | 1354.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 1350.05 | 1342.25 | 1354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1357.05 | 1345.21 | 1355.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 1357.05 | 1345.21 | 1355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1370.00 | 1350.17 | 1356.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1379.50 | 1350.17 | 1356.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 1359.50 | 1352.03 | 1356.67 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 1364.70 | 1358.58 | 1358.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 1369.35 | 1360.74 | 1359.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1365.15 | 1368.20 | 1363.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 10:00:00 | 1365.15 | 1368.20 | 1363.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 1361.00 | 1366.76 | 1363.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 1359.25 | 1366.76 | 1363.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1367.35 | 1366.88 | 1363.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:00:00 | 1371.90 | 1365.33 | 1363.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 1357.10 | 1373.06 | 1370.11 | SL hit (close<static) qty=1.00 sl=1360.15 alert=retest2 |

### Cycle 47 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1353.25 | 1366.37 | 1367.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 1349.65 | 1358.12 | 1362.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1352.00 | 1351.40 | 1357.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1352.00 | 1351.40 | 1357.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1352.00 | 1351.40 | 1357.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1352.00 | 1351.40 | 1357.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1345.00 | 1350.12 | 1356.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1328.05 | 1350.12 | 1356.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1337.00 | 1343.52 | 1348.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:15:00 | 1333.00 | 1343.52 | 1348.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1261.65 | 1298.34 | 1320.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1270.15 | 1298.34 | 1320.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1266.35 | 1298.34 | 1320.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 1195.24 | 1223.12 | 1265.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 13:15:00 | 1219.05 | 1215.84 | 1215.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1227.85 | 1219.22 | 1217.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 1221.00 | 1221.83 | 1219.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 1221.00 | 1221.83 | 1219.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1227.15 | 1222.89 | 1220.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1230.00 | 1222.89 | 1220.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 1215.90 | 1260.69 | 1261.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 1215.90 | 1260.69 | 1261.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 1209.10 | 1243.11 | 1253.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1112.65 | 1090.34 | 1118.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1112.65 | 1090.34 | 1118.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1112.65 | 1090.34 | 1118.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 1130.00 | 1090.34 | 1118.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 998.30 | 969.74 | 995.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:00:00 | 998.30 | 969.74 | 995.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 995.00 | 974.80 | 995.31 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 15:15:00 | 1000.60 | 994.92 | 994.25 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 988.65 | 993.65 | 993.79 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 997.85 | 994.49 | 994.16 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 988.00 | 993.91 | 994.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 979.50 | 991.03 | 992.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1002.45 | 991.23 | 992.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1002.45 | 991.23 | 992.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1002.45 | 991.23 | 992.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 1004.00 | 991.23 | 992.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 10:15:00 | 1007.45 | 994.47 | 993.73 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 15:15:00 | 984.00 | 993.39 | 993.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 978.70 | 990.45 | 992.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 13:15:00 | 976.70 | 975.74 | 980.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 982.90 | 977.17 | 981.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 982.90 | 977.17 | 981.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 982.90 | 977.17 | 981.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 979.85 | 977.71 | 981.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 957.45 | 977.71 | 981.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 971.20 | 959.79 | 959.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 971.20 | 959.79 | 959.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 984.60 | 965.44 | 962.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 973.60 | 977.79 | 971.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 986.00 | 979.83 | 972.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:00:00 | 988.00 | 979.83 | 972.93 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 972.25 | 979.16 | 974.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-07 12:15:00 | 972.25 | 979.16 | 974.48 | SL hit (close<ema400) qty=1.00 sl=974.48 alert=retest1 |

### Cycle 57 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 948.55 | 969.79 | 971.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 10:15:00 | 948.05 | 965.44 | 969.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 951.00 | 948.97 | 957.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 951.00 | 948.97 | 957.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 949.70 | 949.12 | 956.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 938.90 | 944.84 | 951.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 943.00 | 932.24 | 931.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 943.00 | 932.24 | 931.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 952.15 | 937.97 | 934.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 934.95 | 938.31 | 936.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 13:15:00 | 934.95 | 938.31 | 936.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 934.95 | 938.31 | 936.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:45:00 | 934.45 | 938.31 | 936.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 932.30 | 937.10 | 935.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:45:00 | 933.50 | 937.10 | 935.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1024.75 | 1042.85 | 1025.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1025.50 | 1042.85 | 1025.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1013.60 | 1037.00 | 1024.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1014.20 | 1037.00 | 1024.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1008.15 | 1031.23 | 1022.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 998.25 | 1031.23 | 1022.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1020.40 | 1020.71 | 1019.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 1018.10 | 1020.71 | 1019.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 1022.75 | 1021.11 | 1020.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:45:00 | 1030.25 | 1021.11 | 1020.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1024.35 | 1021.81 | 1020.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:30:00 | 1021.00 | 1021.81 | 1020.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1026.65 | 1022.78 | 1021.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:30:00 | 1022.70 | 1022.78 | 1021.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1017.40 | 1021.62 | 1020.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 1030.00 | 1021.54 | 1020.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-03 09:15:00 | 1133.00 | 1068.49 | 1066.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 1045.10 | 1063.28 | 1064.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 13:15:00 | 1044.65 | 1056.79 | 1061.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 965.25 | 949.11 | 976.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 965.25 | 949.11 | 976.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 965.25 | 949.11 | 976.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 968.15 | 949.11 | 976.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 964.95 | 950.05 | 965.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 928.55 | 950.05 | 965.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 12:15:00 | 956.90 | 932.06 | 931.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 956.90 | 932.06 | 931.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 976.90 | 948.71 | 940.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 1069.50 | 1075.59 | 1052.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 1090.50 | 1074.08 | 1062.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1074.70 | 1082.94 | 1077.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 1074.70 | 1082.94 | 1077.88 | SL hit (close<ema400) qty=1.00 sl=1077.88 alert=retest1 |

### Cycle 61 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1037.20 | 1071.95 | 1073.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 1028.10 | 1047.81 | 1060.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 1043.90 | 1040.83 | 1052.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 1043.90 | 1040.83 | 1052.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1030.20 | 1036.10 | 1044.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 1012.10 | 1030.82 | 1039.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1015.90 | 1025.32 | 1030.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1008.40 | 1021.68 | 1025.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1016.30 | 1022.66 | 1024.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1034.70 | 1023.01 | 1023.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 1034.70 | 1023.01 | 1023.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 1038.90 | 1026.19 | 1025.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1038.90 | 1026.19 | 1025.28 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 1002.70 | 1023.79 | 1024.88 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1032.70 | 1023.11 | 1022.98 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1005.20 | 1021.58 | 1022.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 992.20 | 1015.70 | 1020.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 1011.20 | 1010.30 | 1015.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 1011.20 | 1010.30 | 1015.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1037.30 | 1015.06 | 1016.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1037.30 | 1015.06 | 1016.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1039.50 | 1019.95 | 1018.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1048.70 | 1036.48 | 1029.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1031.30 | 1037.66 | 1031.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1033.00 | 1037.66 | 1031.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1037.10 | 1037.54 | 1032.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1039.90 | 1037.44 | 1032.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 1043.90 | 1037.57 | 1033.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 1063.50 | 1071.38 | 1072.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 1063.50 | 1071.38 | 1072.21 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1180.90 | 1093.28 | 1082.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1284.40 | 1145.52 | 1108.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1357.10 | 1360.15 | 1290.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 1355.80 | 1358.53 | 1323.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1355.80 | 1358.53 | 1323.90 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1330.10 | 1338.63 | 1339.15 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1348.40 | 1340.01 | 1339.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 1349.90 | 1343.41 | 1341.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1326.70 | 1341.92 | 1341.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 1326.70 | 1341.92 | 1341.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1336.80 | 1340.90 | 1340.95 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1345.70 | 1333.88 | 1333.18 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1329.10 | 1332.77 | 1332.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 14:15:00 | 1325.00 | 1331.08 | 1332.06 | Break + close below crossover candle low |

### Cycle 74 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 1344.00 | 1333.33 | 1332.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 1358.20 | 1339.49 | 1336.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 1367.10 | 1368.85 | 1361.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 1367.10 | 1368.85 | 1361.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1366.60 | 1369.47 | 1363.37 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 1327.80 | 1356.19 | 1358.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1317.00 | 1328.99 | 1333.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1333.20 | 1318.49 | 1323.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1322.00 | 1318.49 | 1323.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1337.50 | 1322.29 | 1324.95 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1349.90 | 1330.44 | 1328.36 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 1334.80 | 1336.65 | 1336.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 1316.00 | 1331.79 | 1334.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1324.70 | 1319.29 | 1324.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1337.30 | 1319.29 | 1324.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1311.00 | 1317.64 | 1323.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 1305.00 | 1317.64 | 1323.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 1301.70 | 1305.08 | 1313.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1305.30 | 1305.26 | 1312.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:45:00 | 1300.00 | 1305.48 | 1311.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1285.00 | 1299.34 | 1306.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 1282.00 | 1299.34 | 1306.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 1282.10 | 1293.98 | 1302.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:45:00 | 1283.00 | 1287.46 | 1297.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 1283.30 | 1284.83 | 1294.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1285.50 | 1272.92 | 1281.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1285.50 | 1272.92 | 1281.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1284.40 | 1275.21 | 1281.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 1280.00 | 1278.53 | 1282.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 1280.00 | 1278.53 | 1282.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:15:00 | 1239.75 | 1252.75 | 1263.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:15:00 | 1240.03 | 1252.75 | 1263.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1236.62 | 1249.20 | 1261.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:15:00 | 1235.00 | 1249.20 | 1261.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1250.50 | 1246.47 | 1255.00 | SL hit (close>ema200) qty=0.50 sl=1246.47 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1270.00 | 1253.69 | 1251.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 1283.40 | 1269.95 | 1261.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 1358.60 | 1364.50 | 1348.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 1358.60 | 1364.50 | 1348.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1351.60 | 1364.73 | 1362.53 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 1352.00 | 1360.12 | 1360.69 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1399.00 | 1365.74 | 1362.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 1418.00 | 1376.19 | 1367.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 1408.10 | 1411.20 | 1394.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 1408.10 | 1411.20 | 1394.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1427.00 | 1424.68 | 1416.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:30:00 | 1430.70 | 1425.43 | 1417.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:00:00 | 1428.40 | 1425.43 | 1417.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 1408.00 | 1421.11 | 1416.55 | SL hit (close<static) qty=1.00 sl=1413.40 alert=retest2 |

### Cycle 81 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1399.70 | 1411.49 | 1412.99 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 1425.00 | 1414.19 | 1414.08 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1394.90 | 1410.33 | 1412.34 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 1422.10 | 1410.84 | 1410.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1431.00 | 1415.37 | 1412.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1425.00 | 1426.45 | 1420.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1425.00 | 1426.45 | 1420.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1405.90 | 1421.98 | 1419.13 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1390.40 | 1415.67 | 1416.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1374.30 | 1396.63 | 1404.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1413.40 | 1388.27 | 1396.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:15:00 | 1414.00 | 1388.27 | 1396.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1415.00 | 1393.62 | 1398.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 1411.00 | 1393.62 | 1398.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 1412.10 | 1402.54 | 1401.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1426.70 | 1410.17 | 1405.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1425.00 | 1426.53 | 1417.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:15:00 | 1425.00 | 1426.53 | 1417.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1427.00 | 1426.62 | 1418.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:15:00 | 1430.10 | 1427.60 | 1420.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 1437.00 | 1426.65 | 1421.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 1387.30 | 1419.15 | 1420.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 1387.30 | 1419.15 | 1420.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 1374.20 | 1393.28 | 1403.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1378.70 | 1377.13 | 1392.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 1379.60 | 1377.13 | 1392.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1392.70 | 1382.79 | 1391.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1394.20 | 1382.79 | 1391.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1382.00 | 1382.63 | 1390.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1374.00 | 1382.63 | 1390.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1397.10 | 1385.52 | 1391.02 | SL hit (close>static) qty=1.00 sl=1395.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1404.10 | 1394.45 | 1394.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1414.30 | 1403.07 | 1398.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 1453.00 | 1453.66 | 1438.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1462.00 | 1453.66 | 1438.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1455.60 | 1454.05 | 1440.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 1450.90 | 1454.05 | 1440.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1450.60 | 1456.69 | 1447.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1450.60 | 1456.69 | 1447.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1451.10 | 1455.57 | 1447.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 1443.90 | 1453.24 | 1447.19 | SL hit (close<ema400) qty=1.00 sl=1447.19 alert=retest1 |

### Cycle 89 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1465.10 | 1488.31 | 1489.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1422.80 | 1464.19 | 1475.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1395.70 | 1392.07 | 1408.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 1395.70 | 1392.07 | 1408.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1391.00 | 1379.45 | 1385.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:15:00 | 1395.30 | 1379.45 | 1385.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1392.40 | 1382.04 | 1386.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:15:00 | 1379.90 | 1384.88 | 1386.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:30:00 | 1374.80 | 1379.91 | 1382.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 1382.20 | 1374.67 | 1375.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1384.70 | 1374.67 | 1375.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 1385.30 | 1376.79 | 1376.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1385.30 | 1376.79 | 1376.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1389.10 | 1379.25 | 1377.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1375.50 | 1379.15 | 1377.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1375.50 | 1379.15 | 1377.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1367.20 | 1376.76 | 1376.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1367.20 | 1376.76 | 1376.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 1361.00 | 1373.61 | 1375.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 1357.70 | 1370.43 | 1373.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1375.90 | 1368.40 | 1371.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1377.90 | 1368.40 | 1371.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1374.20 | 1369.56 | 1371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1375.20 | 1369.56 | 1371.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1384.60 | 1375.05 | 1373.81 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 1360.50 | 1373.82 | 1373.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 1348.40 | 1367.13 | 1370.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1360.60 | 1354.83 | 1358.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 1341.70 | 1351.71 | 1354.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 1341.10 | 1346.44 | 1350.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:45:00 | 1343.90 | 1344.51 | 1348.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1337.00 | 1344.81 | 1348.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1329.90 | 1328.08 | 1330.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 1328.40 | 1328.56 | 1330.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1328.50 | 1328.56 | 1330.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1334.60 | 1329.76 | 1330.56 | SL hit (close>static) qty=1.00 sl=1330.40 alert=retest2 |

### Cycle 94 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1304.80 | 1293.72 | 1292.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1328.70 | 1305.20 | 1298.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1410.00 | 1413.97 | 1389.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1410.00 | 1413.97 | 1389.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1401.20 | 1410.21 | 1391.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1422.80 | 1404.59 | 1398.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 1390.00 | 1409.00 | 1403.94 | SL hit (close<static) qty=1.00 sl=1390.10 alert=retest2 |

### Cycle 95 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1379.80 | 1399.24 | 1400.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 1368.90 | 1389.55 | 1395.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1362.20 | 1352.55 | 1367.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1361.20 | 1352.55 | 1367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1364.70 | 1354.98 | 1367.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1364.20 | 1354.98 | 1367.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1361.50 | 1356.28 | 1366.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1366.20 | 1356.28 | 1366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1363.50 | 1358.99 | 1365.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:15:00 | 1365.80 | 1358.99 | 1365.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1365.80 | 1360.35 | 1365.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 1367.00 | 1360.35 | 1365.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1361.00 | 1360.48 | 1365.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1354.60 | 1360.48 | 1365.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 1356.00 | 1358.11 | 1363.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1356.30 | 1354.61 | 1358.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 1380.90 | 1358.21 | 1357.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1380.90 | 1358.21 | 1357.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 1383.40 | 1363.25 | 1359.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1384.00 | 1386.64 | 1377.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1384.00 | 1386.64 | 1377.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1383.90 | 1385.64 | 1378.62 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1354.70 | 1374.30 | 1374.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 1348.40 | 1359.87 | 1365.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1347.40 | 1337.45 | 1346.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1347.40 | 1337.45 | 1346.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1343.00 | 1338.56 | 1346.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 1342.20 | 1338.56 | 1346.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 1342.60 | 1338.67 | 1344.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 1338.80 | 1336.86 | 1339.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1373.50 | 1330.36 | 1327.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 1373.50 | 1330.36 | 1327.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 1381.50 | 1340.59 | 1331.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1335.00 | 1352.69 | 1342.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 1372.00 | 1361.25 | 1349.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1404.80 | 1359.46 | 1352.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1371.00 | 1389.83 | 1388.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1340.90 | 1380.05 | 1384.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1340.90 | 1380.05 | 1384.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 11:15:00 | 1326.60 | 1365.11 | 1376.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1405.40 | 1359.93 | 1367.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1405.40 | 1359.93 | 1367.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1395.00 | 1366.94 | 1370.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:15:00 | 1381.90 | 1366.94 | 1370.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1388.30 | 1374.55 | 1373.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 1388.30 | 1374.55 | 1373.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1420.60 | 1384.68 | 1379.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1348.00 | 1384.30 | 1383.68 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1354.60 | 1378.36 | 1381.04 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 1369.90 | 1362.85 | 1362.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 1373.30 | 1364.94 | 1363.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1385.00 | 1391.74 | 1379.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1413.00 | 1400.95 | 1392.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1417.90 | 1400.95 | 1392.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1415.00 | 1402.84 | 1394.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 1417.80 | 1408.08 | 1398.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1418.10 | 1428.18 | 1420.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1418.10 | 1428.18 | 1420.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1409.30 | 1424.41 | 1419.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1410.30 | 1424.41 | 1419.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1414.50 | 1422.43 | 1418.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1383.20 | 1414.58 | 1415.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1383.20 | 1414.58 | 1415.56 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1409.40 | 1403.74 | 1403.68 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1395.50 | 1402.29 | 1403.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 1394.60 | 1399.99 | 1401.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1398.10 | 1396.54 | 1399.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 1393.30 | 1394.47 | 1397.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1406.90 | 1392.34 | 1393.66 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1408.10 | 1395.49 | 1394.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 1418.80 | 1404.16 | 1399.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 1406.00 | 1407.49 | 1402.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:45:00 | 1405.00 | 1407.49 | 1402.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1401.30 | 1406.25 | 1402.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 1402.10 | 1406.25 | 1402.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1404.30 | 1405.86 | 1402.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 1401.30 | 1405.86 | 1402.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1403.00 | 1405.29 | 1402.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 1403.00 | 1405.29 | 1402.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1411.00 | 1406.43 | 1403.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 1415.00 | 1406.45 | 1404.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1429.10 | 1406.95 | 1405.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 1405.90 | 1416.32 | 1417.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 1405.90 | 1416.32 | 1417.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 1401.70 | 1413.40 | 1415.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1391.70 | 1385.13 | 1391.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1392.80 | 1385.13 | 1391.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1399.00 | 1387.90 | 1392.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1386.70 | 1387.90 | 1392.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 1387.00 | 1387.43 | 1391.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 1385.80 | 1386.31 | 1390.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 1382.90 | 1388.15 | 1389.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1387.00 | 1386.98 | 1388.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1389.70 | 1386.98 | 1388.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1385.00 | 1386.59 | 1388.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:30:00 | 1383.80 | 1386.59 | 1388.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1385.60 | 1386.14 | 1387.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 1383.20 | 1385.33 | 1387.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 1383.00 | 1386.06 | 1387.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:00:00 | 1384.60 | 1386.06 | 1387.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 1411.10 | 1390.04 | 1388.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 10:15:00 | 1411.10 | 1390.04 | 1388.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 1414.50 | 1394.93 | 1390.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1406.00 | 1410.13 | 1401.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1406.00 | 1410.13 | 1401.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1400.20 | 1408.15 | 1400.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1400.20 | 1408.15 | 1400.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1386.10 | 1403.74 | 1399.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 1409.20 | 1403.74 | 1399.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 1401.00 | 1406.56 | 1403.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 1401.50 | 1405.04 | 1403.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:00:00 | 1403.30 | 1407.39 | 1406.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1405.00 | 1406.82 | 1406.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1392.20 | 1406.82 | 1406.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1386.30 | 1402.71 | 1404.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1386.30 | 1402.71 | 1404.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1381.40 | 1398.45 | 1402.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1389.50 | 1386.67 | 1394.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1389.50 | 1386.67 | 1394.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1402.70 | 1389.87 | 1395.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1382.20 | 1389.87 | 1395.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1381.20 | 1388.14 | 1393.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:30:00 | 1376.90 | 1386.17 | 1392.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1375.50 | 1386.17 | 1392.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1403.60 | 1385.47 | 1384.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 1403.60 | 1385.47 | 1384.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1412.20 | 1393.30 | 1388.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 14:15:00 | 1528.00 | 1532.70 | 1506.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1528.00 | 1532.70 | 1506.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1518.80 | 1536.77 | 1522.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1518.80 | 1536.77 | 1522.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1524.40 | 1534.30 | 1522.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1509.30 | 1534.30 | 1522.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1526.40 | 1532.72 | 1522.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 1503.80 | 1532.72 | 1522.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1534.20 | 1533.01 | 1523.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 1522.20 | 1533.01 | 1523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1523.80 | 1531.17 | 1523.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 1521.60 | 1531.17 | 1523.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1537.80 | 1532.50 | 1525.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 1545.30 | 1532.50 | 1525.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1541.90 | 1534.02 | 1526.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 1545.50 | 1535.61 | 1527.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1496.00 | 1528.81 | 1526.27 | SL hit (close<static) qty=1.00 sl=1522.00 alert=retest2 |

### Cycle 111 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 1496.10 | 1522.27 | 1523.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 1481.20 | 1514.05 | 1519.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1537.00 | 1509.26 | 1513.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 1537.00 | 1509.26 | 1513.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1541.20 | 1515.65 | 1516.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 1541.20 | 1515.65 | 1516.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 1536.80 | 1519.88 | 1518.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 1550.60 | 1531.79 | 1524.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1554.20 | 1557.95 | 1545.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 1554.20 | 1557.95 | 1545.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1551.70 | 1556.70 | 1545.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1552.30 | 1556.70 | 1545.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1561.00 | 1558.99 | 1552.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1549.50 | 1558.99 | 1552.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1552.60 | 1557.72 | 1552.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 1553.70 | 1557.72 | 1552.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1553.20 | 1556.81 | 1552.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:30:00 | 1565.20 | 1557.45 | 1553.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 1544.60 | 1553.93 | 1552.11 | SL hit (close<static) qty=1.00 sl=1547.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 1532.20 | 1549.62 | 1550.94 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 1559.10 | 1550.24 | 1549.82 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1549.20 | 1551.64 | 1551.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1480.00 | 1537.31 | 1545.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1548.20 | 1539.49 | 1545.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 1547.10 | 1539.49 | 1545.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 1520.50 | 1535.69 | 1543.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 1513.90 | 1530.17 | 1540.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 13:15:00 | 1438.20 | 1473.50 | 1500.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 1442.50 | 1440.74 | 1462.61 | SL hit (close>ema200) qty=0.50 sl=1440.74 alert=retest2 |

### Cycle 116 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1497.30 | 1463.10 | 1461.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1512.80 | 1480.95 | 1475.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1474.00 | 1511.51 | 1503.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1474.00 | 1511.51 | 1503.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1486.60 | 1506.53 | 1501.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 1482.20 | 1506.53 | 1501.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 1475.40 | 1496.62 | 1497.85 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1516.40 | 1496.09 | 1495.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 15:15:00 | 1520.00 | 1503.75 | 1498.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 1536.70 | 1536.90 | 1523.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:45:00 | 1533.70 | 1536.90 | 1523.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1533.90 | 1534.68 | 1524.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 1530.90 | 1534.68 | 1524.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1510.00 | 1529.75 | 1523.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 1510.00 | 1529.75 | 1523.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1510.80 | 1525.96 | 1522.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 1512.20 | 1525.96 | 1522.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1491.20 | 1515.88 | 1518.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1480.90 | 1500.20 | 1508.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 1475.20 | 1474.43 | 1485.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:45:00 | 1475.80 | 1474.43 | 1485.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1489.00 | 1477.11 | 1485.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 1489.00 | 1477.11 | 1485.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 1485.00 | 1478.69 | 1485.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 1490.60 | 1478.69 | 1485.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1476.30 | 1478.21 | 1484.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 1463.20 | 1478.21 | 1484.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:30:00 | 1465.00 | 1475.23 | 1481.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1464.60 | 1473.44 | 1478.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 1465.10 | 1471.22 | 1476.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1466.80 | 1468.38 | 1473.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:15:00 | 1476.40 | 1468.38 | 1473.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1476.40 | 1469.98 | 1473.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1487.60 | 1469.98 | 1473.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1480.30 | 1472.05 | 1474.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 1461.50 | 1467.92 | 1471.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 1460.90 | 1467.25 | 1469.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:15:00 | 1391.75 | 1430.25 | 1443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:15:00 | 1391.37 | 1430.25 | 1443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:15:00 | 1391.84 | 1430.25 | 1443.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1390.04 | 1417.22 | 1435.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1388.42 | 1417.22 | 1435.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1387.86 | 1417.22 | 1435.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1427.70 | 1418.16 | 1432.37 | SL hit (close>ema200) qty=0.50 sl=1418.16 alert=retest2 |

### Cycle 120 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1448.50 | 1437.84 | 1436.97 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 15:15:00 | 1431.90 | 1435.90 | 1436.19 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1438.50 | 1436.42 | 1436.40 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 10:15:00 | 1431.90 | 1435.51 | 1435.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 1417.90 | 1431.27 | 1433.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1431.90 | 1431.03 | 1433.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 1431.90 | 1431.03 | 1433.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1425.00 | 1429.83 | 1432.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1426.90 | 1429.83 | 1432.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1464.80 | 1436.82 | 1435.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 12:15:00 | 1475.00 | 1456.10 | 1448.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1436.50 | 1458.60 | 1452.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1430.00 | 1458.60 | 1452.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1427.20 | 1452.32 | 1449.89 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1425.10 | 1444.01 | 1446.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1417.20 | 1438.65 | 1443.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1446.20 | 1439.54 | 1443.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:45:00 | 1442.60 | 1439.54 | 1443.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1450.00 | 1441.63 | 1443.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1450.90 | 1441.63 | 1443.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 09:15:00 | 1490.00 | 1451.31 | 1448.01 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 1438.20 | 1451.82 | 1453.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 12:15:00 | 1429.60 | 1445.55 | 1449.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1454.80 | 1428.29 | 1437.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 1454.80 | 1428.29 | 1437.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 12:15:00 | 1514.10 | 1445.45 | 1444.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1522.00 | 1460.76 | 1451.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1549.70 | 1554.99 | 1528.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:45:00 | 1546.40 | 1554.99 | 1528.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1543.60 | 1555.05 | 1537.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1543.20 | 1555.05 | 1537.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1534.80 | 1551.00 | 1536.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1533.50 | 1551.00 | 1536.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1523.40 | 1545.48 | 1535.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 1523.40 | 1545.48 | 1535.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 1515.00 | 1528.19 | 1529.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1470.20 | 1516.59 | 1524.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 1455.00 | 1452.59 | 1473.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 1455.00 | 1452.59 | 1473.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1507.90 | 1465.58 | 1474.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1507.90 | 1465.58 | 1474.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1498.00 | 1472.07 | 1476.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1486.30 | 1472.07 | 1476.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 1489.90 | 1475.07 | 1477.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 1501.40 | 1480.34 | 1479.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1501.40 | 1480.34 | 1479.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1506.60 | 1493.13 | 1486.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1508.00 | 1520.91 | 1508.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1499.30 | 1520.91 | 1508.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1505.50 | 1517.83 | 1507.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 1510.10 | 1517.83 | 1507.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1501.10 | 1514.49 | 1507.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1501.10 | 1514.49 | 1507.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1475.90 | 1506.77 | 1504.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1475.90 | 1506.77 | 1504.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1474.50 | 1500.31 | 1501.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1468.20 | 1493.89 | 1498.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1499.90 | 1492.09 | 1496.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:30:00 | 1482.20 | 1490.11 | 1495.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1509.80 | 1481.65 | 1481.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1509.80 | 1481.65 | 1481.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 1534.30 | 1492.18 | 1486.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1592.20 | 1623.83 | 1597.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 1592.20 | 1623.83 | 1597.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 1596.00 | 1618.27 | 1597.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 1603.50 | 1618.27 | 1597.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 1605.00 | 1614.61 | 1597.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 14:00:00 | 1604.00 | 1612.49 | 1597.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1652.90 | 1606.46 | 1597.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1600.30 | 1615.92 | 1606.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 1599.60 | 1615.92 | 1606.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1598.60 | 1612.46 | 1605.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1593.00 | 1612.46 | 1605.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1588.80 | 1607.73 | 1604.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 1593.00 | 1607.73 | 1604.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1506.40 | 1584.62 | 1594.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1506.40 | 1584.62 | 1594.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 1478.20 | 1563.34 | 1583.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1551.50 | 1547.41 | 1568.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1551.50 | 1547.41 | 1568.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1558.70 | 1550.42 | 1565.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1566.50 | 1550.42 | 1565.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1549.20 | 1550.17 | 1564.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 1549.20 | 1550.17 | 1564.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1560.00 | 1552.14 | 1563.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:00:00 | 1560.00 | 1552.14 | 1563.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 1558.20 | 1553.35 | 1563.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:45:00 | 1575.70 | 1553.35 | 1563.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1560.20 | 1554.53 | 1560.87 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1574.40 | 1563.40 | 1562.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 1594.30 | 1571.93 | 1566.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1601.40 | 1608.13 | 1596.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 1602.80 | 1608.13 | 1596.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1600.00 | 1606.50 | 1597.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1591.20 | 1606.50 | 1597.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1603.00 | 1605.80 | 1597.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1624.70 | 1605.80 | 1597.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:00:00 | 1610.00 | 1628.93 | 1626.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1628.00 | 1636.84 | 1637.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1628.00 | 1636.84 | 1637.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 09:15:00 | 1617.10 | 1632.89 | 1635.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 1608.50 | 1607.39 | 1618.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:00:00 | 1608.50 | 1607.39 | 1618.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1609.20 | 1602.25 | 1609.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 1612.30 | 1602.25 | 1609.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1600.00 | 1601.80 | 1608.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 1597.50 | 1600.68 | 1607.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 1594.00 | 1600.31 | 1606.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 1633.00 | 1606.00 | 1607.81 | SL hit (close>static) qty=1.00 sl=1624.10 alert=retest2 |

### Cycle 136 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 1657.10 | 1616.22 | 1612.29 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 1607.70 | 1636.69 | 1636.77 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1703.10 | 1646.50 | 1641.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1751.90 | 1681.45 | 1658.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1697.20 | 1707.75 | 1682.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 1697.20 | 1707.75 | 1682.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1668.80 | 1697.74 | 1682.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 1668.80 | 1697.74 | 1682.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1684.50 | 1695.09 | 1682.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 1687.70 | 1692.66 | 1682.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 1686.10 | 1688.04 | 1681.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1695.50 | 1688.04 | 1681.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1643.00 | 1716.32 | 1722.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 1643.00 | 1716.32 | 1722.75 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-05-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-11 12:15:00 | 1689.60 | 1673.51 | 1671.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-11 13:15:00 | 1698.20 | 1678.45 | 1673.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-11 14:15:00 | 1678.00 | 1678.36 | 1674.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-11 14:15:00 | 1678.00 | 1678.36 | 1674.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 14:15:00 | 1678.00 | 1678.36 | 1674.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 15:00:00 | 1678.00 | 1678.36 | 1674.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 15:15:00 | 1674.80 | 1677.65 | 1674.28 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-24 12:15:00 | 1301.05 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-07-24 12:45:00 | 1300.60 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-07-25 09:15:00 | 1293.00 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-25 11:30:00 | 1300.15 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-26 10:15:00 | 1294.90 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-29 09:15:00 | 1293.00 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-29 10:30:00 | 1291.00 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-07-29 15:00:00 | 1292.05 | 2024-07-30 15:15:00 | 1303.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1315.60 | 2024-08-02 09:15:00 | 1292.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest1 | 2024-08-21 09:15:00 | 1335.15 | 2024-08-23 10:15:00 | 1401.91 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 10:15:00 | 1337.05 | 2024-08-23 10:15:00 | 1403.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 11:00:00 | 1339.50 | 2024-08-23 10:15:00 | 1406.48 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 09:15:00 | 1335.15 | 2024-08-27 10:15:00 | 1403.70 | STOP_HIT | 0.50 | 5.13% |
| BUY | retest1 | 2024-08-21 10:15:00 | 1337.05 | 2024-08-27 10:15:00 | 1403.70 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest1 | 2024-08-21 11:00:00 | 1339.50 | 2024-08-27 10:15:00 | 1403.70 | STOP_HIT | 0.50 | 4.79% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1412.30 | 2024-09-06 09:15:00 | 1392.70 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1436.95 | 2024-09-18 10:15:00 | 1576.03 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2024-09-16 11:15:00 | 1432.75 | 2024-09-19 13:15:00 | 1449.00 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2024-09-24 11:15:00 | 1436.10 | 2024-09-25 14:15:00 | 1458.30 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-09-24 11:45:00 | 1433.35 | 2024-09-25 14:15:00 | 1458.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-10-01 11:45:00 | 1509.85 | 2024-10-04 09:15:00 | 1446.10 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-10-01 14:00:00 | 1494.65 | 2024-10-04 09:15:00 | 1446.10 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-10-03 09:30:00 | 1492.65 | 2024-10-04 09:15:00 | 1446.10 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-10-14 09:15:00 | 1472.00 | 2024-10-21 09:15:00 | 1468.20 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-10-24 11:15:00 | 1400.10 | 2024-10-28 14:15:00 | 1405.65 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-10-24 13:45:00 | 1399.95 | 2024-10-28 14:15:00 | 1405.65 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-10-30 09:15:00 | 1432.00 | 2024-11-07 15:15:00 | 1431.05 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-11-11 11:30:00 | 1445.40 | 2024-11-11 12:15:00 | 1456.70 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-11-25 13:15:00 | 1293.55 | 2024-11-26 09:15:00 | 1416.25 | STOP_HIT | 1.00 | -9.49% |
| SELL | retest2 | 2024-12-06 11:15:00 | 1369.80 | 2024-12-11 09:15:00 | 1380.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-12-06 14:00:00 | 1369.35 | 2024-12-11 09:15:00 | 1380.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-12-18 09:15:00 | 1399.10 | 2024-12-19 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-12-19 09:30:00 | 1395.05 | 2024-12-19 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-19 11:15:00 | 1392.90 | 2024-12-19 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-19 14:00:00 | 1393.10 | 2024-12-19 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-30 09:15:00 | 1447.25 | 2025-01-06 10:15:00 | 1445.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-12-30 12:00:00 | 1447.70 | 2025-01-06 10:15:00 | 1445.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-12-31 10:45:00 | 1442.10 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-12-31 12:00:00 | 1444.30 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-01-01 13:15:00 | 1458.35 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-01-01 14:30:00 | 1464.00 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-01-02 10:45:00 | 1458.30 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-01-02 11:15:00 | 1458.05 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-01-02 12:15:00 | 1472.80 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-01-02 14:45:00 | 1476.80 | 2025-01-06 11:15:00 | 1440.80 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-01-20 11:00:00 | 1371.90 | 2025-01-21 10:15:00 | 1357.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1328.05 | 2025-01-27 09:15:00 | 1261.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1337.00 | 2025-01-27 09:15:00 | 1270.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 10:15:00 | 1333.00 | 2025-01-27 09:15:00 | 1266.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1328.05 | 2025-01-28 09:15:00 | 1195.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1337.00 | 2025-01-28 09:15:00 | 1203.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 10:15:00 | 1333.00 | 2025-01-28 09:15:00 | 1199.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 15:15:00 | 1230.00 | 2025-02-07 10:15:00 | 1215.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-03-03 09:15:00 | 957.45 | 2025-03-05 14:15:00 | 971.20 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-03-07 09:30:00 | 986.00 | 2025-03-07 12:15:00 | 972.25 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest1 | 2025-03-07 10:00:00 | 988.00 | 2025-03-07 12:15:00 | 972.25 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-03-12 09:30:00 | 938.90 | 2025-03-18 13:15:00 | 943.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-03-27 14:15:00 | 1030.00 | 2025-04-03 09:15:00 | 1133.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 928.55 | 2025-04-15 12:15:00 | 956.90 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest1 | 2025-04-23 10:00:00 | 1090.50 | 2025-04-24 13:15:00 | 1074.70 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-04-29 14:15:00 | 1012.10 | 2025-05-06 11:15:00 | 1038.90 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-05-02 09:15:00 | 1015.90 | 2025-05-06 11:15:00 | 1038.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-05 09:15:00 | 1008.40 | 2025-05-06 11:15:00 | 1038.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1016.30 | 2025-05-06 11:15:00 | 1038.90 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1039.90 | 2025-05-22 13:15:00 | 1063.50 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-05-14 10:15:00 | 1043.90 | 2025-05-22 13:15:00 | 1063.50 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-06-27 11:15:00 | 1305.00 | 2025-07-07 10:15:00 | 1239.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 09:45:00 | 1301.70 | 2025-07-07 10:15:00 | 1240.03 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1305.30 | 2025-07-07 11:15:00 | 1236.62 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-06-30 12:45:00 | 1300.00 | 2025-07-07 11:15:00 | 1235.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-27 11:15:00 | 1305.00 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-06-30 09:45:00 | 1301.70 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1305.30 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-06-30 12:45:00 | 1300.00 | 2025-07-08 09:15:00 | 1250.50 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-07-01 10:15:00 | 1282.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-07-01 12:00:00 | 1282.10 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-07-01 14:45:00 | 1283.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-07-02 09:30:00 | 1283.30 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-07-03 14:30:00 | 1280.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-07-03 15:15:00 | 1280.00 | 2025-07-10 09:15:00 | 1270.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-07-25 10:30:00 | 1430.70 | 2025-07-25 12:15:00 | 1408.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-07-25 11:00:00 | 1428.40 | 2025-07-25 12:15:00 | 1408.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-08-06 13:15:00 | 1430.10 | 2025-08-07 13:15:00 | 1387.30 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-08-07 09:15:00 | 1437.00 | 2025-08-07 13:15:00 | 1387.30 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1374.00 | 2025-08-12 09:15:00 | 1397.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2025-08-18 09:15:00 | 1462.00 | 2025-08-19 09:15:00 | 1443.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-19 11:15:00 | 1462.50 | 2025-08-26 09:15:00 | 1465.10 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-09-04 14:15:00 | 1379.90 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-05 14:30:00 | 1374.80 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-09 12:30:00 | 1382.20 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-09-09 13:00:00 | 1384.70 | 2025-09-09 13:15:00 | 1385.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-18 14:30:00 | 1341.70 | 2025-09-25 09:15:00 | 1334.60 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-09-19 11:45:00 | 1341.10 | 2025-09-25 09:15:00 | 1334.60 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-09-19 14:45:00 | 1343.90 | 2025-09-25 13:15:00 | 1331.30 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1337.00 | 2025-09-25 13:15:00 | 1331.30 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1328.40 | 2025-09-26 15:15:00 | 1276.70 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1328.50 | 2025-09-29 09:15:00 | 1274.62 | PARTIAL | 0.50 | 4.06% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1328.90 | 2025-09-29 09:15:00 | 1274.04 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2025-09-25 13:00:00 | 1328.70 | 2025-09-29 09:15:00 | 1270.15 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1328.40 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1328.50 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1328.90 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-09-25 13:00:00 | 1328.70 | 2025-09-29 13:15:00 | 1288.50 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1422.80 | 2025-10-13 10:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-10-16 10:15:00 | 1354.60 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1356.00 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 11:00:00 | 1356.30 | 2025-10-20 12:15:00 | 1380.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-29 12:15:00 | 1342.20 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-29 14:45:00 | 1342.60 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-10-31 09:45:00 | 1338.80 | 2025-11-06 09:15:00 | 1373.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-11-07 11:30:00 | 1372.00 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-11-10 09:15:00 | 1404.80 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1371.00 | 2025-11-12 09:15:00 | 1340.90 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-11-13 11:15:00 | 1381.90 | 2025-11-13 12:15:00 | 1388.30 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1413.00 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-11-27 14:15:00 | 1417.90 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1415.00 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-11-28 09:45:00 | 1417.80 | 2025-12-02 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1393.30 | 2025-12-09 10:15:00 | 1406.90 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-11 14:15:00 | 1415.00 | 2025-12-15 13:15:00 | 1405.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1429.10 | 2025-12-15 13:15:00 | 1405.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1386.70 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-18 11:00:00 | 1387.00 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-18 12:30:00 | 1385.80 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-19 11:45:00 | 1382.90 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-22 10:30:00 | 1383.20 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-12-22 12:30:00 | 1383.00 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-12-22 13:00:00 | 1384.60 | 2025-12-23 10:15:00 | 1411.10 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-24 12:15:00 | 1409.20 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-26 11:15:00 | 1401.00 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-26 13:30:00 | 1401.50 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-29 14:00:00 | 1403.30 | 2025-12-30 09:15:00 | 1386.30 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-31 10:30:00 | 1376.90 | 2026-01-01 14:15:00 | 1403.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1375.50 | 2026-01-01 14:15:00 | 1403.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-01-09 13:15:00 | 1545.30 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-09 13:45:00 | 1541.90 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-01-09 14:30:00 | 1545.50 | 2026-01-12 09:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2026-01-19 12:30:00 | 1565.20 | 2026-01-19 14:15:00 | 1544.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-19 14:30:00 | 1556.00 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-20 09:45:00 | 1557.90 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-01-20 10:30:00 | 1556.80 | 2026-01-20 13:15:00 | 1532.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1513.90 | 2026-01-28 13:15:00 | 1438.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1513.90 | 2026-01-29 15:15:00 | 1442.50 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2026-02-16 10:15:00 | 1463.20 | 2026-02-23 09:15:00 | 1391.75 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2026-02-16 12:30:00 | 1465.00 | 2026-02-23 09:15:00 | 1391.37 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1464.60 | 2026-02-23 09:15:00 | 1391.84 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1465.10 | 2026-02-23 11:15:00 | 1390.04 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-02-18 15:15:00 | 1461.50 | 2026-02-23 11:15:00 | 1388.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:15:00 | 1460.90 | 2026-02-23 11:15:00 | 1387.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 10:15:00 | 1463.20 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2026-02-16 12:30:00 | 1465.00 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1464.60 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.52% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1465.10 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-02-18 15:15:00 | 1461.50 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2026-02-19 12:15:00 | 1460.90 | 2026-02-23 13:15:00 | 1427.70 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1486.30 | 2026-03-17 12:15:00 | 1501.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-17 11:45:00 | 1489.90 | 2026-03-17 12:15:00 | 1501.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-20 10:30:00 | 1482.20 | 2026-03-24 10:15:00 | 1509.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-30 12:15:00 | 1603.50 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.06% |
| BUY | retest2 | 2026-03-30 13:15:00 | 1605.00 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest2 | 2026-03-30 14:00:00 | 1604.00 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -6.08% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1652.90 | 2026-04-02 09:15:00 | 1506.40 | STOP_HIT | 1.00 | -8.86% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1624.70 | 2026-04-20 15:15:00 | 1628.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-04-16 14:00:00 | 1610.00 | 2026-04-20 15:15:00 | 1628.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-04-23 11:30:00 | 1597.50 | 2026-04-23 15:15:00 | 1633.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-23 14:15:00 | 1594.00 | 2026-04-23 15:15:00 | 1633.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-04-30 13:45:00 | 1687.70 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-04-30 14:45:00 | 1686.10 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-30 15:15:00 | 1695.50 | 2026-05-06 09:15:00 | 1643.00 | STOP_HIT | 1.00 | -3.10% |
