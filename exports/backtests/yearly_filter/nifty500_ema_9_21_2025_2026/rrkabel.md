# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 1928.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 27 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 82 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 50
- **Target hits / Stop hits / Partials:** 2 / 81 / 14
- **Avg / median % per leg:** 0.57% / -0.06%
- **Sum % (uncompounded):** 55.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 12 | 27.9% | 2 | 41 | 0 | -0.68% | -29.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| BUY @ 3rd Alert (retest2) | 42 | 12 | 28.6% | 2 | 40 | 0 | -0.66% | -27.9% |
| SELL (all) | 54 | 35 | 64.8% | 0 | 40 | 14 | 1.56% | 84.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 35 | 64.8% | 0 | 40 | 14 | 1.56% | 84.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| retest2 (combined) | 96 | 47 | 49.0% | 2 | 80 | 14 | 0.59% | 56.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 1304.70 | 1309.47 | 1310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 15:15:00 | 1294.90 | 1306.55 | 1308.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 1309.90 | 1306.96 | 1308.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1310.50 | 1307.67 | 1308.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:15:00 | 1311.00 | 1307.67 | 1308.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1309.10 | 1307.95 | 1308.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 1303.80 | 1307.95 | 1308.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1304.00 | 1293.70 | 1297.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 1300.70 | 1293.70 | 1297.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 1304.30 | 1298.30 | 1298.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 1301.10 | 1298.86 | 1298.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1301.10 | 1298.86 | 1298.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1323.00 | 1304.62 | 1301.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1311.00 | 1313.08 | 1307.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1311.00 | 1313.08 | 1307.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1307.00 | 1311.86 | 1307.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1329.00 | 1311.86 | 1307.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 1326.00 | 1321.05 | 1315.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1318.00 | 1316.68 | 1316.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 1298.20 | 1312.98 | 1314.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 1298.20 | 1312.98 | 1314.85 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1319.60 | 1312.99 | 1312.98 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 1311.70 | 1312.73 | 1312.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 1309.90 | 1312.17 | 1312.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1326.60 | 1312.71 | 1312.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1321.40 | 1314.45 | 1313.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 1337.70 | 1319.10 | 1315.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1378.60 | 1381.47 | 1353.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:00:00 | 1378.60 | 1381.47 | 1353.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1375.00 | 1379.13 | 1367.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1380.30 | 1376.57 | 1370.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 1383.20 | 1377.80 | 1371.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 1382.00 | 1378.64 | 1372.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 1382.10 | 1380.71 | 1374.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1379.20 | 1382.21 | 1376.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1403.80 | 1382.21 | 1376.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 13:15:00 | 1387.50 | 1400.54 | 1400.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 13:15:00 | 1387.50 | 1400.54 | 1400.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 10:15:00 | 1381.10 | 1391.37 | 1396.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:30:00 | 1368.70 | 1377.15 | 1383.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 1367.20 | 1377.15 | 1383.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1368.20 | 1374.14 | 1380.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1369.00 | 1374.14 | 1380.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1377.80 | 1374.08 | 1379.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1378.60 | 1374.08 | 1379.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1384.80 | 1376.23 | 1379.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:30:00 | 1388.90 | 1376.23 | 1379.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1364.20 | 1373.82 | 1378.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 1361.00 | 1373.03 | 1377.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 1361.00 | 1372.74 | 1375.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 1361.90 | 1370.28 | 1374.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 1360.00 | 1353.14 | 1359.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1356.80 | 1353.87 | 1359.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 1346.70 | 1352.44 | 1357.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1300.26 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1298.84 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1299.79 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1300.55 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 1325.40 | 1319.95 | 1330.68 | SL hit (close>ema200) qty=0.50 sl=1319.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1350.50 | 1327.68 | 1326.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1383.50 | 1352.97 | 1346.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1358.80 | 1363.09 | 1355.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1356.00 | 1361.67 | 1355.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1356.00 | 1361.67 | 1355.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1347.90 | 1358.92 | 1354.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1352.30 | 1358.92 | 1354.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1346.00 | 1356.33 | 1353.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 1349.60 | 1354.99 | 1353.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 1349.90 | 1353.59 | 1352.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 1350.00 | 1352.39 | 1352.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 1350.00 | 1352.39 | 1352.42 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 1358.80 | 1353.12 | 1352.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 10:15:00 | 1368.80 | 1356.25 | 1354.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 12:15:00 | 1373.00 | 1374.78 | 1367.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 13:00:00 | 1373.00 | 1374.78 | 1367.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1366.40 | 1373.10 | 1367.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1368.30 | 1373.10 | 1367.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1362.80 | 1371.04 | 1367.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 1362.50 | 1371.04 | 1367.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1362.00 | 1369.23 | 1366.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 1370.00 | 1369.23 | 1366.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 15:15:00 | 1372.00 | 1378.77 | 1379.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 1372.00 | 1378.77 | 1379.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 1348.40 | 1372.70 | 1376.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1344.60 | 1343.81 | 1353.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 1344.60 | 1343.81 | 1353.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1351.60 | 1345.37 | 1353.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 1315.90 | 1345.37 | 1353.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 1335.00 | 1343.13 | 1348.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 1335.70 | 1343.11 | 1346.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:45:00 | 1335.00 | 1340.52 | 1342.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 1371.70 | 1342.69 | 1342.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1371.70 | 1342.69 | 1342.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1405.30 | 1381.56 | 1369.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 1380.00 | 1381.25 | 1370.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:30:00 | 1380.00 | 1381.25 | 1370.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1439.60 | 1462.23 | 1444.72 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 1430.40 | 1437.81 | 1438.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 1428.00 | 1434.68 | 1437.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1457.00 | 1441.31 | 1439.27 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1436.50 | 1446.93 | 1447.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 1403.90 | 1433.33 | 1440.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1382.10 | 1380.23 | 1397.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 1382.10 | 1380.23 | 1397.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1387.40 | 1382.57 | 1395.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 1387.10 | 1382.57 | 1395.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1392.80 | 1386.76 | 1394.42 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1412.90 | 1399.65 | 1398.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 1428.30 | 1410.23 | 1403.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1421.90 | 1430.91 | 1420.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1421.90 | 1429.11 | 1420.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:45:00 | 1425.20 | 1429.11 | 1420.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1414.40 | 1426.17 | 1419.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1414.40 | 1426.17 | 1419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1375.80 | 1416.10 | 1415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1375.80 | 1416.10 | 1415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 1334.60 | 1399.80 | 1408.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 1307.40 | 1371.27 | 1393.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 1258.00 | 1256.49 | 1283.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 1258.00 | 1256.49 | 1283.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1281.00 | 1259.04 | 1271.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1281.00 | 1259.04 | 1271.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1284.00 | 1264.03 | 1272.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1268.30 | 1264.03 | 1272.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1258.30 | 1262.00 | 1270.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1254.20 | 1260.44 | 1268.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 1253.90 | 1259.13 | 1267.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1226.00 | 1219.24 | 1219.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1226.00 | 1219.24 | 1219.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1228.50 | 1221.64 | 1220.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 1223.70 | 1223.30 | 1221.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1220.30 | 1222.70 | 1221.51 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1210.00 | 1220.16 | 1220.46 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1236.90 | 1222.04 | 1221.17 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1219.00 | 1221.01 | 1221.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1213.50 | 1219.50 | 1220.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 1215.80 | 1219.26 | 1219.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1202.80 | 1190.75 | 1190.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1202.80 | 1190.75 | 1190.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1209.30 | 1195.63 | 1193.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1210.10 | 1215.84 | 1210.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1206.70 | 1214.01 | 1210.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1206.70 | 1214.01 | 1210.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1212.50 | 1213.71 | 1210.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 1213.80 | 1213.97 | 1210.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 1205.00 | 1209.01 | 1209.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 1205.00 | 1209.01 | 1209.29 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1219.20 | 1211.00 | 1210.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1228.00 | 1220.07 | 1215.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 1218.50 | 1220.93 | 1217.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 12:00:00 | 1218.50 | 1220.93 | 1217.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1214.40 | 1219.63 | 1216.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1214.40 | 1219.63 | 1216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1214.00 | 1218.50 | 1216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1205.50 | 1218.50 | 1216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1220.50 | 1218.90 | 1216.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1229.90 | 1219.12 | 1217.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1242.40 | 1247.00 | 1247.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 1242.40 | 1247.00 | 1247.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 1239.00 | 1245.40 | 1246.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 1243.60 | 1240.89 | 1243.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1247.00 | 1242.11 | 1243.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 1247.30 | 1242.11 | 1243.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1248.00 | 1243.29 | 1244.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1248.00 | 1243.29 | 1244.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 1250.20 | 1245.76 | 1245.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1256.80 | 1247.97 | 1246.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1267.10 | 1267.40 | 1260.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 1265.10 | 1267.40 | 1260.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1275.20 | 1288.78 | 1279.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1275.20 | 1288.78 | 1279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1273.50 | 1285.72 | 1279.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1273.50 | 1285.72 | 1279.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1255.80 | 1274.36 | 1275.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1247.00 | 1260.86 | 1266.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1233.90 | 1232.44 | 1246.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 1234.70 | 1232.44 | 1246.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1230.20 | 1231.95 | 1240.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1238.60 | 1231.95 | 1240.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1240.00 | 1233.56 | 1240.74 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 1254.90 | 1244.82 | 1244.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 1260.50 | 1249.58 | 1246.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1236.20 | 1247.82 | 1246.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1240.40 | 1246.34 | 1245.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 1236.00 | 1246.34 | 1245.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1258.40 | 1248.46 | 1246.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:30:00 | 1265.40 | 1254.17 | 1249.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:30:00 | 1266.50 | 1263.83 | 1257.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1269.60 | 1260.20 | 1256.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 1266.20 | 1263.61 | 1259.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1257.80 | 1263.66 | 1260.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 1270.00 | 1263.26 | 1261.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 1269.00 | 1264.19 | 1262.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:45:00 | 1270.00 | 1265.21 | 1263.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1255.00 | 1262.96 | 1262.71 | SL hit (close<static) qty=1.00 sl=1256.70 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 1254.40 | 1261.25 | 1261.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1251.10 | 1259.22 | 1260.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1256.10 | 1254.62 | 1257.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1256.10 | 1254.62 | 1257.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1268.00 | 1257.74 | 1258.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1281.00 | 1257.74 | 1258.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1270.90 | 1260.37 | 1259.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 1283.60 | 1266.88 | 1262.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1261.60 | 1268.63 | 1265.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1269.80 | 1268.86 | 1265.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 1265.00 | 1268.86 | 1265.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1265.90 | 1268.27 | 1265.97 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1258.00 | 1264.33 | 1264.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1257.00 | 1262.86 | 1263.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1255.70 | 1259.30 | 1261.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1248.60 | 1257.16 | 1260.00 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1272.20 | 1260.44 | 1260.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 1278.00 | 1268.68 | 1264.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1270.00 | 1272.10 | 1268.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:45:00 | 1270.90 | 1272.10 | 1268.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1262.70 | 1270.22 | 1267.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 1262.70 | 1270.22 | 1267.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1261.20 | 1268.41 | 1267.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1271.40 | 1267.13 | 1266.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1254.30 | 1264.56 | 1265.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1254.30 | 1264.56 | 1265.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 14:15:00 | 1253.10 | 1257.87 | 1261.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1264.40 | 1257.66 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1258.00 | 1257.73 | 1260.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1258.00 | 1257.73 | 1260.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1264.50 | 1259.08 | 1260.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 1256.30 | 1259.34 | 1260.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 1252.60 | 1258.85 | 1260.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1335.90 | 1265.96 | 1260.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1335.90 | 1265.96 | 1260.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 1355.20 | 1316.99 | 1293.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 1409.80 | 1410.12 | 1382.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 1409.80 | 1410.12 | 1382.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1406.00 | 1413.22 | 1405.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:30:00 | 1404.80 | 1413.22 | 1405.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1406.70 | 1411.91 | 1405.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1443.80 | 1403.21 | 1402.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 1428.10 | 1414.82 | 1408.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1432.40 | 1414.82 | 1408.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 1427.40 | 1418.54 | 1410.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1420.80 | 1420.58 | 1413.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 1414.70 | 1420.58 | 1413.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1399.00 | 1416.17 | 1412.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1399.00 | 1416.17 | 1412.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1393.00 | 1411.53 | 1411.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1391.20 | 1411.53 | 1411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1390.00 | 1407.23 | 1409.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1390.00 | 1407.23 | 1409.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1356.30 | 1393.04 | 1401.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 1358.20 | 1353.75 | 1364.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:00:00 | 1358.20 | 1353.75 | 1364.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1388.00 | 1361.23 | 1365.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:15:00 | 1392.20 | 1361.23 | 1365.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1394.90 | 1367.97 | 1367.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 1403.90 | 1382.06 | 1374.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1390.90 | 1393.27 | 1384.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 13:00:00 | 1390.90 | 1393.27 | 1384.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1372.30 | 1388.22 | 1384.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1372.30 | 1388.22 | 1384.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1365.00 | 1383.57 | 1382.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1395.50 | 1383.57 | 1382.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1374.80 | 1382.25 | 1381.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 1378.60 | 1381.52 | 1381.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1378.60 | 1381.52 | 1381.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1376.50 | 1380.51 | 1381.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 1386.70 | 1378.13 | 1379.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1384.60 | 1379.42 | 1379.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 1382.40 | 1379.42 | 1379.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1383.20 | 1380.61 | 1380.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1383.20 | 1380.61 | 1380.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1389.50 | 1382.39 | 1381.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 1361.20 | 1382.77 | 1381.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1353.70 | 1376.96 | 1379.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 1349.20 | 1363.20 | 1371.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:45:00 | 1363.50 | 1355.37 | 1363.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1360.00 | 1356.30 | 1363.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1353.90 | 1359.23 | 1362.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1385.50 | 1364.53 | 1364.72 | SL hit (close>static) qty=1.00 sl=1363.70 alert=retest2 |

### Cycle 40 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1379.60 | 1367.54 | 1366.07 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1355.50 | 1366.59 | 1366.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1344.00 | 1360.14 | 1363.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1346.70 | 1341.72 | 1350.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1347.50 | 1342.87 | 1350.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1339.00 | 1342.87 | 1350.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1352.00 | 1344.70 | 1350.37 | SL hit (close>static) qty=1.00 sl=1350.70 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1365.40 | 1354.09 | 1353.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1409.00 | 1364.58 | 1357.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1387.10 | 1387.87 | 1375.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1387.40 | 1387.87 | 1375.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1382.00 | 1386.32 | 1379.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1379.00 | 1386.32 | 1379.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1391.60 | 1401.71 | 1395.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1391.60 | 1401.71 | 1395.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1391.80 | 1399.73 | 1395.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 1397.90 | 1399.73 | 1395.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1391.40 | 1398.06 | 1395.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1391.80 | 1398.06 | 1395.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1401.00 | 1399.39 | 1396.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1396.50 | 1399.39 | 1396.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1404.50 | 1400.41 | 1397.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1388.10 | 1400.41 | 1397.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1397.00 | 1400.08 | 1397.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 1397.00 | 1400.08 | 1397.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1394.80 | 1399.02 | 1397.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 1395.40 | 1399.02 | 1397.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1399.70 | 1399.16 | 1397.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1423.10 | 1398.72 | 1397.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 1400.00 | 1403.27 | 1400.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 1399.80 | 1402.80 | 1400.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1392.60 | 1400.76 | 1399.91 | SL hit (close<static) qty=1.00 sl=1393.50 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 1389.60 | 1397.93 | 1398.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1373.30 | 1390.78 | 1394.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 10:15:00 | 1369.90 | 1367.66 | 1377.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 1369.90 | 1367.66 | 1377.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1374.10 | 1365.43 | 1371.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 1374.10 | 1365.43 | 1371.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1382.60 | 1368.86 | 1372.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1382.60 | 1368.86 | 1372.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 1392.00 | 1377.07 | 1375.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1408.00 | 1385.29 | 1379.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1409.00 | 1409.01 | 1396.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 1409.70 | 1409.01 | 1396.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1438.90 | 1448.54 | 1440.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1438.90 | 1448.54 | 1440.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1449.00 | 1448.63 | 1441.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1430.80 | 1448.63 | 1441.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1424.50 | 1443.80 | 1439.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1428.40 | 1443.80 | 1439.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1427.30 | 1437.25 | 1437.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1421.90 | 1434.18 | 1436.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1414.40 | 1434.10 | 1435.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1464.80 | 1435.81 | 1433.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1464.80 | 1435.81 | 1433.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 1502.70 | 1449.19 | 1439.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1523.40 | 1525.64 | 1500.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 1521.80 | 1525.64 | 1500.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1513.70 | 1524.02 | 1513.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1513.70 | 1524.02 | 1513.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1509.00 | 1521.02 | 1513.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1509.00 | 1521.02 | 1513.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1510.60 | 1518.93 | 1512.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 1508.50 | 1518.93 | 1512.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1510.00 | 1517.15 | 1512.53 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1493.60 | 1509.77 | 1509.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1486.60 | 1505.14 | 1507.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1458.10 | 1457.06 | 1472.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:45:00 | 1460.90 | 1457.06 | 1472.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1455.00 | 1456.65 | 1471.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1450.00 | 1455.32 | 1469.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 1449.60 | 1447.69 | 1455.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 1451.30 | 1448.81 | 1455.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1442.90 | 1449.83 | 1454.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1456.00 | 1451.07 | 1455.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1462.10 | 1451.07 | 1455.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1458.20 | 1452.49 | 1455.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 1456.00 | 1452.49 | 1455.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1465.80 | 1455.15 | 1456.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1465.80 | 1455.15 | 1456.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1467.70 | 1457.66 | 1457.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1467.70 | 1457.66 | 1457.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 1470.00 | 1462.26 | 1459.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 1531.00 | 1534.30 | 1521.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 1534.30 | 1534.30 | 1521.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1557.00 | 1541.74 | 1532.67 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1515.50 | 1530.91 | 1531.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1482.20 | 1509.86 | 1519.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 1514.10 | 1495.76 | 1504.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1506.10 | 1497.82 | 1504.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1498.90 | 1499.49 | 1504.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 1498.20 | 1500.19 | 1504.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 1497.90 | 1498.65 | 1501.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1498.00 | 1498.90 | 1501.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1498.00 | 1498.72 | 1500.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1481.00 | 1498.72 | 1500.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1505.40 | 1500.06 | 1501.37 | SL hit (close>static) qty=1.00 sl=1501.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 1373.50 | 1345.94 | 1344.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1426.80 | 1367.55 | 1355.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1376.30 | 1381.48 | 1365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1372.00 | 1379.49 | 1368.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1353.80 | 1379.49 | 1368.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1342.50 | 1372.09 | 1366.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 1319.90 | 1372.09 | 1366.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1330.00 | 1363.67 | 1363.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1330.00 | 1363.67 | 1363.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1329.60 | 1356.86 | 1360.09 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1417.90 | 1368.60 | 1363.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 1491.50 | 1437.35 | 1407.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 1450.00 | 1453.05 | 1428.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 1465.40 | 1453.05 | 1428.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1452.20 | 1466.18 | 1450.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1452.20 | 1466.18 | 1450.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1460.00 | 1464.94 | 1451.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1462.60 | 1462.76 | 1453.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 1468.00 | 1463.81 | 1456.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 12:00:00 | 1462.30 | 1463.55 | 1457.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 1465.30 | 1463.90 | 1458.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1447.00 | 1461.42 | 1459.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1447.00 | 1461.42 | 1459.09 | SL hit (close<ema400) qty=1.00 sl=1459.09 alert=retest1 |

### Cycle 53 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 1448.90 | 1456.56 | 1457.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 12:15:00 | 1442.60 | 1453.77 | 1455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1452.90 | 1440.97 | 1447.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1458.00 | 1444.38 | 1448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1458.00 | 1444.38 | 1448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1416.90 | 1412.76 | 1417.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1418.40 | 1412.76 | 1417.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1411.30 | 1412.47 | 1417.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1415.60 | 1412.47 | 1417.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1415.70 | 1413.12 | 1417.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1415.70 | 1413.12 | 1417.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1412.70 | 1413.03 | 1416.67 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1423.30 | 1419.27 | 1418.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1442.00 | 1425.87 | 1422.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1421.20 | 1430.60 | 1425.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1418.80 | 1428.24 | 1425.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 1432.90 | 1429.81 | 1426.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 1408.00 | 1426.52 | 1426.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1408.00 | 1426.52 | 1426.78 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1444.00 | 1426.40 | 1426.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1458.10 | 1437.41 | 1431.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1432.60 | 1436.45 | 1431.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1443.50 | 1437.86 | 1432.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 1443.50 | 1437.86 | 1432.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1492.00 | 1455.15 | 1443.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 1525.00 | 1484.01 | 1464.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:00:00 | 1516.10 | 1504.03 | 1479.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1550.10 | 1503.72 | 1487.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:00:00 | 1518.10 | 1513.12 | 1497.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1556.00 | 1551.88 | 1532.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 1496.00 | 1524.21 | 1526.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 1496.00 | 1524.21 | 1526.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1487.30 | 1516.83 | 1523.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1498.10 | 1481.96 | 1495.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1498.00 | 1485.17 | 1495.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1506.00 | 1485.17 | 1495.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1532.70 | 1494.67 | 1498.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1532.70 | 1494.67 | 1498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1534.00 | 1502.54 | 1502.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 1548.70 | 1517.49 | 1509.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1498.50 | 1509.94 | 1510.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 1478.90 | 1497.56 | 1503.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 1433.40 | 1430.97 | 1450.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 1433.40 | 1430.97 | 1450.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1388.70 | 1370.40 | 1384.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1388.70 | 1370.40 | 1384.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1376.80 | 1371.68 | 1383.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1372.10 | 1371.68 | 1383.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1375.10 | 1374.07 | 1381.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1392.80 | 1377.82 | 1382.70 | SL hit (close>static) qty=1.00 sl=1390.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1414.00 | 1387.64 | 1386.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 1428.00 | 1404.28 | 1395.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1416.80 | 1419.73 | 1406.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 1411.80 | 1419.73 | 1406.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1395.70 | 1412.64 | 1407.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 1395.70 | 1412.64 | 1407.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1394.90 | 1409.09 | 1406.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1394.90 | 1409.09 | 1406.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 1393.10 | 1403.21 | 1403.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1332.60 | 1383.94 | 1394.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1352.90 | 1335.99 | 1353.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1352.90 | 1335.99 | 1353.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1353.10 | 1339.41 | 1353.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1353.10 | 1339.41 | 1353.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1350.10 | 1341.55 | 1353.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1358.10 | 1341.55 | 1353.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1355.00 | 1344.24 | 1353.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1379.00 | 1344.24 | 1353.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1368.60 | 1349.11 | 1354.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1369.00 | 1349.11 | 1354.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 1359.10 | 1355.97 | 1356.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:30:00 | 1358.00 | 1355.97 | 1356.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 1352.10 | 1355.19 | 1356.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:45:00 | 1349.30 | 1353.62 | 1355.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1338.00 | 1323.41 | 1323.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1338.00 | 1323.41 | 1323.08 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1299.00 | 1320.18 | 1321.89 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1333.00 | 1321.98 | 1320.89 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 1314.20 | 1318.99 | 1319.63 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1326.10 | 1320.45 | 1320.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1327.40 | 1321.84 | 1320.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1345.00 | 1323.76 | 1323.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1375.10 | 1380.66 | 1381.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1375.10 | 1380.66 | 1381.24 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1396.70 | 1383.87 | 1382.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 12:15:00 | 1407.90 | 1394.14 | 1389.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1471.10 | 1478.53 | 1458.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 1471.10 | 1478.53 | 1458.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1460.70 | 1472.87 | 1461.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 1461.40 | 1472.87 | 1461.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1460.20 | 1470.33 | 1460.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1460.00 | 1470.33 | 1460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1473.50 | 1470.97 | 1462.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:30:00 | 1456.50 | 1470.97 | 1462.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1459.00 | 1467.69 | 1462.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1455.60 | 1467.69 | 1462.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1458.80 | 1465.91 | 1462.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1450.80 | 1465.91 | 1462.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1463.00 | 1468.24 | 1465.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:30:00 | 1463.00 | 1468.24 | 1465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1460.00 | 1466.59 | 1465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1431.80 | 1466.59 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1435.20 | 1460.31 | 1462.40 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1487.00 | 1451.98 | 1447.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1494.30 | 1460.44 | 1451.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 11:15:00 | 1598.80 | 1608.62 | 1581.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 12:00:00 | 1598.80 | 1608.62 | 1581.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1580.00 | 1603.48 | 1586.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1534.80 | 1603.48 | 1586.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1571.00 | 1596.98 | 1584.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1633.10 | 1596.98 | 1584.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 14:15:00 | 1796.41 | 1700.40 | 1647.57 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 1297.80 | 2025-05-16 09:15:00 | 1354.98 | TARGET_HIT | 1.00 | 4.41% |
| SELL | retest2 | 2025-05-20 13:15:00 | 1303.80 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-05-22 09:30:00 | 1304.00 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-05-22 10:15:00 | 1300.70 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-05-22 13:30:00 | 1304.30 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1329.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-27 09:15:00 | 1326.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-28 09:30:00 | 1318.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-04 10:00:00 | 1380.30 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-06-04 10:30:00 | 1383.20 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-06-04 12:00:00 | 1382.00 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-06-04 12:30:00 | 1382.10 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1403.80 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1368.70 | 2025-06-19 12:15:00 | 1300.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 14:00:00 | 1367.20 | 2025-06-19 12:15:00 | 1298.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1368.20 | 2025-06-19 12:15:00 | 1299.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1369.00 | 2025-06-19 12:15:00 | 1300.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1368.70 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-06-11 14:00:00 | 1367.20 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1368.20 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1369.00 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1292.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-06-13 12:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1292.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 12:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-06-13 12:45:00 | 1361.90 | 2025-06-23 09:15:00 | 1293.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 1361.90 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-06-17 09:45:00 | 1360.00 | 2025-06-23 09:15:00 | 1292.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:45:00 | 1360.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-06-17 12:00:00 | 1346.70 | 2025-06-23 09:15:00 | 1279.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:00:00 | 1346.70 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest2 | 2025-06-30 12:00:00 | 1349.60 | 2025-06-30 14:15:00 | 1350.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-06-30 13:15:00 | 1349.90 | 2025-06-30 14:15:00 | 1350.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-03 09:15:00 | 1370.00 | 2025-07-04 15:15:00 | 1372.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-07-09 09:15:00 | 1315.90 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2025-07-09 14:15:00 | 1335.00 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-07-10 11:15:00 | 1335.70 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-07-11 11:45:00 | 1335.00 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1254.20 | 2025-08-20 09:15:00 | 1226.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2025-08-08 13:00:00 | 1253.90 | 2025-08-20 09:15:00 | 1226.00 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-08-25 15:15:00 | 1215.80 | 2025-09-02 09:15:00 | 1202.80 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-09-05 13:45:00 | 1213.80 | 2025-09-08 11:15:00 | 1205.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1229.90 | 2025-09-17 12:15:00 | 1242.40 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-10-01 14:30:00 | 1265.40 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-03 12:30:00 | 1266.50 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-06 09:30:00 | 1269.60 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-06 13:15:00 | 1266.20 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-07 14:15:00 | 1270.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-08 09:30:00 | 1269.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-08 11:45:00 | 1270.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1271.40 | 2025-10-20 09:15:00 | 1254.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-23 12:15:00 | 1256.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1252.60 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1443.80 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-11-04 10:30:00 | 1428.10 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-11-04 11:00:00 | 1432.40 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-11-04 11:30:00 | 1427.40 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-11-14 09:15:00 | 1395.50 | 2025-11-14 11:15:00 | 1378.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1374.80 | 2025-11-14 11:15:00 | 1378.60 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-11-17 11:15:00 | 1382.40 | 2025-11-17 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1353.90 | 2025-11-20 10:15:00 | 1385.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1339.00 | 2025-11-25 09:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-25 11:00:00 | 1344.40 | 2025-11-25 11:15:00 | 1352.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-03 09:15:00 | 1423.10 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-03 12:30:00 | 1400.00 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-12-03 13:45:00 | 1399.80 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1414.40 | 2025-12-19 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1450.00 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-31 13:15:00 | 1449.60 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-31 15:00:00 | 1451.30 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1442.90 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1498.90 | 2026-01-16 09:15:00 | 1505.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-13 15:15:00 | 1498.20 | 2026-01-19 11:15:00 | 1423.95 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1497.90 | 2026-01-19 11:15:00 | 1423.29 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1498.00 | 2026-01-19 11:15:00 | 1423.01 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1481.00 | 2026-01-19 11:15:00 | 1423.10 | PARTIAL | 0.50 | 3.91% |
| SELL | retest2 | 2026-01-16 10:30:00 | 1492.10 | 2026-01-19 12:15:00 | 1417.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 1498.20 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1497.90 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1498.00 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1481.00 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 10:30:00 | 1492.10 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.06% |
| BUY | retest1 | 2026-02-05 09:15:00 | 1465.40 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1462.60 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-09 09:45:00 | 1468.00 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-09 12:00:00 | 1462.30 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-09 13:00:00 | 1465.30 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-19 09:45:00 | 1432.90 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-25 09:15:00 | 1525.00 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-25 12:00:00 | 1516.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1550.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-02-26 13:00:00 | 1518.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1372.10 | 2026-03-17 14:15:00 | 1392.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-03-17 14:15:00 | 1375.10 | 2026-03-17 14:15:00 | 1392.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-25 14:45:00 | 1349.30 | 2026-04-01 13:15:00 | 1338.00 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1345.00 | 2026-04-13 15:15:00 | 1375.10 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1633.10 | 2026-05-04 14:15:00 | 1796.41 | TARGET_HIT | 1.00 | 10.00% |
