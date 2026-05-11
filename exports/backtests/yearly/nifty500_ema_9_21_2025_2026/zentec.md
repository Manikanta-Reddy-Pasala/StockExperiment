# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2025-12-22 09:15:00 → 2026-05-08 15:15:00 (644 bars)
- **Last close:** 1626.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 31 |
| ALERT1 | 20 |
| ALERT2 | 20 |
| ALERT2_SKIP | 14 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 20
- **Target hits / Stop hits / Partials:** 2 / 21 / 1
- **Avg / median % per leg:** -0.35% / -1.19%
- **Sum % (uncompounded):** -8.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.37% | -19.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.37% | -19.2% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 7 | 1 | 1.08% | 10.8% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 1 | 7 | 1 | 0.09% | 0.8% |
| retest1 (combined) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| retest2 (combined) | 23 | 3 | 13.0% | 1 | 21 | 1 | -0.80% | -18.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 1403.40 | 1396.64 | 1396.36 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1393.90 | 1395.82 | 1396.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1374.30 | 1391.13 | 1393.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 1388.90 | 1379.05 | 1385.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 1388.90 | 1379.05 | 1385.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1388.90 | 1379.05 | 1385.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1388.90 | 1379.05 | 1385.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1394.40 | 1382.12 | 1385.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1374.90 | 1382.12 | 1385.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1367.30 | 1357.37 | 1364.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1367.30 | 1357.37 | 1364.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1373.00 | 1360.50 | 1365.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1365.40 | 1360.50 | 1365.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1355.80 | 1356.81 | 1361.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 1361.40 | 1356.81 | 1361.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1359.60 | 1357.10 | 1360.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 1363.00 | 1357.10 | 1360.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1362.70 | 1358.22 | 1360.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 1362.70 | 1358.22 | 1360.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1362.00 | 1358.97 | 1360.73 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1384.20 | 1365.16 | 1363.05 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1351.70 | 1363.49 | 1364.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1346.20 | 1358.26 | 1361.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 1331.00 | 1330.38 | 1341.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 1306.90 | 1320.91 | 1329.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1241.56 | 1301.90 | 1317.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1247.70 | 1246.59 | 1265.84 | SL hit (close>ema200) qty=0.50 sl=1246.59 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 1322.10 | 1269.21 | 1265.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 15:15:00 | 1336.60 | 1311.06 | 1297.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 1302.00 | 1309.24 | 1297.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1295.20 | 1306.44 | 1297.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1295.20 | 1306.44 | 1297.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1288.10 | 1302.77 | 1296.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1288.10 | 1302.77 | 1296.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1284.60 | 1299.13 | 1295.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1284.60 | 1299.13 | 1295.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1291.70 | 1295.27 | 1294.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1279.20 | 1295.27 | 1294.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1294.60 | 1295.14 | 1294.27 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1279.00 | 1291.91 | 1292.88 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 1318.50 | 1296.10 | 1293.91 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 1281.30 | 1296.35 | 1297.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 1272.90 | 1291.66 | 1295.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1309.00 | 1290.02 | 1288.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1339.80 | 1303.81 | 1295.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1380.20 | 1401.50 | 1378.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1346.00 | 1390.40 | 1375.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1356.80 | 1390.40 | 1375.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1320.80 | 1376.48 | 1370.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1320.80 | 1376.48 | 1370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1317.00 | 1357.23 | 1362.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1312.90 | 1323.26 | 1332.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1341.70 | 1325.33 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1338.00 | 1327.86 | 1332.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1319.30 | 1326.33 | 1331.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 1319.90 | 1325.98 | 1331.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 10:15:00 | 1322.80 | 1328.84 | 1329.52 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 1345.10 | 1327.60 | 1327.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 13:15:00 | 1352.70 | 1339.61 | 1333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1313.60 | 1335.92 | 1333.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1315.10 | 1331.76 | 1332.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 1305.00 | 1320.12 | 1325.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 1334.90 | 1318.98 | 1322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1326.00 | 1320.39 | 1322.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1333.20 | 1320.39 | 1322.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1354.40 | 1327.19 | 1325.29 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1325.50 | 1331.52 | 1331.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1319.00 | 1329.02 | 1330.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 1366.90 | 1328.69 | 1329.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 1345.50 | 1332.05 | 1330.75 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1322.00 | 1329.75 | 1330.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 1313.10 | 1326.42 | 1328.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 1323.90 | 1318.08 | 1322.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1320.00 | 1318.46 | 1322.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1309.40 | 1318.46 | 1322.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1319.80 | 1319.61 | 1320.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1339.30 | 1326.25 | 1323.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 1346.00 | 1348.70 | 1340.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 1346.00 | 1348.70 | 1340.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1366.30 | 1355.66 | 1347.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:30:00 | 1414.80 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 10:00:00 | 1426.00 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 1414.90 | 1387.02 | 1371.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:45:00 | 1411.60 | 1400.51 | 1380.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1408.70 | 1420.36 | 1410.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1448.00 | 1420.88 | 1413.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1432.20 | 1425.10 | 1417.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 1429.80 | 1430.98 | 1424.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 1424.00 | 1428.64 | 1424.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1419.60 | 1426.83 | 1424.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1419.60 | 1426.83 | 1424.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1420.50 | 1425.57 | 1423.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1418.90 | 1425.57 | 1423.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1391.00 | 1410.05 | 1414.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1407.20 | 1364.75 | 1378.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1419.00 | 1375.60 | 1382.39 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1412.80 | 1388.87 | 1387.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 1426.00 | 1400.38 | 1393.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 1432.70 | 1440.20 | 1428.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 1432.70 | 1440.20 | 1428.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1416.80 | 1435.52 | 1427.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1416.80 | 1435.52 | 1427.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1424.40 | 1433.29 | 1427.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1439.60 | 1433.29 | 1427.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 1402.20 | 1419.34 | 1422.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1375.00 | 1367.02 | 1381.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1375.00 | 1367.02 | 1381.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1362.00 | 1367.64 | 1379.06 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 1387.00 | 1380.56 | 1380.26 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1351.40 | 1374.73 | 1377.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1344.60 | 1368.70 | 1374.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1381.90 | 1316.07 | 1334.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1329.50 | 1318.75 | 1333.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1300.90 | 1335.68 | 1337.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1362.00 | 1335.17 | 1335.22 | SL hit (close>static) qty=1.00 sl=1352.10 alert=retest2 |

### Cycle 25 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1368.30 | 1341.80 | 1338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1372.10 | 1357.52 | 1347.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1539.90 | 1542.05 | 1509.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 15:00:00 | 1539.90 | 1542.05 | 1509.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1499.30 | 1533.17 | 1510.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1524.00 | 1529.07 | 1512.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1521.00 | 1524.91 | 1513.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 1523.90 | 1525.43 | 1514.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1545.70 | 1524.34 | 1515.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1519.60 | 1530.72 | 1523.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 1519.60 | 1530.72 | 1523.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1520.30 | 1528.64 | 1523.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1529.80 | 1528.64 | 1523.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 1505.80 | 1524.07 | 1521.76 | SL hit (close<static) qty=1.00 sl=1513.40 alert=retest2 |

### Cycle 26 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1534.30 | 1522.01 | 1520.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1623.40 | 1552.33 | 1536.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 1739.00 | 1751.66 | 1718.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 1719.10 | 1745.15 | 1718.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 1729.00 | 1741.92 | 1719.12 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1672.60 | 1708.05 | 1711.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1654.40 | 1697.32 | 1706.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1697.00 | 1686.66 | 1696.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1709.90 | 1691.31 | 1697.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 1709.00 | 1691.31 | 1697.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1718.50 | 1696.75 | 1699.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1718.50 | 1696.75 | 1699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1727.00 | 1702.80 | 1701.78 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1683.90 | 1700.00 | 1702.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1680.70 | 1690.74 | 1696.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1690.00 | 1688.48 | 1693.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1659.00 | 1688.48 | 1693.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1647.90 | 1680.36 | 1689.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 1640.70 | 1680.36 | 1689.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1491.40 | 1670.22 | 1678.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 09:15:00 | 1493.10 | 1636.13 | 1662.17 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 31 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1613.70 | 1566.18 | 1563.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 1665.50 | 1616.99 | 1594.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1626.00 | 1635.56 | 1614.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:45:00 | 1623.80 | 1635.56 | 1614.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-12 09:15:00 | 1241.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-14 09:15:00 | 1247.70 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1319.30 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-06 10:30:00 | 1319.90 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1309.40 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-25 09:30:00 | 1319.80 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-04 09:30:00 | 1414.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-03-04 10:00:00 | 1426.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-04 15:15:00 | 1414.90 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-03-05 09:45:00 | 1411.60 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-03-09 14:30:00 | 1448.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-10 10:45:00 | 1432.20 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-03-11 10:15:00 | 1429.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-11 12:00:00 | 1424.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1439.60 | 2026-03-20 11:15:00 | 1404.40 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1300.90 | 2026-04-02 13:15:00 | 1362.00 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1524.00 | 2026-04-16 09:15:00 | 1505.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1521.00 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-13 14:45:00 | 1523.90 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1545.70 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1529.80 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-04-30 09:15:00 | 1659.00 | 2026-05-04 09:15:00 | 1493.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1640.70 | 2026-05-04 09:15:00 | 1476.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 09:15:00 | 1491.40 | 2026-05-07 09:15:00 | 1613.70 | STOP_HIT | 1.00 | -8.20% |
